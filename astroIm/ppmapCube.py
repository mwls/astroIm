"""
astroIm module to provide astroImage object and other useful functions

Task to feather SCUBA-2 with Planck and/or Herschel data was written in colloboration with Thomas Williams

Author: Matthew Smith
Email: matthew.smith@astro.cf.ac.uk
Date: 2018-02-28 (first development)
"""

# import modules
import numpy as np
from astropy import wcs
from astropy.io import fits as pyfits
from astropy.modeling.models import BlackBody as blackbody_nu
import astropy.constants as con
import os
import glob
import warnings
warnings.filterwarnings("ignore")
import astropy.units as u
import copy
import pickle
from astroIm import astroImage

# PPMAP cube class
class ppmapCube(object):
    
    def __init__(self, filename, ext=0, load=True, betaValues=None, sigmaCube=None, loadSig=True, sigExt=0):
        # load in the fits file
        if load:
            # load fits file
            fits = pyfits.open(filename)
            self.cube = fits[ext].data
            self.header = fits[ext].header
            fits.close()
        else:
            fits = filename
            self.cube = fits[ext].data
            self.header = fits[ext].header
        
        # if provided load sigma cube
        if sigmaCube is not None:
            if loadSig:
                # load fits file
                sigFits = pyfits.open(sigmaCube)
                self.error = sigFits[sigExt].data
                
                # check has the same dimensions as cube
                if self.cube.shape != self.error.shape:
                    raise Exception("Error cube dimensions do not match signal cube.")
                sigFits.close()
            else:
                sigFits = filename
                self.error = sigFits[sigExt].data
                
                # check has the same dimensions as cube
                if self.cube.shape != self.error.shape:
                    raise Exception("Error cube dimensions do not match signal cube.")
                
        
        # get number of temperature and beta bins
        if self.cube.ndim == 4:
            self.nTemperature = self.cube.shape[1]
            self.nBeta = self.cube.shape[0]
        else:
            self.nTemperature = self.cube.shape[0]
            self.nBeta = 1
        
        # calculate temperature of each bin
        self.temperatures = 10**(np.linspace(np.log10(self.header['TMIN']),np.log10(self.header['TMAX']),self.nTemperature)) * u.K
        
        # see if any beta information in header
        if "BETA01" in self.header:
            Bvalues = np.array([])
            for i in range(0,self.nBeta):
                headerKey = f"BETA{i+1:02d}"
                Bvalues = np.append(Bvalues, self.header[headerKey])
        else:
            if betaValues is None:
                raise Exception("Need information on beta")
            if isinstance(betaValues,float):
                if self.nBeta != 1:
                    raise Exception("Only 1 Beta value given, but multiple betas in cube")
            else:
                if len(betaValues) != self.nBeta:
                    raise Exception("Provided betas does not match shape of PPMAP cube")
                if isinstance(betaValues,list):
                    betaValues = np.array(betaValues)
                Bvalues = betaValues
        self.betas = Bvalues
        
        # get distance from header
        self.distance = self.header['DISTANCE'] * u.kpc
        
        # check image is in standard PPMAP units
        if self.header['BUNIT'] != "10^20 cm^-2":
            raise Exception("Not Programmed to handle different units")
        
        # add the correct units to the cube
        self.cube = self.cube * u.cm**-2.0
        
        # convert the cube to something more useful
        self.cube = self.cube * 1.0e20 * 2.8 * con.u  # mass per cm^-2
        self.cube = self.cube.to(u.Msun * u.pc**-2.0) # solar mass per parsec^2
        
        # have to also convert the error cube if loaded
        if hasattr(self,'error'):
             self.error = self.error * u.cm**-2.0
             self.error = self.error * 1.0e20 * 2.8 * con.u
             self.error = self.error.to(u.Msun * u.pc**-2.0)
        
    # define method to get pixel sizes 
    def getPixelScale(self):
        # function to get pixel size
        WCSinfo = wcs.WCS(self.header)
        pixSizes = wcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        if np.abs(pixSizes[0]-pixSizes[1]) > 0.0001:
            raise ValueError("PANIC - program does not cope with non-square pixels")
        self.pixSize = round(pixSizes[0], 6) * u.arcsecond
        return round(pixSizes[0], 6)
    
    # define method to mask cube to total column density above S/N threshold
    def totalSNcut(self, sigToNoise=5.0):
        if hasattr(self,'error') is False:
            raise Exception("To perform S/N cut, need to have loaded error cube")
        print("RUNNING TEST")
        # sum the column density over all temperatures and betas
        if self.cube.ndim == 4:
            totalCD = np.sum(self.cube, axis=(0,1))
        else:
            totalCD = np.sum(self.cube, axis=(0))
        
        # calculate total error
        if self.cube.ndim == 4:
            totalCDerr =  np.sqrt(np.sum(self.error**2.0, axis=(0,1)))
        else:
            totalCDerr =  np.sqrt(np.sum(self.error**2.0, axis=(0)))
        
        # find where above threshold
        sel = np.where(totalCD / totalCDerr < sigToNoise)
        
        # change slices that do not correspond to nan's
        if self.cube.ndim == 4:
            self.cube[:,:,sel[0],sel[1]] = np.nan
            self.error[:,:,sel[0],sel[1]] = np.nan
        else:
            self.cube[:,sel[0],sel[1]] = np.nan
            self.error[:,sel[0],sel[1]] = np.nan
        
        
    # define method to mask individual channels based on S/N threshold
    def channelSNcut(self, sigToNoise=5.0):
        if hasattr(self,'error') is False:
            raise Exception("To perform S/N cut, need to have loaded error cube")
        
        
        # find where above threshold
        sel = np.where(self.cube / self.error < sigToNoise)
        
        # modify values in object
        self.cube[sel] = np.nan
        self.error[sel] = np.nan
    
    
    # define function to create an artificial image
    def artificialImage(self, wavelength, tau, tauWavelength, ccVals=None):
        
        # see if found pixel size, otherwise do it now
        if hasattr(self, 'pixSize') is False:
            self.getPixelScale()
        
        # if no cc values provided pass an array of ones
        if ccVals is None:
            ccVals = np.ones((self.nBeta, self.nTemperature))
        
        # change to mass per pixel
        massCube = self.cube * (self.distance * np.tan(self.pixSize))**2.0
        
        # create emission map
        emission = np.zeros((massCube.shape[-2], massCube.shape[-1]))
        
        # convert wavlength to frequency
        frequency = con.c / wavelength
        
        # convert rest wavelength to frequency
        refFrequency = con.c / tauWavelength
        
        # create mask to see if all pixels were nan's
        mask = np.zeros(emission.shape)
        
        # loop over every beta value
        for i in range(0,self.nBeta):
            for j in range(0,self.nTemperature):
                if massCube.ndim == 4:
                    slice = massCube[i,j,:,:] * ccVals[i,j]
                else:
                    slice = massCube[j,:,:] * ccVals[i,j]
                
                # set any nan pixels to zero
                nanSel = np.where(np.isnan(slice) == True)
                nonNaNSel = np.where(np.isnan(slice) == False)
                slice[nanSel] = 0.0
                
                # add slice to total emission
                blackbody = blackbody_nu(temperature=self.temperatures[j])
                emission = emission + slice * tau * (frequency / refFrequency)**self.betas[i] *  blackbody(frequency) / self.distance**2.0 * u.sr
        
                # add if non-nan value to adjust mask
                mask[nonNaNSel] = 1
                
        
        # if all channels in slice are nan restore nan's to emission map
        maskSel = np.where(mask < 0.5)
        emission[maskSel] = np.nan
        
        # convert emission map to Jy per arcsec^2
        emission = emission.to(u.Jy) / (self.pixSize)**2.0
        
        # make new 2D header
        outHeader = self.header.copy()
        outHeader['NAXIS'] = 2
        outHeader["i_naxis"] = 2
        del(outHeader['NAXIS3'])
        if self.cube.ndim == 4:
            del(outHeader['NAXIS4'])
        # add unit to header
        outHeader['BUNIT'] = "Jy/arcsec^2"
        # add wavelength to header
        outHeader['WAVELNTH'] = (wavelength.to(u.um).value, "Wavelength in Microns")
        
        # make astro image object from 
        fitsHdu = pyfits.PrimaryHDU(emission.value, outHeader)
        fitsHduList = pyfits.HDUList([fitsHdu])
        artificialImage = astroImage(fitsHduList, load=False, instrument='PPMAP')
        
        return artificialImage


