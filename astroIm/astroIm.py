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
from astropy.stats import sigma_clipped_stats
from reproject import reproject_from_healpix, reproject_interp, reproject_exact
from photutils import make_source_mask
from astropy.convolution import convolve_fft as APconvolve_fft
from astropy.convolution import Gaussian2DKernel
from astropy.modeling.blackbody import blackbody_nu
import astropy.constants as con
import os
import glob
import warnings
warnings.filterwarnings("ignore")
import astropy.units as u
import astropy.wcs as pywcs
from astropy.coordinates import SkyCoord
from scipy import interpolate
import copy
import pickle

# define classes

# image class to make adjustments to image required
class astroImage(object):
    
    
    def __init__(self, filename, ext=0, instrument=None, band=None, unit=None, load=True, FWHM=None):
        if load:
            # load fits file
            fits = pyfits.open(filename)
            self.image = fits[ext].data
            self.header = fits[ext].header
        else:
            fits = filename
            self.image = fits[ext].data
            self.header = fits[ext].header
        
        # identify instrument
        if instrument is None:
            if 'INSTRUME' in self.header:
                self.instrument = self.header['INSTRUME']
            elif 'INSTRMNT' in self.header:
                self.instrument = self.header['INSTRMNT']
            elif ext > 0:
                try:
                    primeHeader = fits[0].header
                    if 'INSTRUME' in primeHeader:
                        self.instrument = primeHeader['INSTRUME']
                    else:
                        self.instrument = primeHeader['INSTRMNT']
                except:
                    raise ValueError("Unable to find instrument, please specify")
            else:
                raise ValueError("Unable to find instrument, please specify: ", filename)
        else:
            self.instrument = instrument
        
        # identify band
        if band is None:
            if 'FILTER' in self.header:
                self.band = self.header['FILTER']
            elif 'WAVELNTH' in self.header:
                self.band = self.header['WAVELNTH']
            elif 'WVLNGTH' in self.header:
                self.band = self.header['WVLNGTH']
            elif 'FREQ' in self.header:
                self.band = self.header['FREQ']
            elif ext > 0:
                try:
                    primeHeader = fits[0].header
                    if 'FILTER' in primeHeader:
                        self.band = primeHeader['FILTER']
                    elif 'WAVELNTH' in primeHeader:
                        self.band = primeHeader['WAVELNTH']
                    elif 'WVLNGTH' in primeHeader:
                        self.band = primeHeader['WVLNGTH']
                    else:
                        self.band = primeHeader['FREQ']
                except:
                    raise ValueError("Band not indentified please specify")
            else:
                raise ValueError("Band not indentified please specify")
        else:
            self.band = band
        
        # if PACS or SPIRE make sure band is integer
        if self.instrument == "PACS" or self.instrument == "SPIRE":
            self.band = str(int(self.band))
        
        # set unit in header if provided
        if unit is not None:
            self.header['BUNIT'] = unit
        
        # For Dustpedia files strip out the micron as not really compatible yet
        if isinstance(self.band,str):
            if self.band.count("um") > 0:
                self.band = self.band.split("um")[0]
                self.bandUnits = "um"
    
        # if SCUBA-2, see if standard SCUBA2 image with 3 dimensions, and remove it and extra headings
        if self.instrument == "SCUBA-2" and len(self.image.shape) == 3:
            self.image = self.image[0,:,:]
            self.header['NAXIS'] = 2
            self.header['i_naxis'] = 2
            
            extraHeaders = ['NAXIS3', 'LBOUND3', 'CRPIX3', 'CRVAL3', 'CTYPE3', 'CDELT3', 'CUNIT3']
            for extHeader in extraHeaders:
                try:
                    del self.header[extHeader]
                except:
                    pass
       
        # see if bunit in header, if planck add it
        if "BUNIT" not in self.header:
            if self.instrument == "Planck":
                self.header['BUNIT'] = self.header['TUNIT1']
                
        # if bunit not present but zunit is add that
        if "BUNIT" not in self.header:
            if "ZUNITS" in self.header:
                self.header['BUNIT'] = self.header["ZUNITS"]
        
        # if bunit not present but zunit is add that
        if "BUNIT" not in self.header:
            if "SIGUNIT" in self.header:
                self.header['BUNIT'] = self.header["SIGUNIT"]
        
        try:
            self.wavelength = self.standardCentralWavelengths(self.instrument, self.band)
        except:
            pass
        
        # if information provided add in FWHM information
        if FWHM is not None:
            try:
                self.fwhm = FWHM.to(u.arcsecond)
            except:
                self.fwhm = FWHM * u.arcsecond
        
        # close fits file
        if load:    
            fits.close()
    
    def getPixelScale(self):
        # function to get pixel size
        WCSinfo = wcs.WCS(self.header)
        pixSizes = wcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        if np.abs(pixSizes[0]-pixSizes[1]) > 0.0001:
            raise ValueError("PANIC - program does not cope with non-square pixels")
        self.pixSize = round(pixSizes[0], 6)
        return round(pixSizes[0], 6)
    
        
    def background_rms(self, snr=2, npixels=5, dilate_size=11, sigClip=3.0, iterations=20, maskMatch=None):
        # function to get background level and noise
        if maskMatch is None:
            mask = make_source_mask(self.image, nsigma=snr, npixels=npixels, dilate_size=dilate_size)
        else:
            mask = maskMatch
        _,median,std = sigma_clipped_stats(self.image, mask=mask, sigma=sigClip, maxiters=iterations)
        self.bkgMedian = median
        self.bkgStd = std
        return median, std, mask
    
    
    def constantBackSub(self, backConstant):
        # function to subtract a constant from the image
        
        # subtract constant from image
        self.image = self.image - backConstant
        
        return
    
    
    def ellipseAnnulusBackSub(self, backInfo, backNoise=False):
        # function to select pixels within an elliptical aperture
        
        # extract background annulus parameters
        centreRA = backInfo["centreRA"]
        centreDEC = backInfo["centreDEC"]
        innerRad = backInfo["innerRad"]
        outerRad = backInfo["outerRad"]
        PA = backInfo['PA']
        
        # see if ra and dec maps already exist, if needed run
        if hasattr(self, 'raMap') is False:
            self.coordMaps()
        
        # convert PA to radians with correct axes
        PA = (90.0-PA) / 180.0 * np.pi
        
        # adjust D25 to degrees
        innerMajorRad = innerRad[0]/(60.0)
        innerMinorRad = innerRad[1]/(60.0)
        outerMajorRad = outerRad[0]/(60.0)
        outerMinorRad = outerRad[1]/(60.0)
        
        # select pixels 
        pixelSel = np.where((((self.raMap - centreRA)*np.cos(self.decMap / 180.0 * np.pi)*np.cos(PA) + (self.decMap - centreDEC)*np.sin(PA))**2.0 / outerMajorRad**2.0 + \
                            (-(self.raMap - centreRA)*np.cos(self.decMap / 180.0 * np.pi)*np.sin(PA) + (self.decMap - centreDEC)*np.cos(PA))**2.0/ outerMinorRad**2.0 <= 1.0) &\
                            (((self.raMap - centreRA)*np.cos(self.decMap / 180.0 * np.pi)*np.cos(PA) + (self.decMap - centreDEC)*np.sin(PA))**2.0 / innerMajorRad**2.0 + \
                            (-(self.raMap - centreRA)*np.cos(self.decMap / 180.0 * np.pi)*np.sin(PA) + (self.decMap - centreDEC)*np.cos(PA))**2.0/ innerMinorRad**2.0 > 1.0) &\
                            (np.isnan(self.image) == False))
        
        # calculate the mean of the pixels and subtract from the image
        print(len(pixelSel[0]))
        self.image = self.image - self.image[pixelSel].mean()
        
        # if back noise is set, find standard deviation and return value
        if backNoise:
            noise = self.image[pixelSel].std() 
            return noise
        else:
            return
    
    def coordMaps(self):
        # function to find ra and dec co-ordinates of every pixel
        
        # Parse the WCS keywords in the primary HDU
        header = self.header
        wcs = pywcs.WCS(self.header)
        
        # Make input arrays for every pixel on the map
        xpix = np.zeros((header["NAXIS1"]*header["NAXIS2"]),dtype=int)
        for i in range(0,header["NAXIS2"]):
            xpix[i*header["NAXIS1"]:(i+1)*header["NAXIS1"]] = np.arange(0,header["NAXIS1"],1)
        ypix = np.zeros((header["NAXIS1"]*header["NAXIS2"]),dtype=int)
        for i in range(1,header["NAXIS2"]):
            ypix[(i)*header["NAXIS1"]:(i+1)*header["NAXIS1"]] = i
        
        # Convert all pixels into sky co-ordinates
        sky = wcs.wcs_pix2world(xpix,ypix, 0)
        if wcs.world_axis_physical_types[0].count('galactic') > 0:
            gc = SkyCoord(l=sky[0]*u.degree, b =sky[1]*u.degree,frame='galactic')
            icrs = gc.transform_to('icrs')
            raMap = icrs.ra.value
            decMap = icrs.dec.value
        else:
            raMap = sky[0]
            decMap = sky[1]
    
        # Change shape so dimensions and positions match or the stored image (ie python image y,x co-ordinates)
        raMap = raMap.reshape(header["NAXIS2"],header["NAXIS1"])
        decMap = decMap.reshape(header["NAXIS2"],header["NAXIS1"])
        xpix = xpix.reshape(raMap.shape)
        ypix = ypix.reshape(decMap.shape)
        
        # see if all raMap is negative
        if raMap.max() < 0.0:
            raMap = raMap + 360.0
        
        # raise exception if ra crosses the zero line
        if raMap.min() < 0.0:
            raise Exception("Not programmed to deal with ra that crosses ra=0")
        
        # return two maps
        self.raMap = raMap
        self.decMap = decMap
        
        return
    
    def standardBeamAreas(self, instrument=None, band=None):
        # define standard beam areas
        beamAreas = {"SCUBA-2":{"450":141.713, "850":246.729}, "SPIRE":{"250":469.4, "350":831.3, "500":1804.3},\
                     "Planck":{"353":96170.4}, "SCUBA-2&Planck":{"850":246.729}, "SCUBA-2&SPIRE":{"450":141.713}}
        if instrument is not None:
            return beamAreas[instrument][band]
        else:
            return beamAreas
    
    def standardCentralWavelengths(self, instrument=None, band=None):
        # define central wavelengths for bands in micron
        centralWavelengths = {"SCUBA-2":{"450":450.0, "850":850.0}, "SPIRE":{"250":250.0, "350":350.0, "500":500.0},\
                              "Planck":{"353":850.0}}
        if instrument is not None:
            return centralWavelengths[instrument][band]
        else:
            return centralWavelengths
    
    def standardFWHM(self, instrument=None, band=None):
        # define central wavelengths for bands in micron
        FWHMs = {"SCUBA-2":{"450":7.9*u.arcsecond, "850":13.0*u.arcsecond},\
                 "SPIRE":{"250":17.6*u.arcsecond, "350":23.9*u.arcsecond, "500":35.2*u.arcsecond},\
                 "Planck":{"353":289.08*u.arcsecond}}
        if instrument is not None:
            return FWHMs[instrument][band]
        else:
            return FWHMs
    
    def convertUnits(self, newUnit, conversion=None, beamArea=None):
        # function to convert units of map
        
        # if a conversion value given use that, if not calculate
        if conversion is not None:
            self.image = self.image * conversion
            self.header['BUNIT'] = newUnit
            if "SIGUNIT" in self.header:
                self.header['SIGUNIT'] = newUnit
            if "ZUNITS" in self.header:
                self.header['ZUNITS'] = newUnit
            print(self.band, " image converted to ", newUnit, " using provided conversion")
        else:
            
            # check if unit is programmed
            units = ["pW", "Jy/arcsec^2", "mJy/arcsec^2", "mJy/arcsec**2", "MJy/sr", "Jy/beam", "mJy/beam", "Jy/pix", "K_CMB"]
            oldUnit = self.header["BUNIT"]
                        
            # programmed beam areas
            beamAreas = self.standardBeamAreas()
            if beamArea is not None:
                beamAreas[self.instrument][self.band] = beamArea
            
            # program conversion SCUBA-2 pW to Jy/arcsec^2
            scubaConversions = {"450":{"Jy/beam":497.6, "Jy/arcsec^2":3.51}, "850":{"Jy/beam":480.5, "Jy/arcsec^2":1.95}}
            
            # program to convert planck to MJy/sr
            planckConversion = {"353":287.450}
            
            if oldUnit == newUnit:
                # check that not already in correct unit
                print("Image is already in correct units")
            else:
                # see if in a pre-progammed unit
                if oldUnit not in units:
                    print("Image Unit: ", oldUnit, " not programmed - result maybe unreliable")
                if newUnit not in units:
                    print("Image Unit: ", newUnit, " not programmed - result maybe unreliable")
                
                # check if SCUBA-2 instrument units of pW and if so convert first to Jy/arcsec^2
                if self.instrument == "SCUBA-2" and self.header['BUNIT'] == 'pW':
                    if newUnit == 'Jy/beam':
                        self.image = self.image * scubaConversions[self.band]['Jy/beam']
                        self.header['BUNIT'] = 'Jy/beam'
                        oldUnit = 'Jy/beam'
                    else:
                        self.image = self.image * scubaConversions['Jy/arcsec^2']
                        self.header['BUNIT'] = 'Jy/arcsec^2'
                        oldUnit = 'Jy/arcsec^2'
                    if oldUnit == newUnit:
                        print("Image converted to ", newUnit)
                        return
                elif self.header['BUNIT'] == 'pW':
                    raise ValueError("Can only process pW from SCUBA-2")
                
                # check if Planck instruments in unit of K_CMB
                if self.instrument == "Planck" and self.header['BUNIT'] == "K_CMB":
                    self.image = self.image * planckConversion[self.band]
                    self.header['BUNIT'] = "MJy/sr"
                    oldUnit = "MJy/sr"
                elif self.header['BUNIT'] == "K_CMB":
                    raise ValueError("Can only process K_CMB from Planck")
                
                ### process the old units
                if oldUnit == "Jy/pix":
                    conversion = 1.0 * u.Jy
                    pixArea = self.pixSize * self.pixSize * u.arcsecond**2.0
                    conversion = conversion / pixArea
                elif oldUnit == "Jy/beam":
                    conversion = 1.0 * u.Jy
                    #pixArea = self.pixSize * self.pixSize
                    #conversion = conversion * pixArea / beamAreas[self.instrument][self.band]
                    conversion = conversion / (beamAreas[self.instrument][self.band] * u.arcsecond**2.0)
                elif oldUnit == "mJy/beam":
                    conversion = 0.001 * u.Jy
                    #pixArea = self.pixSize * self.pixSize
                    #conversion = conversion * pixArea / beamAreas[self.instrument][self.band]
                    conversion = conversion / (beamAreas[self.instrument][self.band] * u.arcsecond**2.0)
                elif oldUnit == "Jy/arcsec^2":
                    conversion = 1.0 * u.Jy / u.arcsecond**2.0
                elif oldUnit == "mJy/arcsec^2" or oldUnit == "mJy/arcsec**2":
                    conversion = 0.001 * u.Jy / u.arcsecond**2.0
                elif oldUnit == "MJy/sr":
                    conversion = 1.0e6 * u.Jy / u.sr
                else:
                    raise ValueError("Unit not programmed: ", oldUnit)
                                
                # convert to new unit
                if newUnit == "Jy/pix" or newUnit == "Jy/beam" or newUnit == "mJy/beam":
                    # convert to Jy per arcsec^2
                    conversion = conversion.to(u.Jy/u.arcsecond**2.0).value
                    if newUnit == "Jy/pix":
                        conversion = conversion * self.pixSize * self.pixSize
                    elif newUnit == "Jy/beam":
                        conversion = conversion * beamAreas[self.instrument][self.band]
                    elif newUnit == "mJy/beam":
                        conversion = conversion * beamAreas[self.instrument][self.band] * 1000.0
                elif newUnit == "Jy/arcsec^2":
                    conversion = conversion.to(u.Jy/u.arcsecond**2.0).value
                elif newUnit == "mJy/arcsec^2" or newUnit == "mJy/arcsec**2":
                    conversion = conversion.to(u.Jy/u.arcsecond**2.0).value * 1000.0
                elif newUnit == "MJy/sr":
                    conversion = conversion.to(u.Jy/u.sr).value * 1.0e-6
                elif newUnit == "pW" and self.instrument == "SCUBA-2":
                    conversion = conversion * beamAreas[self.instrument][self.band]
                    conversion = conversion / scubaConversions[self.band]['Jy/beam']
                else:
                    raise ValueError("Unit not programmed")
                
                self.image = self.image * conversion
                self.header['BUNIT'] = newUnit
                if "SIGUNIT" in self.header:
                        self.header['SIGUNIT'] = newUnit
                if "ZUNITS" in self.header:
                        self.header['ZUNITS'] = newUnit
                print("Image converted to: ", newUnit)
    
    def centralWaveAdjust(self, newWavelength, adjustSettings):
        # function to adjust for difference in central wavelengths
        print("Performing Central Wavelength Adjustment")
        
        # get current central wavelength
        currentWavelength = self.standardCentralWavelengths(instrument=self.instrument, band=self.band)
        
        
        # see if have a PPMAP cube
        if "ppmapCube" in adjustSettings:
            if "ppmapCubeErr" in adjustSettings:
                # load PPMAP cube 
                ppMap = ppmapCube(adjustSettings["ppmapCube"], sigmaCube=adjustSettings["ppmapCubeErr"])
            else:
                # load PPMAP cube 
                ppMap = ppmapCube(adjustSettings["ppmapCube"])
            
            if "applySNcut" in adjustSettings and adjustSettings["applySNcut"] is False:
                pass
            else:
                if hasattr(ppMap,"error"):
                    # apply signal-to-noise cut
                    if "sigCut" in adjustSettings:
                        sigCut =  adjustSettings["sigCut"]
                    else:
                        sigCut = 5.0
                    #ppMap.totalSNcut(sigToNoise=sigCut)
                    ppMap.channelSNcut(sigToNoise=sigCut)
            
            
            # create artficial ppmap image at both new and old wavelength
            predictedNewWave = ppMap.artificialImage(newWavelength*u.um, adjustSettings["tau"], adjustSettings["tauWavelength"])
            predictedCurrWave = ppMap.artificialImage(self.wavelength*u.um, adjustSettings["tau"], adjustSettings["tauWavelength"])
            
            # set variable that using a map based (rather than a constant across whole image
            mapMethod = True
        # see if the case of using a constant correction across entire image
        elif adjustSettings["temperature"] is not None and isinstance(adjustSettings["temperature"],str) is False and adjustSettings["beta"] is not None and isinstance(adjustSettings["beta"],str) is False:
            # if constant just compare what a blackbody would be before and after
            newLevel = (con.c/(newWavelength*u.um))**adjustSettings["beta"] * blackbody_nu(newWavelength*u.um, adjustSettings["temperature"]*u.K)
            currLevel = (con.c/(currentWavelength*u.um))**adjustSettings["beta"] * blackbody_nu(currentWavelength*u.um, adjustSettings["temperature"]*u.K)
            factor = (newLevel / currLevel).value
            mapMethod = False
        else:
            # see for case where have either a temperature or beta map
            raise Exception("Temperature/Beta map not Programmed Yet")
        
        # if map method have to do further processing
        if mapMethod:
            ## smooth the data to match the resolution of the image
            # get the image FWHM
            if "imageFWHM" in adjustSettings:
                imageFWHM = adjustSettings['imageFWHM']
            elif hasattr(self,'fwhm'):
                imageFWHM = self.fwhm
            else:
                # see if low res in our standard FWHM
                imageFWHM = self.standardFWHM(instrument=self.instrument, band=self.band)
            
            # get the reference data FWHM
            refFWHM =  adjustSettings['refFWHM']
            
            # perform convolution if image lower resolution than reference information
            if imageFWHM > refFWHM:
                # create kernel ant do convolution
                predictedNewWave.getPixelScale()
                kernel = ((np.sqrt(imageFWHM**2.0 - refFWHM**2.0)).to(u.arcsecond).value) / (2.35482 * predictedNewWave.pixSize)
                convolvedNewWave = predictedNewWave.convolve(kernel, boundary=['extend'])
                convolvedCurrWave = predictedCurrWave.convolve(kernel, boundary=['extend'])
                
                ratioMap = copy.deepcopy(convolvedNewWave)
                ratioMap.image = convolvedNewWave.image / convolvedCurrWave.image
                
            else:
                # create ratio map of the two
                ratioMap = copy.deepcopy(predictedNewWave)
                ratioMap.image = predictedNewWave.image / predictedCurrWave.image
            
            # get median ratio for outer boundaries later on
            medianRatio = np.nanmedian(ratioMap.image)
                        
            # fill in nan gaps by interpolation
            maskedRatio = np.ma.masked_invalid(ratioMap.image)
            xx, yy = np.meshgrid(np.arange(0,maskedRatio.shape[1]), np.arange(0,maskedRatio.shape[0]))
            x1 = xx[~maskedRatio.mask]
            y1 = yy[~maskedRatio.mask]
            newValues = maskedRatio[~maskedRatio.mask]
            ratioMap.image = interpolate.griddata((x1,y1), newValues.ravel(), (xx,yy), method='linear')
            
            
            # check no values above or below previous max/min in interpolation
            if ratioMap.image.max() > np.nanmax(maskedRatio):
                sel = np.where(ratioMap.image > np.nanmax(maskedRatio))
                ratioMap.image[sel] = np.nanmax(maskedRatio)
            if ratioMap.image.min() < np.nanmin(maskedRatio):
                sel = np.where(ratioMap.image < np.nanmin(maskedRatio))
                ratioMap.image[sel] = np.nanmin(maskedRatio)
            
            # reproject ratio map to match input image
            ratioMap.reproject(self.header, exact=False)
            
            # replace nan's caused by no coverage to nan value
            nanPos = np.where(np.isnan(ratioMap.image) == True)
            ratioMap.image[nanPos] = medianRatio
            
            self.image = self.image * ratioMap.image
            if hasattr(self,"error"):
                self.error = self.error * ratioMap.image
        else:
            self.image = self.image * factor
            if hasattr(self,"error"):
                self.error = self.error * factor
        
    
    def ccAdjuster(self, adjustSettings, ccValues, saveCCinfo=False):
        # function to adjust image for colour corrections
        print("Performing Colour Correction Adjustment")
        
        # define function that gets cc value for beta/temperature combination
        def ccValueFind(temperature, beta, ccInfo):
            Tgrid = ccInfo["temperatures"]
            Bgrid = ccInfo["betas"]
            ccvalues = ccInfo["ccValues"]
            
            if "gridInfo" in ccInfo:
                gridInfo = ccInfo["gridInfo"]
            else:
                gridInfo = None
            
            if gridInfo is None:
                # find index of closest Temperature
                indexT = np.where(Tgrid-temperature > 0)[0]
                
                # find index of closest Beta
                indexB = np.where(Bgrid-beta > 0)[0]
                
                # change the index values if out of range
                if len(indexT) == 0:
                    indexT = -2
                elif indexT[0] == 0:
                    indexT = 0
                else:
                    indexT = indexT[0] - 1
                if len(indexB) == 0:
                    indexB = -2
                elif indexB[0] == 0:
                    indexB = 0
                else:
                    indexB = indexB[0] - 1
                
            else:
                # find index of closest Temperature
                #indexT = np.int(np.floor((temperature-gridInfo['T']['start'])/gridInfo['T']['step']))
                indexT = np.int((temperature-gridInfo['T']['start'])/gridInfo['T']['step'])
                
                # find index of closest Beta
                indexB = np.int((beta-gridInfo['B']['start'])/gridInfo['B']['end'])
                # change the index values if out of range
                if indexT < 0:
                    indexT = 0
                elif indexT >= len(Tgrid) - 1:
                    indexT = -2
                
                if indexB < 0:
                    indexB = 0
                elif indexB >= len(Bgrid) - 1:
                    indexB = -2
           
            # iterpolate along T-axis first
            ccStep = (ccvalues[indexB, indexT+1] - ccvalues[indexB, indexT])/(Tgrid[indexT+1]-Tgrid[indexT]) * (temperature-Tgrid[indexT]) + ccvalues[indexB, indexT]
            ccValue = (ccvalues[indexB+1, indexT] - ccvalues[indexB, indexT])/(Bgrid[indexB+1]-Bgrid[indexB]) * (beta-Bgrid[indexB]) + ccStep
        
            return ccValue
        
        
        
        # see if have a PPMAP cube
        if "ppmapCube" in adjustSettings:
            if "ppmapCubeErr" in adjustSettings:
                # load PPMAP cube 
                ppMap = ppmapCube(adjustSettings["ppmapCube"], sigmaCube=adjustSettings["ppmapCubeErr"])
            else:
                # load PPMAP cube 
                ppMap = ppmapCube(adjustSettings["ppmapCube"])
            
            if "applySNcut" in adjustSettings and adjustSettings["applySNcut"] is False:
                pass
            else:
                if hasattr(ppMap,"error"):
                    # apply signal-to-noise cut
                    if "sigCut" in adjustSettings:
                        sigCut =  adjustSettings["sigCut"]
                    else:
                        sigCut = 5.0
                    #ppMap.totalSNcut(sigToNoise=sigCut)
                    ppMap.channelSNcut(sigToNoise=sigCut)
            
            # loop over each temperature/beta value and get colour-correction
            ccPPMAPvals = np.ones((ppMap.nBeta,ppMap.nTemperature))
            for i in range(0,ppMap.nBeta):
                for j in range(0,ppMap.nTemperature):
                    ccPPMAPvals[i,j] = ccValueFind(ppMap.temperatures[j].to(u.K).value, ppMap.betas[i], ccValues)
            
            # create artficial ppmap image both with and without colour corrections
            predictedMapWithCC = ppMap.artificialImage(self.wavelength*u.um, adjustSettings["tau"], adjustSettings["tauWavelength"],ccVals=ccPPMAPvals)
            predictedMapNoCC = ppMap.artificialImage(self.wavelength*u.um, adjustSettings["tau"], adjustSettings["tauWavelength"])
            
                        
            # set variable that using a map based (rather than a constant across whole image
            mapMethod = True
        # see if the case of using a constant correction across entire image
        elif adjustSettings["temperature"] is not None and isinstance(adjustSettings["temperature"],str) is False and adjustSettings["beta"] is not None and isinstance(adjustSettings["beta"],str) is False:
            # if constant just look up ccValue
            ccFactor = ccValueFind(adjustSettings["temperature"], adjustSettings["beta"], ccValues)
            
            mapMethod = False
        else:
            # see for case where have either a temperature or beta map
            raise Exception("Temperature/Beta map not Programmed Yet")
        
        # if map method have to do further processing
        if mapMethod:
            ## smooth the data to match the resolution of the image
            # get the image FWHM
            if "imageFWHM" in adjustSettings:
                imageFWHM = adjustSettings['imageFWHM']
            elif hasattr(self,'fwhm'):
                imageFWHM = self.fwhm
            else:
                # see if low res in our standard FWHM
                imageFWHM = self.standardFWHM(instrument=self.instrument, band=self.band)
            
            # get the reference data FWHM
            refFWHM =  adjustSettings['refFWHM']
            
            # perform convolution if image lower resolution than reference information
            if imageFWHM > refFWHM:
                # create kernel ant do convolution
                predictedMapWithCC.getPixelScale()
                kernel = ((np.sqrt(imageFWHM**2.0 - refFWHM**2.0)).to(u.arcsecond).value) / (2.35482 * predictedMapWithCC.pixSize)
                convolvedCCMapImage = predictedMapWithCC.convolve(kernel, boundary=['extend'])
                convolvedNoCCMapImage = predictedMapNoCC.convolve(kernel, boundary=['extend'])
                
            
                # create ratio map of the two
                ccMap = copy.deepcopy(convolvedCCMapImage)
                ccMap.image = convolvedCCMapImage.image / convolvedNoCCMapImage.image
            else:
                ccMap = copy.deepcopy(predictedMapWithCC)
                ccMap.image = predictedMapWithCC.image / predictedMapNoCC.image
            
            # get median ratio for outer boundaries later on
            medianCC = np.nanmedian(ccMap.image)
            
                            
            # fill in nan gaps by interpolation
            maskedRatio = np.ma.masked_invalid(ccMap.image)
            xx, yy = np.meshgrid(np.arange(0,maskedRatio.shape[1]), np.arange(0,maskedRatio.shape[0]))
            x1 = xx[~maskedRatio.mask]
            y1 = yy[~maskedRatio.mask]
            newValues = maskedRatio[~maskedRatio.mask]
            ccMap.image = interpolate.griddata((x1,y1), newValues.ravel(), (xx,yy), method='linear')
            
            
            # check no values above or below previous max/min in interpolation
            if ccMap.image.max() > np.nanmax(maskedRatio):
                sel = np.where(ccMap.image > np.nanmax(maskedRatio))
                ccMap.image[sel] = np.nanmax(maskedRatio)
            if ccMap.image.min() < np.nanmin(maskedRatio):
                sel = np.where(ccMap.image < np.nanmin(maskedRatio))
                ccMap.image[sel] = np.nanmin(maskedRatio)
            
            # reproject ratio map to match input image
            ccMap.reproject(self.header, exact=False)
            
            # replace nan's caused by no coverage to median value
            nanPos = np.where(np.isnan(ccMap.image) == True)
            ccMap.image[nanPos] = medianCC
            
            self.image = self.image * ccMap.image
            if hasattr(self,"error"):
                self.error = self.error * ratioMap.image
            
            if saveCCinfo:
                self.ccData = ccMap.image
            
        else:
            self.image = self.image * ccFactor
            if hasattr(self,"error"):
                self.error = self.error * ccFactor
    
            if saveCCinfo:
                self.ccData = ccFactor
    
    
    def restoreDefaultCC(self):
        # function to restore the image to default colour-corrections
        
        # update image
        self.image = self.image / self.ccData
        
        # update error
        if hasattr(self,"error"):
            self.error = self.error / self.ccData
    
    
    def reproject(self, projHead, exact=True):
        # function to reproject the fits image
        
        # create new hdu
        if "PIXTYPE" in self.header and self.header["PIXTYPE"] == "HEALPIX":
            #hdu = pyfits.hdu.table._TableLikeHDU(self.image, self.header)
            hdu = pyfits.hdu.table.BinTableHDU(self.image, self.header)
        else:
            hdu = pyfits.PrimaryHDU(self.image, self.header)
        
        # see if a healpix image
        if "PIXTYPE" in self.header and self.header["PIXTYPE"] == "HEALPIX":
            resampleMap,_ = reproject_from_healpix(hdu, projHead)
        else:
            if exact:
                resampleMap, _ = reproject_exact(hdu, projHead)
            else:
                resampleMap, _ = reproject_interp(hdu, projHead)
        
        self.image = resampleMap
        
        # modify original header
        # projection keywords
        projKeywords = ["NAXIS1", "NAXIS2", "LBOUND1", "LBOUND2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",\
                        "CTYPE1", "CTYPE2", "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",\
                        "RADESYS", "EQUINOX", "CROTA2", "CROTA1"]
        header = self.header
        for keyword in projKeywords:
            if keyword in projHead:
                header[keyword] = projHead[keyword]
            else:
                try:
                    del(header[keyword])
                except:
                    pass
        
        # save new header
        self.header = header
            
    def imageManipulation(self, operation, value):
        # function to manipulate fits file
        
        # see if a 2D map, or single value
        if isinstance(value, np.ndarray):
            if value.shape != self.image.shape:
                raise ValueError("Image do not have the same shape")
        
        if operation == "+":
            self.image = self.image + value
        elif operation == "-":
            self.image = self.image - value
        elif operation == "*":
            self.image = self.image * value
        elif operation == "/":
            self.image = self.image / value
        elif operation == "**":
            self.image = self.image ** value
        else:
            raise ValueError("Operation not programmed")
    
    
    def convolve(self, kernel, boundary='extend', fill_value=0.0):
        
        # see if kernel is a number or an array
        if isinstance(kernel,type(float())):
            kernelImage = Gaussian2DKernel(x_stddev = kernel)
            kernelImage = kernelImage.array
        else:
            kernelImage = kernel
            
        # renormalise so peak is one
        kernelImage = kernelImage / kernelImage.max()
        
        # find positions that are NaNs
        NaNsel = np.where(np.isnan(self.image) == True)
        
        if boundary == 'fill':
            convolvedArray = APconvolve_fft(self.image, kernelImage, boundary=boundary, fill_value=fill_value, allow_huge=True)
        else:
            convolvedArray = APconvolve_fft(self.image, kernelImage, boundary=boundary, allow_huge=True)
        
        # restore NaNs
        convolvedArray[NaNsel] = np.nan
        
        # create combined image hdu
        convolveHeader = self.header
        convolveHdu = pyfits.PrimaryHDU(convolvedArray, convolveHeader)
        convolveHdulist = pyfits.HDUList([convolveHdu])
        
        # create combine astro image
        convolvedImage = astroImage(convolveHdulist, load=False, instrument=self.instrument, band=self.band)
        
        return convolvedImage
    
        
    def imageFFTcombine(self, lowresImage, filterScale=None, beamArea=None, filterType="gauss", butterworthOrder=None, sigmoidScaling=None, beamMatchedMode=True):
        # function to combine this image with another
        
        # check that this is an allowed combination
        
        # # programmed beam areas
        beamAreas = self.standardBeamAreas()
        if beamArea is not None:
            for instrument in beamArea.keys():
                for band in beamArea[instrument].keys():
                    if instrument in beamAreas:
                        beamAreas[instrument][band] = beamArea[instrument][band]
                    else:
                        beamAreas[instrument] = {band:beamArea[instrument][band]}
        
        # get the two images
        hires = self.image
        lowres = lowresImage.image
        
        # subtract background from both
        hires = hires  - self.bkgMedian
        lowres = lowres - lowresImage.bkgMedian
        
        # see if either have NaNs
        NaNmask = np.where( (np.isnan(lowres) == True) | (np.isnan(hires) == True) )
        lowres[np.isnan(lowres) == True] = 0
        hires[np.isnan(hires) == True] = 0
        
        # create radius in arcsecond from centre for all pixels
        x_centre,y_centre = hires.shape[0]/2.0,hires.shape[1]/2.0
        x,y = np.meshgrid(np.linspace(-x_centre,x_centre,hires.shape[0]), 
                           np.linspace(-y_centre,y_centre,hires.shape[1]))
        
        d = np.sqrt(x*x+y*y)
        d = np.transpose(d)
        d *= self.pixSize
        
        # Calculate the frequencies in the Fourier plane to create a filter
        x_f,y_f = np.meshgrid(np.fft.fftfreq(hires.shape[0],self.pixSize),
                              np.fft.fftfreq(hires.shape[1],self.pixSize))
        #d_f = np.sqrt(x_f**2 + y_f**2) *2.0#Factor of 2 due to Nyquist sampling
        d_f = np.sqrt(x_f**2 + y_f**2)
        d_f = np.transpose(d_f)
       
        
        # create filter scale
        if filterScale is None:
            if self.instrument == "SCUBA-2":
                if self.band == "450":
                    filterScale = 36
                elif self.band == "850":
                    filterScale = 480
            else:
                raise ValueError("Filter Scale needs to be defined")
        
        # create filter
        if filterType == "butterworth":
            d_f = d_f**-1
            if butterworthOrder is None:
                butterworthOrder = 4.0
            
            # Create a butterworth filter
            filter = (np.sqrt(1.0+(d_f/filterScale)**(2.0*butterworthOrder)))**-1.0
        elif filterType == "gauss":
            # Create a Gaussian given the filter scale, taking into account pixel scale.
            filter_scale = np.float(filterScale)
            filter_std = filter_scale / (2.0*np.sqrt(2.0*np.log(2.0)))
            filter = np.exp(-( (d_f*2.0*np.pi)**2.0 * filter_std**2.0 / 2.0))
            #filter = np.exp(-(d)**2.0 / (2.0*filter_std**2.0))
        elif filterType == "sigmoid":
            d_f = d_f**-1
            if sigmoidScaling is None:
                sigmoidScaling = 1.0
            filter_scale = np.float(filterScale)
            filter = 1.0 - 1.0 / (1.0 + np.exp(-1.0*(d_f - filter_scale)/sigmoidScaling))
        else:
            raise Exception("Must specify combination type")
        
        # Force in the amplitude at (0,0) since d_f here is undefined
        filter[0,0] = 0
        
        # Fourier transform all these things
        filter_fourier = np.fft.fftshift(filter)
        #filter_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(filter)))
        filter_fourier /= np.nanmax(filter_fourier)
        hires_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(hires)))
        lowres_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(lowres)))
        print('Fourier transforms complete')
        
        # Calculate the volume ratio (high to low res)
        ratio = beamAreas[self.instrument][self.band] / beamAreas[lowresImage.instrument][lowresImage.band]
        lowres_fourier *= ratio
        
        # Weight image the based on the filter
        if filterType == "gauss":
            hires_fourier_weighted = hires_fourier * (1.0-filter_fourier)
            if beamMatchedMode:
                lowres_fourier_weighted = lowres_fourier 
            else:
                lowres_fourier_weighted = lowres_fourier * filter_fourier
        else:
            hires_fourier_weighted = hires_fourier * (1.0-filter_fourier)
            lowres_fourier_weighted = lowres_fourier *filter_fourier
        #hires_fourier_weighted = hires_fourier * filter_fourier
        #lowres_fourier_weighted = lowres_fourier * (1.0-filter_fourier)
        #hires_fourier_weighted = hires_fourier * (1.0-filter_fourier)
        #lowres_fourier_weighted = lowres_fourier *filter_fourier
        #lowres_fourier_weighted = lowres_fourier
        
        combined_fourier=hires_fourier_weighted+lowres_fourier_weighted
        
        combined_fourier_shift = np.fft.ifftshift(combined_fourier)
        combined = np.fft.fftshift(np.real(np.fft.ifft2(combined_fourier_shift)))
        
        print('Data combined')
        
        # restore nans
        combined[NaNmask] = np.nan
        
        # add background back to image
        combined = combined + lowresImage.bkgMedian
        print('Background restored to image')
        
        # create combined image hdu
        combineHeader = self.header
        combineHeader['INSTRUME'] = self.instrument + '&' + lowresImage.instrument
        combineHdu = pyfits.PrimaryHDU(combined, combineHeader)
        combineHdulist = pyfits.HDUList([combineHdu])
        
        # create combine astro image
        try:
            combineImage = astroImage(combineHdulist, load=False)
        except:
            combineImage = astroImage(combineHdulist, load=False, band=self.band)
        
        # copy attributes from high-res image if available
        if hasattr(self,"fwhm"):
            combineImage.fwhm = self.fwhm
        if hasattr(self,"ccData"):
            combineImage.ccData = self.ccData
        
        return combineImage
    
    def saveToFits(self, outPath, overwrite=False):
        # function to save to fits
        
        fitsHdu = pyfits.PrimaryHDU(self.image, self.header)
        fitsHduList = pyfits.HDUList([fitsHdu])
        
        fitsHduList.writeto(outPath, overwrite=overwrite)

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
        self.pixSize = round(pixSizes[0], 6)
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
        massCube = self.cube * (self.distance * np.tan(self.pixSize*u.arcsecond))**2.0
        
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
                emission = emission + slice * tau * (frequency / refFrequency)**self.betas[i] *  blackbody_nu(frequency, self.temperatures[j]) / self.distance**2.0 * u.sr
        
                # add if non-nan value to adjust mask
                mask[nonNaNSel] = 1
                
        
        # if all channels in slice are nan restore nan's to emission map
        maskSel = np.where(mask < 0.5)
        emission[maskSel] = np.nan
        
        # convert emission map to Jy per arcsec^2
        emission = emission.to(u.Jy) / (self.pixSize*u.arcsecond)**2.0
        
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

# create function which loads in colour-corrections
def loadColourCorrect(colFile, SPIREtype):
    # function to load in polynomial colour correction information
    
    # check in SPIRE type only one value set to True
    if np.array(list(SPIREtype.values())).sum() != 1:
        raise Exception("Can only set one SPIRE cc type")
    
    # load in colour correct data
    filein = open(colFile, 'rb')
    ccinfo = pickle.load(filein)
    filein.close()
    
    # have to choose required SPIRE colour corrections
    ccType = [i for i in SPIREtype if SPIREtype[i] is True][0]
    
    # move appropiate SPIRE values to root of dictionary then pop SPIRE
    for key in ccinfo["SPIRE"][ccType].keys():
        ccinfo[key] = ccinfo["SPIRE"][ccType][key]
    ccinfo.pop("SPIRE")
    
    # loop over all ccInfo keys:
    newCCinfo = {}
    planckConvert = {"350":"857", "550":"545", "850":"353", "1382":"217", "2100":"143", "3000":"100"}
    for key in ccinfo.keys():
        if key[0:4] == 'PACS' or key[0:4] == 'IRAS' or key[0:4] == 'MIPS':
            if key[0:4] not in newCCinfo:
                newCCinfo[key[0:4]] = {} 
            newCCinfo[key[0:4]][key[4:]] = ccinfo[key]
        elif key[0:5] == 'SPIRE':
            if key[0:5] not in newCCinfo:
                newCCinfo[key[0:5]] = {} 
            newCCinfo[key[0:5]][key[5:]] = ccinfo[key]
        elif key[0:5] == 'SCUBA':
            if 'SCUBA-2' not in newCCinfo:
                newCCinfo['SCUBA-2'] = {} 
            newCCinfo['SCUBA-2'][key[5:]] = ccinfo[key]
        elif key[0:6] == "Planck":
            if "Planck" not in newCCinfo:
                newCCinfo["Planck"] = {}
            newCCinfo["Planck"][planckConvert[key[6:]]] = ccinfo[key]
        else:
            raise "Instrument/band not programmed for cc load"

    
    # return colour correction information
    return newCCinfo
            