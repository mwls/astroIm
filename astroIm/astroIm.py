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
from scipy import interpolate
import copy
import pickle

# define classes

# image class to make adjustments to image required
class astroImage(object):
    
    
    def __init__(self, filename, ext=0, telescope=None, instrument=None, band=None, unit=None, load=True, FWHM=None, slices=None, dustpediaHeaderCorrect=False, verbose=True):
        if load:
            # load fits file
            fits = pyfits.open(filename)
            self.image = fits[ext].data
            self.header = fits[ext].header
        else:
            fits = filename
            self.image = fits[ext].data
            self.header = fits[ext].header
        
        # see if more than two dimensions are prsent
        if self.image.ndim > 2:
            
            # check if only two axes have more than size = 1
            if len((np.where(np.array(self.image.shape) > 1))[0]) > 2:
                # if slices is not defined raise an exception
                if slices is None:
                    raise Exception("To use astroImage class must indicate what slices to use for data with more than 2D data")
                
                # see what oder slices is given in, and if using zero index
                if "fitsAxisOrder" in slices:
                    fitsOrder = slices["fitsAxisOrder"]
                else:
                    fitsOrder = True
                if "zeroIndex" in slices:
                    if slices['zeroIndex']:
                        zeroIndex = 0
                    else:
                        zeroIndex = 1
                else:
                    if fitsOrder:
                        zeroIndex = 1
                    else:
                        zeroIndex = 0
                
                # create dictionary to store what slices to make - initiate with no restrictions
                imgSlices = [slice(None)]*self.image.ndim
                                
                # adjust for restrictions from slices
                for key in slices:
                    if key == "fitsAxisOrder" or key == "zeroIndex":
                        continue
                    if fitsOrder:
                        imgSlices[self.image.ndim-key+zeroIndex-1] = slice(slices[key] - zeroIndex, slices[key] - zeroIndex+1)
                    else:
                        imgSlices[key-zeroIndex] = slice(slices[key] - zeroIndex, slices[key] - zeroIndex+1)
                
                # now perform slices to the array
                self.image = self.image[imgSlices]
                
                # check only two axis with more than 1 dimension now
                if len((np.where(np.array(self.image.shape) > 1))[0]) > 2:
                    raise Exception("Slices not specified to only give 2D image")
            
                
            # create new WCS from array
            origHeader = self.header.copy()
            naxisSel = (self.image.ndim - np.where(np.array(self.image.shape) > 1)[0])[::-1].tolist()
            imgWCS = wcs.WCS(self.header, naxis=naxisSel)
            
            # remove all WCS keywords from header
            self.deleteWCSheaders()
            
            # update NAXIS keywords
            axisN = 1
            for i in range(1,self.header['NAXIS']+1):
                if self.header['NAXIS'+str(i)] == "1":
                    continue
                else:
                    self.header['NAXIS'+str(axisN)] = self.header['NAXIS'+str(i)]
                    axisN += 1
            # delete any remaining extra NAXIS headers
            for i in range(3,self.header['NAXIS']+1):
                try:
                    del(self.header['NAXIS'+str(i)])
                except:
                    pass
            # update NAXIS and i_naxis keyword
            self.header['NAXIS'] = 2
            self.header['i_naxis'] = 2
            
            # add new keywords to header
            self.header.update(imgWCS.to_header())
                            
            # remove extra axis from array
            self.image = np.squeeze(self.image)
                
            
        elif self.image.ndim < 2:
            if 'PIXTYPE' in self.header and self.header['PIXTYPE'] == 'HEALPIX':
                pass
            else:
                raise Exception("Less than 2 spatial axis discovered")
        
        
        
        # correct dustpedia header 
        if dustpediaHeaderCorrect:
            keywordAdjust = ["COORDSYS", "SIGUNIT", "TELESCOP", "INSTRMNT", "DETECTOR", "WVLNGTH", "HIPE_CAL"]
            for keyword in keywordAdjust:
                if keyword in self.header:
                    info = self.header[keyword].split("/")
                    if keyword == "SIGUNIT":
                        if self.header[keyword][0:10].count("/") > 0:
                            self.header[keyword] = (info[0]+"/"+info[1],info[2])
                    else:
                        self.header[keyword] = (info[0], info[1])    

        # identify telescope
        if telescope is None:
            if 'TELESCOP' in self.header:
                self.telescope = self.header['TELESCOP']
            elif ext != 0:
                primeHeader = fits[0].header
                if 'TELESCOP' in primeHeader:
                    self.telescope = primeHeader['TELESCOP']
                else:
                    self.telescope = None
            else:
                self.telescope = None
        else:
            self.telescope = telescope
        
        # identify instrument
        if instrument is None:
            if 'INSTRUME' in self.header:
                self.instrument = self.header['INSTRUME']
            elif 'INSTRMNT' in self.header:
                self.instrument = self.header['INSTRMNT']
            elif ext != 0:
                try:
                    primeHeader = fits[0].header
                    if 'INSTRUME' in primeHeader:
                        self.instrument = primeHeader['INSTRUME']
                        self.header['INSTRUME'] = primeHeader['INSTRUME']
                    else:
                        self.instrument = primeHeader['INSTRMNT']
                        self.header['INSTRMNT'] = primeHeader['INSTRMNT']
                except:
                    if verbose:
                        print("Warning - Unable to find instrument, recommended to specify")
                    self.instrument = None
            else:
                if verbose:
                    print("Warning - Unable to find instrument, recommended to specify")
                self.instrument = None
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
            elif ext != 0:
                bandFound = False
                primeHeader = fits[0].header
                for bandHeader in ['FILTER', 'WAVELNTH', 'WVLNGTH' 'FREQ']:
                    if bandHeader in primeHeader:
                        self.band = primeHeader[bandHeader]
                        self.header[bandHeader] = primeHeader[bandHeader]
                        bandFound = True
                        break
                
                if bandFound is False:
                    print("Warning - Band not identified, recommended to specify")
                    self.band = None
            elif self.telescope == "ALMA":
                almaBands = {"Band1":[31.0,45.0], "Band2":[67.0,90.0], "Band3":[84.0,116.0], "Band4":[125.0,163.0],\
                             "Band5":[163.0,211.0], "Band6":[211.0,275.0], "Band7":[275.0,373.0], "Band8":[385.0,500.0],\
                             "Band9":[602.0,720.0], "Band10":[787.0,950.0]}
                try:
                    bandFound = False
                    if origHeader["CTYPE3"] == "FREQ":
                        freqGHz = origHeader['CRVAL3'] / 1.0e9
                        for almaBand in almaBands:
                            if freqGHz >= almaBands[almaBand][0] and freqGHz <= almaBands[almaBand][1]:
                                self.band = almaBand
                                bandFound = True
                                break
                except:
                    bandFound = False
                    
                if bandFound is False:
                    if verbose:
                        print("Warning - Band not identified, recommended to specify")
                    self.band = None      
            else:
                if verbose:
                    print("Warning - Band not identified, recommended to specify")
                self.band = None
        else:
            self.band = band
        
        
        # set unit in header if provided
        if unit is not None:
            self.header['BUNIT'] = unit
        
        # For Dustpedia files strip out the micron as not really compatible yet
        if isinstance(self.band,str):
            if self.band.count("um") > 0:
                self.band = self.band.split("um")[0]
                self.bandUnits = "um"
        
        # if PACS or SPIRE make sure band is integer
        if self.instrument == "PACS" or self.instrument == "SPIRE":
            self.band = str(int(self.band))
    
       
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
        
        if "BUNIT" in self.header:
            self.unit = self.header['BUNIT']
        else:
            self.unit = None
        
        # try and get the wavelength of the observation
        try:
            self.wavelength = self.standardCentralWavelengths(self.instrument, self.band)
        except:
            pass
        
        
        # see if beam information is provided in header
        if "BMAJ" in self.header and "BMIN" in self.header:
            # extract 
            if "BPA" in self.header:
                self.beam  = {"BMAJ": (self.header['BMAJ'] * u.degree).to(u.arcsecond),\
                              "BMIN": (self.header['BMIN'] * u.degree).to(u.arcsecond),\
                              "BPA": self.header['BPA'] * u.degree}
            else:
                if FWHM is None:
                    FWHM = (self.header['BMAJ'] + self.header['BMIN'])/2.0 * u.degree
        
        # see if can get FWHM
        if FWHM is not None:
            try:
                self.fwhm = FWHM.to(u.arcsecond)
            except:
                self.fwhm = FWHM * u.arcsecond
        else:
            try:
                self.fwhm = self.standardFWHM(self.instrument, self.band)
            except:
                pass
        
        # see if can load the pixel size
        try:
            self.getPixelScale()
        except:
            pass 
        
        # close fits file
        if load:    
            fits.close()
        
        return
    
    
    def deleteWCSheaders(self):
        # function to remove headers involved in providing the WCS, so can update keywords with new header from WCS.to_header
        
        # all headers with axis number after
        headsToAdjust = ['CRPIX', 'CDELT', 'CRVAL', 'CTYPE', 'LBOUND', 'CUNIT']
        
        # loop over all header and axis number and delete if present
        for i in range(1,self.header['NAXIS']+1):
            for keyword in headsToAdjust:
                try:
                    del(self.header[keyword+str(i)])
                except:
                    pass
        
        # also remove and CDX_Y or PCX_Y
        for code in ["CD", "PC"]:
            # see if present in header
            if code+"1_1" in self.header or code+"01_01" in self.header or code+"001_001" in self.header:
                # get number format
                if code+"1_1" in self.header:
                    numOrder = "01"
                elif code+"1_1" in self.header:
                    numOrder = "02"
                else:
                    numOrder = "03"
                    
                # loop over entire matrix
                for i in range(1,self.header['NAXIS']+1):
                    for j in range(1,self.header['NAXIS']+1):
                        try:
                            del(self.header[f"{code}{i:{numOrder}}"])
                        except:
                            pass
        
        return
        
    
    def getPixelScale(self):
        # function to get pixel size
        WCSinfo = wcs.WCS(self.header)
        pixSizes = wcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        if np.abs(pixSizes[0]-pixSizes[1]) > 0.0001:
            raise ValueError("PANIC - program does not cope with non-square pixels")
        self.pixSize = round(pixSizes[0], 6) * u.arcsecond
        return round(pixSizes[0], 6)
    
        
    def background_sigmaClip(self, snr=2, npixels=5, dilate_size=11, sigClip=3.0, iterations=20, maskMatch=None, apply=False):
        # function to get background level and noise
        
        # import modules
        from astropy.stats import sigma_clipped_stats
        from photutils import make_source_mask
        
        if maskMatch is None:
            mask = make_source_mask(self.image, nsigma=snr, npixels=npixels, dilate_size=dilate_size)
        else:
            mask = maskMatch
        _,median,std = sigma_clipped_stats(self.image, mask=mask, sigma=sigClip, maxiters=iterations)
        self.bkgMedian = median
        self.bkgStd = std
        
        if apply:
            self.image = self.image - self.bkgMedian
            self.bkgMedian = 0.0
        
        return mask
    
    
    def constantBackSub(self, backConstant):
        # function to subtract a constant from the image
        
        # subtract constant from image
        self.image = self.image - backConstant
        
        return
    
    
    def ellipticalAnnulusBackSub(self, centre, inner=None, outer=None, axisRatio=None, PA=None, outerCircle=False, backNoise=False,\
                               method='exact', subpixels=None, maskNaN=True, apply=False):
        # function to select pixels within an elliptical aperture
        
        # import required modules
        from photutils import aperture_photometry
        from astropy.table import Column
        from astropy.table import join as tableJoin
        from astropy.table import Table
        from astropy.coordinates import SkyCoord
        from photutils import SkyEllipticalAnnulus
        from photutils import EllipticalAnnulus
        
        # if axis ratio has been set then calculate minor
        if inner is None or outer is None:
            raise Exception("No Radius/Semi-major axis info given")
        
        if PA is None:
            raise Exception("No Angle information is given")
        
        # see if inner is just one value or two
        if isinstance(inner, u.Quantity) and len(inner.shape) == 0:
            if axisRatio is not None:
                inner = np.array([inner.value, inner.value*axisRatio])*inner.unit
            elif isinstance(outer, u.Quantity) and len(outer.shape) > 0:
                inner = np.array([inner.value, inner.value * (outer[1]/outer[0]).value])*inner.unit
            elif isinstance(outer, (list, tuple, np.ndarray)):
                if isinstance(outer[0],u.Quantity):
                    inner = np.array([inner.value, inner.value * (outer[1]/outer[0]).value])*inner.unit
                else:
                    inner = np.array([inner.value, inner.value * outer[1]/outer[0]])*inner.unit
            else:
                raise Exception("No information provided about minor axis")
            
        elif isinstance(inner, (list, tuple, np.ndarray)) is False:
            if axisRatio is not None:
                inner = np.array([inner, inner*axisRatio])
            elif isinstance(outer, u.Quantity) and len(outer.shape) > 0:
                inner = np.array([inner, inner * outer[1]/outer[0]])
            elif isinstance(outer, (list, tuple, np.ndarray)):
                inner = np.array([inner, inner * outer[1]/outer[0]])
            else:
                raise Exception("No information provided about minor axis")
        
        
        # see if outer is just one value or two
        if isinstance(outer, u.Quantity) and len(outer.shape) == 0:
            if axisRatio is not None:
                outer = np.array([outer.value, outer.value*axisRatio])*outer.unit
            elif isinstance(inner, u.Quantity) and len(inner.shape) > 0:
                outer = np.array([outer.value, outer.value * (inner[1]/inner[0]).value])*outer.unit
            elif isinstance(inner, (list, tuple, np.ndarray)):
                if isinstance(inner[0],u.Quantity):
                    outer = np.array([outer.value, outer.value * (inner[1]/inner[0]).value])*outer.unit
                else:
                    outer = np.array([outer.value, outer.value * inner[1]/inner[0]])*outer.unit
            else:
                raise Exception("No information provided about minor axis")
            
        elif isinstance(outer, (list, tuple, np.ndarray)) is False:
            if axisRatio is not None:
                outer = np.array([outer, inner*axisRatio])
            elif isinstance(inner, u.Quanitity) and len(inner.shape) > 0:
                outer = np.array([outer, outer * inner[1]/inner[0]])
            elif isinstance(inner, (list, tuple, np.ndarray)):
                outer = np.array([outer, outer * inner[1]/inner[0]])
            else:
                raise Exception("No information provided about minor axis")
        
        # if outerCircle is set change outer minor axis to match primary
        if outerCircle is True:
            outer[1] = outer[0]
        
        # set flag whether needed to load WCS info
        try:
            imgWCS = wcs.WCS(self.header)
            pixOnly = False
        except:
            imgWCS = None
            pixOnly = True
        
        # create mask to remove any NaN's
        if maskNaN:
            nanMask = np.zeros(self.image.shape, dtype=bool)
            nanMask[np.isnan(self.image)] = True
        else:
            nanMask = False
        
        # check if PA is a quantity otherwise assume its degrees
        if isinstance(PA, u.Quantity) is False:
            PA = PA * u.degree
        
        
        # see if centre is a sky coordinate use Sky aperture otherwise assume its pixel
        if isinstance(centre, SkyCoord):
            # see if inner is in pixels or angular units
            if isinstance(inner[0], u.Quantity) is False:
                # convert to angular size by multiplying by pixel size
                if hasattr(self,'pixSize') is False:
                    self.getPixelScale()
                inner = inner * self.pixSize
            
            # see if outer is in pixels or angular units
            if isinstance(outer[0], u.Quantity) is False:
                # convert to angular size by multiplying by pixel size
                if hasattr(self,'pixSize') is False:
                    self.getPixelScale()
                outer = outer * self.pixSize
        
            
        
            if pixOnly:
                raise ValueError("Unable to read WCS and specified in Sky Coordinates")
            
            # create aperture object
            aperture = SkyEllipticalAnnulus(centre, inner[0], outer[0], outer[1], inner[1], theta=PA)
      
        else:
            # see if inner is in pixels or angular units
            if isinstance(inner[0], u.Quantity):
                # see if have the pixel size loaded
                if hasattr(self,'pixSize') is False:
                    self.getPixelSize()
                
                # convert to pixels by dividing by pixel size
                inner = (inner / self.pixSize).value
            
            # see if inner is in pixels or angular units
            if isinstance(outer[0], u.Quantity):
                # see if have the pixel size loaded
                if hasattr(self,'pixSize') is False:
                    self.getPixelSize()
                
                # convert to pixels by dividing by pixel size
                outer = (outer / self.pixSize).value
            
            # convert angle to be from x-axes not PA from north (both counter-clockwise
            apPA = apPA - 90.0*u.degree
            
            # create aperture object
            if pixOnly:
                aperture = EllipticalAnnulus(centre, inner[0], outer[0], outer[1], inner[1], theta=PA.to(u.radian).value)
            else:
                aperture = EllipticalAnnulus(centre, inner[0], outer[0], outer[1], inner[1], theta=PA.to(u.radian).value).to_sky(imgWCS)
            
        # perform aperture photometry to find sum in the annulus
        phot_table = aperture_photometry(self.image, aperture, wcs=imgWCS, method=method, subpixels=subpixels, mask=nanMask)
        nPixTable = aperture_photometry(np.ones(self.image.shape), aperture, wcs=imgWCS, method=method, subpixels=subpixels, mask=nanMask)
            
        # calculate background mean value
        backValue = phot_table['aperture_sum'][0] / nPixTable['aperture_sum'][0]
        
        # if desired calculate standard deviation of values in backgound region
        if backNoise:
            # first make sure aperture is converted to a pixel aperture
            if pixOnly:
                pixAperture = aperture
            else:
                pixAperture = aperture.to_pixel(imgWCS)
            
            # now create mask object
            mask = pixAperture.to_mask(method='center', subpixels=subpixels)
            
            # create a zoomed in mask and select where values are one not zerp
            sel = np.where(mask.multiply(np.ones(self.image.shape)) > 0.5)
            noise = mask.cutout(self.image)[sel].std()
            
        
        # if apply is set apply background subtraction to image
        if apply:
            self.image = self.image - backValue
        
        # if back noise is set, find standard deviation and return value
        if backNoise:
            return backValue, noise
        else:
            return backValue
        
        
    def circularAnnulusBackSub(self, centre, inner=None, outer=None, backNoise=False,\
                               method='exact', subpixels=None, maskNaN=True, apply=False):
        
        # call the elliptical annulus function set to give results as if a circle
        if backNoise:
            backValue, noise = self.ellipticalAnnulusBackSub(centre, inner=inner, outer=outer, axisRatio=1.0, PA=0.0, outerCircle=True, backNoise=backNoise,\
                                                             method=method, subpixels=subpixels, maskNaN=maskNaN, apply=apply)
        else:
            backValue = self.ellipticalAnnulusBackSub(centre, inner=inner, outer=outer, axisRatio=1.0, PA=0.0, outerCircle=True, backNoise=backNoise,\
                                                      method=method, subpixels=subpixels, maskNaN=maskNaN, apply=apply)
        
        # if back noise is set, find standard deviation and return value
        if backNoise:
            return backValue, noise
        else:
            return backValue
    
    
    def circularAperture(self, galInfo, radius=None, multiRadius = False, localBackSubtract=None, names=None, method='exact', subpixels=None, backMedian=False, maskNaN = True, error=None):
        # function to perform circular aperture photometry 
        
        # set mode to circle
        mode = "circle"
        
        # see if variable provided is a dictionary of dictionaries or of skyCoord
        if isinstance(galInfo,dict):
            allKeys = list(galInfo.keys())
            
            if isinstance(galInfo[allKeys[0]], dict):
                # extract info from galInfo variables
                
                if multiRadius is False:
 
                    # setup new arrays
                    centres = {}
                    tempRad = []
                    tempLocalBackSubtract = []
                    for i in range(0,len(allKeys)):
                        # add centres to dictionary so retain names
                        centres[allKeys[i]] = galInfo[allKeys[i]]["centre"]
                        
                        # append radius to array
                        if "radius" in galInfo[allKeys[i]]:
                            tempRad.append(galInfo[allKeys[i]]['radius'])
                        else:
                            tempRad.append(radius)
                        
                        # append local background subtract
                        if "localBackSubtract" in galInfo[allKeys[i]]:
                            tempLocalBackSubtract.append(galInfo[allKeys[i]]['localBackSubtract'])
                        else:
                            tempLocalBackSubtract.append(localBackSubtract)
                    
                    # restore arrays
                    radius = tempRad
                    localBackSubtract = tempLocalBackSubtract
                else:
                    # just extract centre information
                    centres = {}
                    for i in range(0,len(allKeys)):
                        centres[allKeys[i]] = galInfo[allKeys[i]]["centre"]
                
            else:
                centres = galInfo
        else:
            centres = galInfo
                    
        
        # if doing one radius per object see if centres and radius have multiple values, that they are the same length 
        if multiRadius is False:
            if isinstance(centres, (list, tuple, np.ndarray)) and isinstance(radius, (list, tuple, np.ndarray)) and isinstance(radius, u.Quantity) is False:
                if len(centres) != len(radius):
                    raise ValueError("List of centres is not same length as list of radius (if want multiple radii at one position set multiRadius to True)")
            
        # check if doing local background subtraction that only one background radius, or same as centres
        if localBackSubtract is not None and isinstance(centres, (list, tuple, np.ndarray)) and isinstance(localBackSubtract, (list, tuple, np.ndarray)):
            if len(centres) != len(localBackSubtract):
                raise ValueError("List of background radius values is not same length as list of centres")
        elif localBackSubtract is not None and isinstance(centres, (list, tuple, np.ndarray)) and "inner" in localBackSubtract and isinstance(localBackSubtract['inner'], (list, tuple, np.ndarray)) and isinstance(localBackSubtract['inner'], u.Quantity) is False:
            if len(centres) != len(localBackSubtract['inner']):
                raise ValueError("List of background radius values is not same length as list of centres")
        elif localBackSubtract is not None and isinstance(centres, (list, tuple, np.ndarray)) and "outer" in localBackSubtract and isinstance(localBackSubtract['outer'], (list, tuple, np.ndarray)) and isinstance(localBackSubtract['outer'], u.Quantity) is False:
            if len(centres) != len(localBackSubtract['inner']):
                raise ValueError("List of background outer radius values is not same length as list of centres")
        
        
    
        # perform aperture photometry
        phot_table = self.aperturePhotometry(mode, centres, radius, multiRadius=multiRadius, localBackSubtract=localBackSubtract, names=names, method=method, subpixels=subpixels, backMedian=backMedian, maskNaN=maskNaN, error=error)
    
        return phot_table
    
    
    def ellipticalAperture(self, galInfo, major=None, minor=None, axisRatio=None, PA=None, multiRadius = False, localBackSubtract=None, names=None, method='exact', subpixels=None, backMedian=False, maskNaN = True, error=None):
        # function to perform circular aperture photometry 
        
        # set mode to circle
        mode = "ellipse"
        
        # if doing multi radius perform checks
        if multiRadius:
            if minor is not None:
                raise ValueError("For multiple radial apertures, set the 'axisRatio' parameter with a fixed value, rather than setting 'minor'")
            if isinstance(axisRatio,(list,tuple,np.ndarray)):
                if isinstance(galInfo,(list,tuple,np.ndarray)) and len(galInfo) != len(axisRatio):
                    raise ValueError("For multiple radial apertures, 'axisRatio' can only have one value per object")
        
        # see if variable provided is a dictionary of dictionaries or of skyCoord
        if isinstance(galInfo,dict):
            allKeys = list(galInfo.keys())
            
            if isinstance(galInfo[allKeys[0]], dict):
                # extract info from galInfo variables
                
                if multiRadius is False:
 
                    # setup new arrays
                    centres = {}
                    tempRad = []
                    tempLocalBackSubtract = []
                    tempPA = []
                    tempMinor = []
                    for i in range(0,len(allKeys)):
                        # add centres to dictionary so retain names
                        centres[allKeys[i]] = galInfo[allKeys[i]]["centre"]
                        
                        # append major radius to array
                        if "major" in galInfo[allKeys[i]]:
                            tempRad.append(galInfo[allKeys[i]]['major'])
                        else:
                            tempRad.append(major)
                        
                        # see if PA is created if not put in list
                        if "PA" in galInfo[allKeys[i]]:
                            tempPA.append(galInfo[allKeys[i]]["PA"])
                        else:
                            tempPA.append(PA)
                            
                        # see if either axis ratio or is specified
                        if "axisRatio" in galInfo[allKeys[i]]:
                            tempMinor.append(tempRad[-1] * galInfo[allKeys[i]]["axisRatio"])
                        elif "minor" in  galInfo[allKeys[i]]:
                            tempMinor.append(galInfo[allKeys[i]]["minor"])
                        elif minor is not None:
                            tempMinor.append(minor)
                        else:
                            tempMinor.append(tempRad[-1] * axisRatio)
                            
                        # append local background subtract
                        if "localBackSubtract" in galInfo[allKeys[i]]:
                            tempLocalBackSubtract.append(galInfo[allKeys[i]]['localBackSubtract'])
                        else:
                            tempLocalBackSubtract.append(localBackSubtract)
                    
                    # restore arrays
                    major = tempRad
                    localBackSubtract = tempLocalBackSubtract
                    PA = tempPA
                    minor = tempMinor 
                else:
                    # just extract centre information
                    centres = galInfo
                    
                    # extract centre information and any PA or axis Ratio information
                    centres = {}
                    tempPA = []
                    tempAxisRatio = []
                    tempLocalBackSubtract = []
                    for i in range(0,len(allKeys)):
                        # add centres to dictionary so retain names
                        centres[allKeys[i]] = galInfo[allKeys[i]]["centre"]
                    
                        # append PA to array
                        if "PA" in galInfo[allKeys[i]]:
                            tempPA.append(galInfo[allKeys[i]]['PA'])
                        else:
                            tempPA.append(PA)
                    
                        # append axis ratio information
                        if "axisRatio" in galInfo[allKeys[i]]:
                            tempAxisRatio.append(galInfo[allKeys[i]]["axisRatio"])
                        else:
                            tempAxisRatio.append(axisRatio)
                        
                        # append local background subtract
                        if "localBackSubtract" in galInfo[allKeys[i]]:
                            tempLocalBackSubtract.append(galInfo[allKeys[i]]['localBackSubtract'])
                        else:
                            tempLocalBackSubtract.append(localBackSubtract)
                    
                    PA = tempPA
                    axisRatio = tempAxisRatio
                    localBackSubtract = tempLocalBackSubtract
                        
            else:
                centres = galInfo
        else:
            centres = galInfo
                    
        # see what is defined minor/axisRatio and create uniform
        if minor is None and axisRatio is not None:
            if isinstance(major, (list,tuple)) and isinstance(axisRatio,(list,tuple)):
                minor = []
                for i in range(0,len(radius)):
                    minor.append(major[i] * axisRatio[i])
            elif isinstance(major,(list,tuple)) and isinstance(axisRatio,(list,tuple)) is False:
                minor = []
                for i in range(0,len(major)):
                    minor.append(major[i] * axisRatio)
            elif isinstance(major,(list,tuple)) is False and isinstance(axisRatio,(list,tuple)):
                minor = []
                for i in range(0,len(axisRatio)):
                    minor.append(major * axisRatio[i])
            else:
                minor = major * axisRatio
        
        
        # if doing one radius per object see if centres and radius have multiple values, that they are the same length 
        if multiRadius is False:
            if isinstance(centres, (list, tuple, np.ndarray)) and isinstance(major, (list, tuple, np.ndarray)) and isinstance(minor, u.Quantity) is False:
                if len(centres) != len(major):
                    raise ValueError("List of centres is not same length as list of semi-major axis (if want multiple radii at one position set multiRadius to True)")
        
            # check that if minor supplied is the same length as radius array (or single values)
            if isinstance(minor, (list, tuple, np.ndarray)) and isinstance(major, u.Quantity) is False:
                if len(major) != len(minor):
                    raise ValueError("Semi-minor axis list is not same length as list of semi-major axos")
        
            
        # check if doing local background subtraction that only one background radius, or same as centres
        if localBackSubtract is not None and isinstance(centres, (list, tuple, np.ndarray)) and isinstance(localBackSubtract, (list, tuple, np.ndarray)):
            if len(centres) != len(localBackSubtract):
                raise ValueError("List of background radius values is not same length as list of centres")
        elif localBackSubtract is not None and isinstance(centres, (list, tuple, np.ndarray)) and "inner" in localBackSubtract and isinstance(localBackSubtract['inner'], (list, tuple, np.ndarray)) and isinstance(localBackSubtract['inner'], u.Quantity) is False:
            if len(centres) != len(localBackSubtract['inner']):
                raise ValueError("List of background radius values is not same length as list of centres")
        elif localBackSubtract is not None and isinstance(centres, (list, tuple, np.ndarray)) and "outer" in localBackSubtract and isinstance(localBackSubtract['outer'], (list, tuple, np.ndarray)) and isinstance(localBackSubtract['outer'], u.Quantity) is False:
            if len(centres) != len(localBackSubtract['inner']):
                raise ValueError("List of background outer radius values is not same length as list of centres")
    
        # check number of PA angles matches centres
        if PA is None:
            raise ValueError("PA information must be provided")
        else:
            if isinstance(PA, (list,tuple, np.ndarray)) and isinstance(PA, u.Quantity) is False:
                if len(centres) != len(PA):
                    raise ValueError("List of Posistion Angles must have same length as number of centres")
        
        # check either minor defined
        if minor is None:
            raise ValueError("Semi-minor axis must be defined")
    
        # perform aperture photometry
        phot_table = self.aperturePhotometry(mode, centres, major, minor=minor, PA=PA, multiRadius=multiRadius, localBackSubtract=localBackSubtract, names=names, method=method, subpixels=subpixels, backMedian=backMedian, maskNaN=maskNaN, error=error)

        # calculate surface brightness profile if desired
        calculateSB = True
        if multiRadius and calculateSB:
            ### create a table where radii is midway between the bins
            
            # calculate halfway bins for major and minor
            halfMajor = self.halfBinArrays(major)
            halfMinor = self.halfBinArrays(minor)
            
            # run photometry in these values
            halfbin_phot_table = self.aperturePhotometry(mode, centres, halfMajor, minor=halfMinor, PA=PA, multiRadius=multiRadius, localBackSubtract=localBackSubtract, names=names, method=method, subpixels=subpixels, backMedian=backMedian, maskNaN=maskNaN, error=error)

            self.surfaceBrightness(phot_table, halfbin_phot_table)

            print("here")
            
    
        return phot_table   


    def rectangularAperture(self, galInfo, length=None, width=None, ratio=None, PA=None, multiRadius = False, localBackSubtract=None, names=None, method='exact', subpixels=None, backMedian=False, maskNaN = True, error=None):
        # function to perform circular aperture photometry 
        
        # set mode to circle
        mode = "rectangle"
        
        # if doing multi radius perform checks
        if multiRadius:
            if width is not None:
                raise ValueError("For multiple size apertures, set the 'axisRatio' parameter with a fixed value, rather than setting 'width'")
            if isinstance(ratio,(list,tuple,np.ndarray)):
                if isinstance(galInfo,(list,tuple,np.ndarray)) and len(galInfo) != len(ratio):
                    raise ValueError("For multiple radial apertures, 'ratio' can only have one value per object")
        
        
        # see if variable provided is a dictionary of dictionaries or of skyCoord
        if isinstance(galInfo,dict):
            allKeys = list(galInfo.keys())
            
            if isinstance(galInfo[allKeys[0]], dict):
                # extract info from galInfo variables
                
                if multiRadius is False:
 
                    # setup new arrays
                    centres = {}
                    tempLen = []
                    tempLocalBackSubtract = []
                    tempPA = []
                    tempWidth = []
                    for i in range(0,len(allKeys)):
                        # add centres to dictionary so retain names
                        centres[allKeys[i]] = galInfo[allKeys[i]]["centre"]
                        
                        # append major radius to array
                        if "length" in galInfo[allKeys[i]]:
                            tempLen.append(galInfo[allKeys[i]]['length'])
                        else:
                            tempLen.append(length)
                        
                        # see if PA is created if not put in list
                        if "PA" in galInfo[allKeys[i]]:
                            tempPA.append(galInfo[allKeys[i]]["PA"])
                        else:
                            tempPA.append(PA)
                            
                        # see if either axis ratio or is specified
                        if "ratio" in galInfo[allKeys[i]]:
                            tempWidth.append(tempLen[-1] * galInfo[allKeys[i]]["ratio"])
                        elif "width" in  galInfo[allKeys[i]]:
                            tempWidth.append(galInfo[allKeys[i]]["width"])
                        elif width is not None:
                            tempWidth.append(width)
                        else:
                            tempWidth.append(tempLen[-1] * ratio)
                            
                        # append local background subtract
                        if "localBackSubtract" in galInfo[allKeys[i]]:
                            tempLocalBackSubtract.append(galInfo[allKeys[i]]['localBackSubtract'])
                        else:
                            tempLocalBackSubtract.append(localBackSubtract)
                    
                    # restore arrays
                    major = tempLen
                    localBackSubtract = tempLocalBackSubtract
                    PA = tempPA
                    minor = tempWidth 
                else:
                    # just extract centre information
                    centres = galInfo
                    
                    # extract centre information and any PA or axis Ratio information
                    centres = {}
                    tempPA = []
                    tempRatio = []
                    for i in range(0,len(allKeys)):
                        # add centres to dictionary so retain names
                        centres[allKeys[i]] = galInfo[allKeys[i]]["centre"]
                    
                        # append PA to array
                        if "PA" in galInfo[allKeys[i]]:
                            tempPA.append(galInfo[allKeys[i]]['PA'])
                        else:
                            tempPA.append(PA)
                    
                        # append axis ratio information
                        if "ratio" in galInfo[allKeys[i]]:
                            tempRatio.append(galInfo[allKeys[i]]["ratio"])
                        else:
                            tempRatio.append(ratio)
                    
                    PA = tempPA
                    ratio = tempRatio
                    minor = width
                    if isinstance(length, (list,tuple)):
                        length = np.array(length) * 2.0
                    else:
                        major = length * 2.0
                        
            else:
                centres = galInfo
                minor = width
                major = length
        else:
            centres = galInfo
            minor = width
            major = length
                    
        # see what is defined width or length ratio and create uniform
        if minor is None and ratio is not None:
            if isinstance(major, (list,tuple)) and isinstance(ratio,(list,tuple)):
                minor = []
                for i in range(0,len(radius)):
                    minor.append(major[i] * ratio[i])
            elif isinstance(major,(list,tuple)) and isinstance(ratio,(list,tuple)) is False:
                minor = []
                for i in range(0,len(major)):
                    minor.append(major[i] * ratio)
            elif isinstance(major,(list,tuple)) is False and isinstance(ratio,(list,tuple)):
                minor = []
                for i in range(0,len(ratio)):
                    minor.append(major * ratio[i])
            else:
                minor = major * ratio
        
        
        # if doing one radius per object see if centres and radius have multiple values, that they are the same length 
        if multiRadius is False:
            if isinstance(centres, (list, tuple, np.ndarray)) and isinstance(major, (list, tuple, np.ndarray)) and isinstance(minor, u.Quantity) is False:
                if len(centres) != len(major):
                    raise ValueError("List of centres is not same length as list of semi-major axis (if want multiple radii at one position set multiRadius to True)")
        
            # check that if minor supplied is the same length as radius array (or single values
            if isinstance(minor, (list, tuple, np.ndarray)) and isinstance(major, u.Quantity) is False:
                if len(major) != len(minor):
                    raise ValueError("Semi-minor axis list is not same length as list of semi-major axos")
        
            
        # check if doing local background subtraction that only one background radius, or same as centres
        if localBackSubtract is not None and isinstance(centres, (list, tuple, np.ndarray)) and isinstance(localBackSubtract, (list, tuple, np.ndarray)):
            if len(centres) != len(localBackSubtract):
                raise ValueError("List of background radius values is not same length as list of centres")
        elif localBackSubtract is not None and isinstance(centres, (list, tuple, np.ndarray)) and "inner" in localBackSubtract and isinstance(localBackSubtract['inner'], (list, tuple, np.ndarray)) and isinstance(localBackSubtract['inner'], u.Quantity) is False:
            if len(centres) != len(localBackSubtract['inner']):
                raise ValueError("List of background radius values is not same length as list of centres")
        elif localBackSubtract is not None and isinstance(centres, (list, tuple, np.ndarray)) and "outer" in localBackSubtract and isinstance(localBackSubtract['outer'], (list, tuple, np.ndarray)) and isinstance(localBackSubtract['outer'], u.Quantity) is False:
            if len(centres) != len(localBackSubtract['inner']):
                raise ValueError("List of background outer radius values is not same length as list of centres")
    
        # check number of PA angles matches centres
        if PA is None:
            raise ValueError("PA information must be provided")
        else:
            if isinstance(PA, (list,tuple, np.ndarray)) and isinstance(PA, u.Quantity) is False:
                if len(centres) != len(PA):
                    raise ValueError("List of Posistion Angles must have same length as number of centres")
        
        # check either minor defined
        if minor is None:
            raise ValueError("Semi-minor axis must be defined")
        
        # perform aperture photometry
        phot_table = self.aperturePhotometry(mode, centres, major, minor=minor, PA=PA, multiRadius=multiRadius, localBackSubtract=localBackSubtract, names=names, method=method, subpixels=subpixels, backMedian=backMedian, maskNaN=maskNaN, error=error)
    
        return phot_table
    
    
    def aperturePhotometry(self, mode, centres, radius, minor=None, PA=None, multiRadius = False, localBackSubtract=None, names=None, method='exact', subpixels=None, backMedian=False, maskNaN=True, error=None):
        # function to perform photometry
        
        # import required modules
        from photutils import aperture_photometry
        from astropy.table import Column
        from astropy.table import join as tableJoin
        from astropy.table import Table
        from astropy.coordinates import SkyCoord
        
        # check mode programmed
        if mode not in ["circle", "ellipse", "rectangle"]:
            raise ValueError("Shape Not programmed")
        
        # import relevant photutil function
        if mode == "circle":
            from photutils import SkyCircularAperture
            from photutils import CircularAperture as PixCircularAperture
            if localBackSubtract:
                from photutils import SkyCircularAnnulus
                from photutils import CircularAnnulus
        elif mode == "ellipse":
            from photutils import SkyEllipticalAperture
            from photutils import EllipticalAperture as PixEllipticalAperture
            if localBackSubtract:
                from photutils import SkyEllipticalAnnulus
                from photutils import EllipticalAnnulus
        elif mode == "rectangle":
            from photutils import SkyRectangularAperture
            from photutils import RectangularAperture as PixRectangularAperture
            if localBackSubtract:
                from photutils import SkyRectangularAnnulus
                from photutils import RectangularAnnulus

        # if names is not none, see if a list, if not make it one
        if names is not None:
            if isinstance(names,(list,tuple)) is False:
                names = [names]
        
        # if centres is a dictionary split into names
        if isinstance(centres, dict):
            names = list(centres.keys())
            centreCopy = centres.copy()
            centres = []
            for objName in names:
                centres.append(centreCopy[objName])
        
        
        # if the image is in surface brightness units return the mean of aperture not the sum
        calculateMean = False
        if hasattr(self,'unit'):
            programedUnits, SBinfo = self.programmedUnits(SBinfo=True)
            if self.unit in SBinfo:
                if SBinfo[self.unit] is True:
                    print("Image is in Surface-Brightness Units - Calculating Mean")
                    calculateMean = True
        
        # look at what error info is provided
        if error is not None:
            
            # if error image is provided, check that its the same shape as the image
            errorImage = False 
            if isinstance(error,np.ndarray):
                if np.array_equal(error.shape, self.image.shape) is False:
                    print("Error parameter is an array, that does not match image shape - error analysis not performed")
                    error = None
                errorImage = True
            elif isinstance(error, bool):
                if error is True:
                    if localBackSubtract is None:
                        print("Setting Error to True requires localBackSubtract to be used (alternatively provide error map or uncertainty value)")
                        error = None
                else:
                    error = None
            elif isinstance(error, (list,tuple)):
                print("Error has been set to a list or Tuple - this is not a programmed method - error analysis not performed")
                error = None
                       
        
        # set flag whether needed to load WCS info
        try:
            imgWCS = wcs.WCS(self.header)
            pixOnly = False
        except:
            imgWCS = None
            pixOnly = True
        
        # create mask to remove any NaN's
        if maskNaN:
            nanMask = np.zeros(self.image.shape, dtype=bool)
            nanMask[np.isnan(self.image)] = True
        else:
            nanMask = False
        
        
        # create list of apertures
        apertures = []
        
        # if doing a local background subtract, create array to store these apertures
        if localBackSubtract is not None:
            backApertures = []
        
        # if centers is only one value embed in list
        if isinstance(centres, SkyCoord):
            centres  = [centres]
        elif isinstance(centres, (list, tuple, np.ndarray)) is False:
            centres = [centres]
        else:
            if len(centres) == 2 and isinstance(centres[0], (float, int, np.float, np.int)) and isinstance(centres[0], (float, int, np.float, np.int)):
                centres = [centres]
        
        # arrays to hold indicies
        if multiRadius:
            MRi = []
            MRj = []
        
        # loop over each centre, see if pixel or SkyCoord
        for i in range(0,len(centres)):
            # if doing multiRadius loop over all otherwise put in single element list to loop over
            if multiRadius:
                masterRadius = radius
            else:
                # see if radius varies for each centre or is a constant
                if isinstance(radius, u.Quantity):
                    masterRadius = [radius]
                elif isinstance(radius, (list, tuple, np.ndarray)):
                    masterRadius = [radius[i]]
                else:
                    masterRadius = [radius]                    
            
            # if doing ellipse format the minor radius and get PA
            if mode == "ellipse" or mode == "rectangle":
                if multiRadius:
                    if isinstance(minor,list) and len(minor) == len(centres):
                        masterMinor = minor[i]
                    else:
                        masterMinor = minor
                else:
                    # see if radius varies for each centre or is a constant
                    if isinstance(minor, u.Quantity):
                        masterMinor = [minor]
                    elif isinstance(minor, (list, tuple, np.ndarray)):
                        masterMinor = [minor[i]]
                    else:
                        masterMinor = [minor]
                
                # get the position angle
                if isinstance(PA, (list,tuple,np.ndarray)) is True and isinstance(PA, u.Quantity) is False:
                    apPA = PA[i]
                else:
                    apPA = PA
                # check if PA is a quantity otherwise assume its degrees
                if isinstance(apPA, u.Quantity) is False:
                    apPA = apPA * u.degree
                
            # if first radius in a multi-radius run is zero 
            if multiRadius:
                zeroFirst = False
                if isinstance(masterRadius[0], u.Quantity):
                    if masterRadius[0].value == 0.0:
                        zeroFirst = True
                else:
                    if masterRadius[0] == 0.0:
                        zeroFirst = True
                
            
            # loop over every radius if multi-radius
            for j in range(0,len(masterRadius)):
                rad = masterRadius[j]
                if mode == "ellipse" or mode == "rectangle":
                    minorRad = masterMinor[j]                        
                
                if multiRadius and zeroFirst and j == 0:
                    continue
                
                # get the background radius for each centre or see if same for each
                if localBackSubtract is not None:
                    if isinstance(localBackSubtract, (list,tuple,np.ndarray)):
                        backRadInfo = localBackSubtract[i]
                        if backRadInfo is not None:
                            if mode == "ellipse" and "outerCircle" not in backRadInfo:
                                backRadInfo["outerCircle"] = False
                    elif isinstance(localBackSubtract['inner'],u.Quantity) is False and isinstance(localBackSubtract['inner'], (list,tuple, np.ndarray)):
                        backRadInfo = {"inner":localBackSubtract['inner'][i], "outer":localBackSubtract['outer'][i]}
                        if mode == "ellipse":
                            if "outerCircle" in localBackSubtract:
                                backRadInfo["outerCircle"] = localBackSubtract["outerCircle"]
                            else:
                                backRadInfo["outerCircle"] = False
                    else:
                        backRadInfo = {"inner":localBackSubtract['inner'], "outer":localBackSubtract['outer']}
                        if mode == "ellipse":
                            if "outerCircle" in localBackSubtract:
                                backRadInfo["outerCircle"] = localBackSubtract["outerCircle"]
                            else:
                                backRadInfo["outerCircle"] = False
                
                # if centre is a sky coordinate use Sky aperture otherwise assume its pixel
                if isinstance(centres[i], SkyCoord):
                    # see if radius is in pixels or angular units
                    if isinstance(rad, u.Quantity) is False:
                        # convert to angular size by multiplying by pixel size
                        if hasattr(self,'pixSize') is False:
                            self.getPixelScale()
                        rad = rad * self.pixSize
                    
                    if mode == "ellipse" or mode == "rectangle":
                        if isinstance(minorRad, u.Quantity) is False:
                            # convert to angular size by multiplying by pixel size
                            if hasattr(self,'pixSize') is False:
                                self.getPixelScale()
                            minorRad = minorRad * self.pixSize
                    
                    
                    # see if background radius is in pixel or angular units
                    if localBackSubtract is not None and backRadInfo is not None:
                        # see if back inner radius is in pixels or angular units
                        if isinstance(backRadInfo["inner"], u.Quantity) is False:
                            # convert to angular size by multiplying by pixel size
                            if hasattr(self,'pixSize') is False:
                                self.getPixelScale()
                            backRadInfo["inner"] = backRadInfo["inner"] * self.pixSize
                        
                        # see if back outer                         
                        if isinstance(backRadInfo["outer"], u.Quantity) is False:
                            # convert to angular size by multiplying by pixel size
                            if hasattr(self,'pixSize') is False:
                                self.getPixelScale()
                            backRadInfo["outer"] = backRadInfo["outer"] * self.pixSize
                            
                                            
                    if pixOnly:
                        raise ValueError("Unable to read WCS and specified in Sky Coordinates")
                    
                    # create aperture object
                    if mode == "circle":
                        apertures.append(SkyCircularAperture(centres[i], r=rad))
                    elif mode == "ellipse":
                        apertures.append(SkyEllipticalAperture(centres[i], rad, minorRad, theta=apPA))
                    elif mode == "rectangle":
                        apertures.append(SkyRectangularAperture(centres[i], rad, minorRad, theta=apPA))
                    
                    # if doing local subtraction create background aperture
                    if localBackSubtract is not None:
                        if backRadInfo is not None:
                            if mode == "ellipse" or mode == "rectangle":
                                if multiRadius:
                                    backgroundRatio = (masterMinor[-1] / masterRadius[-1]).value
                                else:
                                    backgroundRatio = minorRad/rad
                            
                            if mode == "circle":
                                backApertures.append(SkyCircularAnnulus(centres[i], r_in=backRadInfo["inner"], r_out=backRadInfo["outer"]))
                            elif mode == "ellipse":
                                if backRadInfo["outerCircle"]:
                                    backApertures.append(SkyEllipticalAnnulus(centres[i], backRadInfo["inner"], backRadInfo["outer"], backRadInfo["outer"], b_in=backRadInfo["inner"]*backgroundRatio, theta=apPA))
                                else:
                                    backApertures.append(SkyEllipticalAnnulus(centres[i], backRadInfo["inner"], backRadInfo["outer"], backRadInfo["outer"]*backgroundRatio, theta=apPA))
                            elif mode == "rectangle":
                                backApertures.append(SkyRectangularAnnulus(centres[i], backRadInfo["inner"], backRadInfo["outer"], backRadInfo["outer"]*backgroundRatio, theta=apPA))
                        else:
                            backApertures.append(None)
                else:
                    # see if radius is in pixels or angular units
                    if isinstance(rad, u.Quantity):
                        # see if have the pixel size loaded
                        if hasattr(self,'pixSize') is False:
                            self.getPixelScale()
                        
                        # convert to pixels by dividing by pixel size
                        rad  = (rad / self.pixSize).value
                    
                    
                    if mode == "ellipse" or mode == "rectangle":
                        if isinstance(minorRad, u.Quantity):
                            # convert to pixel size by diving by pixel size
                            if hasattr(self,'pixSize') is False:
                                self.getPixelScale()
                            minorRad = (minorRad / self.pixSize).value
                            
                    
                    # see if background radius is in pixel or angular units
                    if localBackSubtract is not None and backRadInfo is not None:
                        # see if back inner radius is in pixels or angular units
                        if isinstance(backRadInfo["inner"], u.Quantity):
                            # see if have the pixel size loaded
                            if hasattr(self,'pixSize') is False:
                                self.getPixelScale()
                                
                            # convert to pixels by dividing by pixel size
                            backRadInfo["inner"] = (backRadInfo["inner"] / self.pixSize).value
                        
                        # see if back inner radius is in pixels or angular units
                        if isinstance(backRadInfo["outer"], u.Quantity):
                            # see if have the pixel size loaded
                            if hasattr(self,'pixSize') is False:
                                self.getPixelScale()
                                
                            # convert to pixels by dividing by pixel size
                            backRadInfo["outer"] = (backRadInfo["outer"] / self.pixSize).value
                            
                    
                    # convert angle to be from x-axes not PA from north (both counter-clockwise
                    if mode == "ellipse" or mode == "rectangle":
                        apPA = apPA - 90.0*u.degree
                    
                    # create aperture object
                    if pixOnly:
                        if mode == "circle":
                            apertures.append(PixCircularAperture(centres[i], r=rad))
                        elif mode == "ellipse":
                            apertures.append(PixEllipticalAperture(centres[i], rad, minorRad, theta=apPA.to(u.radian).value))
                        elif mode == "rectangle":
                            apertures.append(PixRectangularAperture(centres[i], rad, minorRad, theta=apPA.to(u.radian).value))
                            
                        if localBackSubtract:
                            if mode == "ellipse" or mode == "rectangle":
                                if multiRadius:
                                    backgroundRatio = masterMinor[-1] / masterRadius[-1]
                                else:
                                    backgroundRatio = minorRad/rad
                            
                            if mode == "circle":
                                backApertures.append(CicularAnnulus(centres[i], r_in=backRadInfo['inner'], r_out=backRadInfo['outer']))
                            elif mode == "ellipse":
                                if backRadInfo["outerCircle"]:
                                    backApertures.append(EllipticalAnnulus(centres[i], backRadInfo["inner"], backRadInfo["outer"], backRadInfo["outer"], b_in=backRadInfo["inner"]*backgroundRatio, theta=apPA.to(u.radian).value))
                                else:
                                    backApertures.append(EllipticalAnnulus(centres[i], backRadInfo["inner"], backRadInfo["outer"], backRadInfo["outer"]*backgroundRatio, theta=apPA.to(u.radian).value))
                            elif mode == "rectangle":
                                backApertures.append(RectangularAnnulus(centres[i], backRadInfo["inner"], backRadInfo["outer"], backRadInfo["outer"]*backgroundRatio, theta=apPA.to(u.radian).value))
                    else:
                        if mode == "circle":
                            apertures.append(PixCircularAperture(centres[i], r=rad).to_sky(imgWCS))
                        elif mode == "ellipse":
                            apertures.append(PixEllipticalAperture(centres[i], rad, minorRad, theta=apPA.to(u.radian).value).to_sky(imgWCS))
                        elif mode == "rectangle":
                            apertures.append(PixRectangularAperture(centres[i], rad, minorRad, theta=apPA.to(u.radian).value).to_sky(imgWCS))
                            
                        if localBackSubtract:
                            if mode == "ellipse" or mode == "rectangle":
                                if multiRadius:
                                    backgroundRatio = masterMinor[-1] / masterRadius[-1]
                                else:
                                    backgroundRatio = minorRad/rad
                            
                            if mode == "circle":
                                backApertures.append(CircularAnnulus(centres[i], r_in=backRadInfo['inner'], r_out=backRadInfo['outer']).to_sky(imgWCS))
                            elif mode == "ellipse":
                                if backRadInfo["outerCircle"]:
                                    backApertures.append(EllipticalAnnulus(centres[i], backRadInfo["inner"], backRadInfo["outer"], backRadInfo["outer"], b_in=backRadInfo["inner"]*backgroundRatio, theta=apPA.to(u.radian).value).to_sky(imgWCS))
                                else:
                                    backApertures.append(EllipticalAnnulus(centres[i], backRadInfo["inner"], backRadInfo["outer"], backRadInfo["outer"]*backgroundRatio, theta=apPA.to(u.radian).value).to_sky(imgWCS))
                            elif mode == "rectangle":
                                backApertures.append(RectangularAnnulus(centres[i], backRadInfo["inner"], backRadInfo["outer"], backRadInfo["outer"]*backgroundRatio, theta=apPA.to(u.radian).value).to_sky(imgWCS))
                
                # multiple radius mode need to know the centre and radius
                if multiRadius:
                    # save what the centres and the radius is for each entry, so can reconstruct the table
                    MRi.append(i)
                    MRj.append(j)
            
        # perform the aperture photometry and calculate number of pixels
        for i in range(0,len(apertures)):
            # perform aperture photometry
            ind_phot_table = aperture_photometry(self.image, apertures[i], wcs=imgWCS, method=method, subpixels=subpixels, mask=nanMask)
            ind_nPixTable = aperture_photometry(np.ones(self.image.shape), apertures[i], wcs=imgWCS, method=method, subpixels=subpixels, mask=nanMask)
                                   
            # perform backgound subtraction if requested
            if localBackSubtract is not None:
                backApertureExist = False
                if multiRadius:
                    if isinstance(localBackSubtract, (list,tuple,np.ndarray)) is False or localBackSubtract[MRi[i]] is not None:
                        backApertureExist = True
                else:
                    if isinstance(localBackSubtract, (list,tuple,np.ndarray)) is False or localBackSubtract[i] is not None:
                        backApertureExist = True
                
                if backApertureExist:
                    # calculate either median or mean
                    if backMedian:
                        if pixOnly:
                            backMask = backApertures[i].to_mask('center').multiply(np.ones(self.image.shape))
                            backImage = backApertures[i].to_mask('center').multiply(self.image)
                        else:
                            backMask = backApertures[i].to_pixel(imgWCS).to_mask('center').multiply(np.ones(self.image.shape))
                            backImage = backApertures[i].to_pixel(imgWCS).to_mask('center').multiply(self.image)
                        if maskNaN:
                            backValues = np.nanmedian(backImage[backMask > 0])
                        else:
                            backValues = np.median(backImage[backMask > 0])
                        backNpix = len(backImage[backMask > 0])
                        ind_phot_table['aperture_sum'] = ind_phot_table['aperture_sum'] - backValues * ind_nPixTable['aperture_sum']
                    else:
                        ind_back_table = aperture_photometry(self.image, backApertures[i], wcs=imgWCS, method=method, subpixels=subpixels, mask=nanMask)
                        ind_back_nPixTable = aperture_photometry(np.ones(self.image.shape), backApertures[i], wcs=imgWCS, method=method, subpixels=subpixels, mask=nanMask)
    
                        backValues = ind_back_table['aperture_sum'] / ind_back_nPixTable['aperture_sum']
                        backNpix = ind_back_nPixTable['aperture_sum']
                        ind_phot_table['aperture_sum'] = ind_phot_table['aperture_sum'] - backValues * ind_nPixTable['aperture_sum']
            
            # see if using error image
            if error is not None:
                if errorImage is True:
                    var_phot_table = aperture_photometry(error**2.0, apertures[i], wcs=imgWCS, method=method, subpixels=subpixels, mask=nanMask)
                    apError = np.sqrt(var_phot_table['aperture_sum'])
                else:
                    # if error set to true use background region to caclulate uncertainty
                    if isinstance(error, bool):
                        # if median used calculate from mask generated, otherwise measure aperture of x^2 values
                        if backMedian:
                            if maskNaN:
                                noiseBack = np.nanstd(backImage[backMask > 0])
                            else:
                                noiseBack = np.std(backImage[backMask > 0])
                        else:
                            ind_backSquare_table = aperture_photometry(self.image**2.0, backApertures[i], wcs=imgWCS, method=method, subpixels=subpixels, mask=nanMask)
                            noiseBack = np.sqrt(ind_backSquare_table['aperture_sum'][0]/ind_back_nPixTable['aperture_sum'] - backValues**2.0)             
                    else:
                        noiseBack = error
                    apError = noiseBack * np.sqrt(ind_nPixTable['aperture_sum'])
                
                # if local background subtraction done include that contribution
                if localBackSubtract:
                    if isinstance(localBackSubtract, (list,tuple,np.ndarray)) is False or localBackSubtract[i] is not None:
                        if errorImage is True:
                            var_back_table = aperture_photometry(error**2.0, backApertures[i], wcs=imgWCS, method=method, subpixels=subpixels, mask=nanMask)
                            backValError = np.sqrt(var_back_table['aperture_sum']) / backNpix
                        else:
                            backValError = noiseBack / backNpix
                        
                        backError = backValError * ind_nPixTable['aperture_sum']
                    
                        # combine background and aperture error    
                        apError = np.sqrt(apError**2.0 + backError**2.0)
                
                # save error result to table
                ind_phot_table['aperture_error'] = apError
            
            if multiRadius:
                # if first run intiate 2D grid to store results
                if i == 0:
                    multiRadApSum = np.zeros((len(centres),len(masterRadius)))
                    multiRadApSum[:,:] = np.nan
                    multiRadNpix = np.zeros((len(centres),len(masterRadius)))
                    multiRadNpix[:,:] = np.nan
                    if error is not None:
                        multiRadApErr = np.zeros((len(centres),len(masterRadius)))
                        multiRadApErr[:,:] = np.nan
                
                # if first aperture is zero set the first row flux to zero
                if i == 0 and zeroFirst:
                    multiRadApSum[:,0] = 0.0
                    multiRadNpix[:,0] = 0.0
                    if error is not None:
                        multiRadApErr[:,0] = 0.0
                
                # update the 2D arrays to include the values 
                multiRadApSum[MRi[i],MRj[i]] = ind_phot_table['aperture_sum']
                multiRadNpix[MRi[i],MRj[i]] = ind_nPixTable['aperture_sum']
                if error is not None:
                    multiRadApErr[MRi[i],MRj[i]] = ind_phot_table['aperture_error']
                
            else:
                # if first object create master table, otherwise append line.
                if i == 0:
                    phot_xcenter = [ind_phot_table['xcenter'][0].value]
                    phot_ycenter = [ind_phot_table['ycenter'][0].value]
                    if "sky_center" in ind_phot_table.colnames:
                        phot_sky_center = [ind_phot_table['sky_center'][0]]
                    phot_apsum = [ind_phot_table['aperture_sum'][0]]
                    phot_npix = [ind_nPixTable['aperture_sum'][0]]
                    if error is not None:
                        phot_aperr = [ind_phot_table['aperture_error'][0]]
                    #phot_table = ind_phot_table.copy()
                    #nPixTable = ind_nPixTable.copy()
                else:
                    #ind_phot_table['id'][0] = i+1
                    #phot_table.add_row(ind_phot_table[-1])
                    #nPixTable.add_row(ind_nPixTable[-1])
                    phot_xcenter.append(ind_phot_table['xcenter'][0].value)
                    phot_ycenter.append(ind_phot_table['ycenter'][0].value)
                    if "sky_center" in ind_phot_table.colnames:
                        phot_sky_center.append(ind_phot_table['sky_center'][0])
                    phot_apsum.append(ind_phot_table['aperture_sum'][0])
                    phot_npix.append(ind_nPixTable['aperture_sum'][0]) 
                    if error is not None:
                        phot_aperr.append(ind_phot_table['aperture_error'][0])
        
        # now process the table to the correct format
        if multiRadius:
            phot_table = Table()
            if mode == "circle":
                phot_table['Radius'] = radius
            elif mode == "ellipse":
                phot_table['Semi-Major'] = radius
            elif mode == "rectangle":
                phot_table['Semi-Length'] = radius
            
            # loop over each centre and add to table columns
            for i in range(0,len(centres)):
                # extract source name
                if names is not None:
                    sourceName = names[i]
                else:
                    sourceName = "Source" + str(i)
                
                # add columns to table
                phot_table[sourceName+"_number_pixels"] = multiRadNpix[i,:]
                if calculateMean:
                    phot_table[sourceName+"_aperture_mean"] = multiRadApSum[i,:] / multiRadNpix[i,:]
                    if error is not None:
                        phot_table[sourceName+"_aperture_mean_error"] = multiRadApErr[i,:] / multiRadNpix[i,:]
                else:
                    phot_table[sourceName+"_aperture_sum"] = multiRadApSum[i,:]
                    if error is not None:
                        phot_table[sourceName+"_aperture_error"] = multiRadApErr[i,:]
                
                # add units if possible
                if hasattr(self, 'unit'):
                    if calculateMean:
                        phot_table[sourceName+"_aperture_mean"].unit = self.unit
                        if error is not None:
                            phot_table[sourceName+"_aperture_mean_error"].unit = self.unit
                    else:
                         # modify the unit
                        for unitClass in programedUnits:
                            if self.unit in programedUnits[unitClass]:
                                newUnit = self.unit.split("/pix")[0]
                        
                        phot_table[sourceName+'_aperture_sum'].unit = newUnit
                        if error is not None:
                            phot_table[sourceName+'_aperture_error'].unit = newUnit
        else:
            # create table
            phot_table = Table()
            
            # ad rows
            phot_table['id'] = range(1,len(phot_xcenter)+1)
            phot_table['xcentre'] = phot_xcenter
            phot_table['ycentre'] = phot_ycenter
            if "sky_center" in ind_phot_table.colnames:
                phot_table['sky_centre'] = phot_sky_center
            phot_table['number_pixels'] = phot_npix
            phot_table['aperture_sum'] = phot_apsum
            if error is not None:
                phot_table['aperture_error'] = phot_aperr
                       
            # add unit to xcenter and ycenter
            phot_table['xcentre'].unit = u.pix
            phot_table['ycentre'].unit = u.pix
                        
            # change to mean if in surface brightness units
            if calculateMean:    
                phot_table['aperture_mean'] = phot_table['aperture_sum'] / phot_table['number_pixels']
                del(phot_table['aperture_sum'])
                if error is not None:
                    phot_table['aperture_mean_error'] = phot_table['aperture_error'] / phot_table['number_pixels']
                    del(phot_table['aperture_error'])
            
            # try adding in unit
            if hasattr(self, 'unit'):
                #unts = self.programmedUnits()
                if calculateMean:
                    phot_table['aperture_mean'].unit = self.unit
                    if error is not None:
                        phot_table['aperture_mean_error'].unit = self.unit
                else:
                    # modify the unit
                    for unitClass in programedUnits:
                        if self.unit in programedUnits[unitClass]:
                            newUnit = self.unit.split("/pix")[0]
                    
                    phot_table['aperture_sum'].unit = newUnit
                    if error is not None:
                        phot_table['aperture_error'].unit = newUnit
            
            # see if wanted to adjust names
            if names is not None:
                if isinstance(names, (list, tuple, np.ndarray)) is False:
                    names = [names]
                
                # check length matches number of centres
                if len(centres) != len(names):
                    print("Unable to apply names - different length to provided centres")

                phot_table['id'] = names                
                #phot_table.replace_column('id',Column(data=names, name='id', dtype='str'))
        
        
        return phot_table
    
    def halfBinArrays(self, radArray):
        # function which calculates halfway bins for surface brightness function
        
        nestedList = True
        if isinstance(radArray, list):
            if isinstance(radArray[0], (list, tuple, np.ndarray)) is False:
                adjustedArray = [radArray]
                nestedList = False
            else:
                adjustedArray = radArray
        else:
            adjustedArray = [radArray]
            nestedList = False
        
        # if nestedList need to create a list to store half-arrays
        if nestedList:
            outBins = []
        
        # loop over everyset of radii
        for i in range(0,len(adjustedArray)):
            currentBins = adjustedArray[i]
        
            # see if starts with a 'radius' of zero
            zeroFirst=False
            if isinstance(currentBins[0],u.Quantity) is True:
                if currentBins[0].value == 0.0:
                    zeroFirst = True
            else:
                if currentBins[0] == 0.0:
                    zeroFirst = True
            
            if zeroFirst:
                halfBins = np.array([])
                if isinstance(currentBins[0],u.Quantity):
                    halfBins = halfBins * currentBins[0].unit
            else:
                # first see if in the step size we would go below r=0.0 
                if (currentBins[1] - currentBins[0]) / 2.0 >= currentBins[0]:
                    if isinstance(currentBins[0],u.Quantity):
                        halfBins = 0.0 * currentBins[0].unit
                    else:
                        halfBins = 0.0
                else:
                    halfBins = currentBins[0] - (currentBins[1] - currentBins[0]) / 2.0
                
            # now add in all the steps between
            halfBins = np.append(halfBins, currentBins[:-1] + (currentBins[1:]-currentBins[:-1])/2.0)
                
            # now add final bin
            halfBins = np.append(halfBins, currentBins[-1] + (currentBins[-1]-currentBins[-2])/2.)
        
            if nestedList:
                outBins.append(halfBins)
            else:
                outBins = halfBins
        
        return outBins
                
    def surfaceBrightness(self, phot_table, half_bin_table):
        # function to calculate surface brightness profiles
        
        # get list objects, and find if surface brightness or sum
        colnames = phot_table.colnames
        objNames = []
        for colname in colnames[1:]:
            if colname[-12:] == "aperture_sum":
                objNames.append(colname[:-13])
                surfaceBrightnessUnits = False
            if colname[-12:] == "aperture_mean":
                objNames.append(colname[:-13])
                surfaceBrightnessUnits = True
        
        # see if can get pixel area otherwise do in units of per pixel
        if hasattr(self, "pixSize"):
            try:
                self.getPixelScale()
            except:
                pass
        if hasattr(self, "pixSize"):
            pixArea = (self.pixSize)**2.0
            pixArea = pixArea.to(u.arcsec**2.0).value
            pixAreaKnown = True
        else:
            pixArea = 1.0
            pixAreaKnown = False
                
        # loop over each object
        for objName in objNames:
            # see if starts with a 'radius' of zero
            if phot_table[colnames[0]][0] == 0.0:
                if surfaceBrightnessUnits:
                    surfaceBrightness = np.array(half_bin_table[objName+"_aperture_mean"][0]*half_bin_table[objName+"_number_pixels"][0] / (half_bin_table[objName+"_number_pixels"][0]*pixArea))
                else:
                    surfaceBrightness = np.array(half_bin_table[objName+"_aperture_sum"][0] / (half_bin_table[objName+"_number_pixels"][0]*pixArea))
            else:
                surfaceBrightness = np.array([])
            
            # calculate rest of surface brightness points
            if surfaceBrightnessUnits:
                surfaceBrightness = np.append(surfaceBrightness,(half_bin_table[objName+"_aperture_mean"].data[1:]*half_bin_table[objName+"_number_pixels"].data[1:] - half_bin_table[objName+"_aperture_mean"].data[:-1]*half_bin_table[objName+"_number_pixels"].data[:-1]) / \
                                             ((half_bin_table[objName+"_number_pixels"].data[1:] - half_bin_table[objName+"_number_pixels"].data[:-1])*pixArea)) 
            else:
                surfaceBrightness = np.append(surfaceBrightness,(half_bin_table[objName+"_aperture_sum"].data[1:] - half_bin_table[objName+"_aperture_sum"].data[:-1]) / \
                                             ((half_bin_table[objName+"_number_pixels"].data[1:] - half_bin_table[objName+"_number_pixels"].data[:-1])*pixArea)) 
            
            # see if error has been included
            if objName + "_aperture_error" in colnames or objName + "_aperture_mean_error" in colnames:
                if phot_table[colnames[0]][0] == 0.0:
                    if surfaceBrightnessUnits:
                        surfaceBrightnessErr = np.array(half_bin_table[objName+"_aperture_mean_error"][0]*half_bin_table[objName+"_number_pixels"][0] / (half_bin_table[objName+"_number_pixels"][0]*pixArea))
                    else:
                        surfaceBrightnessErr = np.array(half_bin_table[objName+"_aperture_error"][0] / (half_bin_table[objName+"_number_pixels"][0]*pixArea))
                else:
                    surfaceBrightness = np.array([])
                
                # calculate rest of surface brightness points
                if surfaceBrightnessUnits:
                    surfaceBrightnessErr = np.append(surfaceBrightness,np.sqrt((half_bin_table[objName+"_aperture_mean_error"].data[1:]*half_bin_table[objName+"_number_pixels"].data[1:])**2.0 - (half_bin_table[objName+"_aperture_mean_error"].data[:-1]*half_bin_table[objName+"_number_pixels"].data[:-1])**2.0) / \
                                                 ((half_bin_table[objName+"_number_pixels"].data[1:] - half_bin_table[objName+"_number_pixels"].data[:-1])*pixArea)) 
                else:
                    surfaceBrightnessErr = np.append(surfaceBrightness,np.sqrt(half_bin_table[objName+"_aperture_error"].data[1:]**2.0 - half_bin_table[objName+"_aperture_error"].data[:-1]**2.0) / \
                                                 ((half_bin_table[objName+"_number_pixels"].data[1:] - half_bin_table[objName+"_number_pixels"].data[:-1])*pixArea)) 
             
            # add surface brightness to the table
            phot_table[objName+"_surface_brightness"] = surfaceBrightness
            phot_table[objName+"_surface_brightness"].unit = self.unit + " arcsec^-2"
            if objName + "_aperture_error" in colnames or objName + "_aperture_mean_error" in colnames:
               phot_table[objName+"_surface_brightness_error"] = surfaceBrightness
               phot_table[objName+"_surface_brightness_error"].unit = self.unit + " arcsec^-2" 
            
        
    def coordMaps(self):
        # function to find ra and dec co-ordinates of every pixel
        
        # import modules
        from astropy.coordinates import ICRS
        
        # Parse the WCS keywords in the primary HDU
        header = self.header
        wcs = wcs.WCS(self.header)
        
        # Make input arrays for every pixel on the map
        xpix = np.zeros((header["NAXIS1"]*header["NAXIS2"]),dtype=int)
        for i in range(0,header["NAXIS2"]):
            xpix[i*header["NAXIS1"]:(i+1)*header["NAXIS1"]] = np.arange(0,header["NAXIS1"],1)
        ypix = np.zeros((header["NAXIS1"]*header["NAXIS2"]),dtype=int)
        for i in range(1,header["NAXIS2"]):
            ypix[(i)*header["NAXIS1"]:(i+1)*header["NAXIS1"]] = i
        
        # Convert all pixels into sky co-ordinates
        sky = wcs.pixel_to_world(xpix,ypix)
        
        # check that is in IRCS format
        if sky.is_equivalent_frame(ICRS()) is False:
            icrs = sky.transform_to('icrs')
            raMap = icrs.ra.value
            decMap = icrs.dec.value
        else:
            raMap = sky.ra.value
            decMap = sky.dec.value
        
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
        beamAreas = {"SCUBA-2":{"450":141.713*u.arcsecond**2., "850":246.729*u.arcsecond**2.}, "SPIRE":{"250":469.4*u.arcsecond**2., "350":831.3*u.arcsecond**2., "500":1804.3*u.arcsecond**2.},\
                     "Planck":{"353":96170.4*u.arcsecond**2.}, "SCUBA-2&Planck":{"850":246.729*u.arcsecond**2.}, "SCUBA-2&SPIRE":{"450":141.713*u.arcsecond**2.}}
        if instrument is not None:
            return beamAreas[instrument][band]
        else:
            return beamAreas
    
    def standardCentralWavelengths(self, instrument=None, band=None):
        # define central wavelengths for bands in micron
        centralWavelengths = {"SCUBA-2":{"450":450.0*u.micron, "850":850.0*u.micron}, 
                              "SPIRE":{"250":250.0*u.micron, "350":350.0*u.micron, "500":500.0*u.micron},\
                              "PACS":{"70":70.0*u.micron, "100":100*u.micron, "160":160.0*u.micron},\
                              "Planck":{"353":850.0*u.micron}}
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
    
    def programmedUnits(self, SBinfo=False):
        # function to return dictionary of programmed units and groups
        
        # list of programmed units (grouped by just syntax differences)
        units = {"other":["pW", "K_CMB"],\
                 "Jy/arcsec^2":["Jy/arcsec^2", "Jy arcsec^-2", "Jy arcsec-2", "Jy arcsec**-2"],\
                 "mJy/arcsec^2":["mJy/arcsec^2", "mJy arcsec^-2", "mJy arcsec-2", "mJy/arcsec**2"], \
                 "MJy/sr":["MJy/sr", "MJy per sr", "MJy sr^-1", "MJy sr-1", "MJy sr**-1"],\
                 "Jy/beam":["Jy/beam", "Jy beam^-1", "Jy beam-1", "Jy beam**-1"],\
                 "mJy/beam":["mJy/beam", "mJy beam^-1", "mJy beam-1", "mJy beam**-1"],\
                 "Jy/pix":["Jy/pix", "Jy pix^-1", "Jy pix-1", "Jy pix**-1", "Jy/pixel", "Jy pixel^-1", "Jy pixel-1", "Jy pixel**-1"],\
                 "mJy/pix":["mJy/pix", "mJy pix^-1", "mJy pix-1", "mJy pix**-1", "mJy pixel^-1", "mJy pixel-1", "mJy pixel**-1"]}
        
        # is the unit surface brightness or not
        if SBinfo:
            masterGroupSB = {"other":True, "Jy/arcsec^2":True, "mJy/arcsec^2":True, "MJy/sr":True, "Jy/beam":True, "mJy/beam":True,\
                             "Jy/pix":False, "mJy/pix":False}
            SBunits = {}
            for unitClass in units:
                for unit in units[unitClass]:
                    SBunits[unit] = masterGroupSB[unitClass]
        
        if SBinfo:
            return units, SBunits
        else:
            return units
        
    
    def convertUnits(self, newUnit, conversion=None, beamArea=None, verbose=True):
        # function to convert units of map
        
        # if a conversion value given use that, if not calculate
        if conversion is not None:
            self.image = self.image * conversion
            self.header['BUNIT'] = newUnit
            self.unit = newUnit
            if "SIGUNIT" in self.header:
                self.header['SIGUNIT'] = newUnit
            if "ZUNITS" in self.header:
                self.header['ZUNITS'] = newUnit
            if verbose:
                print(self.band, " image converted to ", newUnit, " using provided conversion")
        else:
            
            # get list of programmed units
            units = self.programmedUnits()
            
            # make list of all units
            allUnits = []
            for unitClass in units:
                allUnits = allUnits + units[unitClass] 
            
            # get old unit
            oldUnit = self.header["BUNIT"]
                        
            # load programmed beam areas
            beamAreas = self.standardBeamAreas()
            # if beam area specified save it
            if beamArea is not None:
                beamAreas[self.instrument][self.band] = beamArea
            # if we don't have beam area see if can get from beam information
            if self.instrument not in beamAreas or self.band not in beamAreas[self.instrument]:
                if hasattr(self,'beam') is True:
                    if verbose:
                        print("Calculating Beam Area from Gaussian Beam information")
                    beamAreas[self.instrument][self.band] = 1.1331 * self.beam['BMAJ'] * self.beam['BMIN']
                        
            
            # program conversion SCUBA-2 pW to Jy/arcsec^2
            scubaConversions = {"450":{"Jy/beam":497.6, "Jy/arcsec^2":3.51}, "850":{"Jy/beam":480.5, "Jy/arcsec^2":1.95}}
            
            # program to convert planck to MJy/sr
            planckConversion = {"353":287.450}
            
            if oldUnit == newUnit:
                # check that not already in correct unit
                if verbose:
                    print("Image is already in correct units")
            else:
                # see if in a pre-progammed unit
                if oldUnit not in allUnits:
                    if verbose:
                        print("Image Unit: ", oldUnit, " not programmed - result maybe unreliable")
                if newUnit not in allUnits:
                    if verbose:
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
                        if verbose:
                            print("Image converted to ", newUnit)
                        return
                elif self.header['BUNIT'] == 'pW':
                    raise ValueError("Can only process pW from SCUBA-2")
                
                # check if Planck instruments in unit of K_CMB
                if self.instrument == "Planck" and self.header['BUNIT'] == "K_CMB":
                    self.image = self.image * planckConversion[self.band]
                    self.header['BUNIT'] = "MJy/sr"
                    self.unit = "MJy/sr"
                    oldUnit = "MJy/sr"
                elif self.header['BUNIT'] == "K_CMB":
                    raise ValueError("Can only process K_CMB from Planck")
                
                ### process the old units
                if oldUnit in units["Jy/pix"]:
                    conversion = 1.0 * u.Jy
                    pixArea = self.pixSize * self.pixSize
                    conversion = conversion / pixArea
                elif oldUnit in units["mJy/pix"]:
                    conversion = 0.001 * u.Jy
                    pixArea = self.pixSize * self.pixSize
                    conversion = conversion / pixArea
                elif oldUnit in units["Jy/beam"]:
                    conversion = 1.0 * u.Jy
                    #pixArea = self.pixSize * self.pixSize
                    #conversion = conversion * pixArea / beamAreas[self.instrument][self.band]
                    conversion = conversion / (beamAreas[self.instrument][self.band])
                elif oldUnit in units["mJy/beam"]:
                    conversion = 0.001 * u.Jy
                    #pixArea = self.pixSize * self.pixSize
                    #conversion = conversion * pixArea / beamAreas[self.instrument][self.band]
                    conversion = conversion / (beamAreas[self.instrument][self.band])
                elif oldUnit in units["Jy/arcsec^2"]:
                    conversion = 1.0 * u.Jy / u.arcsecond**2.0
                elif oldUnit in units["mJy/arcsec^2"]:
                    conversion = 0.001 * u.Jy / u.arcsecond**2.0
                elif oldUnit in units["MJy/sr"]:
                    conversion = 1.0e6 * u.Jy / u.sr
                else:
                    raise ValueError("Unit not programmed: ", oldUnit)
                                
                # convert to new unit
                if newUnit in units["Jy/pix"] or newUnit in units["mJy/pix"] or newUnit in units["Jy/beam"] or newUnit in units["mJy/beam"]:
                    # convert to Jy per arcsec^2
                    conversion = conversion.to(u.Jy/u.arcsecond**2.0).value
                    if newUnit in units["Jy/pix"]:
                        pixArea = self.pixSize * self.pixSize
                        conversion = conversion * pixArea.to(u.arcsecond**2.0).value 
                    elif newUnit in units["mJy/pix"]:
                        pixArea = self.pixSize * self.pixSize
                        conversion = conversion * pixArea.to(u.arcsecond**2.0).value * 1000.0
                    elif newUnit in units["Jy/beam"]:
                        conversion = (conversion * beamAreas[self.instrument][self.band]).value
                    elif newUnit in units["mJy/beam"]:
                        conversion = (conversion * beamAreas[self.instrument][self.band] * 1000.0).value
                elif newUnit in units["Jy/arcsec^2"]:
                    conversion = conversion.to(u.Jy/u.arcsecond**2.0).value
                elif newUnit in units["mJy/arcsec^2"]:
                    conversion = conversion.to(u.Jy/u.arcsecond**2.0).value * 1000.0
                elif newUnit in units["MJy/sr"]:
                    conversion = conversion.to(u.Jy/u.sr).value * 1.0e-6
                elif newUnit == "pW" and self.instrument == "SCUBA-2":
                    conversion = (conversion * beamAreas[self.instrument][self.band]).value
                    conversion = conversion / scubaConversions[self.band]['Jy/beam']
                else:
                    raise ValueError("Unit not programmed")
                
                self.image = self.image * conversion
                self.header['BUNIT'] = newUnit
                self.unit = newUnit
                if "SIGUNIT" in self.header:
                    self.header['SIGUNIT'] = newUnit
                if "ZUNITS" in self.header:
                    self.header['ZUNITS'] = newUnit
                if "QTTY____" in self.header:
                    self.header['QTTY____'] = newUnit
                if verbose:
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
            predictedNewWave = ppMap.artificialImage(newWavelength, adjustSettings["tau"], adjustSettings["tauWavelength"])
            predictedCurrWave = ppMap.artificialImage(self.wavelength, adjustSettings["tau"], adjustSettings["tauWavelength"])
            
            # set variable that using a map based (rather than a constant across whole image
            mapMethod = True
        # see if the case of using a constant correction across entire image
        elif adjustSettings["temperature"] is not None and isinstance(adjustSettings["temperature"],str) is False and adjustSettings["beta"] is not None and isinstance(adjustSettings["beta"],str) is False:
            # if constant just compare what a blackbody would be before and after
            blackbody = blackbody_nu(temperature=adjustSettings["temperature"]*u.K)
            newLevel = (con.c/(newWavelength*u.um))**adjustSettings["beta"] * blackbody(newWavelength*u.um)
            currLevel = (con.c/(currentWavelength*u.um))**adjustSettings["beta"] * blackbody(currentWavelength*u.um)
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
                kernel = np.sqrt(imageFWHM**2.0 - refFWHM**2.0) 
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
            ratioMap = ratioMap.reproject(self.header, exact=False)
            
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
            predictedMapWithCC = ppMap.artificialImage(self.wavelength, adjustSettings["tau"], adjustSettings["tauWavelength"],ccVals=ccPPMAPvals)
            predictedMapNoCC = ppMap.artificialImage(self.wavelength, adjustSettings["tau"], adjustSettings["tauWavelength"])
            
                        
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
                kernel = np.sqrt(imageFWHM**2.0 - refFWHM**2.0)
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
            ccMap = ccMap.reproject(self.header, exact=False)
            
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
    
    
    def reproject(self, projHead, exact=True, conserveFlux=False):
        # function to reproject the fits image
        from reproject import reproject_from_healpix, reproject_interp, reproject_exact
        
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
        
        
        # modify original header
        # projection keywords
        projKeywords = ["NAXIS1", "NAXIS2", "LBOUND1", "LBOUND2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",\
                        "CTYPE1", "CTYPE2", "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",\
                        "RADESYS", "EQUINOX", "CROTA2", "CROTA1"]
        header = self.header.copy()
        for keyword in projKeywords:
            if keyword in projHead:
                header[keyword] = projHead[keyword]
            else:
                try:
                    del(header[keyword])
                except:
                    pass
        
        # create reprojected image hdu
        repoHdu = pyfits.PrimaryHDU(resampleMap, header)
        repoHdulist = pyfits.HDUList([repoHdu])
        
        # create combine astro image
        repoImage = astroImage(repoHdulist, load=False, instrument=self.instrument, band=self.band)
          
        
        # see if need to correct image to conserve flux rather than surface brightness and correct
        if conserveFlux or self.unit=="Jy/pix" or self.unit=="mJy/pix":
            # get original pixel size
            if hasattr(self, "pixSize") is False:
                self.getPixelScale()
            origPixSize = self.pixSize
            
            # get output pixel size
            repoImage.getPixelScale()
            outPixSize = repoImage.pixSize

            # adjust image for difference in pixel area
            repoImage.image = repoImage.image * (outPixSize**2.0/origPixSize**2.0).to(u.dimensionless_unscaled).value
            
        # return new image
        return repoImage
    
        
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
    
    
    def convolve(self, kernel, boundary='fill', fill_value=0.0, peakNorm=False, FWHM=True):
        
        # import modules
        from astropy.convolution import convolve_fft as APconvolve_fft
        from astropy.convolution import Gaussian2DKernel
        
        # see if 2D kernel is a number or an array
        if isinstance(kernel, type(1.0*u.arcsecond)) is False:
            kernelImage = kernel
        else:
            if FWHM:
                stddev = (kernel / (self.pixSize * 2.0*np.sqrt(2.0*np.log(2.0)))).to(u.dimensionless_unscaled).value
            else:
                stddev = (kernel / self.pixSize).to(u.dimensionless_unscaled).value
            
            kernelImage = Gaussian2DKernel(x_stddev = stddev)
            kernelImage = kernelImage.array
        
        # renormalise so peak is one
        kernelImage = kernelImage / kernelImage.max()
        
        # find positions that are NaNs
        NaNsel = np.where(np.isnan(self.image) == True)
        
        # set if have to normalise kernel
        if peakNorm:
            normKernel = False
        else:
            normKernel = True
        
        if boundary == 'fill':
            convolvedArray = APconvolve_fft(self.image, kernelImage, boundary=boundary, fill_value=fill_value, allow_huge=True, normalize_kernel=normKernel)
        else:
            convolvedArray = APconvolve_fft(self.image, kernelImage, boundary=boundary, allow_huge=True, normalize_kernel=normKernel)
        
        # restore NaNs
        convolvedArray[NaNsel] = np.nan
        
        # create combined image hdu
        convolveHeader = self.header
        convolveHdu = pyfits.PrimaryHDU(convolvedArray, convolveHeader)
        convolveHdulist = pyfits.HDUList([convolveHdu])
        
        # create combine astro image
        convolvedImage = astroImage(convolveHdulist, load=False, instrument=self.instrument, band=self.band)
        
        return convolvedImage
    
    
    def cutout(self, centre, size, copy=False):
        # function to create a cutout of the image
        
        # import astropy cutout routing
        from astropy.coordinates import SkyCoord
        from astropy.nddata.utils import Cutout2D
        
        # centre can be a SkyCoord or (X, Y pixels)
        if isinstance(centre,SkyCoord) is False:
            if isinstance(centre[0], u.Quantity) is False:
                print("Units on centre not given - assuming in pixel coordinates")
        
        # adjust size info to flip as takes sizeY, sizeX, or warn if not in coordinates
        if isinstance(size, (list, tuple, np.ndarray)) and isinstance(size,u.Quantity) is False:
            newSize = (size[1], size[0])
            if isinstance(size[0],u.Quantity) is False:
                print("Units on size not given - assumin in pixel coordinates")
        elif isinstance(size,u.Quantity):
            if len(size.shape) > 0:
                newSize = (size[1], size[0])
            else:
                newSize = size
        else:
            newSize = size
            print("Units on size not given - assumin in pixel coordinates")
        
        
        # create WCS information
        WCSinfo = wcs.WCS(self.header)
            
        # create cutout
        cutoutOut = Cutout2D(self.image, centre, newSize, wcs=WCSinfo, mode='partial', fill_value=np.nan, copy=copy)
       
        # create new header
        cutHeader = self.header.copy()
        cutHeadWCS = cutoutOut.wcs.to_header()
        for keyword in cutHeadWCS:
            cutHeader[keyword] = cutHeadWCS[keyword]
        
        # create astro image object from output
        cutoutHdu = pyfits.PrimaryHDU(cutoutOut.data, cutHeader)
        cutoutHdulist = pyfits.HDUList([cutoutHdu])
        
        # create combine astro image
        cutoutImage = astroImage(cutoutHdulist, load=False, instrument=self.instrument, band=self.band)
        
        return cutoutImage
        
        
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
        d *= self.pixSize.to(u.arcsecond).value
        
        # Calculate the frequencies in the Fourier plane to create a filter
        x_f,y_f = np.meshgrid(np.fft.fftfreq(hires.shape[0],self.pixSize.to(u.arcsecond).value),
                              np.fft.fftfreq(hires.shape[1],self.pixSize.to(u.arcsecond).value))
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
        ratio = (beamAreas[self.instrument][self.band] / beamAreas[lowresImage.instrument][lowresImage.band]).to(u.dimensionless_unscaled).value
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
    
    
    def plot(self, recentre=None, stretch='linear', vmin=None, vmid=None, vmax=None, cmap=None, facecolour='white', nancolour='black', hide_colourbar=False, save=None):
        # function to make a quick plot of the data using matplotlib and aplpy
        
        # import modules
        import aplpy
        import matplotlib.pyplot as plt
        
        # create figure
        fig = plt.figure()
        
        # repackage into an HDU 
        hdu = pyfits.PrimaryHDU(self.image, self.header)
        
        # create aplpy axes
        f1 = aplpy.FITSFigure(hdu, figure=fig)
        
        # if doing a log stretch find vmax, vmid, vmin
        if stretch == "log":
            if vmin is None or vmax is None or vmid is None:
                # select non-NaN pixels
                nonNAN = np.where(np.isnan(self.image) == False)
                
                # sort pixels
                sortedPix = self.image[nonNAN]
                sortedPix.sort()
                
                # set constants
                minFactor = 1.0
                brightPixCut = 5
                brightClip = 0.9
                midScale = 301.0
                
                if vmin is None:
                    numValues = np.round(len(sortedPix) * 0.95).astype(int)
                    vmin = -1.0 * sortedPix[:-numValues].std() * minFactor
                
                if vmax is None:
                    vmax = sortedPix[-brightPixCut] * brightClip
                
                if vmid is None:
                    vmid=(midScale * vmin - vmax)/100.0
        
        
        # apply colourscale
        f1.show_colorscale(stretch=stretch, cmap=cmap, vmin=vmin, vmax=vmax, vmid=vmid)
        
        # set nan colour to black, and face
        f1.set_nan_color(nancolour)
        f1.ax.set_facecolor(facecolour)
        
        # recentre image
        if recentre is not None:
            # import skycoord object
            from astropy.coordinates import SkyCoord
            
            # creat flag to check if centre found
            noCentre = False
            
            # get/calculate SkyCood object
            if "coord" in recentre:
                centreCoord = recentre["coord"]            
            elif "RA" in recentre and "DEC" in recentre:
                centreCoord = SkyCoord(ra=recentre['RA'], dec=recentre['DEC'], frame='icrs')
            elif "l" in recentre and "b" in recentre:
                centreCoord = SkyCoord(l=recentre["l"], b=recentre['b'], frame='galactic')
            else:
                noCentre = True
                print("Cannot recentre as no coordinate information identified")
            
            
            if noCentre is False:
                # get WCS infomation
                WCSinfo = wcs.WCS(self.header)
                
                # convert to xpix and ypix
                xpix, ypix = wcs.utils.skycoord_to_pixel(centreCoord, WCSinfo)
                
                # convert back to sky coordinates of image for APLpy
                worldCoord = WCSinfo.all_pix2world([xpix],[ypix],0)
            
                
                # see if radius or length/width data present 
                if "rad" in recentre:    
                    f1.recenter(worldCoord[0][0], worldCoord[1][0], radius=recentre['rad'].to(u.degree).value)
                elif "radius" in recentre:
                    f1.recenter(worldCoord[0][0], worldCoord[1][0], radius=recentre['radius'].to(u.degree).value)
                elif "width" in recentre and "height" in recentre:
                    f1.recenter(worldCoord[0][0], worldCoord[1][0], width=recentre['width'].to(u.degree).value, height=recentre['height'].to(u.degree).value)
                else:
                    print("Cannot recentre as no size information identified")
        
        # add colorbar
        if hide_colourbar is False:
            f1.add_colorbar()
            f1.colorbar.show()
            if hasattr(self, 'unit'):
                f1.colorbar.set_axis_label_text(self.unit)
        
        # save plot if desired
        if save is not None:
            plt.savefig(save)
        
        plt.show()
    
    
    def saveToFits(self, outPath, overwrite=False):
        # function to save to fits
        
        fitsHdu = pyfits.PrimaryHDU(self.image, self.header)
        fitsHduList = pyfits.HDUList([fitsHdu])
        
        fitsHduList.writeto(outPath, overwrite=overwrite)

        return
    

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
    