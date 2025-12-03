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

###############################################################################################################
###############################################################################################################
###############################################################################################################

# define classes

# image class to make adjustments to image required
class astroImage(object):
    
    
    def __init__(self, filename, ext=0, telescope=None, instrument=None, band=None, unit=None, load=True, FWHM=None, slices=None, dustpediaHeaderCorrect=None, verbose=True):
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
                
                # see what order slices is given in, and if using zero index
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
                # covert to tuple
                imgSlices = tuple(imgSlices)

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
        
        
        # try and identify if dustpedia file
        if dustpediaHeaderCorrect is None:
            # try to automatically detect if dustpedia image with issue
            try:
                if self.header['COORDSYS'].count("/") > 0 and self.header['SIGUNIT'].count("/") > 0:
                    dustpediaHeaderCorrect = True
                else:
                    dustpediaHeaderCorrect = False
            except:
                dustpediaHeaderCorrect = False

        # correct dustpedia header
        if dustpediaHeaderCorrect:
            keywordAdjust = ["COORDSYS", "SIGUNIT", "TELESCOP", "INSTRMNT", "DETECTOR", "WVLNGTH", "HIPE_CAL", "TARGET"]
            for keyword in keywordAdjust:
                if keyword in self.header and isinstance(self.header[keyword],str):
                    info = self.header[keyword].split("/")
                    if keyword == "SIGUNIT":
                        info2 = self.header[keyword].split("/ Unit of the map")
                        if len(info2) > 1:
                            self.header[keyword] = (info2[0],"Unit of the map")
                        else:
                            self.header[keyword] = info2[0]
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
        
        # correct telescope name if needed
        if self.telescope == "act":
            self.telescope = "ACT"
            self.header['TELESCOP'] = "ACT"
        elif self.telescope == "act+planck":
            self.telescope = "ACT&Planck"
            self.header['TELESCOP'] = "ACT&Planck"
        elif self.telescope == "CTIO 4.0-m telescope":
            self.telescope = "CTIO"
            self.header['TELESCOP'] = "CTIO"

        # identify instrument
        if instrument is None:
            if 'INSTRUME' in self.header:
                self.instrument = self.header['INSTRUME']
            elif 'INSTRMNT' in self.header:
                self.instrument = self.header['INSTRMNT']
            elif 'ORIGIN' in self.header and self.header['ORIGIN'] == "2MASS":
                self.instrument = "2MASS"
            elif self.telescope == "WISE":
                self.instrument = "WISE"
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
            elif 'WAVELN' in self.header:
                self.band = self.header['WAVELN']
            elif 'FREQ' in self.header:
                if self.telescope == "ACT" or self.telescope == "ACT&Planck":
                    if self.header['FREQ'][0] == 'f':
                        self.band = self.header['FREQ'][1:]
                    else:
                        self.band = self.header['FREQ']
                else:
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
            elif 'BAND' in self.header:
                self.band = self.header['BAND']
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
    
        # see if GALEX needs band converting
        if self.instrument == "GALEX":
            if self.band == "1528":
                self.band = "FUV"
            elif self.band == "2271":
                self.band == "NUV"
       
        # see if 2MASS band needs adjusting
        if self.instrument == "2MASS":
            if self.band in ['j', 'h']:
                self.band = self.band.upper()
            if self.band == "K" or self.band == "k":
                self.band = "Ks"
            if self.band == "ks":
                self.band = "Ks"

        # see if WISE band needs adjusting
        if self.instrument == "WISE":
            if self.band == "1" or self.band == 1 or self.band == "W1":
                self.band = "3.4"
            elif self.band == "2" or self.band == 2 or self.band == "W2":
                self.band = "4.6"
            elif self.band == "3" or self.band == 3 or self.band == "W3":
                self.band = "12"
            elif self.band == "4" or self.band == 4 or self.band == "W4":
                self.band = "22"

        # see if DECcam filter needs updating
        if self.instrument == "DECam":
            if len(self.band) > 7 and self.band[1:7] == " DECam":
                self.band = self.band[0]

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
                if self.instrument is not None and self.band is not None:
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
    
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    
    # section of pre-programmed data/instrument properties
    
    
    def standardBeamAreas(self, instrument=None, band=None):
        # define standard beam areas
        beamAreas = {"SCUBA-2":{"450":141.713*u.arcsecond**2., "850":246.729*u.arcsecond**2.},\
                     "SPIRE":{"250":469.4*u.arcsecond**2., "350":831.3*u.arcsecond**2., "500":1804.3*u.arcsecond**2.},\
                     "NIKA-2":{"260":152.5*u.arcsecond**2., "160":367.1*u.arcsecond**2.},\
                     "Planck":{"857":24.244*u.arcmin**2, "545":26.535*u.arcmin**2, "353":26.714*u.arcmin**2, "217":28.447*u.arcmin**2,\
                               "143":59.954*u.arcmin**2, "100":105.778*u.arcmin**2, "070":200.742*u.arcmin**2, "044":832.946*u.arcmin**2, "030":1189.513*u.arcmin**2}, 
                     "SCUBA-2&Planck":{"850":246.729*u.arcsecond**2.}, "SCUBA-2&SPIRE":{"450":141.713*u.arcsecond**2.},\
                     "NIKA-2&Planck":{"260":152.5*u.arcsecond**2., "160":367.1*u.arcsecond**2.},\
                     "ACT":{"220":117.6*u.nsr, "150":188.17*u.nsr, "090":490.29*u.nsr}, # 220 value from Pinceton specification site, other two from ACT-DR5 clusters page
                     "NIKA-2&ACT":{"260":152.5*u.arcsecond**2., "160":367.1*u.arcsecond**2.},\
                     }
        
        if instrument is not None:
            return beamAreas[instrument][band]
        else:
            return beamAreas
    
    ###############################################################################################################
    
    def standardCentralWavelengths(self, instrument=None, band=None):
        # define central wavelengths for bands in micron
        centralWavelengths = {"SCUBA-2":{"450":450.0*u.micron, "850":850.0*u.micron}, 
                              "SPIRE":{"250":250.0*u.micron, "350":350.0*u.micron, "500":500.0*u.micron},\
                              "PACS":{"70":70.0*u.micron, "100":100*u.micron, "160":160.0*u.micron},\
                              "NIKA-2":{"260":1.15*u.mm, "160":1.875*u.mm},\
                              "Planck":{"857":350.0*u.micron, "545":550*u.micron, "353":850.0*u.micron, "217":1.382*u.mm,\
                                        "143":2.097*u.mm, "100":3.0*u.mm, "070":4.286*u.mm, "044":6.818*u.mm, "030":10.0*u.mm},\
                              "GALEX":{"FUV":1528*u.angstrom, "NUV":2271*u.angstrom},\
                              "2MASS":{"J":1.235*u.micron, "H":1.662*u.micron, "Ks":2.159*u.micron},\
                              "UVOT":{"W1":2600*u.angstrom, "M2":2246*u.angstrom, "W2":1928*u.angstrom},\
                              "SDSS":{"u":3543*u.angstrom, "g":4770*u.angstrom, "r":6231*u.angstrom, "i":7625*u.angstrom, "z":9134*u.angstrom},\
                              "WISE":{"3.4":3.3526*u.micron, "4.6":4.6028*u.micron, "12":11.5608*u.micron, "22":22.0883*u.micron},\
                              "ACT":{"220":1.33835919*u.mm, "150":1.99861639*u.mm, "090":3.05910671*u.mm},\
                              "ACT&Planck":{"220":1.33835919*u.mm, "150":1.99861639*u.mm, "090":3.05910671*u.mm},\
                              }            
        
        if instrument is not None:
            return centralWavelengths[instrument][band]
        else:
            return centralWavelengths
    
    ###############################################################################################################
    
    def standardFWHM(self, instrument=None, band=None):
        # define central wavelengths for bands in micron
        FWHMs = {"SCUBA-2":{"450":7.9*u.arcsecond, "850":13.0*u.arcsecond},\
                 "SPIRE":{"250":17.6*u.arcsecond, "350":23.9*u.arcsecond, "500":35.2*u.arcsecond},\
                 "NIKA-2":{"260":11.6*u.arcsecond, "160":18.0*u.arcsecond},\
                 "Planck":{"857":4.325*u.arcmin, "545":4.682*u.arcmin, "353":4.818*u.arcmin, "217":4.990*u.arcmin,\
                           "143":7.248*u.arcmin, "100":9.651*u.arcmin, "070":13.252*u.arcmin, "044":27.005*u.arcmin, "030":32.239*u.arcmin},\
                 "GALEX":{"FUV":4.3*u.arcsec, "NUV":5.3*u.arcsec},\
                 "2MASS":{"J":2.8*u.arcsec, "H":2.7*u.arcsec, "Ks":2.8*u.arcsec},\
                 "UVOT":{"W1":2.37*u.arcsecond, "M2":2.45*u.arcsecond, "W2":2.92*u.arcsecond},\
                 "SDSS":{"u":1.53*u.arcsecond, "g":1.44*u.arcsecond, "r":1.32*u.arcsecond, "i":1.26*u.arcsecond, "z":1.29*u.arcsecond},\
                 "WISE":{"3.4":6.1*u.arcsecond, "4.6":6.4*u.arcsecond, "12":6.5*u.arcsecond, "22":12.0*u.arcsecond},\
                 "ACT":{"220":1.0*u.arcminute, "150":1.3*u.arcminute, "090":2.1*u.arcminute},\
                 "ACT&Planck":{"220":1.0*u.arcminute, "150":1.3*u.arcminute, "090":2.1*u.arcminute},\
                 }
        
        if instrument is not None and band is not None:
            return FWHMs[instrument][band]
        elif instrument is None and band is not None:
            raise Exception("Band specified but not Instrument")
        elif instrument is not None and band is None:
            raise Exception("Instrument specified but not Band")
        else:
            return FWHMs
    
    ###############################################################################################################

    def standardInstrumentalUnit(self, instrument, unit):
        # define standard instrumental units and provide function to test if they are programmed
        instrumentUnits = {"SCUBA-2":"pW",\
                           "Planck":"K_CMB",\
                           "GALEX":"ct/s",\
                           "2MASS":"DN",\
                           "WISE":"DN"}
        
        if unit is None:
            if instrument in instrumentUnits:
                return instrumentUnits[instrument]
            else:
                return None
        else:
            if instrument in instrumentUnits and instrumentUnits[instrument] == unit:
                return True
            else:
                return False
    
    ###############################################################################################################

    def standardInstrumentConversion(self, instrument=None, band=None):
        # standard instrument conversions from instrumental units to alternative units
        instrumentConversions = {"SCUBA-2":{"450":{"outUnit":"Jy/arcsec^2", "value":3.51}, "850":{"outUnit":"Jy/arcsec^2", "value":1.95}},\
                                 "Planck":{"857":{"outUnit":"MJy/sr", "value":2.27}, "545":{"outUnit":"MJy/sr", "value":58.04},\
                                           "353":{"outUnit":"MJy/sr", "value":287.450}, "217":{"outUnit":"MJy/sr", "value":483.690},\
                                           "143":{"outUnit":"MJy/sr", "value":371.74}, "100":{"outUnit":"MJy/sr", "value":244.1}},\
                                 "GALEX":{"FUV":{"outUnit":"Jy/pix", "value":1.076e-4}, "NUV":{"outUnit":"Jy/pix", "value":3.373e-5}},# Morriset 2007 \
                                 "2MASS":{"J":{"outUnit":"Jy/pix", "value":1594.0}, "H":{"outUnit":"Jy/pix", "value":1024.0}, "Ks":{"outUnit":"Jy/pix", "value":666.7}}, # Cohen 2002
                                 "WISE":{"3.4":{"outUnit":"Jy/pix", "value":1.9350e-6}, "4.6":{"outUnit":"Jy/pix", "value":2.7048e-6}, "12":{"outUnit":"Jy/pix", "value":1.8326e-6}, "22":{"outUnit":"Jy/pix", "value":5.2269e-5}},\
                                    } 
        
        if instrument is not None and band is not None:
            if instrument == "WISE":
                print("Conversion assuming ALLWISE values")
            if instrument == "2MASS":
                if "MAGZP" in self.header:
                    conversion = 10.0**(-self.header["MAGZP"]/2.5) * instrumentConversions[instrument][band]["value"]
                else:
                    raise Exception("2MASS conversion requires MAGZP keyword in header")
                return instrumentConversions[instrument][band]["outUnit"], conversion
            else:
                return instrumentConversions[instrument][band]["outUnit"], instrumentConversions[instrument][band]["value"]
        else:
            raise Exception("Unable to provide instrumental unit, module error - report")
        
    ###############################################################################################################
    
    def programmedUnits(self, SBinfo=False):
        # function to return dictionary of programmed units and groups
        
        # list of programmed units (grouped by just syntax differences)
        units = {"other":["pW", "K_CMB", "ct/s"],\
                 "Jy/arcsec^2":["Jy/arcsec^2", "Jy arcsec^-2", "Jy arcsec-2", "Jy arcsec**-2"],\
                 "mJy/arcsec^2":["mJy/arcsec^2", "mJy arcsec^-2", "mJy arcsec-2", "mJy/arcsec**2"], \
                 "MJy/sr":["MJy/sr", "MJy per sr", "MJy sr^-1", "MJy sr-1", "MJy sr**-1", "MJy / sr"],\
                 "Jy/beam":["Jy/beam", "Jy beam^-1", "Jy beam-1", "Jy beam**-1", "Jy / beam"],\
                 "mJy/beam":["mJy/beam", "mJy beam^-1", "mJy beam-1", "mJy beam**-1", "mJy / beam"],\
                 "Jy/pix":["Jy/pix", "Jy pix^-1", "Jy pix-1", "Jy pix**-1", "Jy/pixel", "Jy pixel^-1", "Jy pixel-1", "Jy pixel**-1", "Jy / pix"],\
                 "mJy/pix":["mJy/pix", "mJy/pixel", "mJy pix^-1", "mJy pix-1", "mJy pix**-1", "mJy pixel^-1", "mJy pixel-1", "mJy pixel**-1"],\
                 "uK_CMB":["uK_CMB", "uK CMB"],\
                 "K_CMB":["K_CMB", "K CMB"],\
                 "maggy":["maggy","maggies"],\
                 "nanomaggy":["nanomaggy", "nanomaggies"],\
                 }
        
        # is the unit surface brightness or not
        if SBinfo:
            masterGroupSB = {"other":True, "Jy/arcsec^2":True, "mJy/arcsec^2":True, "MJy/sr":True, "Jy/beam":True, "mJy/beam":True,\
                             "Jy/pix":False, "mJy/pix":False, "uK_CMB":True, "K_CMB":True, "maggy":False, "nanomaggy":False}
            SBunits = {}
            for unitClass in units:
                for unit in units[unitClass]:
                    SBunits[unit] = masterGroupSB[unitClass]
        
        if SBinfo:
            return units, SBunits
        else:
            return units
    
    ###############################################################################################################

    def isSurfaceBrightnessUnit(self):
        # function to see if image unit is a surface brightness unit

        # get programmed units
        programedUnits, SBinfo = self.programmedUnits(SBinfo=True)
        
        # see if image unit is in SB info
        if self.unit in SBinfo:
            if SBinfo[self.unit] is True:
                return True
            else:
                return False
        else:
            return None
    
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    
    
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

    ###############################################################################################################    
    
    def getPixelScale(self):
        # function to get pixel size
        WCSinfo = wcs.WCS(self.header)
        pixSizes = wcs.utils.proj_plane_pixel_scales(WCSinfo)*3600.0
        if np.abs(pixSizes[0]-pixSizes[1])/np.min(pixSizes) > 0.01:
            raise ValueError("PANIC - program does not cope with non-square pixels")
        self.pixSize = round(pixSizes[0], 6) * u.arcsecond
        return round(pixSizes[0], 6)
    
    ###############################################################################################################

    def coordInImage(self, coords, checkNaN=False, pixPosition=False):
        # function to see if a position or a list of positions is in an image

        # create WCS
        WCSinfo = wcs.WCS(self.header)

        # create output array
        inImage = np.zeros(len(coords), dtype=bool)

        # see if its a list
        if isinstance(coords, (list,tuple)):
            # create X, Y array
            pixCoordX = np.zeros(len(coords))
            pixCoordY = np.zeros(len(coords))
            # check each coordinate
            for i in range(0,len(coords)):
                pixCoordX[i], pixCoordY[i] = wcs.utils.skycoord_to_pixel(coords[i], WCSinfo)
        else:
            # get X, Y position from each coordinate
            pixCoordX, pixCoordY = wcs.utils.skycoord_to_pixel(coords, WCSinfo)

        # see if within the boundaries of the image
        sel = np.where((np.isnan(pixCoordX) == False) & (np.isnan(pixCoordY) == False) & (pixCoordX >= 0.0) & (pixCoordX <= self.image.shape[1]-1) & (pixCoordY >= 0.0) & (pixCoordY <= self.image.shape[0]-1))
        inImage[sel] = True

        # check if position in image is a NaN
        if checkNaN:
            sel = np.where(inImage == True)
            pixVals = self.image[np.array(np.round(pixCoordY[sel]), dtype=int), np.array(np.round(pixCoordX[sel]), dtype=int)]
            sel2 = np.where(np.isnan(pixVals) == True)
            inImage[sel][sel2] = False
        
        # check if coordinate was given as a single position and if so adjust output
        if isinstance(coords, (list,tuple)) is False and len(coords) == 1:

            if isinstance(coords.ra.value, (float)):
                inImage = inImage[0]
                pixCoordX = pixCoordX[0]
                pixCoordY = pixCoordY[0]

        if pixPosition:
            return inImage, (pixCoordX, pixCoordY)
        else:
            return inImage

    ###############################################################################################################

    def getPixelPosition(self, coords, oneIndex=False):
        # function to get ra and dec position of a pixel 
        # assumes X, Y

        # set origin
        if oneIndex:
            origin = 1
        else:
            origin = 0

        # create WCS
        WCSinfo = wcs.WCS(self.header)

        # see if just one coordinate provided as embedded single entry
        if len(coords) == 1:
            # create RA and dec array
            outPos = wcs.utils.pixel_to_skycoord(coords[0][0], coords[0][1], WCSinfo, origin=origin)

            return [outPos]
        elif len(coords) == 2:
            # see if just a coordinate or list of two coordinates
            if isinstance(coords[0], (list,tuple,np.ndarray)):
                outPos = [wcs.utils.pixel_to_skycoord(coords[i][0], coords[i][1], WCSinfo, origin=origin) for i in range(0,len(coords))]

                return outPos
            else:
                # create RA and dec array
                outPos = wcs.utils.pixel_to_skycoord(coords[0], coords[1], WCSinfo, origin=origin)

                return outPos

        else:
            outPos = [wcs.utils.pixel_to_skycoord(coords[i][0], coords[i][1], WCSinfo, origin=origin) for i in range(0,len(coords))]

            return outPos

    ###############################################################################################################

    def getPixelFromCoord(self, coords, oneIndex=False):
        # function to get pixel position from ra and dec
        # assumes ra, dec

        # set origin
        if oneIndex:
            origin = 1
        else:
            origin = 0

        # create WCS
        WCSinfo = wcs.WCS(self.header)

        # see if its a list
        if isinstance(coords, (list,tuple)):
            # create X, Y array
            outPix = []
            # check each coordinate
            for i in range(0,len(coords)):
                outPix.append(wcs.utils.skycoord_to_pixel(coords[i], WCSinfo, origin=origin))
        else:
            # get X, Y position from each coordinate
            outPix = wcs.utils.skycoord_to_pixel(coords, WCSinfo, origin=origin)

        # return output
        return outPix

    ###############################################################################################################
        
    def background_sigmaClip(self, snr=2, npixels=5, dilate_size=11, sigClip=3.0, iterations=20, mask=None, apply=False, returnMask=False, parallelMask=False):
        # function to get background level and noise
        
        # import modules
        from astropy.stats import sigma_clipped_stats
        if mask is None:
            #from photutils import make_source_mask
            pass
        if parallelMask:
            import multiprocessing as mp
        
        if mask is None:
            imgMask = make_source_mask(self.image, nsigma=snr, npixels=npixels, dilate_size=dilate_size)
        else:
            if isinstance(mask,(list,tuple)) or str(mask.__class__).count("regions.shapes") > 0:
                imgMask = self.generateMaskFromRegions(mask, parallelMask=parallelMask)
            else:
                # convert mask to boolean
                imgMask = np.array(mask, dtype='bool')


        _,median,std = sigma_clipped_stats(self.image, mask=imgMask, sigma=sigClip, maxiters=iterations)
        self.bkgMedian = median
        self.bkgStd = std
        
        if apply:
            self.image = self.image - self.bkgMedian
            self.bkgMedian = 0.0
        
        if returnMask:
            return mask
        else:
            return
    
    ###############################################################################################################
    
    def constantBackSub(self, backConstant):
        # function to subtract a constant from the image
        
        # subtract constant from image
        self.image = self.image - backConstant
        
        return
    
    ###############################################################################################################

    def background_polySub(self, polyOrder=5, snr=2, npixels=5, dilate_size=11, performSigmaClip=True, sigClip=3.0, iterations=20, mask=None, apply=True, downSample=None, downMethod='mean', parallelMask=False, returnMask=False):
        # function to perform a polynomial fit to the image

        # import modules
        from astropy.stats import sigma_clip
        if mask is None:
            #from photutils import make_source_mask
            pass
        from astropy.modeling.models import Polynomial2D
        from astropy.modeling import fitting as astropyFitter
        
        
        if mask is None:
            print("Creating Automatic Mask")
            imgMask = make_source_mask(self.image, nsigma=snr, npixels=npixels, dilate_size=dilate_size)
        else:
            if isinstance(mask,(list,tuple)) or str(mask.__class__).count("regions.shapes") > 0:
                imgMask = self.generateMaskFromRegions(mask, parallelMask=parallelMask)
            else:
                # convert mask to boolean
                imgMask = np.array(mask, dtype='bool')

        # create a copy of the image to manipulate
        imgData = self.image.copy()
        
        # mask data with nan's
        sel = np.where(imgMask == True)
        imgData[sel] = np.nan

        # perform sigma clipping if desired
        if performSigmaClip:
            imgData = sigma_clip(imgData, sigma=sigClip, maxiters=iterations, axis=(0,1), masked=False)

        # if returning mask generate now based on NaN's
        if returnMask:
            imgMask = np.array(np.zeros(imgData.shape),dtype='bool')
            maskSel = np.where(np.isnan(imgData) == True)
            imgMask[maskSel] = True


        ## fit polynomial 2D model
        # crete x, y points
        y, x = np.mgrid[:imgData.shape[0],:imgData.shape[1]]
        
        # if want to downsample the image to speed up fitting, peform downsampling
        if downSample is not None:
            # make sure is an integer
            downFactor = int(downSample)

            # get dimensions
            dimen = imgData.shape

            # see where the offset in image is to downsample (so edges are even)
            offset_x = dimen[1] % downFactor // 2
            offset_y = dimen[0] % downFactor // 2

            # create estimator object depending on median or mean
            if downMethod == 'median':
                estimator = np.nanmedian
            elif downMethod == "mean":
                estimator = np.nanmean
            
            # create downsampled image
            down_img = estimator(np.dstack([imgData[offset_y+i+(downFactor-1)//2:offset_y+i+(downFactor-1)//2+dimen[0]//downFactor*downFactor:downFactor, offset_x+j+(downFactor-1)//2:offset_x+j+(downFactor-1)//2+dimen[1]//downFactor*downFactor:downFactor] for i in range(-((downFactor-1)//2),(downFactor+1+1)//2) for j in range(-((downFactor-1)//2),(downFactor+1+1)//2)]),axis=2)

            # create coordinate grids which use same coordinate system as original image
            down_x = x[offset_y+(downFactor-1)//2:dimen[0]-offset_y:downFactor, offset_x+(downFactor-1)//2:dimen[1]-offset_x:downFactor]
            down_y = y[offset_y+(downFactor-1)//2:dimen[0]-offset_y:downFactor, offset_x+(downFactor-1)//2:dimen[1]-offset_x:downFactor]
            # if downFactor even account by adding 0.5
            if (downFactor + 1) % 2 == 1:
                down_x += 0.5
                down_y += 0.5
            
        else:
            down_x = x
            down_y = y
            down_img = imgData

        # initiate fitter
        mod_init = Polynomial2D(degree=polyOrder)
        fitMod = astropyFitter.LevMarLSQFitter()

        # select non-nan pixels
        nonNaN = np.where(np.isnan(down_img) == False)

        # perform fit
        pfit = fitMod(mod_init, down_x[nonNaN], down_y[nonNaN], down_img[nonNaN])

        # create 2D polynomial image for full image
        backPoly = pfit(x,y)

        # subtract from image
        if apply:
            self.image = self.image - backPoly
            if returnMask:
                return imgMask
            else:
                return
        else:
            if returnMask:
                return backPoly, imgMask
            else:
                return backPoly

    ###############################################################################################################

    def generateMaskFromRegions(self, regions, parallelMask=False):
        # function to generate mask from a list of regions (from region package)

        # check is in correct format
        if isinstance(regions,(list,tuple)) or str(regions.__class__).count("regions.shapes") > 0:
            print("Making mask based on data provided")
        else:
            raise Exception("Regions not provided as input")

        # if running parallel make sure module loaded
        if parallelMask:
            import multiprocessing as mp

        # if only one region provided embed in list
        if str(regions.__class__).count("regions.shapes") > 0:
            regions = [regions]
        
        # create mask
        imgMask = np.zeros(self.image.shape)
        
        # create blank wcs
        imgWCS = None
        try:
            # create image WCS
            imgWCS = wcs.WCS(self.header)
        except:
            pass
                        
        # see if running mask making in parallel
        if parallelMask:
            # get number of threads
            nthread = mp.cpu_count()

            # create pool
            pool = mp.Pool(processes=nthread)
            
            # see how big to make blocks to split
            chunkSize = len(regions)//nthread

            # calculate the indexes each chunk size is
            indexes = []
            for i in range(0,nthread):
                if i == nthread - 1:
                    indexes.append([i*chunkSize,len(regions)])
                else:
                    indexes.append([i*chunkSize,(i+1)*chunkSize])

            # create list to catch multiprocessing outputs
            output_list = []

            # loop over each chunk and run masking
            for i in range(0,nthread):
                outputs = pool.apply_async(regionsToMask, args=(regions[indexes[i][0]:indexes[i][1]], imgWCS, self.image.shape))
                output_list.append(outputs)
            
            # get outputs and combine mask
            for output in output_list:
                imgMask[output.get() > 0] = 1

            # close pool
            pool.close()
            pool.join()
        else:
            # run mask
            tempMask = regionsToMask(regions, imgWCS, self.image.shape)
                
            imgMask[tempMask > 0] = 1

        # convert mask to boolean
        imgMask = np.array(imgMask, dtype='bool')

        return imgMask

    ###############################################################################################################
    
    def ellipticalAnnulusBackSub(self, centre, inner=None, outer=None, axisRatio=None, PA=None, outerCircle=False, backNoise=False,\
                               method='exact', subpixels=None, maskNaN=True, apply=False):
        # function to select pixels within an elliptical aperture
        
        # import required modules
        import photutils
        if photutils.__version__ < '2.0.0':
            from photutils import aperture_photometry
            from photutils import SkyEllipticalAnnulus
            from photutils import EllipticalAnnulus
        else:
            from photutils.aperture import aperture_photometry
            from photutils.aperture import SkyEllipticalAnnulus
            from photutils.aperture import EllipticalAnnulus
        from astropy.table import Column
        from astropy.table import join as tableJoin
        from astropy.table import Table
        from astropy.coordinates import SkyCoord
        
        
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
        
    ###############################################################################################################

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
    
    ###############################################################################################################
    
    def circularAperture(self, galInfo, radius=None, multiRadius = False, localBackSubtract=None, names=None, method='exact', subpixels=None, backMedian=False, maskNaN = True, error=None):
        # function to perform circular aperture photometry 
        
        # import necessary modules
        from astropy.coordinates import SkyCoord

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
        
        # if skycoord object contains a list of coordinates convert to a list
        elif isinstance(galInfo, SkyCoord) and len(galInfo.shape) > 0:
            centres = list(galInfo)
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
    
    ###############################################################################################################
    
    def ellipticalAperture(self, galInfo, major=None, minor=None, axisRatio=None, PA=None, multiRadius = False, localBackSubtract=None, names=None, method='exact', subpixels=None, backMedian=False, maskNaN = True, error=None):
        # function to perform circular aperture photometry 
        
        # import necessary modules
        from astropy.coordinates import SkyCoord

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
        
        # if skycoord object contains a list of coordinates convert to a list
        elif isinstance(galInfo, SkyCoord) and len(galInfo.shape) > 0:
            centres = list(galInfo)
        
        else:
            centres = galInfo
                    
        # see what is defined minor/axisRatio and create uniform
        if minor is None and axisRatio is not None:
            if isinstance(major, (list,tuple)) and isinstance(axisRatio,(list,tuple)):
                minor = []
                for i in range(0,len(major)):
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
    
        return phot_table   

    ###############################################################################################################

    def rectangularAperture(self, galInfo, length=None, width=None, ratio=None, PA=None, multiRadius = False, localBackSubtract=None, names=None, method='exact', subpixels=None, backMedian=False, maskNaN = True, error=None):
        # function to perform circular aperture photometry 
        
        # import necessary modules
        from astropy.coordinates import SkyCoord

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
        
        # if skycoord object contains a list of coordinates convert to a list
        elif isinstance(galInfo, SkyCoord) and len(galInfo.shape) > 0:
            centres = list(galInfo)
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
                for i in range(0,len(ratio)):
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
    
    ###############################################################################################################

    def aperturePhotometry(self, mode, centres, radius, minor=None, PA=None, multiRadius = False, localBackSubtract=None, names=None, method='exact', subpixels=None, backMedian=False, maskNaN=True, error=None):
        # function to perform photometry
        
        # import required modules
        import photutils
        if photutils.__version__ < '2.0.0':
            from photutils import aperture_photometry
        else:
            from photutils.aperture import aperture_photometry
        from astropy.table import Column
        from astropy.table import join as tableJoin
        from astropy.table import Table
        from astropy.coordinates import SkyCoord
        
        # check mode programmed
        if mode not in ["circle", "ellipse", "rectangle"]:
            raise ValueError("Shape Not programmed")
        
        # import relevant photutil function
        if mode == "circle":
            if photutils.__version__ < '2.0.0':
                from photutils import SkyCircularAperture
                from photutils import CircularAperture as PixCircularAperture
                if localBackSubtract:
                    from photutils import SkyCircularAnnulus
                    from photutils import CircularAnnulus
            else:
                from photutils.aperture import SkyCircularAperture
                from photutils.aperture import CircularAperture as PixCircularAperture
                if localBackSubtract:
                    from photutils.aperture import SkyCircularAnnulus
                    from photutils.aperture import CircularAnnulus
        elif mode == "ellipse":
            if photutils.__version__ < '2.0.0':
                from photutils import SkyEllipticalAperture
                from photutils import EllipticalAperture as PixEllipticalAperture
                if localBackSubtract:
                    from photutils import SkyEllipticalAnnulus
                    from photutils import EllipticalAnnulus
            else:
                from photutils.aperture import SkyEllipticalAperture
                from photutils.aperture import EllipticalAperture as PixEllipticalAperture
                if localBackSubtract:
                    from photutils.aperture import SkyEllipticalAnnulus
                    from photutils.aperture import EllipticalAnnulus
        elif mode == "rectangle":
            if photutils.__version__ < '2.0.0':
                from photutils import SkyRectangularAperture
                from photutils import RectangularAperture as PixRectangularAperture
                if localBackSubtract:
                    from photutils import SkyRectangularAnnulus
                    from photutils import RectangularAnnulus
            else:
                from photutils.aperture import SkyRectangularAperture
                from photutils.aperture import RectangularAperture as PixRectangularAperture
                if localBackSubtract:
                    from photutils.aperture import SkyRectangularAnnulus
                    from photutils.aperture import RectangularAnnulus

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
            SBunit = self.isSurfaceBrightnessUnit()
            if SBunit is True:
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
        if isinstance(centres, SkyCoord) and len(centres.shape) == 0:
            centres  = [centres]
        elif isinstance(centres, (list, tuple, np.ndarray)) is False:
            centres = [centres]
        else:
            if len(centres) == 2 and isinstance(centres[0], (float, int)) and isinstance(centres[0], (float, int)):
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
                                backApertures.append(CircularAnnulus(centres[i], r_in=backRadInfo['inner'], r_out=backRadInfo['outer']))
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
                    if hasattr(ind_phot_table['xcenter'][0], 'value'):
                        phot_xcenter = [ind_phot_table['xcenter'][0].value]
                        phot_ycenter = [ind_phot_table['ycenter'][0].value]
                    else:
                        phot_xcenter = [ind_phot_table['xcenter'][0]]
                        phot_ycenter = [ind_phot_table['ycenter'][0]]
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
                    if hasattr(ind_phot_table['xcenter'][0], 'value'):
                        phot_xcenter.append(ind_phot_table['xcenter'][0].value)
                        phot_ycenter.append(ind_phot_table['ycenter'][0].value)
                    else:
                        phot_xcenter.append(ind_phot_table['xcenter'][0])
                        phot_ycenter.append(ind_phot_table['ycenter'][0])
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
                        programedUnits = self.programmedUnits()
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
            if hasattr(self, 'unit') and self.unit is not None:
                programedUnits = self.programmedUnits()
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
    
    ###############################################################################################################

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

    ###############################################################################################################

    def surfaceBrightness(self, phot_table, half_bin_table):
        # function to calculate surface brightness profiles
        
        # get list objects, and find if surface brightness or sum
        colnames = phot_table.colnames
        objNames = []
        for colname in colnames[1:]:
            if colname[-12:] == "aperture_sum":
                objNames.append(colname[:-13])
                surfaceBrightnessUnits = False
            if colname[-13:] == "aperture_mean":
                objNames.append(colname[:-14])
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
            if objName + "_aperture_error" in colnames or objName + "_aperture_mean_error" in colnames:
               phot_table[objName+"_surface_brightness_error"] = surfaceBrightness 
            if hasattr(self, 'unit'):
                if surfaceBrightnessUnits:
                    apUnit = phot_table[objName+"_aperture_mean"].unit
                else:
                    apUnit = phot_table[objName+"_aperture_sum"].unit
                phot_table[objName+"_surface_brightness"].unit = str(apUnit) + " arcsec^-2"
                if objName + "_aperture_error" in colnames or objName + "_aperture_mean_error" in colnames:
                    if surfaceBrightnessUnits:
                        errUnit = phot_table[objName+"_aperture_mean_error"].unit
                    else:
                        errUnit = phot_table[objName+"_aperture_error"].unit
                    phot_table[objName+"_surface_brightness_error"].unit = str(errUnit) + " arcsec^-2"

        return

    ###############################################################################################################

    def coordMaps(self, returnPixMaps=False):
        # function to find ra and dec co-ordinates of every pixel
        
        # import modules
        from astropy.coordinates import ICRS
        
        # Parse the WCS keywords in the primary HDU
        header = self.header
        wcsInfo = wcs.WCS(self.header)
        
        # Make input arrays for every pixel on the map
        xpix = np.zeros((header["NAXIS1"]*header["NAXIS2"]),dtype=int)
        for i in range(0,header["NAXIS2"]):
            xpix[i*header["NAXIS1"]:(i+1)*header["NAXIS1"]] = np.arange(0,header["NAXIS1"],1)
        ypix = np.zeros((header["NAXIS1"]*header["NAXIS2"]),dtype=int)
        for i in range(1,header["NAXIS2"]):
            ypix[(i)*header["NAXIS1"]:(i+1)*header["NAXIS1"]] = i
        
        # Convert all pixels into sky co-ordinates
        sky = wcsInfo.pixel_to_world(xpix,ypix)
        
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
        self.raMap = raMap * u.degree
        self.decMap = decMap * u.degree
        
        # if want to output 
        if returnPixMaps:
            return xpix, ypix
        else:
            return
    
    ###############################################################################################################

    def pixelRadius(self, coordinate, major=None, minor=None, axisRatio=None, inclin=None, PA=None, specificPixels=None):
        
        # create blank radius map
        radMap = np.zeros(self.image.shape)
        
        # Parse the WCS keywords in the primary HDU
        header = self.header
        wcsInfo = wcs.WCS(self.header)
        
        # get pixel size
        if hasattr(self, "pixSize") is False:
            self.getPixelScale()
        
        if specificPixels is None:
            # Make array of x and y for every pixel on map
            xpix = np.zeros(radMap.shape,dtype=int)
            for i in range(0,radMap.shape[1]):
                xpix[:,i] = i
            ypix = np.zeros(radMap.shape,dtype=int)
            for i in range(0,int(radMap.shape[0])):
                ypix[i,:] = i
        else:
            
            # check correct format could be (X, Y), or ((X1,Y1), (X2,Y2)) in list, tuple or array format
            if isinstance(specificPixels, np.ndarray):
                if specificPixels.ndim != 2:
                    raise Exception("Correct Number of Dimensions Not Specified")
                
                if specificPixels.shape[1] != 2:
                    raise Exception("specificPixels does not have the correct dimensions")

                # create empty array
                xpix = np.zeros((specificPixels.shape[0]))
                ypix = np.zeros((specificPixels.shape[0]))

                # create x and y pix
                for i in range(0,specificPixels.shape[0]):
                    xpix[i] = specificPixels[i,0]
                    ypix[i] = specificPixels[i,1]
                
            # add in list or tuple case
            elif isinstance(specificPixels, (tuple,list)):
                xpix = np.zeros((len(specificPixels)))
                ypix = np.zeros((len(specificPixels)))
                for i in range(0,len(specificPixels)):
                    # check the element is a 2D array
                    if isinstance(specificPixels[i], (tuple,list)):
                        if len(specificPixels[i]) != 2:
                            raise Exception("specificPixels has wrong format")
                    elif isinstance(specificPixels[i], np.ndarray):
                        if specificPixels[i].ndim != 1 and specificPixels[i].shape[0] != 2:
                            raise Exception("specificPixels has wrong format")
                    
                    # create x and y pix
                    xpix[i] = specificPixels[i][0]
                    ypix[i] = specificPixels[i][1]


        # Convert all pixels into sky co-ordinates
        #sky = wcsInfo.pixel_to_world(xpix,ypix)
    
        # get centre
        from astropy.coordinates import SkyCoord
        if isinstance(coordinate, SkyCoord):
            # make sure modules required are imported
            from astropy.wcs.utils import skycoord_to_pixel
            
            # get X, Y pixel of coordinates
            tempCentre = skycoord_to_pixel(coordinate, wcsInfo, origin=0)
            pixCentre=[0.0,0.0]
            pixCentre[0] = float(tempCentre[0])
            pixCentre[1] = float(tempCentre[1])
            
        elif isinstance(coordinate, (list, tuple, np.ndarray)) and len(coordinate) == 2:
            pixCentre = [float(coordinate[0]),float(coordinate[1])]
        else:
            raise Exception("Coordinate is not a recognised SkyCoordinate or pixel coordinate")
        
        # calculate inclination
        if inclin is not None:
            if isinstance(inclin, u.Quantity) is False:
                inclin = inclin * u.deg
            inclin = inclin.to(u.rad)
        elif axisRatio is not None:
            inclin = np.arccos(axisRatio)
        elif major is not None and minor is not None:
            if isinstance(major, u.Quantity) and isinstance(minor, u.Quantity):
                inclin = np.arccos((minor/major).value)
            elif isinstance(major, u.Quantity) and isinstance(minor, u.Quantity) is False:
                raise Exception("Major is provided as a quantity, but minor is a value")
            elif isinstance(major, u.Quantity) is False and isinstance(minor, u.Quantity):
                raise Exception("Minor is provided as a quantity, but major is a value")
            else:
                inclin = np.arccos(minor/major)
        elif major is not None or minor is not None:
            raise Exception("Only one of major or minor is specified")
        else:
            inclin = 0.0
                    
        # see if image is rotated
        if wcsInfo.wcs.has_crota() == True or wcsInfo.wcs.has_cdi_ja() == True or wcsInfo.wcs.has_crotaia() == True or wcsInfo.wcs.has_pci_ja()==True:
            if wcsInfo.wcs.has_cdi_ja():
                if wcsInfo.wcs.cd[1,0] == 0.0 and wcsInfo.wcs.cd[0,1] == 0.0:
                    rotation = 0.0
                else:
                    rotation = np.arctan(wcsInfo.wcs.cd[1][0]/wcsInfo.wcs.cd[0][0])
            elif wcsInfo.wcs.has_pci_ja():
                if wcsInfo.wcs.pc[1,0] == 0.0 and wcsInfo.wcs.pc[0,1] == 0.0:
                    rotation = 0.0
                else:
                    rotation = np.arctan(wcsInfo.wcs.pc[1][0]/wcsInfo.wcs.pc[0][0])
            else:
                try:
                    rotation = wcsInfo.wcs.crota[1]
                    rotation = rotation / 180.0 * np.pi
                except:
                    raise Exception("Rotation calculation not programmed for this header")
        else:
            rotation = 0.0
        
        ## calculate imagePA
        # if no PA defined assumed 0.0
        if PA is None:
            tempPA = 0.0
        elif isinstance(PA, u.Quantity):
            tempPA = PA.to(u.rad).value
        else:
            tempPA = PA * np.pi / 180.0
        
        # PA is east of North, convert to image coordinates including rotation
        PA = -1.0 * (np.pi/2.0 - tempPA) + rotation
        
        # calculate radius (in arcsecond)
        Xsquare = ((xpix - pixCentre[0]) * np.cos(PA) + (ypix - pixCentre[1]) * np.sin(PA))**2.0
        Ysquare = (-(xpix - pixCentre[0]) * np.sin(PA) + (ypix - pixCentre[1]) * np.cos(PA))**2.0
        radMap = np.sqrt(Xsquare + Ysquare / np.cos(inclin)**2.0)
        # adjust for physical size of pixels
        radMap = radMap * self.pixSize           
        
        # return radius map
        return radMap
        
    ###############################################################################################################

    def convertUnits(self, newUnit, conversion=None, beamArea=None, forceInstrumentalUnit=False, verbose=True):
        # function to convert units of map
        
        # if a conversion value given use that, if not calculate
        if conversion is not None:
            self.image = self.image * conversion
            if hasattr(self,'error'):
                self.error = self.error * conversion
            self.header['BUNIT'] = newUnit
            self.unit = newUnit
            if "SIGUNIT" in self.header:
                self.header['SIGUNIT'] = newUnit
            if "ZUNITS" in self.header:
                self.header['ZUNITS'] = newUnit
            
            if hasattr(self, "bkgMedian"):
                self.bkgMedian = self.bkgMedian * conversion
            if hasattr(self, "bkgStd"):
                self.bkgMedian = self.bkgStd * conversion

            if verbose:
                print(self.band, " image converted to ", newUnit, " using provided conversion")
        else:
            # if forcing instrumental units
            if forceInstrumentalUnit:
                self.header['BUNIT'] = self.standardInstrumentalUnit(self.instrument, None)
                self.unit = self.standardInstrumentalUnit(self.instrument, None)

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
            
            if oldUnit == newUnit:
                # check that not already in correct unit
                if verbose:
                    print("Image is already in correct units")
            else:
                # see if in a pre-progammed unit
                if oldUnit not in allUnits:
                    if verbose:
                        if self.standardInstrumentalUnit(self.instrument, self.header['BUNIT']) is False:
                            print("Image Unit: ", oldUnit, " not programmed - result maybe unreliable")
                if newUnit not in allUnits:
                    if verbose:
                        print("Image Unit: ", newUnit, " not programmed - result maybe unreliable")
                
                # check if the image is default instrumental units rather than an astronomical unit
                if self.standardInstrumentalUnit(self.instrument, self.header['BUNIT']):
                    # get conversion value 
                    instConvUnit, instConvValue = self.standardInstrumentConversion(self.instrument, self.band)
                    
                    # apply changes to image
                    self.image = self.image * instConvValue
                    if hasattr(self,'error'):
                        self.error = self.error * instConvValue
                    self.header['BUNIT'] = instConvUnit
                    self.unit = instConvUnit
                    oldUnit = instConvUnit
                    
                    if hasattr(self, "bkgMedian"):
                        self.bkgMedian = self.bkgMedian * instConvValue
                    if hasattr(self, "bkgStd"):
                        self.bkgMedian = self.bkgStd * instConvValue

                    if oldUnit == newUnit:
                        if verbose:
                           print("Image converted to ", newUnit)
                        return
                
                # see if returning image to standard instrumental units
                postInstrumentConversion = False
                if self.standardInstrumentalUnit(self.instrument, newUnit):
                    # see what unit need the map to be in to convert
                    instConvUnit, instConvValue = self.standardInstrumentConversion(instrument=self.instrument, band=self.band)

                    # see if image is in required units already
                    if oldUnit in units[instconvUnit]:
                        # apply changes to image
                        self.image = self.image / instConvValue
                        if hasattr(self,'error'):
                            self.error = self.error / instConvValue
                        self.header['BUNIT'] = newUnit
                        self.unit = newUnit
                                                
                        if hasattr(self, "bkgMedian"):
                            self.bkgMedian = self.bkgMedian / instConvValue
                        if hasattr(self, "bkgStd"):
                            self.bkgMedian = self.bkgStd / instConvValue

                        print("Image converted to ", newUnit)
                        return
                    
                    else:
                        defaultInstUnit = newUnit
                        newUnit = instrConvUnit
                        postInstrumentConversion = True

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
                elif oldUnit in units["uK_CMB"] or oldUnit in units["K_CMB"]:
                    try:
                        import pysm3.units as cmbUnits
                    except:
                        raise Exception("Converting CMB units requires pysm3 to be installed")
                    if oldUnit in units["uK_CMB"]:
                        conversion = 1.0 * cmbUnits.uK_CMB
                    elif oldUnit in units["K_CMB"]:
                        conversion = 1.0 * cmbUnits.K_CMB
                    currentWavelength = self.standardCentralWavelengths(instrument=self.instrument, band=self.band)
                    conversion = conversion.to(u.Jy/u.sr, equivalencies=cmbUnits.cmb_equivalencies((con.c/currentWavelength).to(u.GHz)))
                    
                elif oldUnit in units["maggy"] or oldUnit in units["nanomaggy"]:
                    zero_point_star_equiv = u.zero_point_flux(3631.*u.Jy)
                    if oldUnit in units["maggy"]:
                        conversion = 1.0*u.maggy
                    elif oldUnit in units["nanomaggy"]:
                        conversion = 1.0 * u.nanomaggy
                    conversion = conversion.to(u.Jy, zero_point_star_equiv) 
                    pixArea = self.pixSize * self.pixSize
                    conversion = conversion / pixArea

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
                        conversion = (conversion * beamAreas[self.instrument][self.band].to(u.arcsec**2.0)).value
                    elif newUnit in units["mJy/beam"]:
                        conversion = (conversion * beamAreas[self.instrument][self.band].to(u.arcsec**2.0) * 1000.0).value
                elif newUnit in units["Jy/arcsec^2"]:
                    conversion = conversion.to(u.Jy/u.arcsecond**2.0).value
                elif newUnit in units["mJy/arcsec^2"]:
                    conversion = conversion.to(u.Jy/u.arcsecond**2.0).value * 1000.0
                elif newUnit in units["MJy/sr"]:
                    conversion = conversion.to(u.Jy/u.sr).value * 1.0e-6
                elif newUnit in units["maggy"] or newUnit in units["nanomaggy"]:
                    zero_point_star_equiv = u.zero_point_flux(3631.*u.Jy)
                    pixArea = self.pixSize * self.pixSize
                    if newUnit in units["maggy"]:
                        conversion = (conversion * pixArea).to(u.mgy, zero_point_star_equiv)
                    else:
                        conversion = (conversion * pixArea).to(u.nanomaggy, zero_point_star_equiv)
                elif newUnit == "pW" and self.instrument == "SCUBA-2":
                    conversion = (conversion * beamAreas[self.instrument][self.band]).value
                    conversion = conversion / scubaConversions[self.band]['Jy/beam']
                else:
                    raise ValueError("Unit not programmed")
                
                if postInstrumentConversion:
                    conversion = conversion / instConvValue
                    newUnit = defaultInstUnit

                self.image = self.image * conversion
                if hasattr(self,'error'):
                    self.error = self.error * conversion

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
                
                # remove or adjust attributes which could be present
                if hasattr(self, "bkgMedian"):
                    self.bkgMedian = self.bkgMedian * conversion
                if hasattr(self, "bkgStd"):
                    self.bkgMedian = self.bkgStd * conversion

        return

    ###############################################################################################################

    def centralWaveAdjust(self, newWavelength, adjustSettings):
        # function to adjust for difference in central wavelengths
        print("Performing Central Wavelength Adjustment")
        
        # get current central wavelength
        currentWavelength = self.standardCentralWavelengths(instrument=self.instrument, band=self.band)
        
        
        # see if have a PPMAP cube
        if "ppmapCube" in adjustSettings:
            from astroIm import ppmapCube
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
            newLevel = (con.c/(newWavelength))**adjustSettings["beta"] * blackbody(newWavelength)
            currLevel = (con.c/(currentWavelength))**adjustSettings["beta"] * blackbody(currentWavelength)
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

    ###############################################################################################################    
    
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
                indexT = int((temperature-gridInfo['T']['start'])/gridInfo['T']['step'])
                
                # find index of closest Beta
                indexB = int((beta-gridInfo['B']['start'])/gridInfo['B']['end'])
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
        
        ###############################################################################################################
        
        # see if have a PPMAP cube
        if "ppmapCube" in adjustSettings:
            # import ppmap cube
            from astroIm import ppmapCube
            
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
                self.error = self.error * ccMap.image
            
            if saveCCinfo:
                self.ccData = ccMap.image
            
        else:
            self.image = self.image * ccFactor
            if hasattr(self,"error"):
                self.error = self.error * ccFactor
    
            if saveCCinfo:
                self.ccData = ccFactor
    
    ###############################################################################################################
    
    def restoreDefaultCC(self):
        # function to restore the image to default colour-corrections
        
        # update image
        self.image = self.image / self.ccData
        
        # update error
        if hasattr(self,"error"):
            self.error = self.error / self.ccData

    ###############################################################################################################
    
    def reproject(self, projHead, exact=True, conserveFlux=None, parallel=False):
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
                resampleMap, _ = reproject_exact(hdu, projHead, parallel=parallel)
            else:
                resampleMap, _ = reproject_interp(hdu, projHead, parallel=parallel)
        
        
        # modify original header
        # projection keywords
        projKeywords = ["NAXIS1", "NAXIS2", "LBOUND1", "LBOUND2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",\
                        "CTYPE1", "CTYPE2", "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",\
                        "PC1_1", "PC1_2", "PC2_1", "PC2_2",\
                        "RADESYS", "EQUINOX", "CROTA2", "CROTA1", "LONPOLE", "LATPOLE"]
        header = self.header.copy()
        
        for keyword in projKeywords:
            if keyword in projHead:
                header[keyword] = projHead[keyword]
            else:
                try:
                    del(header[keyword])
                except:
                    pass
        
        ## create reprojected image hdu
        #repoHdu = pyfits.PrimaryHDU(resampleMap, header)
        #repoHdulist = pyfits.HDUList([repoHdu])
        
        ## create combine astro image
        #repoImage = astroImage(repoHdulist, load=False, instrument=self.instrument, band=self.band)
        
        # create new reporjected astro image
        repoImage = copy.deepcopy(self)
        repoImage.image = resampleMap
        repoImage.header = header
        repoImage.getPixelScale()

        if conserveFlux is None:
            if hasattr(self,'unit'):
                SBunit = self.isSurfaceBrightnessUnit()
                if SBunit is True:
                    conserveFlux = False
                elif SBunit is False:
                    conserveFlux = True
                else:
                    conserveFlux = False
            else:
                conserveFlux = False

        # see if need to correct image to conserve flux rather than surface brightness and correct
        if conserveFlux:
            # get original pixel size
            if hasattr(self, "pixSize") is False:
                self.getPixelScale()
            origPixSize = self.pixSize
            
            # get output pixel size
            #repoImage.getPixelScale()
            outPixSize = repoImage.pixSize

            # adjust image for difference in pixel area
            repoImage.image = repoImage.image * (outPixSize**2.0/origPixSize**2.0).to(u.dimensionless_unscaled).value
            
        # return new image
        return repoImage
    
    ###############################################################################################################

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
    
    ###############################################################################################################
    
    def convolve(self, kernel, boundary='fill', fill_value=0.0, peakNorm=False, FWHM=True, fftConvolve=True):
        
        # import modules
        if fftConvolve:
            from astropy.convolution import convolve_fft as APconvolve_fft
        else:
            from astropy.convolution import convolve as APconvolve
        
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
            if fftConvolve:
                convolvedArray = APconvolve_fft(self.image, kernelImage, boundary=boundary, fill_value=fill_value, allow_huge=True, normalize_kernel=normKernel)
            else:
                convolvedArray = APconvolve(self.image, kernelImage, boundary=boundary, fill_value=fill_value, normalize_kernel=normKernel)
        else:
            if fftConvolve:
                convolvedArray = APconvolve_fft(self.image, kernelImage, boundary=boundary, allow_huge=True, normalize_kernel=normKernel)
            else:
                convolvedArray = APconvolve(self.image, kernelImage, boundary=boundary, normalize_kernel=normKernel)
        
        # restore NaNs
        convolvedArray[NaNsel] = np.nan
        
        # create new astroImage object
        convolvedImage = copy.deepcopy(self)
        convolvedImage.image = convolvedArray

        # update FWHM keyword
        if hasattr(convolvedImage, 'fwhm'):
            # if image is provided can't predict output FWHM
            if isinstance(kernel, type(1.0*u.arcsecond)) is False:
                del(convolvedImage.fwhm)
            else:
                if FWHM:
                    convolvedImage.fwhm = np.sqrt(self.fwhm**2.0 + kernel**2.0)
                else:
                    convolvedImage.fwhm = np.sqrt(self.fwhm**2.0 + (kernel * 2.0*np.sqrt(2.0*np.log(2.0)))**2.0)

        ## create combined image hdu
        #convolveHeader = self.header
        #convolveHdu = pyfits.PrimaryHDU(convolvedArray, convolveHeader)
        #convolveHdulist = pyfits.HDUList([convolveHdu])
        
        ## create combine astro image
        #convolvedImage = astroImage(convolveHdulist, load=False, instrument=self.instrument, band=self.band)
        
        return convolvedImage
    
    ###############################################################################################################

    def cutout(self, centre, size, makeCopy=False):
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
        cutoutOut = Cutout2D(self.image, centre, newSize, wcs=WCSinfo, mode='partial', fill_value=np.nan, copy=makeCopy)
       
        # create new header
        cutHeader = self.header.copy()
        cutHeadWCS = cutoutOut.wcs.to_header()
        for keyword in cutHeadWCS:
            cutHeader[keyword] = cutHeadWCS[keyword]
        
        # create astro image object from output
        cutoutImage = copy.deepcopy(self)
        cutoutImage.image = cutoutOut.data
        cutoutImage.header = cutHeader

        ## create astro image object from output
        #cutoutHdu = pyfits.PrimaryHDU(cutoutOut.data, cutHeader)
        #cutoutHdulist = pyfits.HDUList([cutoutHdu])
        
        ## create combine astro image
        #cutoutImage = astroImage(cutoutHdulist, load=False, instrument=self.instrument, band=self.band)
        
        return cutoutImage

    ###############################################################################################################    
        
    def imageFFTcombine(self, lowresImage, filterScale=None, beamArea=None, filterType="gauss", butterworthOrder=None, sigmoidScaling=None, beamMatchedMode=True, medianSubtract=True):
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
        if medianSubtract:
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
            filter_scale = float(filterScale)
            filter_std = filter_scale / (2.0*np.sqrt(2.0*np.log(2.0)))
            filter = np.exp(-( (d_f*2.0*np.pi)**2.0 * filter_std**2.0 / 2.0))
            #filter = np.exp(-(d)**2.0 / (2.0*filter_std**2.0))
        elif filterType == "sigmoid":
            d_f = d_f**-1
            if sigmoidScaling is None:
                sigmoidScaling = 1.0
            filter_scale = float(filterScale)
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
        
        # create new combined image
        combineImage = copy.deepcopy(self)
        combineImage.image = combined
        combineImage.header = combineHeader

        #combineHdu = pyfits.PrimaryHDU(combined, combineHeader)
        #combineHdulist = pyfits.HDUList([combineHdu])
        
        ## create combine astro image
        #try:
        #    combineImage = astroImage(combineHdulist, load=False)
        #except:
        #    combineImage = astroImage(combineHdulist, load=False, band=self.band)
        
        ## copy attributes from high-res image if available
        #if hasattr(self,"fwhm"):
        #    combineImage.fwhm = self.fwhm
        #if hasattr(self,"ccData"):
        #    combineImage.ccData = self.ccData
        
        return combineImage
    
    ###############################################################################################################
    
    def getPowerSpectra(self, oneD=True, mask=None, plot=True, spatialUnits=u.deg, normaliseScale=None, savePlot=None): 
        # function to create power spectrum of the map
        # based on Agpy implementation of Adam Ginsburg
        
        # import psds module
        from astroIm import psds

        if oneD is False:
            raise Exception("Wrapper for 2D power spectrum not implemented yet")
        
        powerSpecImage = self.image.copy()
    
        # if a mask is provide set regions either 0 or false to NaN
        if mask is not None:
            # convert mask to int
            intMask = mask.astype(int)
            
            # set image to NaN where mask == 0
            powerSpecImage[intMask < 0.1] = np.nan
            
        # check pixel size is loaded
        if hasattr(self, "pixSize") is False:
            self.getPixelScale()
            
        # create power spectrum of map
        rawFreq, psd = psds.PSD2(powerSpecImage, oned=oneD)
        
        # adjust scaling
        freq = rawFreq / (self.pixSize.to(spatialUnits))
        spatial = 1.0 / freq
        
        # see if want to apply a normalisation
        if normaliseScale is not None:
            sel = np.where(np.abs(spatial - normaliseScale.to(spatialUnits)) == (np.abs(spatial - normaliseScale.to(spatialUnits))).min())
            psd = psd / psd[sel][0]
        
        # save power spectra to astroImage object
        self.powerSpec = {"frequency":freq, "spatial":spatial, "psd":psd}
        
        
        if plot:
            self.powerSpecPlot(save=savePlot)

    ###############################################################################################################

    def powerSpecPlot(self, powerSpecInfo=None, spatialUnits=None, labels=None, linestyle=None, color=None, show=True, save=None):
        # module which plots a powerspectra
        # import module
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        
        # define figure and axes
        fig = plt.figure(figsize=(6,4))
        f1 = plt.axes([0.15,0.12,0.8,0.76])
        
        # extract data
        if powerSpecInfo is None:
            powerSpecData = [self.powerSpec]
        else:
            if isinstance(powerSpecInfo,dict) is True:
                powerSpecData = [self.powerSpec]
            elif isinstance(powerSpecInfo,list) is True:
                powerSpecData = powerSpecInfo

        # if spatialUnits not specified then take from first psd
        if spatialUnits is None:
            spatialUnits = powerSpecData[0]['spatial'].unit

        # process line property information
        lineProperties = [labels, linestyle, color]
        for i in range(0,len(lineProperties)):
            lineProp = lineProperties[i]
            if lineProp is None:
                lineProperties[i] = [None] * len(powerSpecData)
            else:
                if isinstance(lineProp, str) and len(powerSpecData) == 1:
                    lineProperties[i] = [lineProp]
                elif isinstance(lineProp, list) and len(powerSpecData) == len(lineProp):
                    continue
                elif isinstance(lineProp, list) and len(powerSpecData) != len(lineProp) and len(lineProp) == 1:
                    lineProperties[i] = lineProp * len(powerSpecData)
                else:
                    raise Exception(f"{lineProp} variable input is not understood")
        
        # plot data
        addLegend = False
        for i in range(0,len(powerSpecData)):
            f1.plot(powerSpecData[i]['frequency'], powerSpecData[i]['psd'], label=lineProperties[0][i], linestyle=lineProperties[1][i], color=lineProperties[2][i])
            if lineProperties[0][i] is not None:
                addLegend = True
        
        # add legend if needed
        if addLegend:
            f1.legend()

        # set scales to log
        f1.set_xscale("log")
        f1.set_yscale("log")
    
        # get x-axis bounds
        x1bound = f1.get_xbound()
        y1bound = f1.get_ybound()
        
        # work out what ticks would be present on other axis
        x2low = np.ceil(np.log10(1.0 / x1bound[1]))
        x2high = np.floor(np.log10(1.0 / x1bound[0]))
        x2ticks = 10.0**np.arange(x2low,x2high+1.0,1.0)
        
        # work out as fraction of axis where these would lie
        x2frac = []
        x2label = []
        for tick in x2ticks:
            x2frac.append((np.log10(1.0/tick) - x1bound[0]) / (x1bound[1]-x1bound[0]))
            x2label.append(str(tick))
            
        # create second axes
        ax2 = f1.twiny()
        ax2.set_xscale("log")
        ax2.set_xlim(1.0/x1bound[0],1.0/x1bound[1])
            
        # adjust format
        f1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        f1.set_ylabel("Normalised Power (Arbitrary Units)")
        if spatialUnits == u.deg:
            f1.set_xlabel('Spatial Frequency (1/$^\circ$)')
            ax2.set_xlabel('Spatial Scale ($^\circ$)')
        elif spatialUnits == u.arcmin:
            f1.set_xlabel("Spatial Frequency (1/')")
            ax2.set_xlabel("Spatial Scale (')")
        elif spatialUnits == u.arcsec:
            f1.set_xlabel('Spatial Frequency (1/")')
            ax2.set_xlabel('Spatial Scale (")')
        
        # if save plot specified
        if save is not None:
            plt.savefig(save)

        # show plot
        if show:
            plt.show()
    
        return
        
###############################################################################################################
            
    def plot(self, recentre=None, stretch='linear', vmin=None, vmid=None, vmax=None, cmap=None, facecolour='white', nancolour='black', hide_colourbar=False, show=True, save=None):
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
        
        if show:
            plt.show()
        
        plt.close()

    ###########################################################################################################

    def interactivePlot(self, recentre=None, stretch='linear', vmin=None, vmid=None, vmax=None, cmap=None):
        # function to make a quick plot of the data using matplotlib and aplpy
        
        # import modules
        import aplpy
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RangeSlider
        from matplotlib.widgets import RadioButtons
        
        def updateClim(val):
            # Update plot if slider is changed
            # The val passed to a callback by the RangeSlider will
            # be a tuple of (min, max)

            # Update the image's colormap
            f1.image.set_clim((val[0],val[1]))

            # Redraw the figure to ensure it updates
            fig.canvas.draw_idle()

        # create figure
        fig = plt.figure(figsize=(12,6))
        
        # create binding so when click get the location
        binding_id = plt.connect('button_press_event', clickEvent)

        # repackage into an HDU 
        hdu = pyfits.PrimaryHDU(self.image, self.header)
        
        # create aplpy axes
        axesLocation = [0.07,0.13,0.40,0.80]
        f1 = aplpy.FITSFigure(hdu, figure=fig, subplot=axesLocation)
        
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
        f1.set_nan_color('black')
        f1.ax.set_facecolor('white')
        
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
        f1.add_colorbar()
        f1.colorbar.show()
        if hasattr(self, 'unit'):
            f1.colorbar.set_axis_label_text(self.unit)
        
        # set axis to be adjustable
        #f1.ax.set_adjustable('box', share=True)

        # set start pix 
        f1.show_markers(-1,-1, coords_frame='pixel', marker="+", c='white')

        # get initial colour scale limits
        initialClim = f1.image.get_clim()

        # create range slider
        sliderLoc = [axesLocation[0]+0.005, axesLocation[1]-0.12, axesLocation[2]-0.083, 0.05]
        slider_ax = plt.axes(sliderLoc)
        slider = RangeSlider(slider_ax, "", valmin=np.nanmin(self.image), valmax=np.nanmax(self.image), valinit=initialClim)
        slider.on_changed(updateClim)
        fig.text(axesLocation[0], axesLocation[1]-0.07, "Scalebar", fontsize=10)

        ### create pixel infomation box
        # labels
        leftInfoBorder = 0.52
        fig.text(leftInfoBorder, 0.89, "Pixel Information", fontsize=12)
        fig.text(leftInfoBorder+0.02, 0.86, "Marker", fontsize=10)

        # turn marker on/off
        markerAx = plt.axes([leftInfoBorder+0.0, 0.82, 0.04,0.12])
        markerButton = RadioButtons(markerAx, ('On', 'Off'), activecolor='C0')
        markerButton.ax.set_frame_on(False)
        #markerButton.on_clicked(scaleAdjust)



        plt.show()

    ###########################################################################################################

    def saveToFits(self, outPath, overwrite=False):
        # function to save to fits
        
        fitsHdu = pyfits.PrimaryHDU(self.image, self.header)
        fitsHduList = pyfits.HDUList([fitsHdu])
        
        fitsHduList.writeto(outPath, overwrite=overwrite)

        return
    
    ###########################################################################################################

    def fitsHdu(self):
        fitsHdu = pyfits.PrimaryHDU(self.image, self.header)
        
        return fitsHdu


##############################################################################################################
###############################################################################################################        
###############################################################################################################

# define function to plot multiple power spectra
def multiPowerSpectraPlot(images, units=None, matchProjection=False, refImage=0, oneD=True, mask=None, spatialUnits=u.deg, normaliseScale=None, labels=None, linestyle=None, color=None, beamAreas=None, show=True, save=None):
    # function to plot multiple power spctra on one plot

    ### apply image pre-processing
    # convert units
    if units is None:
        #try:
        # convert all images to Jy/beam
        for imageObj in images:
            imageObj.convertUnits("Jy/beam")
        
        # try to match beam areas
        if beamAreas is None:
            refBeamArea = images[refImage].standardBeamAreas(instrument=images[refImage].instrument, band=images[refImage].band)
            
            for i in range(0,len(images)):
                # skip if refimage
                if i == refImage:
                    continue
                
                # get images beam area
                beamArea = images[i].standardBeamAreas(instrument=images[i].instrument, band=images[i].band)
                
                # scale image by beam ratio
                images[i].image = images[i].image * (refBeamArea/beamArea).to("").value
        else:
            if isinstance(beamAreas, (list, tuple, np.ndarray)):
                if len(beamAreas) != len(images):
                    raise Exception("Beam Areas list is not same length as images provided")
            else:
                raise Exception("Beam Areas provided needs to be a list")
            
            # adjust image
            for i in range(0,len(images)):
                images[i].image = images[i].image * beamAreas[refImage]/beamAreas[i]
        #except:
        #    raise Exception("Unable to automatically scale powerspectra")
    elif units is not None:
        for imageObj in images:
            imageObj.convertUnits(units)
    
    # see if need to reproject
    if matchProjection:
        # extract reference header
        refHead = images[refImage].header

        # loop over each image and reproject
        for i in range(0,len(images)):
            # skip reference image
            if i == refImage:
                continue

            # reproject
            images[i].reproject(refHead)

    # perform powerspectra calculations
    for imageObj in images:
        imageObj.getPowerSpectra(spatialUnits=spatialUnits, normaliseScale=normaliseScale, oneD=oneD, mask=mask, plot=False)

    # create plot
    images[0].powerSpecPlot(powerSpecInfo=[imageObj.powerSpec for imageObj in images], spatialUnits=spatialUnits, labels=labels, linestyle=linestyle, color=color, show=show, save=save)

    return

##############################################################################################################
###############################################################################################################        
###############################################################################################################

### Functions for interactive plot ###

def clickEvent(e):
    # function to click on either masSB, temperature or beta map
    # moves marker and updates plot
    
    # see if click in axis
    if e.inaxes == fMass.ax or e.inaxes == fTemp.ax:
        inAxes = True
    elif varyBeta and e.inaxes == fBeta.ax:
        inAxes = True
    else:
        inAxes = False
    
    if inAxes:
        # get xdata and ydata
        if e.xdata is not None and e.ydata is not None:
            
            # calculate pixel coordinates
            # create pixel coordinates, and work out SEDlabel for pickle
            pixCoordinates = [int(np.round(e.xdata))+1,int(np.round(e.ydata))+1]
            SEDLabel =  f"{objectID}_{int(np.round(e.xdata))+1}x{int(np.round(e.ydata))+1}"
            print(f'Selecting pixel: {int(np.round(e.xdata))+1}x{int(np.round(e.ydata))+1}')
            
            # move marker on fits images
            fMass._layers['marker_set_1'].set_offsets((int(np.round(e.xdata))+1,int(np.round(e.ydata))+1))
            fTemp._layers['marker_set_1'].set_offsets((int(np.round(e.xdata))+1,int(np.round(e.ydata))+1))
            if varyBeta:
                fBeta._layers['marker_set_1'].set_offsets((int(np.round(e.xdata))+1,int(np.round(e.ydata))+1))
            
            # see if SED fit exists for pixel
            if SEDLabel in SEDresults:
                # get SED for this pixel
                compData, totalModelData, dataPoints, scaleLims, info, yUnit = SEDmodel(SEDresults[SEDLabel][model]['inputData'], SEDresults[SEDLabel][model]['model'], SEDresults[SEDLabel][model]['model']['kappa'], includeColourCorrect, ccInfo, SEDresults[SEDLabel][model]['result'], SEDLabel, model)
        
                ### update SED plot
                # plot components
                for i in range(0,len(compData)):
                    compPlot[i][0].set_xdata(compData[i]['x'])
                    compPlot[i][0].set_ydata(compData[i]['y'])
                    
                # update datapoints
                for i in range(0,3):
                    if dataPoints[i] is not None:
                        update_errorbar(dataPlot[i], dataPoints[i]['x'], dataPoints[i]['y'], yerr=dataPoints[i]['e'])
                    else:
                        dataPlot[i][0].set_data(None,None)
                
                # update total model plot
                totalModelPlot[0].set_xdata(totalModelData['x'])
                totalModelPlot[0].set_ydata(totalModelData['y'])
        
                # set scale limits        
                fSED.set_xlim(scaleLims['x'][0],scaleLims['x'][1])
                fSED.set_ylim(scaleLims['y'][0],scaleLims['y'][1])
        
                # update labels
                figPixText.set_text(f"{objectID} - Pixel: {pixCoordinates[0]} x {pixCoordinates[1]}")
                figSEDText.set_text(info)
            else:
                # plot components
                for i in range(0,len(compPlot)):
                    compPlot[i][0].set_xdata(None)
                    compPlot[i][0].set_ydata(None)
                
                # update datapoints
                for i in range(0,3):
                    try:
                        dataPlot[i][0].set_data(None,None)
                    except:
                        pass
                    for j in range(0,2):
                        try:
                            dataPlot[i][1][j].set_xdata(None)
                        except:
                            pass
                        try:
                            dataPlot[i][1][j].set_ydata(None)
                        except:
                            pass
                    try:
                        dataPlot[i][2][0].set_segments(np.array([]))
                    except:
                        pass
                    
                # update total model plot
                totalModelPlot[0].set_xdata(None)
                totalModelPlot[0].set_ydata(None)
                
                # update labels
                figPixText.set_text(f"{objectID} - Pixel: {pixCoordinates[0]} x {pixCoordinates[1]}")
                figSEDText.set_text("No SED fit for this pixel.")
            
            # redraw plot
            plt.draw()

###############################################################################################################


##############################################################################################################
###############################################################################################################        
###############################################################################################################


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
        elif key[0:4] == "NIKA":
            if "NIKA-2" not in newCCinfo:
                newCCinfo['NIKA-2'] = {}
            newCCinfo['NIKA-2'][key[4:]] = ccinfo[key]
        elif key[0:3] == "ACT":
            if "ACT" not in newCCinfo:
                newCCinfo['ACT'] = {}
            newCCinfo['ACT'][key[3:]] = ccinfo[key]
        else:
            raise Exception("Instrument/band not programmed for cc load")

    
    # return colour correction information
    return newCCinfo

###############################################################################################################

# Function that can do mask processing - sepearte so can run in parallel (has to be outside class)
def regionsToMask(mask, imgWCS, shape):
    threadMask = np.zeros(shape)
    for i in range(0,len(mask)):
        
        # if sky region convert to a pixel region
        if hasattr(mask[i],'to_pixel'):
            # convert to pixel mask
            mask[i] = mask[i].to_pixel(imgWCS)
        
        # create individual mask
        tempMask = mask[i].to_mask(mode='center')

        imgRegion = tempMask.to_image(shape)
        
        # see if region is empty (i.e., no overlap)
        if imgRegion is None:
            continue
        
        #imgMask = imgMask + imgRegion
        threadMask[imgRegion > 0] = 1
    return threadMask

###############################################################################################################

def make_source_mask(data, nsigma, npixels, dilate_size):
        # temporary function to deal with depreciation of make_source_mask in photutils
        from photutils.segmentation import detect_threshold
        from photutils.segmentation import detect_sources
        from scipy import ndimage

        # run detect threshold
        threshold = detect_threshold(data, nsigma)

        segm = detect_sources(data, threshold, npixels)
        if segm is None:
            return np.zeros(data.shape, dtype=bool)

        if dilate_size is not None and dilate_size > 1:
            selem = np.ones((dilate_size, dilate_size))
            return ndimage.binary_dilation(segm.data.astype(bool), selem)
        else:
            return segm.data.astype(bool)

###############################################################################################################