# Module to create kernel to match PSF of images
# the fundctions are based on a python implementation of Aniano et al (2011) method
# and also Thomas Williams (Cardiff Uni PhD) python implementation of that code

import numpy as np
import warnings
import time
import copy
from astroIm import astroImage
import astropy.units as u

# create a PSF class that is the same as astro image but has a few additional methods
class psfImage(astroImage):
    # initialise object either based on the input image or astroImage call 
    def __init__(self, imageIn, makeOdd=False, centrePeak=False, fft=False, ext=0, telescope=None, instrument=None, band=None, unit=None, load=True, FWHM=None, slices=None, dustpediaHeaderCorrect=None):
        # if already is an astroImage object can skip loading, else need to load
        if isinstance(imageIn, astroImage):
            self.__dict__.update(imageIn.__dict__)
        else:
            super().__init__(imageIn, ext, telescope=telescope, instrument=instrument, band=band, unit=unit, load=load, FWHM=FWHM, slices=slices, dustpediaHeaderCorrect=dustpediaHeaderCorrect)

        # convert to square image if rectangular
        if self.image.shape[0] != self.image.shape[1]:
            if self.image.shape[0] > self.image.shape[1]:
                addX = self.image.shape[0] - self.image.shape[1]
                addY = 0

                if addX %2 == 1:
                    print("Rectangular PSF image requires odd amount to be added, setting centrePeak to True")
                    centrePeak = True

                bufferX = addX // 2
                bufferY = 0
            else:
                addY = self.image.shape[1] - self.image.shape[0]
                addX = 0

                if addY %2 == 1:
                    print("Rectangular PSF image requires odd amount to be added, setting centrePeak to True")
                    centrePeak = True
                bufferY = addY // 2
                bufferX = 0
            
            # create new image and embed previous
            newImage = np.zeros(self.image.shape[0]+addY, self.image.shape[1]+addX)
            newImage[bufferY:bufferY+self.image[0],bufferX:bufferX+self.image[1]] = self.image

            # overwrite image and header information
            self.image = newImage
            self.header['NAXIS1'] = newImage.shape[1]
            self.header['NAXIS2'] = newImage.shape[0]

        # if asked for make the image odd
        if makeOdd:
            addX = 0
            addY = 0
            if self.image[0] %2 == 0:
                addY = 1
            if self.image[1] %2 == 0:
                addX = 1
        
            if addX > 0 or addY > 0:
                newImage = np.zeros(self.image.shape[0]+addY, self.image.shape[1]+addX)
                newImage[0:self.image[0],0:self.image[1]] = self.image

                self.image = newImage
                self.header['NAXIS1'] = newImage.shape[1]
                self.header['NAXIS2'] = newImage.shape[0]
        
        # centre peak if desired
        if centrePeak:
            # find peak pixel 
            sel = np.where(self.image == self.image.max())
            peakPix = [sel[0][0], sel[1][0]]

            # calculate offset from centre
            offsetY = 0
            offsetX = 0
            if (self.image.shape[0] + 1) / 2 - 1 - peakPix[0] > 0.5:
                offsetY =  (self.image.shape[0] + 1) // 2 - 1 - peakPix[0]
            if (self.image.shape[1] + 1) / 2 - 1 - peakPix[1] > 0.5:
                offsetX =  (self.image.shape[1] + 1) // 2 - 1 - peakPix[1]
            
            if offsetY > 0 or offsetX > 0:
                # apply offset to image
                newImage = np.zeros(self.image.shape)
                newImage[offsetY:self.image.shape[0],offsetX:self.image.shape[1]] = self.image[0:self.image.shape[0]-offsetY,0:self.image.shape[1]-offsetX]
            
                self.image = newImage
            
        # place at RA = 180 and DEC = 0, and adjust project type
        self.header['CRVAL1'] = 180.0
        self.header['CRVAL2'] = 0.0
        self.header['CTYPE1'] = 'RA---TAN'
        self.header['CTYPE2'] = 'DEC--TAN'
        self.header['CRPIX1'] = (self.image.shape[1] + 1)//2
        self.header['CRPIX2'] = (self.image.shape[0] + 1)//2
        
        if hasattr(self,'pixSize') is False:
            raise Exception("pixSize must be defined for PSF image")

        # track whether been FFT'd or not
        self.fft = fft

        return
    

    # function to normalise the image
    def normalisePSF(self, normalisePeak=False):
        # check if any NaN's in image
        if len(np.where(np.isnan(self.image))[0]) > 0:
            raise Exception("NaN's present on PSF image")

        # normalise image depending on type
        if normalisePeak:
            self.image = self.image / self.image.max()
        else:
            self.image = self.image / self.image.sum()
        
        return
    
    # function to resample the PSF to a new pixel scale
    def resamplePSF(self, newPixSize, interp_order=3, forceOdd=True):
        # get ratio of pixel size
        ratio = (self.pixSize / newPixSize).value

        # new psf image size
        newDimen = np.ceil(np.array(self.image.shape) * ratio).astype(int)

        # check that can place in the middle of the new image
        for i in range(0,2):
            if self.image.shape[i] - newDimen[i] %2:
                newDimen[i] += 1
        
        # resample the array
        from scipy.ndimage import zoom
        resamplePSF = zoom(self.image, ratio, order=interp_order) / ratio**2.0

        # force odd-sized array
        if forceOdd:
            if resamplePSF.shape[0] %2 == 0:
                resamplePSF = resamplePSF[0:resamplePSF.shape[0]-1,:]
            if resamplePSF.shape[1] %2 == 0:
                resamplePSF = resamplePSF[:,0:resamplePSF.shape[1]-1]
            
        ## update the object
        # set image
        self.image = resamplePSF

        # update header
        self.header['NAXIS1'] = self.image.shape[1]
        self.header['NAXIS2'] = self.image.shape[0]
        self.header['CRPIX1'] = (self.image.shape[1] + 1)//2
        self.header['CRPIX2'] = (self.image.shape[0] + 1)//2
        # update pixel size
        if "CDELT1" in self.header:
            self.header['CDELT1'] = -newPixSize.to(u.deg).value
            self.header['CDELT2'] = newPixSize.to(u.deg).value
        if 'CD1_1' in self.header:
            self.header['CD1_1'] = -newPixSize.to(u.deg).value
            self.header['CD2_2'] = newPixSize.to(u.deg).value
            self.header['CD1_2'] = 0.0
            self.header['CD2_1'] = 0.0
        if 'PC1_1' in self.header:
            raise Exception("PC headers not yet implemented")

        return

    # fucntion to centroid the PSF
    def centroid(self, gaussFiltLevel=5, pixThreshold=5e-3):
        from scipy.ndimage import filters
        
        # smooth the psf
        psf_smooth = filters.gaussian_filter(self.image, gaussFiltLevel)

        # assume the centre of the PSF os somewhere in the central half of the data
        psf_max = psf_smooth[psf_smooth.shape[0]//4:3*psf_smooth.shape[0]//4,psf_smooth.shape[1]//4:3*psf_smooth.shape[1]//4].max()

        # find pixels close in value to the maximum
        sel = np.where((psf_max-psf_smooth)/psf_max < pixThreshold)

        # set up variables
        x_centroid = 0
        y_centroid = 0
        n = 0

        for i in range(0,len(sel[0])):
            # skip if not in the centre
            if sel[0][i] < psf_smooth.shape[0]//4 or sel[0][i] >= 3*psf_smooth.shape[0]//4 or sel[1][i] < psf_smooth.shape[1]//4 or sel[1][i] >= 3*psf_smooth.shape[1]//4:
                continue
            
            x_centroid += sel[1][i]
            y_centroid += sel[0][i]
            n += 1
        
        # normalise and adjust
        x_centroid = np.round(x_centroid / n).astype(int)
        y_centroid = np.round(y_centroid / n).astype(int)

        # shift the PSF to centre it
        offsetY = (self.image.shape[0]+1)//2 - y_centroid -1
        offsetX = (self.image.shape[1]+1)//2 - x_centroid -1

        if offsetY > 0 or offsetX > 0:
            # apply offset to image
            newImage = np.zeros(self.image.shape)
            newImage[offsetY:self.image.shape[0],offsetX:self.image.shape[1]] = self.image[0:self.image.shape[0]-offsetY,0:self.image.shape[1]-offsetX]
        
            self.image = newImage

        return

    def circulisePSF(self, rotations=14, polyOrder=3, fourierSpacing=None):
        # function to rotate the PSF and take an interative average each time

        # import scipt rotate function
        from scipy.ndimage import rotate

        # decide how rotating
        if fourierSpacing is None:
            if hasattr(self,'fft'):
                if self.fft:
                    fourierSpacing = True
                else:
                    fourierSpacing = False
            else:
                raise Exception("fourierSpacing must be defined for PSF image, if state unknown")           

        # calculate angles
        if fourierSpacing:
            angles = 360/(2**(np.arange(rotations,0,-1)))
        else:
            angles = (360.0 / rotations) * np.arange(0,rotations)
        
        # create new image
        newPSF = np.zeros(self.image.shape)

        # rotate and add psf
        for i in range(0,len(angles)):
            # rotate image
            print(i)
            rotPSF = rotate(self.image, angles[i], order=polyOrder, reshape=False)

            # add to new image
            newPSF += rotPSF
        
        # normalise
        newPSF = newPSF / len(angles)

        # set anything outside the maximum radius contained within the whole square to be 0
        radius = np.min(newPSF.shape) / 2
        for i in range(0,newPSF.shape[0]):
            for j in range(0,newPSF.shape[1]):
                if (i-(newPSF.shape[0]+1)/2-1)**2 + (j-(newPSF.shape[1]+1)/2-1)**2 > radius**2:
                    newPSF[i,j] = 0.0

        self.image = newPSF

        return
    
    def createFourierTransformPSF(self):
        # function that outputs a fourier transform version of the PSF

        # import module
        import astropy.io.fits as pyfits

        # fourier transform the image
        psf_FFT = np.real(np.fft.fft2(np.fft.ifftshift(self.image)))

        # shift the FFT so the centre is in the middle
        psf_FFT = np.fft.fftshift(psf_FFT)

        # create new astroImage PSF object
        fftHeader = self.header
        fftHdu = pyfits.PrimaryHDU(psf_FFT, fftHeader)
        fftHdulist = pyfits.HDUList([fftHdu])
        
        # create combine astro image
        fftPSFobj = psfImage(fftHdulist, fft=True, load=False)

        return fftPSFobj
    
    def createInverseFourierTransformPSF(self):
        # function that creates an inverse fourier transform version of the PSF/kernel

        # shift the FFT centre so centre is in corners (np.fft default)
        psf_FFT = np.fft.ifftshift(self.image)

        #  inverse fourier transform the image
        newPsf = np.fft.fftshift(np.real(np.fft.ifft2(psf_FFT)))

        # create new astroImage PSF object
        newheader = self.header
        newHdu = pyfits.PrimaryHDU(newPsf, newHeader)
        newHdulist = pyfits.HDUList([newHdu])
        
        # create combine astro image
        psfObj = psfImage(newHdulist, fft=False, load=False)

        return psfObj
        
    def highpassFilterPSF(self):
        # function that highpass filters the PSF

        # only proceed on FFT'd PSF
        if self.fft is False:
            raise Exception("PSF must be Fourier transformed to highpass filter")
        
        # must know FWHM information
        if hasattr(self,'FWHM') is False:
            raise Exception("FWHM must be defined for PSF image")
        
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

        # define the filter parameters
        k_b = 8 * np.pi/(fwhm.to(u.arcsecond).value)
        k_a = 0.9 * k_b

        # apply the filter
        sel = np.where(d_f > k_b)
        self.image[sel] = 0.0
        sel = np.where((d_f >= k_a) & (d_f <= k_b))
        self.image[sel] = np.exp(-1.0*(1.8249*(d_f[sel]-k_b)/(k_b-k_a))**4.0)

        return
    
    def createLowpassFilterPSF(self):
        # function that lowpass filters the PSF

        # only proceed on FFT'd PSF
        if self.fft is False:
            raise Exception("PSF must be Fourier transformed to lowpass filter")
        
        # find where maximum power is
        source_fourier_data = psf[int(x_range/2):-1,int(y_range/2)]
        fft_max = np.amax(source_fourier_data)

        # find scale need to go to
        for n in range(len(source_fourier_data)):
            if source_fourier_data[n] < 0.005*fft_max:
                k_h = n*self.pixSize.to(u.arcsecond).value
                break
        
        # define k_l parameter
        k_l = 0.7 * k_h
        
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

        lowPassFilter = np.ones(self.image.shape)
        # apply the filter
        sel = np.where(d_f > k_h)
        lowPassFilter.image[sel] = 0.0
        sel = np.where((d_f >= k_l) & (d_f <= k_h))
        lowPassFilter.image[sel] = 0.5*(1+np.cos(np.pi*(d_f[sel]-k_l)/(k_h-k_l)))

        return lowPassFilter

# master function to create PSF kernel
def createPSFkernel(hiresPSF, lowresPSF, outputPixelSize=0.2*u.arcsec, operatingPixelSize=0.2*u.arcsec, circulisePSFs=False, verbose=True, returnOperatingKernel=False):
    # check if PSFs are astroImage objects
    if isinstance(hiresPSF, psfImage) is False:
        raise Exception("hiresPSF is not an astroImage PSF object")
    if isinstance(lowresPSF, psfImage) is False:
        raise Exception("lowresPSF is not an astroImage PSF object")
    
    # check operating Pixel size is smaller or equal to output pixel size
    if operatingPixelSize > outputPixelSize:
        raise Exception ("Operating pixel size must be smaller or equal to output pixel size")

    # normalise PSFs by total flux
    if verbose:
        print("Normalising PSFs")
    hiresPSF.normalisePSF()
    lowresPSF.normalisePSF()

    # Resample the PSFs to operating pixel scale
    if verbose:
        print("Resampling PSFs to operating pixel scale")
    hiresPSF.resamplePSF(operatingPixelSize)
    lowresPSF.resamplePSF(operatingPixelSize)
    

    # centroid the PSFs
    if verbose:
        print("Centering PSfs")
    hiresPSF.centroid()
    lowresPSF.centroid()

    # circulise PSFs if desired
    if circulisePSFs:
        if verbose:
            print("Circulising PSFs")
        hiresPSF.circulisePSF()
        lowresPSF.circulisePSF()

    # Fourier tranform the PSFs - only take the real part
    print("Performing Fourier transform on PSFs")
    hiresPSF_FFT = hiresPSF.createFourierTransformPSF()
    lowresPSF_FFT = lowresPSF.createFourierTransformPSF()

    # circularise the FFTs
    print("Circulising PSFs FFTs")
    hiresPSF_FFT.circulisePSF()
    lowresPSF_FFT.circulisePSF()

    # highpass filter the PSFs
    print("Highpass filtering PSFs")
    hiresPSF_FFT.highpassFilterPSF()
    lowresPSF_FFT.highpassFilterPSF()

    # Invert the source FFT (treat any /0 as 0)
    print("Inverting High resolution PSF FFT")
    hiresPSF_FFT_invert_image = np.zeros(hiresPSF.image.shape)
    sel = np.where(hiresPSF_FFT.image != 0.0)
    hiresPSF_FFT_invert_image[sel] = 1.0 / hiresPSF_FFT.image[sel]

    # lowpass filter the low resolution PSF
    print("Creating Lowpass filter")
    lowpassFilter = highresPSF_FFT.lowpassFilterPSF()

    # calculate the FT of convolution kernel
    print("Creating FFT of the kernel")
    kernel_FFT_image = lowresPSF_FFT.image * (lowpassFilter*hiresPSF_FFT_invert_image)

    # create astroimage object of the kernel
    kernel_FFT = copy.deepcopy(hiresPSF_FFT)
    kernel_FFT.image = kernel_FFT_image

    # remove keywords to stop future confusion
    if hasattr(kernel_FFT,'FWHM'):
        del(kernel_FFT.FWHM)
    if hasattr(kernel_FFT,'band'):
        del(kernel_FFT.band)
    if hasattr(kernel_FFT,'instrument'):
        del(kernel_FFT.instrument)
    if hasattr(kernel_FFT,'telescope'):
        del(kernel_FFT.telescope)
    
    # Inverse FFT the kernel
    print("Inverse Fourier Transforming the Kernel")
    kernel = kernel_FFT.createInverseFourierTransformPSF()

    # circulise kernel
    print("Circulising the Kernel")
    kernel.circulisePSF()

    # normalise kernel
    kernel.normalisePSF()

    if outputPixelSize > operatingPixelSize:
        ## resample kernel to output pixel scale
        # calculate see size of new image
        newSize = np.round(np.array(kernel.image.shape) * (kernel.pixSize / outputPixelSize).value).astype(int)
        # check is odd
        if newSize[0] %2 == 0:
            newSize[0] += 1
        if newSize[1] %2 == 0:
            newSize[1] += 1

        # create new header
        newHeader = kernel.header.copy()
        newHeader['NAXIS1'] = newSize[1]
        newHeader['NAXIS2'] = newSize[0]
        if 'CDELT1' in newHeader:
            newHeader['CDELT1'] = -outputPixelSize.to(u.deg).value
            newHeader['CDELT2'] = outputPixelSize.to(u.deg).value
        if 'CD1_1' in newHeader:
            newHeader['CD1_1'] = -outputPixelSize.to(u.deg).value
            newHeader['CD2_2'] = outputPixelSize.to(u.deg).value
            newHeader['CD1_2'] = 0.0
            newHeader['CD2_1'] = 0.0
        if 'PC1_1' in newHeader:
            raise Exception("PC headers not yet implemented")
        newHeader['CRPIX1'] = (newSize[1] + 1)//2
        newHeader['CRPIX2'] = (newSize[0] + 1)//2

        # perform reprojection
        reproKernel = kernel.reproject(newHeader, conserveFlux=True)

        # Not needed but renomalise
        reproKernel.normalisePSF()
    else:
        reproKernel = kernel
    
    # return kernels
    if returnOperatingKernel:
        return reproKernel, kernel
    else:
        return reproKernel

    

