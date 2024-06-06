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
    
    # function to remove zero edges from the PSF
    def removeZeroEdges(self, minPadding=1, matchKernel=None):
        # calculate indicies to loop over
        startI = 0
        endI = np.array(self.image.shape).min() // 2

        # loop progressively inwards
        zeroPadding = 0
        for i in range(startI,endI):
            # check if any non-zero pixels
            if np.sum(self.image[i,:]) == 0.0 and np.sum(self.image[self.image.shape[0]-1-i,:]) == 0.0 and np.sum(self.image[:,i]) == 0.0 and np.sum(self.image[:,self.image.shape[1]-1-i]) == 0.0:
                zeroPadding = i + 1
                continue
            else:
                # if not all zero then break
                break
        
        # if matching zero removal with another kernel see how many zeros can remove that kernel
        if matchKernel is not None:
            matchZeroPadding = 0
            for i in range(startI, zeroPadding+1):
                if np.sum(matchKernel.image[i,:]) == 0.0 and np.sum(matchKernel.image[matchKernel.image.shape[0]-1-i,:]) == 0.0 and np.sum(matchKernel.image[:,i]) == 0.0 and np.sum(matchKernel.image[:,matchKernel.image.shape[1]-1-i]) == 0.0:
                    matchZeroPadding = i + 1
                    continue
                else:
                    # if not all zero then break
                    break
            
            # limit the zero padding to the matched value if its less 
            if matchZeroPadding < zeroPadding:
                zeroPadding = matchZeroPadding

        
        # calculate number of rows/columns to remove
        if zeroPadding - minPadding > 0:
            nRemove = zeroPadding - minPadding

            # adjust image size and header
            self.image = self.image[nRemove:-nRemove,nRemove:-nRemove]
            self.header['NAXIS1'] = self.image.shape[1]
            self.header['NAXIS2'] = self.image.shape[0]
            self.header['CRPIX1'] = self.header['CRPIX1'] - nRemove
            self.header['CRPIX2'] = self.header['CRPIX2'] - nRemove

            # if matched removal apply
            if matchKernel is not None:
                matchKernel.image = matchKernel.image[nRemove:-nRemove,nRemove:-nRemove]
                matchKernel.header['NAXIS1'] = matchKernel.image.shape[1]
                matchKernel.header['NAXIS2'] = matchKernel.image.shape[0]
                matchKernel.header['CRPIX1'] = matchKernel.header['CRPIX1'] - nRemove
                matchKernel.header['CRPIX2'] = matchKernel.header['CRPIX2'] - nRemove
        
        if matchKernel is not None:
            return matchKernel
        else:
            return
    
    # function to apply a radial mask
    def radialMask(self, maskRadius, maskValue=0.0):
        # calculate radius array
        xx, yy = np.meshgrid(np.arange(0,self.image.shape[1]), np.arange(0,self.image.shape[0]))
        rad = np.sqrt((xx-((self.image.shape[1]+1)/2-1))**2 + (yy-((self.image.shape[0]+1)/2-1))**2)

        # scale radius map for pixel size
        if hasattr(self,'pixSize') is False:
            self.getPixelScale()
        rad = rad * self.pixSize

        # select where to mask
        sel = np.where(rad > maskRadius)
        self.image[sel] = maskValue

        return

        

    # function to add zero edges for padding
    def zeroPad(self, newShape):
        # if new shape is only one dimension or float create a 2D array
        if isinstance(newShape, list) or isinstance(newShape, np.ndarray) or isinstance(newShape, tuple):
            pass
        else:
            newShape = np.array([newShape,newShape])

        # check if new shape is larger than current shape
        if newShape[0] < self.image.shape[0] or newShape[1] < self.image.shape[1]:
            raise Exception("New shape must be larger than current shape")
        
        # check if padding is even on both sides
        if (newShape[0] - self.image.shape[0]) %2 == 1 or (newShape[1] - self.image.shape[1]) %2 == 1:
            raise Exception("Difference in sizes must be even so same on both sides")
        
        # calculate padding
        nPadY = (newShape[0] - self.image.shape[0])//2
        nPadX = (newShape[1] - self.image.shape[1])//2

        # create new image
        newImage = np.zeros(newShape)

        # put in current image
        newImage[nPadY:nPadY+self.image.shape[0],nPadX:nPadX+self.image.shape[1]] = self.image

        # update image and header
        self.image = newImage
        self.header['NAXIS1'] = self.image.shape[1]
        self.header['NAXIS2'] = self.image.shape[0]
        self.header['CRPIX1'] = self.header['CRPIX1'] + nPadX
        self.header['CRPIX2'] = self.header['CRPIX2'] + nPadY

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
    def resamplePSF(self, newPixSize, interp_order=3, forceOdd=True, onlyReturn=False):
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
        if onlyReturn is False:
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

            try:
                self.getPixelScale()
            except:
                pass

        if onlyReturn:
            return resamplePSF
        else:
            return

    # fucntion to centroid the PSF
    def centroid(self, gaussFiltLevel=5, pixThreshold=5e-3):
        from scipy.ndimage import filters
        
        # smooth the psf
        psf_smooth = filters.gaussian_filter(self.image, gaussFiltLevel)

        # assume the centre of the PSF is somewhere in the central half of the data
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
        offsetY = y_centroid - ((self.image.shape[0]+1)//2 - 1)
        offsetX = x_centroid - ((self.image.shape[1]+1)//2 - 1)

        if np.abs(offsetY) > 0 or np.abs(offsetX) > 0:
            # create new image
            newImage = np.zeros(self.image.shape)

            # calculate image indices
            if offsetY >= 0:
                y1 = 0
                y2 = self.image.shape[0]-offsetY
                y3 = offsetY
                y4 = self.image.shape[0]
            else:
                y1 = -offsetY
                y2 = self.image.shape[0]
                y3 = 0
                y4 = self.image.shape[0]+offsetY
            if offsetX >= 0:
                x1 = 0
                x2 = self.image.shape[1]-offsetX
                x3 = offsetX
                x4 = self.image.shape[1]
            else:
                x1 = -offsetX
                x2 = self.image.shape[1]
                x3 = 0
                x4 = self.image.shape[1]+offsetX

            # shift the image
            newImage[y1:y2,x1:x2] = self.image[y3:y4,x3:x4]

            # store back to object
            self.image = newImage

        return

    def circulisePSF(self, polyOrder=3, upScaleFactor=3.0):
        # function to make PSF circularly symmetric
        
        # import modules
        from scipy.interpolate import interp1d

        # upscale the image
        upScaleImage = self.resamplePSF(self.pixSize/upScaleFactor, interp_order=polyOrder, onlyReturn=True)

        # create radius positions (in pixels) for upscaled image
        xx, yy = np.meshgrid(np.arange(0,upScaleImage.shape[1]), np.arange(0,upScaleImage.shape[0]))
        rad = np.sqrt((xx-((upScaleImage.shape[1]+1)/2-1))**2 + (yy-((upScaleImage.shape[0]+1)/2-1))**2)

        # divide radius by upScale Factor
        rad = rad / upScaleFactor

        # decide the maximum radius required
        sel = np.where(upScaleImage==0.0)
        if len(sel[0]) == 0:
            maxRadius = np.min(self.image.shape)+1 /2
        else:
            maxRadius = np.min(rad[sel])
        
        # select only values with maxRadius
        sel = np.where(rad <= maxRadius)
        cutRad = rad[sel]
        cutImage = upScaleImage[sel]

        # convert radius to integer
        cutRad = np.round(cutRad).astype(int)
        
        # sum all values at each integer radius
        totBin = np.bincount(cutRad, cutImage)
        
        # count number of values at each integer radius
        nBin = np.bincount(cutRad)
        
        # create radial profile by normalising sum
        kernelProfile = totBin/nBin

        # adjust r=0
        kernelProfile[0] = upScaleImage.max()

        # create scipy interpolater
        radFunc = interp1d(np.arange(cutRad.min(),cutRad.max()+1,1), kernelProfile, kind=polyOrder)

        # need to create radius array for original image
        xx, yy = np.meshgrid(np.arange(0,self.image.shape[1]), np.arange(0,self.image.shape[0]))
        rad = np.sqrt((xx-((self.image.shape[1]+1)/2-1))**2 + (yy-((self.image.shape[0]+1)/2-1))**2)

        # create for interpolater
        newKernel = np.zeros(self.image.shape)
        sel = np.where(rad <= cutRad.max())
        newKernel[sel] = radFunc(rad[sel])

        # save image to kernel
        self.image = newKernel
        
        return

    def createFourierTransformPSF(self):
        # function that outputs a fourier transform version of the PSF

        # fourier transform the image
        psf_FFT = np.real(np.fft.fft2(np.fft.ifftshift(self.image)))

        # shift the FFT so the centre is in the middle
        psf_FFT = np.fft.fftshift(psf_FFT)

        # create new astroImage PSF object
        fftPSFobj = copy.deepcopy(self)
        fftPSFobj.image = psf_FFT
        
        # set the FFT keyword
        fftPSFobj.fft = True

        return fftPSFobj
    
    def createInverseFourierTransformPSF(self):
        # function that creates an inverse fourier transform version of the PSF/kernel

        # shift the FFT centre so centre is in corners (np.fft default)
        psf_FFT = np.fft.ifftshift(self.image)

        #  inverse fourier transform the image
        newPsf = np.fft.fftshift(np.real(np.fft.ifft2(psf_FFT)))

        # create new astroImage PSF object
        psfObj = copy.deepcopy(self)
        psfObj.image = newPsf

        # set the FFT keyword
        psfObj.fft = False

        return psfObj
        
    def highpassFilterPSF(self, applyFilter=True, returnFilter=False, filterExtent=4.0):
        # function that highpass filters the PSF

        # only proceed on FFT'd PSF
        if self.fft is False:
            raise Exception("PSF must be Fourier transformed to highpass filter")
        
        # must know FWHM information
        if hasattr(self,'fwhm') is False:
            raise Exception("FWHM must be defined for PSF image")
                
        # Calculate the frequencies in the Fourier plane to create a filter
        x_f,y_f = np.meshgrid(np.fft.fftfreq(self.image.shape[0],self.pixSize.to(u.arcsecond).value),
                              np.fft.fftfreq(self.image.shape[1],self.pixSize.to(u.arcsecond).value))
        #d_f = np.sqrt(x_f**2 + y_f**2) *2.0#Factor of 2 due to Nyquist sampling
        d_f = np.sqrt(x_f**2 + y_f**2)
        d_f = np.transpose(d_f)

        # define the filter parameters
        k_b = filterExtent * 2.0 * np.pi/(self.fwhm.to(u.arcsecond).value)
        k_a = 0.9 * k_b

        # create the filter
        filter = np.ones(self.image.shape)
        sel = np.where(d_f > k_b)
        filter[sel] = 0.0
        sel = np.where((d_f >= k_a) & (d_f <= k_b))
        filter[sel] = np.exp(-1.0*(1.8249*(d_f[sel]-k_a)/(k_b-k_a))**4.0)

        ## Force in the amplitude at (0,0) since d_f here is undefined
        #filter[0,0] = 0

        # shift the filter
        fourierFilter = np.fft.fftshift(filter)
        #fourierFilter = filter

        # apply the filter
        if applyFilter:
            self.image = self.image * fourierFilter

        # return the filter if desired
        if returnFilter:
            return fourierFilter
        else:
            return
    
    def lowpassFilterPSF(self, applyFilter=True, returnFilter=False):
        # function that lowpass filters the PSF

        # only proceed on FFT'd PSF
        if self.fft is False:
            raise Exception("PSF must be Fourier transformed to lowpass filter")
        
        # Calculate the frequencies in the Fourier plane to create a filter
        x_f,y_f = np.meshgrid(np.fft.fftfreq(self.image.shape[0],self.pixSize.to(u.arcsecond).value),
                              np.fft.fftfreq(self.image.shape[1],self.pixSize.to(u.arcsecond).value))
        #d_f = np.sqrt(x_f**2 + y_f**2) *2.0#Factor of 2 due to Nyquist sampling
        d_f = np.sqrt(x_f**2 + y_f**2)
        d_f = np.transpose(d_f)

        # shift the d_f for fft
        d_f = np.fft.fftshift(d_f)

        # find where maximum power is
        source_fourier_data = self.image[int(self.image.shape[0]/2):-1,int(self.image.shape[1]/2)]
        fft_max = np.amax(source_fourier_data)

        # find scale need to go to
        for n in range(len(source_fourier_data)):
            if source_fourier_data[n] < 0.005*fft_max:
                k_h = d_f[n+int(self.image.shape[0]/2),int(self.image.shape[1]/2)]
                break
        
        # define k_l parameter
        k_l = 0.7 * k_h

        lowPassFilter = np.ones(self.image.shape)
        # apply the filter
        sel = np.where(d_f > k_h)
        lowPassFilter[sel] = 0.0
        sel = np.where((d_f >= k_l) & (d_f <= k_h))
        lowPassFilter[sel] = 0.5*(1+np.cos(np.pi*(d_f[sel]-k_l)/(k_h-k_l)))

        # Force in the amplitude at (0,0) since d_f here is undefined
        #lowPassFilter[0,0] = 0

        # apply the filter
        if applyFilter:
            self.image = self.image * lowPassFilter

        # return the filter if desired
        if returnFilter:
            return lowPassFilter
        else:
            return

    def reprojectPSF(self, outputPixelSize, circulisePSF=True, parallel=False):    
        # function to reproject kernel with a call to standard reproject image 

        ## resample kernel to output pixel scale
        # calculate see size of new image
        newSize = np.round(np.array(self.image.shape) * (self.pixSize / outputPixelSize).value).astype(int)
        # check is odd
        if newSize[0] %2 == 0:
            newSize[0] += 1
        if newSize[1] %2 == 0:
            newSize[1] += 1

        # create new header
        newHeader = self.header.copy()
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
        reproKernel = self.reproject(newHeader, conserveFlux=True, parallel=parallel)

        if circulisePSF:
            # circulise the PSF
            reproKernel.circulisePSF()

        # Not needed but renomalise
        reproKernel.normalisePSF()

        return reproKernel

    def trimKernel(self, trimLevel=0.999):
        # function to trim the kernel based on enclosed energy

        # calculate pixel radius at each point
        xx, yy = np.meshgrid(np.arange(0,self.image.shape[1]), np.arange(0,self.image.shape[0]))
        rad = np.sqrt((xx-((self.image.shape[1]+1)/2-1))**2 + (yy-((self.image.shape[0]+1)/2-1))**2)
        
        # select maximum radius (removing corners)
        maximRad = (np.min(self.image.shape)+1)//2

        # select only values within maximum radius
        sel = np.where(rad <= maximRad)
        cutRad = rad[sel]
        cutImage = self.image[sel]

        # convert radius to integer
        cutRad = np.round(cutRad).astype(int)

        # sum all values at each integer radius
        totBin = np.bincount(cutRad, cutImage)

        # create profle by doing cummulative sum
        trimProfile = np.cumsum(totBin)

        # loop outwards in whole pixels to find point where kernel is above trim level
        trimProfile = np.append(0.0,trimProfile)
                       
        # select the first point where all points are above trim level
        sel = np.where(trimProfile > trimLevel)
        for i in range(0,len(sel[0])):
            if np.all(trimProfile[sel[0][i]:] > trimLevel):
                maxRad = sel[0][i]
                break

        # check maxRad is below size of current kernel
        if maxRad < (np.min(self.image.shape)+1)//2:
            # copy kernel
            trimReproKernel = copy.deepcopy(self)

            # extract image
            trimReproKernel.image = self.image[(self.image.shape[0]+1)//2-1-maxRad:(self.image.shape[0]+1)//2+maxRad,(self.image.shape[1]+1)//2-1-maxRad:(self.image.shape[1]+1)//2+maxRad]

            # update header
            trimReproKernel.header['NAXIS1'] = trimReproKernel.image.shape[1]
            trimReproKernel.header['NAXIS2'] = trimReproKernel.image.shape[0]
            trimReproKernel.header['CRPIX1'] = (trimReproKernel.image.shape[1] + 1)//2
            trimReproKernel.header['CRPIX2'] = (trimReproKernel.image.shape[0] + 1)//2
        else:
            trimReproKernel = copy.deepcopy(self)
        
        return trimReproKernel

# master function to create PSF kernel
def createPSFkernel(hiresPSF, lowresPSF, outputPixelSize=0.2*u.arcsec, operatingPixelSize=0.2*u.arcsec, circulisePSFs=True, overCirculise=False, maxSize=None, verbose=True, returnOperatingKernel=False, trimKernel=True, trimLevel=0.999, parallel=False):
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
        print("\t\t Normalising PSFs")
    hiresPSF.normalisePSF()
    lowresPSF.normalisePSF()

    # Resample the PSFs to operating pixel scale
    if verbose:
        print("\t\t Resampling PSFs to operating pixel scale")
    hiresPSF.resamplePSF(operatingPixelSize)
    lowresPSF.resamplePSF(operatingPixelSize)
    
    # Make sure both kernels are the same size
    if verbose:
        print("\t\t Optimising and matching size of the PSFs")

    # see if extra zeros could be culled from the edges
    hiresPSF.removeZeroEdges()
    lowresPSF.removeZeroEdges()

    if maxSize is not None:
        raise Exception("\t\t Max Size is not implemented yet")
    
    # check if sizes are the same:
    if hiresPSF.image.shape != lowresPSF.image.shape:
        # see which is larger and zero pad the smaller one
        if hiresPSF.image.shape[0] > lowresPSF.image.shape[0] and hiresPSF.image.shape[1] > lowresPSF.image.shape[1]:
            lowresPSF.zeroPad(hiresPSF.image.shape)
        elif hiresPSF.image.shape[0] < lowresPSF.image.shape[0] and hiresPSF.image.shape[1] < lowresPSF.image.shape[1]:
            hiresPSF.zeroPad(lowresPSF.image.shape)
        else:
            newSize = np.array([np.max([hiresPSF.image.shape[0],lowresPSF.image.shape[0]]),np.max([hiresPSF.image.shape[1],lowresPSF.image.shape[1]])])
            hiresPSF.zeroPad(newSize)
            lowresPSF.zeroPad(newSize)

    # centroid the PSFs
    if verbose:
        print("\t\t Centering PSfs")
    hiresPSF.centroid()
    lowresPSF.centroid()

    # circulise PSFs if desired
    if circulisePSFs:
        if verbose:
            print("\t\t Circulising PSFs")
        hiresPSF.circulisePSF()
        lowresPSF.circulisePSF()
        # remove any extra zeros created by circulising
        lowresPSF = hiresPSF.removeZeroEdges(minPadding=1, matchKernel=lowresPSF)

    # Fourier tranform the PSFs - only take the real part
    if verbose:
        print("\t\t Performing Fourier transform on PSFs")
    hiresPSF_FFT = hiresPSF.createFourierTransformPSF()
    lowresPSF_FFT = lowresPSF.createFourierTransformPSF()

    # circularise the FFTs
    if circulisePSFs and overCirculise:
        if verbose:
            print("\t\t Circulising PSFs FFTs")
        hiresPSF_FFT.circulisePSF()
        lowresPSF_FFT.circulisePSF()

    # highpass filter the PSFs
    if verbose:
        print("\t\t Highpass filtering PSFs")
    hiresPSF_FFT.highpassFilterPSF()
    lowresPSF_FFT.highpassFilterPSF()

    # Invert the source FFT (treat any /0 as 0)
    if verbose:
        print("\t\t Inverting High resolution PSF FFT")
    hiresPSF_FFT_invert_image = np.zeros(hiresPSF_FFT.image.shape)
    sel = np.where(hiresPSF_FFT.image != 0.0)
    hiresPSF_FFT_invert_image[sel] = 1.0 / hiresPSF_FFT.image[sel]

    # lowpass filter the low resolution PSF
    if verbose:
        print("\t\t Creating Lowpass filter")
    lowpassFilter = hiresPSF_FFT.lowpassFilterPSF(returnFilter=True, applyFilter=False)

    # calculate the FT of convolution kernel
    if verbose:
        print("\t\t Creating FFT of the kernel")
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
    if verbose:
        print("\t\t Inverse Fourier Transforming the Kernel")
    kernel = kernel_FFT.createInverseFourierTransformPSF()

    # circulise kernel
    if circulisePSFs and overCirculise:
        if verbose:
            print("\t\t Circulising the Kernel")
        kernel.circulisePSF()

    # normalise kernel
    if verbose:
        print("\t\t Normalising the Kernel")
    kernel.normalisePSF()

    # reproject kernel to output pixel scale
    if outputPixelSize > operatingPixelSize:
        if verbose:
            print("\t\t Reprojecting kernel to output resolution")

        # call reproject method
        reproKernel = kernel.reprojectPSF(outputPixelSize, circulisePSF=circulisePSFs, parallel=parallel)
    else:
        reproKernel = kernel
    
    # trim the kernel
    if trimKernel:
        if verbose:
            print("\t\t Trimming kernel")
        
        # trim kernel
        trimReproKernel = reproKernel.trimKernel(trimLevel=trimLevel)
    else:
        trimReproKernel = reproKernel
    
    # return kernels
    if returnOperatingKernel:
        return trimReproKernel, kernel
    else:
        return trimReproKernel

# function to create a psf from an EEF file
def createPSFfromEEF(radius, EEF, outputPixelSize, normalise=True, polyOrder=3, fitGaussian=True, nGaussianPoints=2, FWHM=1.0*u.arcsecond, plotProfile=False):
    # function to create a psf from an EEF file

    # import modules
    from scipy.interpolate import interp1d
    import astropy.io.fits as pyfits
    if plotProfile:
        import matplotlib.pyplot as plt

    # normalise based on last radius
    if normalise:
        EEF = EEF / EEF[-1]

    # set up profile arrays
    profileRad = np.array([]) *u.arcsecond
    profileSB = np.array([])

    # loop through and find surface brightness
    for i in range(1,len(radius)):
        # calculate surface brightness of PSF based on difference in EEF
        profileSB = np.append(profileSB,(EEF[i]-EEF[i-1]) / (np.pi * (radius[i].to(u.arcsecond).value**2 - radius[i-1].to(u.arcsecond).value**2)))
        # calculate radius
        profileRad = np.append(profileRad, (radius[i]+radius[i-1])/2.0)
    
    # decide size of image to create
    imageSize = np.ceil(profileRad[-1] / outputPixelSize).astype(int) * 2 + 1
    
    # create radius array
    xx, yy = np.meshgrid(np.arange(0,imageSize), np.arange(0,imageSize))
    rad = np.sqrt((xx-((imageSize+1)/2-1))**2 + (yy-((imageSize+1)/2-1))**2) * outputPixelSize.to(u.arcsecond).value

    # create scipy interpolater
    psfProFunc = interp1d(profileRad.to(u.arcsecond).value, profileSB, kind=polyOrder, fill_value='extrapolate')

    # apply the interpolater
    imagePSF = psfProFunc(rad)

    # fit a gaussian to the first two data points
    if fitGaussian:
        from scipy.optimize import curve_fit

        # initial guess
        p0 = [1.0, FWHM.to(u.arcsecond).value/2.355]

        def gaussian(x, a, sigma):
            return a*np.exp(-(x)**2/(2*sigma**2))
        popt, pcov = curve_fit(gaussian, profileRad[0:nGaussianPoints].to(u.arcsecond).value, profileSB[0:nGaussianPoints], p0=p0)

        # apply gaussian fit below where fit is defined
        sel = np.where(rad <= profileRad[nGaussianPoints-1].to(u.arcsecond).value)
        imagePSF[sel] = np.exp(-1.0*rad[sel]**2.0/(2*popt[1]**2.0))*popt[0]
    
    if plotProfile:
        fig = plt.figure()
        f1 = plt.axes([0.1,0.1,0.85,0.85])
        f1.plot(profileRad, profileSB, 'o')
        plotRad = np.arange(0,profileRad[-1].to(u.arcsecond).value,0.01)
        f1.plot(plotRad, psfProFunc(plotRad))
        if fitGaussian:
            sel = np.where(plotRad <= profileRad[nGaussianPoints-1].to(u.arcsecond).value)
            f1.plot(plotRad[sel], np.exp(-1.0*plotRad[sel]**2.0/(2*popt[1]**2.0))*popt[0])
        plt.show()

    # as extrapolating set anywhere above profileRad to zero
    sel = np.where(rad > profileRad[-1].to(u.arcsecond).value)
    imagePSF[sel] = 0.0

    # create a HDU to crete header object
    hdu = pyfits.PrimaryHDU(imagePSF)

    # add in pixel size
    hdu.header['CDELT1'] = -outputPixelSize.to(u.deg).value
    hdu.header['CDELT2'] = outputPixelSize.to(u.deg).value

    # put in HDUList
    hduList = pyfits.HDUList([hdu])

    # create PSF object
    psfObj = psfImage(hduList, load=False)

    # normalise PSF
    psfObj.normalisePSF()

    return psfObj



    

