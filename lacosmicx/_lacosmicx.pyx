# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: profile=True, boundscheck=False, nonecheck=False, wraparound=False
# cython: cdivision=True
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
"""
Name : Lacosmicx
Author : Curtis McCully
Date : October 2014
"""
import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
np.import_array()

from libcpp cimport bool

from cython.parallel cimport parallel, prange

from libc.stdint cimport uint8_t
from libc.stdlib cimport abort, malloc, free


cdef extern from "laxutils.h":
    float PyMedian(float * a, int n) nogil
    float PyOptMed3(float * a) nogil
    float PyOptMed5(float * a) nogil
    float PyOptMed7(float * a) nogil
    float PyOptMed9(float * a) nogil
    float PyOptMed25(float * a) nogil
    void PyMedFilt3(float * data, float * output, int nx, int ny) nogil
    void PyMedFilt5(float * data, float * output, int nx, int ny) nogil
    void PyMedFilt7(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt3(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt5(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt7(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt9(float * data, float * output, int nx, int ny) nogil
    void PySubsample(float * data, float * output, int nx, int ny) nogil
    void PyRebin(float * data, float * output, int nx, int ny) nogil
    void PyConvolve(float * data, float * kernel, float * output, int nx,
                    int ny, int kernx, int kerny) nogil
    void PyLaplaceConvolve(float * data, float * output, int nx, int ny) nogil
    void PyDilate3(uint8_t * data, uint8_t * output, int nx, int ny) nogil
    void PyDilate5(uint8_t * data, uint8_t * output, int niter, int nx,
                   int ny) nogil


def lacosmicx(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] indat,
              np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] inmask=None,
              float sigclip=4.5, float sigfrac=0.3, float objlim=5.0,
              float gain=1.0, float readnoise=6.5, float satlevel=65536.0,
              float pssl=0.0, int niter=4, sepmed=True, cleantype='meanmask',
              fsmode='median', psfmodel='gauss', float psffwhm=2.5,
              int psfsize=7, psfk=None, float psfbeta=4.765, verbose=False):
    """lacosmicx(indat, inmask=None, sigclip=4.5, sigfrac=0.3, objlim=5.0,
                 gain=1.0, readnoise=6.5, satlevel=65536.0, pssl=0.0, niter=4,
                 sepmed=True, cleantype='meanmask', fsmode='median',
                 psfmodel='gauss', psffwhm=2.5,psfsize=7, psfk=None,
                 psfbeta=4.765, verbose=False)\n
    Run the LACosmic algorithm to detect cosmic rays in a numpy array.

    If you use this code, please add this repository address in a footnote:
    https://github.com/cmccully/lacosmicx

    Please cite the original paper which can be found at:
    U{http://www.astro.yale.edu/dokkum/lacosmic/}

    van Dokkum 2001, PASP, 113, 789, 1420
    (article : U{http://adsabs.harvard.edu/abs/2001PASP..113.1420V})

    Parameters
    ----------
    indat : float numpy array
        Input data array that will be used for cosmic ray detection.

    inmask : boolean numpy array, optional
        Input bad pixel mask. Values of True will be ignored in the cosmic ray
        detection/cleaning process. Default: None.

    sigclip : float, optional
        Laplacian-to-noise limit for cosmic ray detection. Lower values will
        flag more pixels as cosmic rays. Default: 4.5.

    sigfrac : float, optional
        Fractional detection limit for neighboring pixels. For cosmic ray
        neighbor pixels, a lapacian-to-noise detection limit of
        sigfrac * sigclip will be used. Default: 0.3.

    objlim : float, optional
        Minimum contrast between Laplacian image and the fine structure image.
        Increase this value if cores of bright stars are flagged as cosmic
        rays. Default: 5.0.

    pssl : float, optional
        Previously subtracted sky level in ADU. We always need to work in
        electrons for cosmic ray detection, so we need to know the sky level
        that has been subtracted so we can add it back in. Default: 0.0.

    gain : float, optional
        Gain of the image (electrons / ADU). We always need to work in
        electrons for cosmic ray detection. Default: 1.0

    readnoise : float, optional
        Read noise of the image (electrons). Used to generate the noise model
        of the image. Default: 6.5.

    satlevel : float, optional
        Saturation of level of the image (electrons). This value is used to
        detect saturated stars and pixels at or above this level are added to
        the mask. Default: 65536.0.

    niter : int, optional
        Number of iterations of the LA Cosmic algorithm to perform. Default: 4.

    sepmed : boolean, optional
        Use the separable median filter instead of the full median filter.
        The separable median is not identical to the full median filter, but
        they are approximately the same and the separable median filter is
        significantly faster and still detects cosmic rays well. Default: True

    cleantype : {'median', 'medmask', 'meanmask', 'idw'}, optional
        Set which clean algorithm is used:\n
        'median': An umasked 5x5 median filter\n
        'medmask': A masked 5x5 median filter\n
        'meanmask': A masked 5x5 mean filter\n
        'idw': A masked 5x5 inverse distance weighted interpolation\n
        Default: "meanmask".

    fsmode : {'median', 'convolve'}, optional
        Method to build the fine structure image:\n
        'median': Use the median filter in the standard LA Cosmic algorithm
        'convolve': Convolve the image with the psf kernel to calculate the
        fine structure image.
        Default: 'median'.

    psfmodel : {'gauss', 'gaussx', 'gaussy', 'moffat'}, optional
        Model to use to generate the psf kernel if fsmode == 'convolve' and
        psfk is None. The current choices are Gaussian and Moffat profiles.
        'gauss' and 'moffat' produce circular PSF kernels. The 'gaussx' and
        'gaussy' produce Gaussian kernels in the x and y directions
        respectively. Default: "gauss".

    psffwhm : float, optional
        Full Width Half Maximum of the PSF to use to generate the kernel.
        Default: 2.5.

    psfsize : int, optional
        Size of the kernel to calculate. Returned kernel will have size
        psfsize x psfsize. psfsize should be odd. Default: 7.

    psfk : float numpy array, optional
        PSF kernel array to use for the fine structure image if
        fsmode == 'convolve'. If None and fsmode == 'convolve', we calculate
        the psf kernel using 'psfmodel'. Default: None.

    psfbeta : float, optional
        Moffat beta parameter. Only used if fsmode=='convolve' and
        psfmodel=='moffat'. Default: 4.765.

    verbose : boolean, optional
        Print to the screen or not. Default: False.

    Returns
    -------
    crmask : boolean numpy array
        The cosmic ray mask (boolean) array with values of True where there are
        cosmic ray detections.

    cleanarr : float numpy array
        The cleaned data array.

    Notes
    -----
    To reproduce the most similar behavior to the original LA Cosmic
    (written in IRAF), set  inmask = None, satlevel = np.inf, sepmed=False,
    cleantype='medmask', and fsmode='median'.

    The original IRAF version distinguishes between spectroscopic and imaging
    data. This version does not. After sky subtracting the spectroscopic data,
    this version will work well. The 1-d 'gaussx' and 'gaussy' values for
    psfmodel can also be used for spectroscopic data (and may even alleviate
    the need to do sky subtraction, but this still requires more testing).
    """

    # Grab the sizes of the input array
    cdef int nx = indat.shape[1]
    cdef int ny = indat.shape[0]

    # Tell the compiler about the loop indices so it can optimize them.
    cdef int i, j = 0

    # Make a copy of the data as the cleanarr that we work on
    # This guarantees that that the data will be contiguous and makes sure we
    # don't edit the input data.
    cleanarr = np.empty((ny, nx), dtype=np.float32)
    # Set the initial values to those of the data array
    cleanarr[:, :] = indat[:, :]

    # Setup the mask
    if inmask is None:
        # By default don't mask anything
        mask = np.zeros((ny, nx), dtype=np.uint8, order='C')
    else:
        # Make a copy of the input mask
        mask = np.empty((ny, nx), dtype=np.uint8, order='C')
        mask[:, :] = inmask[:, :]

    # Add back in the previously subtracted sky level and multiply by the gain
    # The statistics only work properly with electrons.
    cleanarr += pssl
    cleanarr *= gain

    # Find the saturated stars and add them to the mask
    updatemask(np.asarray(cleanarr), np.asarray(mask), satlevel, sepmed)

    # Find the unmasked pixels to calculate the sky.
    gooddata = np.zeros(int(nx * ny - np.asarray(mask).sum()), dtype=np.float32,
                        order='c')

    igoodpix = 0

    gooddata[:] = cleanarr[np.logical_not(mask)]

    # Get the default background level for large cosmic rays.
    backgroundlevel = median(gooddata, len(gooddata))
    del gooddata

    # Set up the psf kernel if necessary.
    if psfk is None and fsmode == 'convolve':
        # calculate the psf kernel psfk
        if psfmodel == 'gauss':
            psfk = gausskernel(psffwhm, psfsize)
        elif psfmodel == 'gaussx':
            psfk = gaussxkernel(psffwhm, psfsize)
        elif psfmodel == 'gaussy':
            psfk = gaussykernel(psffwhm, psfsize)
        elif psfmodel == 'moffat':
            psfk = moffatkernel(psffwhm, psfbeta, psfsize)
        else:
            raise ValueError('Please choose a supported PSF model.')

    # Define a cosmic ray mask
    # This is what will be returned at the end
    crmask = np.zeros((ny, nx), dtype=np.uint8, order='C')

    # Calculate the detection limit for neighbor pixels
    cdef float sigcliplow = sigfrac * sigclip

    # Run lacosmic for up to maxiter iterations
    # We stop if no more cosmic ray pixels are found (quite rare)
    if verbose:
        print("Starting {} L.A.Cosmic iterations".format(niter))
    for i in range(niter):
        if verbose:
            print("Iteration {}:".format(i + 1))

        # Detect the cosmic rays

        # We subsample, convolve, clip negative values,
        # and rebin to original size
        subsam = subsample(cleanarr)

        conved = laplaceconvolve(subsam)
        del subsam

        conved[conved < 0] = 0.0
        # This is called L+ in the original LA Cosmic/cosmics.py
        s = rebin(conved)
        del conved

        # Build a the noise map, to compare the laplacian to
        if sepmed:
            m5 = sepmedfilt7(cleanarr)
        else:
            m5 = medfilt5(cleanarr)

        # Clip noise so that we can take a square root
        m5[m5 < 0.00001] = 0.00001
        noise = np.sqrt(m5 + readnoise * readnoise)

        if cleantype != 'median':
            del m5

        # Laplacian signal to noise ratio :
        s /= 2.0 * noise
        # the 2.0 is from the 2x2 subsampling
        # This s is called sigmap in the original lacosmic.cl

        if sepmed:
            sp = sepmedfilt7(s)
        else:
            sp = medfilt5(s)

        # Remove the large structures (s prime) :
        sp = s - sp
        del s

        # Build the fine structure image :
        if fsmode == 'convolve':
            f = convolve(cleanarr, psfk)
        elif fsmode == 'median':
            if sepmed:
                f = sepmedfilt5(cleanarr)
            else:
                f = medfilt3(cleanarr)
        else:
            raise ValueError('Please choose a valid fine structure mode.')

        if sepmed:
            m7 = sepmedfilt9(f)
        else:
            m7 = medfilt7(f)

        f = (f - m7) / noise
        # Clip f as we will divide by f. Similar to the IRAF version.
        f[f < 0.01] = 0.01

        del m7
        del noise

        # Find the candidate cosmic rays
        goodpix = np.logical_not(mask)
        cosmics = np.logical_and(sp > sigclip, goodpix)
        # Note the sp/f and not lplus/f due to the f = f/noise above.
        cosmics = np.logical_and(cosmics, (sp / f) > objlim)
        del f

        # What follows is a special treatment for neighbors, with more relaxed
        # constraints.
        # We grow these cosmics a first time to determine the immediate
        # neighborhood.
        cosmics = dilate3(cosmics)
        cosmics = np.logical_and(cosmics, goodpix)
        # From this grown set, we keep those that have sp > sigmalim
        cosmics = np.logical_and(sp > sigclip, cosmics)

        # Now we repeat this procedure, but lower the detection limit to siglow
        cosmics = dilate3(cosmics)
        cosmics = np.logical_and(cosmics, goodpix)

        del goodpix
        cosmics = np.logical_and(sp > sigcliplow, cosmics)
        del sp

        # Our CR counter
        numcr = cosmics.sum()

        # Update the crmask with the cosmics we have found
        crmask[:, :] = np.logical_or(crmask, cosmics)[:, :]
        del cosmics
        if verbose:
            print("{} cosmic pixels this iteration".format(numcr))

        # If we didn't find anything, we're done.
        if numcr == 0:
            break

        # otherwise clean the image and iterate
        if cleantype == 'median':
            # Unmasked median filter
            cleanarr[crmask] = m5[crmask]
            del m5
        # Masked mean filter
        elif cleantype == 'meanmask':
            clean_meanmask(cleanarr, crmask, mask, nx, ny, backgroundlevel)
        # Masked median filter
        elif cleantype == 'medmask':
            clean_medmask(cleanarr, crmask, mask, nx, ny, backgroundlevel)
        # Inverse distance weighted interpolation
        elif cleantype == 'idw':
            clean_idwinterp(cleanarr, crmask, mask, nx, ny, backgroundlevel)
        else:
            raise ValueError("""cleantype must be one of the following values:
                            [median, meanmask, medmask, idw]""")

    return (crmask.astype(np.bool), cleanarr)


def updatemask(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] data,
               np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] mask,
               float satlevel, bool sepmed):
    """updatemask(data, mask, satlevel, sepmed)\n
     Find staturated stars and puts them in the mask.

     This can then be used to avoid these regions in cosmic detection and
     cleaning procedures. The median filter is used to find large symmetric
     regions of saturated pixels (i.e. saturated stars).

    Parameters
    ----------
    data : float numpy array
        The data array in which we look for saturated stars.

    mask : boolean numpy array
        Bad pixel mask. This mask will be dilated using dilate3 and then
        combined with the saturated star mask.

    satlevel : float
        Saturation level of the image. This value can be lowered if the cores
        of bright (saturated) stars are not being masked.

    sepmed : boolean
        Use the separable median or not. The separable median is not identical
        to the full median filter, but they are approximately the same and the
        separable median filter is significantly faster.
    """

    # Find all of the saturated pixels
    satpixels = data >= satlevel

    # Use the median filter to estimate the large scale structure
    if sepmed:
        m5 = sepmedfilt7(data)
    else:
        m5 = medfilt5(data)

    # Use the median filtered image to find the cores of saturated stars
    # The 10 here is arbitray. Malte Tewes uses 2.0 in cosmics.py, but I
    # wanted to get more of the cores of saturated stars.
    satpixels = np.logical_and(satpixels, m5 > (satlevel / 10.0))

    # Grow the input mask by one pixel to make sure we cover bad pixels
    grow_mask = dilate3(mask)

    # Dilate the saturated star mask to remove edge effects in the mask
    dilsatpixels = dilate5(satpixels, 2)
    del satpixels
    # Combine the saturated pixels with the given input mask
    # Note, we work on the mask pixels in place
    mask[:, :] = np.logical_or(dilsatpixels, grow_mask)[:, :]
    del grow_mask


cdef void clean_meanmask(float[:, ::1] cleanarr, bool[:, ::1] crmask,
                         bool[:, ::1] mask, int nx, int ny,
                         float backgroundlevel):
    """ clean_meanmask(cleanarr, crmask, mask, nx, ny, backgroundlevel)\n
    Clean the bad pixels in cleanarr using a 5x5 masked mean filter.

    Parameters
    ----------
    cleanarr : float numpy array
        The array to be cleaned.

    crmask : boolean numpy array
        Cosmic ray mask. Pixels with a value of True in this mask will be
        cleaned.

    mask : boolean numpy array
        Bad pixel mask. Values of True indicate bad pixels.

    nx : int
        Size of cleanarr in the x-direction. Note cleanarr has dimensions
        ny x nx.

    ny : int
        Size of cleanarr in the y-direction. Note cleanarr has dimensions
        ny x nx.

    backgroundlevel : float
        Average value of the background. This value will be used if there are
        no good pixels in a 5x5 region.
    """
    # Go through all of the pixels, ignore the borders
    cdef int i, j, k, l, numpix
    cdef float s
    cdef bool badpix

    with nogil, parallel():
        # For each pixel
        for j in prange(2, ny - 2):
            for i in range(2, nx - 2):
                # if the pixel is in the crmask
                if crmask[j, i]:
                    numpix = 0
                    s = 0.0

                    # sum the 25 pixels around the pixel
                    # ignoring any pixels that are masked
                    for l in range(-2, 3):
                        for k in range(-2, 3):
                            badpix = crmask[j + l, i + k]
                            badpix = badpix or mask[j + l, i + k]
                            if not badpix:
                                s = s + cleanarr[j + l, i + k]
                                numpix = numpix + 1

                    # if the pixels count is 0
                    # then put in the background of the image
                    if numpix == 0:
                        s = backgroundlevel
                    else:
                        # else take the mean
                        s = s / float(numpix)

                    cleanarr[j, i] = s


cdef void clean_medmask(float[:, ::1] cleanarr, bool[:, ::1] crmask,
                        bool[:, ::1] mask, int nx, int ny,
                        float backgroundlevel):
    """ clean_medmask(cleanarr, crmask, mask, nx, ny, backgroundlevel)\n
    Clean the bad pixels in cleanarr using a 5x5 masked median filter.

    Parameters
    ----------
    cleanarr : float numpy array
        The array to be cleaned.

    crmask : boolean numpy array
        Cosmic ray mask. Pixels with a value of True in this mask will be
        cleaned.

    mask : boolean numpy array
        Bad pixel mask. Values of True indicate bad pixels.

    nx : int
        size of cleanarr in the x-direction. Note cleanarr has dimensions
        ny x nx.

    ny : int
        size of cleanarr in the y-direction. Note cleanarr has dimensions
        ny x nx.

    backgroundlevel : float
        Average value of the background. This value will be used if there are
        no good pixels in a 5x5 region.
    """
    # Go through all of the pixels, ignore the borders
    cdef int k, l, i, j, numpix
    cdef float * medarr
    cdef bool badpixel

    # For each pixel
    with nogil, parallel():
        medarr = < float * > malloc(25 * sizeof(float))
        for j in prange(2, ny - 2):
            for i in range(2, nx - 2):
                # if the pixel is in the crmask
                if crmask[j, i]:
                    numpix = 0
                    # median the 25 pixels around the pixel ignoring
                    # any pixels that are masked
                    for l in range(-2, 3):
                        for k in range(-2, 3):
                            badpixel = crmask[j + l, i + k]
                            badpixel = badpixel or mask[j + l, i + k]
                            if not badpixel:
                                medarr[numpix] = cleanarr[j + l, i + k]
                                numpix = numpix + 1

                    # if the pixels count is 0 then put in the background
                    # of the image
                    if numpix == 0:
                        cleanarr[j, i] = backgroundlevel
                    else:
                        # else take the mean
                        cleanarr[j, i] = PyMedian(medarr, numpix)
        free(medarr)


cdef void clean_idwinterp(float[:, ::1] cleanarr, bool[:, ::1] crmask,
                          bool[:, ::1] mask, int nx, int ny,
                          float backgroundlevel):
    """clean_idwinterp(cleanarr, crmask, mask, nx, ny, backgroundlevel)\n
    Clean the bad pixels in cleanarr using a 5x5 using inverse distance
    weighted interpolation.

    Parameters
    ----------
    cleanarr : float numpy array
        The array to be cleaned.

    crmask : boolean numpy array
        Cosmic ray mask. Pixels with a value of True in this mask will be
        cleaned.

    mask : boolean numpy array
        Bad pixel mask. Values of True indicate bad pixels.

    nx : int
        Size of cleanarr in the x-direction (int). Note cleanarr has dimensions
        ny x nx.

    ny : int
        Size of cleanarr in the y-direction (int). Note cleanarr has dimensions
        ny x nx.

    backgroundlevel : float
        Average value of the background. This value will be used if there are
        no good pixels in a 5x5 region.
    """

    # Go through all of the pixels, ignore the borders
    cdef int i, j, k, l
    cdef float f11, f12, f21, f22 = backgroundlevel
    cdef int x1, x2, y1, y2
    weightsarr = np.array([[0.35355339, 0.4472136, 0.5, 0.4472136, 0.35355339],
                          [0.4472136, 0.70710678, 1., 0.70710678, 0.4472136],
                          [0.5, 1., 0., 1., 0.5],
                          [0.4472136, 0.70710678, 1., 0.70710678, 0.4472136],
                          [0.35355339, 0.4472136, 0.5, 0.4472136, 0.35355339]],
                          dtype=np.float32)
    cdef float[:, ::1] weights = weightsarr
    cdef float wsum
    cdef float val
    cdef int x, y
    # For each pixel
    with nogil, parallel():

        for j in prange(2, ny - 2):
            for i in range(2, nx - 2):
                # if the pixel is in the crmask
                if crmask[j, i]:
                    wsum = 0.0
                    val = 0.0
                    for l in range(-2, 3):
                        y = j + l
                        for k in range(-2, 3):
                            x = i + k
                            if not (crmask[y, x] or mask[y, x]):
                                val = val + weights[l, k] * cleanarr[y, x]
                                wsum = wsum + weights[l, k]
                    if wsum < 1e-6:
                        cleanarr[j, i] = backgroundlevel
                    else:
                        cleanarr[j, i] = val / wsum


def gausskernel(float psffwhm, int kernsize):
    """gausskernel(psffwhm, kernsize)\n
    Calculate a circular Gaussian psf kernel.

    Parameters
    ----------
    psffwhm : float
        Full Width Half Maximum of the PSF to use to generate the kernel.

    kernsize : int
        Size of the kernel to calculate. kernsize should be odd.
        Returned kernel will have size kernsize x kernsize.

    Returns
    -------
    kernel : float numpy array
        Gaussian PSF kernel with size kernsize x kernsize.
    """
    kernel = np.zeros((kernsize, kernsize), dtype=np.float32)
    # Make a grid of x and y values
    x = np.tile(np.arange(kernsize) - kernsize / 2, (kernsize, 1))
    y = x.transpose().copy()
    # Calculate the offset, r
    r2 = x * x + y * y
    # Calculate the kernel
    sigma2 = psffwhm * psffwhm / 2.35482 / 2.35482
    kernel[:, :] = np.exp(-0.5 * r2 / sigma2)[:, :]
    # Normalize the kernel
    kernel /= kernel.sum()
    return kernel


def gaussxkernel(float psffwhm, int kernsize):
    """gaussxkernel(psffwhm, kernsize)\n
    Calculate a Guassian kernel in the x-direction.

    This can be used for spectroscopic data.

    Parameters
    ----------
    psffwhm : float
        Full Width Half Maximum of the PSF to use to generate the kernel.

    kernsize : int
        Size of the kernel to calculate. kernsize should be odd.
        Returned kernel will have size kernsize x kernsize.

    Returns
    -------
    kernel : float numpy array
        Gaussian(x) kernel with size kernsize x kernsize.
    """
    kernel = np.zeros((kernsize, kernsize), dtype=np.float32)
    # Make a grid of x and y values
    x = np.tile(np.arange(kernsize) - kernsize / 2, (kernsize, 1))
    # Calculate the kernel
    sigma2 = psffwhm * psffwhm / 2.35482 / 2.35482
    kernel[:, :] = np.exp(-0.5 * x * x / sigma2)[:, :]
    # Normalize the kernel
    kernel /= kernel.sum()
    return kernel


def gaussykernel(float psffwhm, int kernsize):
    """gaussykernel(psffwhm, kernsize)\n
    Calculate a Guassian kernel in the y-direction.

    This can be used for spectroscopic data.

    Parameters
    ----------
    psffwhm : float
        Full Width Half Maximum of the PSF to use to generate the kernel.

    kernsize : int
        Size of the kernel to calculate. kernsize should be odd.
        Returned kernel will have size kernsize x kernsize.

    Returns
    -------
    kernel : float numpy array
        Gaussian(y) kernel with size kernsize x kernsize.
    """
    kernel = np.zeros((kernsize, kernsize), dtype=np.float32)
    # Make a grid of x and y values
    x = np.tile(np.arange(kernsize) - kernsize / 2, (kernsize, 1))
    y = x.transpose().copy()
    # Calculate the kernel
    sigma2 = psffwhm * psffwhm / 2.35482 / 2.35482
    kernel[:, :] = np.exp(-0.5 * y * y / sigma2)[:, :]
    # Normalize the kernel
    kernel /= kernel.sum()
    return kernel


cdef moffatkernel(float psffwhm, float beta, int kernsize):
    """moffatkernel(psffwhm, beta, kernsize)\n
    Calculate a Moffat psf kernel.

    Parameters
    ----------
    psffwhm : float
        Full Width Half Maximum of the PSF to use to generate the kernel.

    beta : float
        Moffat beta parameter

    kernsize : int
        Size of the kernel to calculate. Returned kernel will have size
        kernsize x kernsize. kernsize should be odd.

    Returns
    -------
    kernel : float numpy array
        Moffat kernel with size kernsize x kernsize.
    """
    kernel = np.zeros((kernsize, kernsize), dtype=np.float32)
    # Make a grid of x and y values
    x = np.tile(np.arange(kernsize) - kernsize / 2, (kernsize, 1))
    y = x.transpose().copy()
    # Calculate the offset r
    r = np.sqrt(x * x + y * y)
    # Calculate the kernel
    hwhm = psffwhm / 2.0
    alpha = hwhm / np.sqrt(np.power(2.0, (1.0 / beta)) - 1.0)
    kernel[:, :] = (np.power(1.0 + (r * r / alpha / alpha), -1.0 * beta))[:, :]
    # Normalize the kernel.
    kernel /= kernel.sum()
    return kernel


"""
Below are wrappers for the C functions in laxutils.c
"""


def median(np.ndarray[np.float32_t, mode='c', cast=True] a, int n):
    """median(a, n)\n
    Find the median of the first n elements of an array.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median.

    n : int
        Number of elements of the array to median.

    Returns
    -------
    med : float
        The median value.

    Notes
    -----
    Wrapper for PyMedian in laxutils.
    """
    cdef float * aptr = < float * > np.PyArray_DATA(a)
    cdef float med = 0.0
    with nogil:
        med = PyMedian(aptr, n)
    return med


def optmed3(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed3(a)\n
    Optimized method to find the median value of an array of length 3.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median. Must be length 3.

    Returns
    -------
    med3 : float
        The median of the 3-element array.

    Notes
    -----
    Wrapper for PyOptMed3 in laxutils.
    """
    cdef float * aptr3 = < float * > np.PyArray_DATA(a)
    cdef float med3 = 0.0
    with nogil:
        med3 = PyOptMed3(aptr3)
    return med3


def optmed5(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed5(a)\n
    Optimized method to find the median value of an array of length 5.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median. Must be length 5.

    Returns
    -------
    med5 : float
        The median of the 5-element array.

    Notes
    -----
    Wrapper for PyOptMed5 in laxutils.
    """
    cdef float * aptr5 = < float * > np.PyArray_DATA(a)
    cdef float med5 = 0.0
    with nogil:
        med5 = PyOptMed5(aptr5)
    return med5


def optmed7(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed7(a)\n
    Optimized method to find the median value of an array of length 7.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median. Must be length 7.

    Returns
    -------
    med7 : float
        The median of the 7-element array.

    Notes
    -----
    Wrapper for PyOptMed7 in laxutils.
    """
    cdef float * aptr7 = < float * > np.PyArray_DATA(a)
    cdef float med7 = 0.0
    with nogil:
        med7 = PyOptMed7(aptr7)
    return med7


def optmed9(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed9(a)\n
    Optimized method to find the median value of an array of length 9.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median. Must be length 9.

    Returns
    -------
    med9 : float
        The median of the 9-element array.

    Notes
    -----
    Wrapper for PyOptMed9 in laxutils.
    """
    cdef float * aptr9 = < float * > np.PyArray_DATA(a)
    cdef float med9 = 0.0
    with nogil:
        med9 = PyOptMed9(aptr9)
    return med9


def optmed25(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed25(a)\n
    Optimized method to find the median value of an array of length 25.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median. Must be length 25.

    Returns
    -------
    med25 : float
        The median of the 25-element array.

    Notes
    -----
    Wrapper for PyOptMed25 in laxutils.
    """
    cdef float * aptr25 = < float * > np.PyArray_DATA(a)
    cdef float med25 = 0.0
    with nogil:
        med25 = PyOptMed25(aptr25)
    return med25


def medfilt3(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d3):
    """medfilt3(d3)\n
    Calculate the 3x3 median filter of an array.

    Parameters
    ----------
    d3 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The median filter is not calculated for a 1 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt3 in laxutils.
    """
    cdef int nx = d3.shape[1]
    cdef int ny = d3.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)
    cdef float * d3ptr = < float * > np.PyArray_DATA(d3)
    cdef float * outd3ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyMedFilt3(d3ptr, outd3ptr, nx, ny)

    return output


def medfilt5(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d5):
    """medfilt5(d5)\n
    Calculate the 5x5 median filter of an array.

    Parameters
    ----------
    d5 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The median filter is not calculated for a 2 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt5 in laxutils.
    """
    cdef int nx = d5.shape[1]
    cdef int ny = d5.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)
    cdef float * d5ptr = < float * > np.PyArray_DATA(d5)
    cdef float * outd5ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyMedFilt5(d5ptr, outd5ptr, nx, ny)
    return output


def medfilt7(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d7):
    """medfilt7(d7)\n
    Calculate the 7x7 median filter of an array.

    Parameters
    ----------
    d7 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The median filter is not calculated for a 3 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt7 in laxutils.
    """
    cdef int nx = d7.shape[1]
    cdef int ny = d7.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * d7ptr = < float * > np.PyArray_DATA(d7)
    cdef float * outd7ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyMedFilt7(d7ptr, outd7ptr, nx, ny)
    return output


def sepmedfilt3(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep3):
    """sepmedfilt3(dsep3)\n
    Calculate the 3x3 separable median filter of an array.

    Parameters
    ----------
    dsep3 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The separable median medians the rows followed by the columns instead of
    using a square window. Therefore it is not identical to the full median
    filter but it is approximatly the same, but it is signifcantly faster.
    The median filter is not calculated for a 1 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt3 in laxutils.
    """
    cdef int nx = dsep3.shape[1]
    cdef int ny = dsep3.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep3ptr = < float * > np.PyArray_DATA(dsep3)
    cdef float * outdsep3ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt3(dsep3ptr, outdsep3ptr, nx, ny)
    return np.asarray(output)


def sepmedfilt5(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep5):
    """sepmedfilt5(dsep5)\n
    Calculate the 5x5 separable median filter of an array.

    Parameters
    ----------
    dsep5 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The separable median medians the rows followed by the columns instead of
    using a square window. Therefore it is not identical to the full median
    filter but it is approximatly the same, but it is signifcantly faster.
    The median filter is not calculated for a 2 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt5 in laxutils.
    """
    cdef int nx = dsep5.shape[1]
    cdef int ny = dsep5.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep5ptr = < float * > np.PyArray_DATA(dsep5)
    cdef float * outdsep5ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt5(dsep5ptr, outdsep5ptr, nx, ny)

    return output


def sepmedfilt7(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep7):
    """sepmedfilt7(dsep7)\n
    Calculate the 7x7 separable median filter of an array.

    Parameters
    ----------
    dsep7 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The separable median medians the rows followed by the columns instead of
    using a square window. Therefore it is not identical to the full median
    filter but it is approximatly the same, but it is signifcantly faster.
    The median filter is not calculated for a 3 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt7 in laxutils.
    """
    cdef int nx = dsep7.shape[1]
    cdef int ny = dsep7.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep7ptr = < float * > np.PyArray_DATA(dsep7)
    cdef float * outdsep7ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt7(dsep7ptr, outdsep7ptr, nx, ny)
    return output


def sepmedfilt9(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep9):
    """sepmedfilt9(dsep9)\n
    Calculate the 9x9 separable median filter of an array.

    Parameters
    ----------
    dsep9 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The separable median medians the rows followed by the columns instead of
    using a square window. Therefore it is not identical to the full median
    filter but it is approximatly the same, but it is signifcantly faster.
    The median filter is not calculated for a 4 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt9 in laxutils.
    """

    cdef int nx = dsep9.shape[1]
    cdef int ny = dsep9.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep9ptr = < float * > np.PyArray_DATA(dsep9)
    cdef float * outdsep9ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt9(dsep9ptr, outdsep9ptr, nx, ny)
    return output


def subsample(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsub):
    """subsample(dsub)\n
    Subsample an array 2x2 given an input array dsub.

    Parameters
    ----------
    dsub : float numpy array
        Array to be subsampled.

    Returns
    -------
    output : float numpy array
        Subsampled array. Output dimensions will be 2 times the input
        dimensions.

    Notes
    -----
    Each pixel is replicated into 4 pixels; no averaging is performed.
    The array needs to be C-contiguous order. Wrapper for PySubsample in
    laxutils.
    """
    cdef int nx = dsub.shape[1]
    cdef int ny = dsub.shape[0]
    cdef int nx2 = 2 * nx
    cdef int ny2 = 2 * ny

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny2, nx2), dtype=np.float32)

    cdef float * dsubptr = < float * > np.PyArray_DATA(dsub)
    cdef float * outdsubptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySubsample(dsubptr, outdsubptr, nx, ny)
    return output


def rebin(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] drebin):
    """rebin(drebin)\n
    Rebin an array 2x2.

    Rebin the array by block averaging 4 pixels back into 1.

    Parameters
    ----------
    drebin : float numpy array
        Array to be rebinned 2x2.

    Returns
    -------
    output : float numpy array
        Rebinned array. The size of the output array will be 2 times smaller
        than drebin.

    Notes
    -----
    This is effectively the opposite of subsample (although subsample does not
    do an average). The array needs to be C-contiguous order. Wrapper for
    PyRebin in laxutils.
    """
    cdef int nx = drebin.shape[1] / 2
    cdef int ny = drebin.shape[0] / 2

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * drebinptr = < float * > np.PyArray_DATA(drebin)
    cdef float * outdrebinptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyRebin(drebinptr, outdrebinptr, nx, ny)
    return output


def convolve(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dconv,
             np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] kernel):
    """convolve(dconv, kernel)\n
    Convolve an array with a kernel.

    Parameters
    ----------
    dconv : float numpy array
        Array to be convolved.

    kernel : float numpy array
        Kernel to use in the convolution.

    Returns
    -------
    output : float numpy array
        Convolved array.

    Notes
    -----
    Both the data and kernel arrays need to be C-contiguous order. Wrapper for
    PyConvolve in laxutils.
    """
    cdef int nx = dconv.shape[1]
    cdef int ny = dconv.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dconvptr = < float * > np.PyArray_DATA(dconv)
    cdef float * outdconvptr = < float * > np.PyArray_DATA(output)

    cdef int knx = kernel.shape[1]
    cdef int kny = kernel.shape[0]
    cdef float * kernptr = < float * > np.PyArray_DATA(kernel)

    with nogil:
        PyConvolve(dconvptr, kernptr, outdconvptr, nx, ny, knx, kny)
    return output


def laplaceconvolve(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dl):
    """laplaceconvolve(dl)\n
    Convolve an array with the Laplacian kernel.

    Convolve with the discrete version of the Laplacian operator with kernel:\n
     0 -1  0\n
    -1  4 -1\n
     0 -1  0\n

    Parameters
    ----------
    dl : float numpy array
        Array to be convolved.

    Returns
    -------
    output: float numpy array
        Convolved array.

    Notes
    -----
    The array needs to be C-contiguous order. Wrapper for PyLaplaceConvolve
    in laxutils.
    """
    cdef int nx = dl.shape[1]
    cdef int ny = dl.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dlapptr = < float * > np.PyArray_DATA(dl)
    cdef float * outdlapptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyLaplaceConvolve(dlapptr, outdlapptr, nx, ny)
    return output


def dilate3(np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] dgrow):
    """dilate3(dgrow)\n
    Perform a boolean dilation on an array.

    Parameters
    ----------
    dgrow : boolean numpy array
        Array to dilate.

    Returns
    -------
    output : boolean numpy array
        Dilated array.

    Notes
    -----
    Dilation is the boolean equivalent of a convolution but using logical ors
    instead of a sum.
    We apply the following kernel:\n
    1 1 1\n
    1 1 1\n
    1 1 1\n
    The binary dilation is not computed for a 1 pixel border around the image.
    These pixels are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyDilate3 in laxutils.
    """
    cdef int nx = dgrow.shape[1]
    cdef int ny = dgrow.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.bool)

    cdef uint8_t * dgrowptr = < uint8_t * > np.PyArray_DATA(dgrow)
    cdef uint8_t * outdgrowptr = < uint8_t * > np.PyArray_DATA(output)
    with nogil:
        PyDilate3(dgrowptr, outdgrowptr, nx, ny)
    return output


def dilate5(np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] ddilate,
            int niter):
    """dilate5(data, niter)\n
    Do niter iterations of boolean dilation on an array.

    Parameters
    ----------
    ddilate : boolean numpy array
        Array to dilate.

    niter : int
        Number of iterations.

    Returns
    -------
    output : boolean numpy array
        Dilated array.

    Notes
    -----
    Dilation is the boolean equivalent of a convolution but using logical ors
    instead of a sum.
    We apply the following kernel:\n
    0 1 1 1 0\n
    1 1 1 1 1\n
    1 1 1 1 1\n
    1 1 1 1 1\n
    0 1 1 1 0\n
    The edges are padded with zeros so that the dilation operator is defined
    for all pixels. The array needs to be C-contiguous order. Wrapper for
    PyDilate5 in laxutils.
    """
    cdef int nx = ddilate.shape[1]
    cdef int ny = ddilate.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.bool)

    cdef uint8_t * ddilateptr = < uint8_t * > np.PyArray_DATA(ddilate)
    cdef uint8_t * outddilateptr = < uint8_t * > np.PyArray_DATA(output)
    with nogil:
        PyDilate5(ddilateptr, outddilateptr, niter, nx, ny)
    return output
