/*
 * Author: Curtis McCully
 * October 2014
 * Licensed under a 3-clause BSD style license - see LICENSE.rst
 *
 * Originally written in C++ in 2011
 * See also https://github.com/cmccully/lacosmicx
 *
 * This file contains utility functions for Lacosmicx. These are the most
 * computationally expensive pieces of the calculation so they have been ported
 * to C.
 *
 * Many thanks to Nicolas Devillard who wrote the optimized methods for finding
 * the median and placed them in the public domain. I have noted in the
 * comments places that use Nicolas Devillard's code.
 *
 * Parallelization has been achieved using OpenMP. Using a compiler that does
 * not support OpenMP, e.g. clang currently, the code should still compile and
 * run serially without issue. I have tried to be explicit as possible about
 * specifying which variables are private and which should be shared, although
 * we never actually have any shared variables. We use firstprivate instead.
 * This does mean that it is important that we never have two threads write to
 * the same memory position at the same time.
 *
 * All calculations are done with 32 bit floats to keep the memory footprint
 * small.
 */
#include<stdlib.h>
#include<math.h>
#include<Python.h>
#include "laxutils.h"
#define ELEM_SWAP(a,b) { float t=(a);(a)=(b);(b)=t; }

float
PyMedian(float* a, int n)
{
    /* Get the median of an array "a" with length "n"
     * using the Quickselect algorithm. Returns a float.
     * This Quickselect routine is based on the algorithm described in
     * "Numerical recipes in C", Second Edition, Cambridge University Press,
     * 1992, Section 8.5, ISBN 0-521-43108-5
     * This code by Nicolas Devillard - 1998. Public domain.
     */

    PyDoc_STRVAR(PyMedian__doc__, "PyMedian(a, n) -> float\n\n"
        "Get the median of array a of length n using the Quickselect "
        "algorithm.");

    /* Make a copy of the array so that we don't alter the input array */
    float* arr = (float *) malloc(n * sizeof(float));
    /* Indices of median, low, and high values we are considering */
    int low = 0;
    int high = n - 1;
    int median = (low + high) / 2;
    /* Running indices for the quick select algorithm */
    int middle, ll, hh;
    /* The median to return */
    float med;

    /* running index i */
    int i;
    /* Copy the input data into the array we work with */
    for (i = 0; i < n; i++) {
        arr[i] = a[i];
    }

    /* Start an infinite loop */
    while (true) {

        /* Only One or two elements left */
        if (high <= low + 1) {
            /* Check if we need to swap the two elements */
            if ((high == low + 1) && (arr[low] > arr[high]))
                ELEM_SWAP(arr[low], arr[high]);
            med = arr[median];
            free(arr);
            return med;
        }

        /* Find median of low, middle and high items;
         * swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])
            ELEM_SWAP(arr[middle], arr[high]);
        if (arr[low] > arr[high])
            ELEM_SWAP(arr[low], arr[high]);
        if (arr[middle] > arr[low])
            ELEM_SWAP(arr[middle], arr[low]);

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low + 1]);

        /* Nibble from each end towards middle,
         * swap items when stuck */
        ll = low + 1;
        hh = high;
        while (true) {
            do
                ll++;
            while (arr[low] > arr[ll]);
            do
                hh--;
            while (arr[hh] > arr[low]);

            if (hh < ll)
                break;

            ELEM_SWAP(arr[ll], arr[hh]);
        }

        /* Swap middle item (in position low) back into
         * the correct position */
        ELEM_SWAP(arr[low], arr[hh]);

        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }

}

#undef ELEM_SWAP

/* All of the optimized median methods below were written by
 * Nicolas Devillard and are in the public domain.
 */

#define PIX_SORT(a,b) { if (a>b) PIX_SWAP(a,b); }
#define PIX_SWAP(a,b) { float temp=a; a=b; b=temp; }

/* ----------------------------------------------------------------------------
 Function :   PyOptMed3()
 In       :   pointer to array of 3 pixel values
 Out      :   a pixel value
 Job      :   optimized search of the median of 3 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made on the nature of the input
 signal.
 Code adapted from Nicolas Devillard.
 --------------------------------------------------------------------------- */
float
PyOptMed3(float* p)
{
    PyDoc_STRVAR(PyOptMed3__doc__, "PyOptMed3(a) -> float\n\n"
        "Get the median of array a of length 3 using a search tree.");

    PIX_SORT(p[0], p[1]);
    PIX_SORT(p[1], p[2]);
    PIX_SORT(p[0], p[1]);
    return p[1];
}

/* ----------------------------------------------------------------------------
 Function :   PyOptMed5()
 In       :   pointer to array of 5 pixel values
 Out      :   a pixel value
 Job      :   optimized search of the median of 5 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made on the nature of the input
 signal.
 Code adapted from Nicolas Devillard.
 --------------------------------------------------------------------------- */
float
PyOptMed5(float* p)
{
    PyDoc_STRVAR(PyOptMed5__doc__, "PyOptMed5(a) -> float\n\n"
        "Get the median of array a of length 5 using a search tree.");

    PIX_SORT(p[0], p[1]);
    PIX_SORT(p[3], p[4]);
    PIX_SORT(p[0], p[3]);
    PIX_SORT(p[1], p[4]);
    PIX_SORT(p[1], p[2]);
    PIX_SORT(p[2], p[3]);
    PIX_SORT(p[1], p[2]);
    return p[2];
}

/* ----------------------------------------------------------------------------
 Function :   PyOptMed7()
 In       :   pointer to array of 7 pixel values
 Out      :   a pixel value
 Job      :   optimized search of the median of 7 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made on the nature of the input
 signal.
 Code adapted from Nicolas Devillard.
 --------------------------------------------------------------------------- */
float
PyOptMed7(float* p)
{
    PyDoc_STRVAR(PyOptMed7__doc__, "PyOptMed7(a) -> float\n\n"
        "Get the median of array a of length 7 using a search tree.");

    PIX_SORT(p[0], p[5]);
    PIX_SORT(p[0], p[3]);
    PIX_SORT(p[1], p[6]);
    PIX_SORT(p[2], p[4]);
    PIX_SORT(p[0], p[1]);
    PIX_SORT(p[3], p[5]);
    PIX_SORT(p[2], p[6]);
    PIX_SORT(p[2], p[3]);
    PIX_SORT(p[3], p[6]);
    PIX_SORT(p[4], p[5]);
    PIX_SORT(p[1], p[4]);
    PIX_SORT(p[1], p[3]);
    PIX_SORT(p[3], p[4]);
    return p[3];
}

/* ----------------------------------------------------------------------------
 Function :   PyOptMed9()
 In       :   pointer to an array of 9 pixel values
 Out      :   a pixel value
 Job      :   optimized search of the median of 9 pixel values
 Notice   :   in theory, cannot go faster without assumptions on the
 signal.
 Formula from:
 XILINX XCELL magazine, vol. 23 by John L. Smith

 The input array is modified in the process
 The result array is guaranteed to contain the median
 value in middle position, but other elements are NOT sorted.
 Code adapted from Nicolas Devillard.
 --------------------------------------------------------------------------- */
float
PyOptMed9(float* p)
{
    PyDoc_STRVAR(PyOptMed9__doc__, "PyOptMed9(a) -> float\n\n"
        "Get the median of array a of length 9 using a search tree.");

    PIX_SORT(p[1], p[2]);
    PIX_SORT(p[4], p[5]);
    PIX_SORT(p[7], p[8]);
    PIX_SORT(p[0], p[1]);
    PIX_SORT(p[3], p[4]);
    PIX_SORT(p[6], p[7]);
    PIX_SORT(p[1], p[2]);
    PIX_SORT(p[4], p[5]);
    PIX_SORT(p[7], p[8]);
    PIX_SORT(p[0], p[3]);
    PIX_SORT(p[5], p[8]);
    PIX_SORT(p[4], p[7]);
    PIX_SORT(p[3], p[6]);
    PIX_SORT(p[1], p[4]);
    PIX_SORT(p[2], p[5]);
    PIX_SORT(p[4], p[7]);
    PIX_SORT(p[4], p[2]);
    PIX_SORT(p[6], p[4]);
    PIX_SORT(p[4], p[2]);
    return p[4];
}

/* ----------------------------------------------------------------------------
 Function :   PyOptMed25()
 In       :   pointer to an array of 25 pixel values
 Out      :   a pixel value
 Job      :   optimized search of the median of 25 pixel values
 Notice   :   in theory, cannot go faster without assumptions on the
 signal.
 Code taken from Graphic Gems.
 Code adapted from Nicolas Devillard.
 --------------------------------------------------------------------------- */
float
PyOptMed25(float* p)
{
    PyDoc_STRVAR(PyOptMed25__doc__, "PyOptMed25(a) -> float\n\n"
        "Get the median of array a of length 25 using a search tree.");

    PIX_SORT(p[0], p[1]);
    PIX_SORT(p[3], p[4]);
    PIX_SORT(p[2], p[4]);
    PIX_SORT(p[2], p[3]);
    PIX_SORT(p[6], p[7]);
    PIX_SORT(p[5], p[7]);
    PIX_SORT(p[5], p[6]);
    PIX_SORT(p[9], p[10]);
    PIX_SORT(p[8], p[10]);
    PIX_SORT(p[8], p[9]);
    PIX_SORT(p[12], p[13]);
    PIX_SORT(p[11], p[13]);
    PIX_SORT(p[11], p[12]);
    PIX_SORT(p[15], p[16]);
    PIX_SORT(p[14], p[16]);
    PIX_SORT(p[14], p[15]);
    PIX_SORT(p[18], p[19]);
    PIX_SORT(p[17], p[19]);
    PIX_SORT(p[17], p[18]);
    PIX_SORT(p[21], p[22]);
    PIX_SORT(p[20], p[22]);
    PIX_SORT(p[20], p[21]);
    PIX_SORT(p[23], p[24]);
    PIX_SORT(p[2], p[5]);
    PIX_SORT(p[3], p[6]);
    PIX_SORT(p[0], p[6]);
    PIX_SORT(p[0], p[3]);
    PIX_SORT(p[4], p[7]);
    PIX_SORT(p[1], p[7]);
    PIX_SORT(p[1], p[4]);
    PIX_SORT(p[11], p[14]);
    PIX_SORT(p[8], p[14]);
    PIX_SORT(p[8], p[11]);
    PIX_SORT(p[12], p[15]);
    PIX_SORT(p[9], p[15]);
    PIX_SORT(p[9], p[12]);
    PIX_SORT(p[13], p[16]);
    PIX_SORT(p[10], p[16]);
    PIX_SORT(p[10], p[13]);
    PIX_SORT(p[20], p[23]);
    PIX_SORT(p[17], p[23]);
    PIX_SORT(p[17], p[20]);
    PIX_SORT(p[21], p[24]);
    PIX_SORT(p[18], p[24]);
    PIX_SORT(p[18], p[21]);
    PIX_SORT(p[19], p[22]);
    PIX_SORT(p[8], p[17]);
    PIX_SORT(p[9], p[18]);
    PIX_SORT(p[0], p[18]);
    PIX_SORT(p[0], p[9]);
    PIX_SORT(p[10], p[19]);
    PIX_SORT(p[1], p[19]);
    PIX_SORT(p[1], p[10]);
    PIX_SORT(p[11], p[20]);
    PIX_SORT(p[2], p[20]);
    PIX_SORT(p[2], p[11]);
    PIX_SORT(p[12], p[21]);
    PIX_SORT(p[3], p[21]);
    PIX_SORT(p[3], p[12]);
    PIX_SORT(p[13], p[22]);
    PIX_SORT(p[4], p[22]);
    PIX_SORT(p[4], p[13]);
    PIX_SORT(p[14], p[23]);
    PIX_SORT(p[5], p[23]);
    PIX_SORT(p[5], p[14]);
    PIX_SORT(p[15], p[24]);
    PIX_SORT(p[6], p[24]);
    PIX_SORT(p[6], p[15]);
    PIX_SORT(p[7], p[16]);
    PIX_SORT(p[7], p[19]);
    PIX_SORT(p[13], p[21]);
    PIX_SORT(p[15], p[23]);
    PIX_SORT(p[7], p[13]);
    PIX_SORT(p[7], p[15]);
    PIX_SORT(p[1], p[9]);
    PIX_SORT(p[3], p[11]);
    PIX_SORT(p[5], p[17]);
    PIX_SORT(p[11], p[17]);
    PIX_SORT(p[9], p[17]);
    PIX_SORT(p[4], p[10]);
    PIX_SORT(p[6], p[12]);
    PIX_SORT(p[7], p[14]);
    PIX_SORT(p[4], p[6]);
    PIX_SORT(p[4], p[7]);
    PIX_SORT(p[12], p[14]);
    PIX_SORT(p[10], p[14]);
    PIX_SORT(p[6], p[7]);
    PIX_SORT(p[10], p[12]);
    PIX_SORT(p[6], p[10]);
    PIX_SORT(p[6], p[17]);
    PIX_SORT(p[12], p[17]);
    PIX_SORT(p[7], p[17]);
    PIX_SORT(p[7], p[10]);
    PIX_SORT(p[12], p[18]);
    PIX_SORT(p[7], p[12]);
    PIX_SORT(p[10], p[18]);
    PIX_SORT(p[12], p[20]);
    PIX_SORT(p[10], p[20]);
    PIX_SORT(p[10], p[12]);

    return p[12];
}

#undef PIX_SORT
#undef PIX_SWAP

/* We have slightly unusual boundary conditions for all of the median filters
 * below. Rather than padding the data, we just don't calculate the median
 * filter for pixels around the border of the output image (n - 1) / 2 from
 * the edge, where we are using an n x n median filter. Edge effects often
 * look like cosmic rays and the edges are often blank so this shouldn't
 * matter. We fill the border with the original data values.
 */

/* Calculate the 3x3 median filter of an array data that has dimensions
 * nx x ny. The results are saved in the output array. The output array should
 * already be allocated as we work on it in place. The median filter is not
 * calculated for a 1 pixel border around the image. These pixel values are
 * copied from the input data. The data should be striped along the x
 * direction, such that pixel i,j in the 2D image should have memory location
 * data[i + nx *j].
 */
void
PyMedFilt3(float* data, float* output, int nx, int ny)
{
    PyDoc_STRVAR(PyMedFilt3__doc__,
        "PyMedFilt3(data, output, nx, ny) -> void\n\n"
            "Calculate the 3x3 median filter on an array data with dimensions "
            "nx x ny. The results are saved in the output array. The output "
            "array should already be allocated as we work on it in place. The "
            "median filter is not calculated for a 1 pixel border around the "
            "image. These pixel values are copied from the input data. Note "
            "that the data array needs to be striped in the x direction such "
            "that pixel i,j has memory location data[i + nx * j]");

    /*Total size of the array */
    int nxny = nx * ny;

    /* Loop indices */
    int i, j, nxj;
    int k, l, nxk;

    /* 9 element array to calculate the median and a counter index. Note that
     * these both need to be unique for each thread so they both need to be
     * private and we wait to allocate memory until the pragma below.*/
    float* medarr;
    int medcounter;

    /* Each thread needs to access the data and the output so we make them
     * firstprivate. We make sure that our algorithm doesn't have multiple
     * threads read or write the same piece of memory. */
#pragma omp parallel firstprivate(output, data, nx, ny) \
    private(i, j, k, l, medarr, nxj, nxk, medcounter)
    {
        /*Each thread allocates its own array. */
        medarr = (float *) malloc(9 * sizeof(float));

        /* Go through each pixel excluding the border.*/
#pragma omp for nowait
        for (j = 1; j < ny - 1; j++) {
            /* Precalculate the multiplication nx * j, minor optimization */
            nxj = nx * j;
            for (i = 1; i < nx - 1; i++) {
                medcounter = 0;
                /* The compiler should optimize away these loops */
                for (k = -1; k < 2; k++) {
                    nxk = nx * k;
                    for (l = -1; l < 2; l++) {
                        medarr[medcounter] = data[nxj + i + nxk + l];
                        medcounter++;
                    }
                }
                /* Calculate the median in the fastest way possible */
                output[nxj + i] = PyOptMed9(medarr);
            }
        }
        /* Each thread needs to free its own copy of medarr */
        free(medarr);
    }

#pragma omp parallel firstprivate(output, data, nx, nxny) private(i)
    /* Copy the border pixels from the original data into the output array */
    for (i = 0; i < nx; i++) {
        output[i] = data[i];
        output[nxny - nx + i] = data[nxny - nx + i];
    }
#pragma omp parallel firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        output[nxj] = data[nxj];
        output[nxj + nx - 1] = data[nxj + nx - 1];
    }

    return;
}

/* Calculate the 5x5 median filter of an array data that has dimensions
 * nx x ny. The results are saved in the output array. The output array should
 * already be allocated as we work on it in place. The median filter is not
 * calculated for a 2 pixel border around the image. These pixel values are
 * copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory
 * location data[i + nx *j].
 */
void
PyMedFilt5(float* data, float* output, int nx, int ny)
{
    PyDoc_STRVAR(PyMedFilt5__doc__,
        "PyMedFilt5(data, output, nx, ny) -> void\n\n"
            "Calculate the 5x5 median filter on an array data with dimensions "
            "nx x ny. The results are saved in the output array. The output "
            "array should already be allocated as we work on it in place. The "
            "median filter is not calculated for a 2 pixel border around the "
            "image. These pixel values are copied from the input data. Note "
            "that the data array needs to be striped in the x direction such "
            "that pixel i,j has memory location data[i + nx * j]");

    /*Total size of the array */
    int nxny = nx * ny;

    /* Loop indices */
    int i, j, nxj;
    int k, l, nxk;

    /* 25 element array to calculate the median and a counter index. Note that
     * these both need to be unique for each thread so they both need to be
     * private and we wait to allocate memory until the pragma below. */
    float* medarr;
    int medcounter;

    /* Each thread needs to access the data and the output so we make them
     * firstprivate. We make sure that our algorithm doesn't have multiple
     * threads read or write the same piece of memory. */
#pragma omp parallel firstprivate(output, data, nx, ny) \
    private(i, j, k, l, medarr, nxj, nxk, medcounter)
    {
        /*Each thread allocates its own array. */
        medarr = (float *) malloc(25 * sizeof(float));

        /* Go through each pixel excluding the border.*/
#pragma omp for nowait
        for (j = 2; j < ny - 2; j++) {
            /* Precalculate the multiplication nx * j, minor optimization */
            nxj = nx * j;
            for (i = 2; i < nx - 2; i++) {
                medcounter = 0;
                /* The compiler should optimize away these loops */
                for (k = -2; k < 3; k++) {
                    nxk = nx * k;
                    for (l = -2; l < 3; l++) {
                        medarr[medcounter] = data[nxj + i + nxk + l];
                        medcounter++;
                    }
                }
                /* Calculate the median in the fastest way possible */
                output[nxj + i] = PyOptMed25(medarr);
            }
        }
        /* Each thread needs to free its own copy of medarr */
        free(medarr);
    }

#pragma omp parallel firstprivate(output, data, nx, nxny) private(i)
    /* Copy the border pixels from the original data into the output array */
    for (i = 0; i < nx; i++) {
        output[i] = data[i];
        output[i + nx] = data[i + nx];
        output[nxny - nx + i] = data[nxny - nx + i];
        output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
    }

#pragma omp parallel firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        output[nxj] = data[nxj];
        output[nxj + 1] = data[nxj + 1];
        output[nxj + nx - 1] = data[nxj + nx - 1];
        output[nxj + nx - 2] = data[nxj + nx - 2];
    }

    return;
}

/* Calculate the 7x7 median filter of an array data that has dimensions
 * nx x ny. The results are saved in the output array. The output array should
 * already be allocated as we work on it in place. The median filter is not
 * calculated for a 3 pixel border around the image. These pixel values are
 * copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory
 * location data[i + nx *j].
 */
void
PyMedFilt7(float* data, float* output, int nx, int ny)
{
    PyDoc_STRVAR(PyMedFilt7__doc__,
        "PyMedFilt7(data, output, nx, ny) -> void\n\n"
            "Calculate the 7x7 median filter on an array data with dimensions "
            "nx x ny. The results are saved in the output array. The output "
            "array should already be allocated as we work on it in place. The "
            "median filter is not calculated for a 3 pixel border around the "
            "image. These pixel values are copied from the input data. Note "
            "that the data array needs to be striped in the x direction such "
            "that pixel i,j has memory location data[i + nx * j]");

    /*Total size of the array */
    int nxny = nx * ny;

    /* Loop indices */
    int i, j, nxj;
    int k, l, nxk;

    /* 49 element array to calculate the median and a counter index. Note that
     * these both need to be unique for each thread so they both need to be
     * private and we wait to allocate memory until the pragma below. */
    float* medarr;
    int medcounter;

    /* Each thread needs to access the data and the output so we make them
     * firstprivate. We make sure that our algorithm doesn't have multiple
     * threads read or write the same piece of memory. */
#pragma omp parallel firstprivate(output, data, nx, ny) \
    private(i, j, k, l, medarr, nxj, nxk, medcounter)
    {
        /*Each thread allocates its own array. */
        medarr = (float *) malloc(49 * sizeof(float));

        /* Go through each pixel excluding the border.*/
#pragma omp for nowait
        for (j = 3; j < ny - 3; j++) {
            /* Precalculate the multiplication nx * j, minor optimization */
            nxj = nx * j;
            for (i = 3; i < nx - 3; i++) {
                medcounter = 0;
                /* The compiler should optimize away these loops */
                for (k = -3; k < 4; k++) {
                    nxk = nx * k;
                    for (l = -3; l < 4; l++) {
                        medarr[medcounter] = data[nxj + i + nxk + l];
                        medcounter++;
                    }
                }
                /* Calculate the median in the fastest way possible */
                output[nxj + i] = PyMedian(medarr, 49);
            }
        }
        /* Each thread needs to free its own copy of medarr */
        free(medarr);
    }

#pragma omp parallel firstprivate(output, data, nx, nxny) private(i)
    /* Copy the border pixels from the original data into the output array */
    for (i = 0; i < nx; i++) {
        output[i] = data[i];
        output[i + nx] = data[i + nx];
        output[i + nx + nx] = data[i + nx + nx];
        output[nxny - nx + i] = data[nxny - nx + i];
        output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
        output[nxny - nx - nx - nx + i] = data[nxny - nx - nx - nx + i];
    }

#pragma omp parallel firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        output[nxj] = data[nxj];
        output[nxj + 1] = data[nxj + 1];
        output[nxj + 2] = data[nxj + 2];
        output[nxj + nx - 1] = data[nxj + nx - 1];
        output[nxj + nx - 2] = data[nxj + nx - 2];
        output[nxj + nx - 3] = data[nxj + nx - 3];
    }

    return;
}

/* Calculate the 3x3 separable median filter of an array data that has
 * dimensions nx x ny. The results are saved in the output array. The output
 * array should already be allocated as we work on it in place. The median
 * filter is not calculated for a 1 pixel border around the image. These pixel
 * values are copied from the input data. The data should be striped along
 * the x direction, such that pixel i,j in the 2D image should have memory
 * location data[i + nx *j]. Note that the rows are median filtered first,
 * followed by the columns.
 */
void
PySepMedFilt3(float* data, float* output, int nx, int ny)
{
    PyDoc_STRVAR(PySepMedFilt3__doc__,
        "PySepMedFilt3(data, output, nx, ny) -> void\n\n"
            "Calculate the 3x3 separable median filter on an array data with"
            "dimensions nx x ny. The results are saved in the output array "
            "which should already be allocated as we work on it in place. The "
            "median filter is not calculated for a 1 pixel border which is "
            "copied from the input data. The data array should be striped in "
            "the x direction such that pixel i,j has memory location "
            "data[i + nx * j]. Note that the rows are median filtered first, "
            "followed by the columns.");

    /* Total number of pixels */
    int nxny = nx * ny;

    /* Output array for the median filter of the rows. We later median filter
     * the columns of this array. */
    float* rowmed = (float *) malloc(nxny * sizeof(float));

    /* Loop indices */
    int i, j, nxj;

    /* 3 element array to calculate the median and a counter index. Note that
     * this array needs to be unique for each thread so it needs to be
     * private and we wait to allocate memory until the pragma below. */
    float* medarr;

    /* Median filter the rows first */

    /* Each thread needs to access the data and rowmed so we make them
     * firstprivate. We make sure that our algorithm doesn't have multiple
     * threads read or write the same piece of memory. */
#pragma omp parallel firstprivate(data, rowmed, nx, ny) \
    private(i, j, nxj, medarr)
    {
        /*Each thread allocates its own array. */
        medarr = (float *) malloc(3 * sizeof(float));

        /* For each pixel excluding the border */
#pragma omp for nowait
        for (j = 0; j < ny; j++) {
            nxj = nx * j;
            for (i = 1; i < nx - 1; i++) {
                medarr[0] = data[nxj + i];
                medarr[1] = data[nxj + i - 1];
                medarr[2] = data[nxj + i + 1];
                /* Calculate the median in the fastest way possible */
                rowmed[nxj + i] = PyOptMed3(medarr);
            }
        }
        /* Each thread needs to free its own medarr */
        free(medarr);
    }

    /* Fill in the borders of rowmed with the original data values */
#pragma omp parallel for firstprivate(data, rowmed, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        rowmed[nxj] = data[nxj];
        rowmed[nxj + nx - 1] = data[nxj + nx - 1];
    }

    /* Median filter the columns */
#pragma omp parallel firstprivate(rowmed, output, nx, ny) \
    private(i, j, nxj, medarr)
    {
        /* Each thread needs to reallocate a new medarr */
        medarr = (float *) malloc(3 * sizeof(float));

        /* For each pixel excluding the border */
#pragma omp for nowait
        for (j = 1; j < ny - 1; j++) {
            nxj = nx * j;
            for (i = 1; i < nx - 1; i++) {
                medarr[0] = rowmed[i + nxj];
                medarr[1] = rowmed[i + nxj - nx];
                medarr[2] = rowmed[i + nxj + nx];
                /* Calculate the median in the fastest way possible */
                output[nxj + i] = PyOptMed3(medarr);
            }
        }
        /* Each thread needs to free its own medarr */
        free(medarr);
    }
    /* Clean up rowmed */
    free(rowmed);

    /* Copy the border pixels from the original data into the output array */
#pragma omp parallel for firstprivate(output, data, nx, nxny) private(i)
    for (i = 0; i < nx; i++) {
        output[i] = data[i];
        output[nxny - nx + i] = data[nxny - nx + i];
    }
#pragma omp parallel for firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        output[nxj] = data[nxj];
        output[nxj + nx - 1] = data[nxj + nx - 1];
    }

    return;
}

/* Calculate the 5x5 separable median filter of an array data that has
 * dimensions nx x ny. The results are saved in the output array. The output
 * array should already be allocated as we work on it in place.The median
 * filter is not calculated for a 2 pixel border around the image. These pixel
 * values are copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory location
 * data[i + nx *j]. Note that the rows are median filtered first, followed by
 * the columns.
 */
void
PySepMedFilt5(float* data, float* output, int nx, int ny)
{
    PyDoc_STRVAR(PySepMedFilt5__doc__,
        "PySepMedFilt5(data, output, nx, ny) -> void\n\n"
            "Calculate the 5x5 separable median filter on an array data with "
            "dimensions nx x ny. The results are saved in the output array "
            "which should already be allocated as we work on it in place. The "
            "median filter is not calculated for a 2 pixel border which is "
            "copied from the input data. The data array should be striped in "
            "the x direction such that pixel i,j has memory location "
            "data[i + nx * j]. Note that the rows are median filtered first, "
            "followed by the columns.");

    /* Total number of pixels */
    int nxny = nx * ny;

    /* Output array for the median filter of the rows. We later median filter
     * the columns of this array. */
    float* rowmed = (float *) malloc(nxny * sizeof(float));

    /* Loop indices */
    int i, j, nxj;

    /* 5 element array to calculate the median and a counter index. Note that
     * this array needs to be unique for each thread so it needs to be
     * private and we wait to allocate memory until the pragma below. */
    float* medarr;

    /* Median filter the rows first */

    /* Each thread needs to access the data and rowmed so we make them
     * firstprivate. We make sure that our algorithm doesn't have multiple
     * threads read or write the same piece of memory. */
#pragma omp parallel firstprivate(data, rowmed, nx, ny) \
    private(i, j, nxj, medarr)
    {
        /*Each thread allocates its own array. */
        medarr = (float *) malloc(5 * sizeof(float));

        /* For each pixel excluding the border */
#pragma omp for nowait
        for (j = 0; j < ny; j++) {
            nxj = nx * j;
            for (i = 2; i < nx - 2; i++) {
                medarr[0] = data[nxj + i];
                medarr[1] = data[nxj + i - 1];
                medarr[2] = data[nxj + i + 1];
                medarr[3] = data[nxj + i - 2];
                medarr[4] = data[nxj + i + 2];
                /* Calculate the median in the fastest way possible */
                rowmed[nxj + i] = PyOptMed5(medarr);
            }
        }
        /* Each thread needs to free its own medarr */
        free(medarr);
    }

    /* Fill in the borders of rowmed with the original data values */
#pragma omp parallel for firstprivate(rowmed, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        rowmed[nxj] = data[nxj];
        rowmed[nxj + 1] = data[nxj + 1];
        rowmed[nxj + nx - 1] = data[nxj + nx - 1];
        rowmed[nxj + nx - 2] = data[nxj + nx - 2];
    }

    /* Median filter the columns */
#pragma omp parallel firstprivate(rowmed, output, nx, ny) \
    private(i, j, nxj, medarr)
    {
        /* Each thread needs to reallocate a new medarr */
        medarr = (float *) malloc(5 * sizeof(float));

        /* For each pixel excluding the border */
#pragma omp for nowait
        for (j = 2; j < ny - 2; j++) {
            nxj = nx * j;
            for (i = 2; i < nx - 2; i++) {
                medarr[0] = rowmed[i + nxj];
                medarr[1] = rowmed[i + nxj - nx];
                medarr[2] = rowmed[i + nxj + nx];
                medarr[3] = rowmed[i + nxj + nx + nx];
                medarr[4] = rowmed[i + nxj - nx - nx];

                /* Calculate the median in the fastest way possible */
                output[nxj + i] = PyOptMed5(medarr);
            }
        }
        /* Each thread needs to free its own medarr */
        free(medarr);
    }
    /* Clean up rowmed */
    free(rowmed);

    /* Copy the border pixels from the original data into the output array */
#pragma omp parallel for firstprivate(output, data, nx, nxny) private(i)
    for (i = 0; i < nx; i++) {
        output[i] = data[i];
        output[i + nx] = data[i + nx];
        output[nxny - nx + i] = data[nxny - nx + i];
        output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
    }
#pragma omp parallel for firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        output[nxj] = data[nxj];
        output[nxj + 1] = data[nxj + 1];
        output[nxj + nx - 1] = data[nxj + nx - 1];
        output[nxj + nx - 2] = data[nxj + nx - 2];
    }

    return;
}

/* Calculate the 7x7 separable median filter of an array data that has
 * dimensions nx x ny. The results are saved in the output array. The output
 * array should already be allocated as we work on it in place. The median
 * filter is not calculated for a 3 pixel border around the image. These pixel
 * values are copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory location
 * data[i + nx *j]. Note that the rows are median filtered first, followed by
 * the columns.
 */
void
PySepMedFilt7(float* data, float* output, int nx, int ny)
{
    PyDoc_STRVAR(PySepMedFilt7__doc__,
        "PySepMedFilt7(data, output, nx, ny) -> void\n\n"
            "Calculate the 7x7 separable median filter on an array data with "
            "dimensions nx x ny. The results are saved in the output array "
            "which should already be allocated as we work on it in place. The "
            "median filter is not calculated for a 3 pixel border which is "
            "copied from the input data. The data array should be striped in "
            "the x direction such that pixel i,j has memory location "
            "data[i + nx * j]. Note that the rows are median filtered first, "
            "followed by the columns.");

    /* Total number of pixels */
    int nxny = nx * ny;

    /* Output array for the median filter of the rows. We later median filter
     * the columns of this array. */
    float* rowmed = (float *) malloc(nxny * sizeof(float));

    /* Loop indices */
    int i, j, nxj;

    /* 7 element array to calculate the median and a counter index. Note that
     * this array needs to be unique for each thread so it needs to be
     * private and we wait to allocate memory until the pragma below. */
    float* medarr;

    /* Median filter the rows first */

    /* Each thread needs to access the data and rowmed so we make them
     * firstprivate. We make sure that our algorithm doesn't have multiple
     * threads read or write the same piece of memory. */
#pragma omp parallel firstprivate(data, rowmed, nx, ny) \
    private(i, j, nxj, medarr)
    {
        /*Each thread allocates its own array. */
        medarr = (float *) malloc(7 * sizeof(float));

        /* For each pixel excluding the border */
#pragma omp for nowait
        for (j = 0; j < ny; j++) {
            nxj = nx * j;
            for (i = 3; i < nx - 3; i++) {
                medarr[0] = data[nxj + i];
                medarr[1] = data[nxj + i - 1];
                medarr[2] = data[nxj + i + 1];
                medarr[3] = data[nxj + i - 2];
                medarr[4] = data[nxj + i + 2];
                medarr[5] = data[nxj + i - 3];
                medarr[6] = data[nxj + i + 3];

                /* Calculate the median in the fastest way possible */
                rowmed[nxj + i] = PyOptMed7(medarr);
            }
        }
        /* Each thread needs to free its own medarr */
        free(medarr);
    }

    /* Fill in the borders of rowmed with the original data values */
#pragma omp parallel for firstprivate(rowmed, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        rowmed[nxj] = data[nxj];
        rowmed[nxj + 1] = data[nxj + 1];
        rowmed[nxj + 2] = data[nxj + 2];
        rowmed[nxj + nx - 1] = data[nxj + nx - 1];
        rowmed[nxj + nx - 2] = data[nxj + nx - 2];
        rowmed[nxj + nx - 3] = data[nxj + nx - 3];
    }

    /* Median filter the columns */
#pragma omp parallel firstprivate(rowmed, output, nx, ny) \
    private(i, j, nxj, medarr)
    {
        /* Each thread needs to reallocate a new medarr */
        medarr = (float *) malloc(7 * sizeof(float));

        /* For each pixel excluding the border */
#pragma omp for nowait
        for (j = 3; j < ny - 3; j++) {
            nxj = nx * j;
            for (i = 3; i < nx - 3; i++) {
                medarr[0] = rowmed[i + nxj - nx];
                medarr[1] = rowmed[i + nxj + nx];
                medarr[2] = rowmed[i + nxj + nx + nx];
                medarr[3] = rowmed[i + nxj - nx - nx];
                medarr[4] = rowmed[i + nxj];
                medarr[5] = rowmed[i + nxj + nx + nx + nx];
                medarr[6] = rowmed[i + nxj - nx - nx - nx];
                /* Calculate the median in the fastest way possible */
                output[nxj + i] = PyOptMed7(medarr);
            }
        }
        /* Each thread needs to free its own medarr */
        free(medarr);
    }
    /* Clean up rowmed */
    free(rowmed);

    /* Copy the border pixels from the original data into the output array */
#pragma omp parallel for firstprivate(output, data, nx, nxny) private(i)
    for (i = 0; i < nx; i++) {
        output[i] = data[i];
        output[i + nx] = data[i + nx];
        output[i + nx + nx] = data[i + nx + nx];
        output[nxny - nx + i] = data[nxny - nx + i];
        output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
        output[nxny - nx - nx - nx + i] = data[nxny - nx - nx - nx + i];

    }
#pragma omp parallel for firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        output[nxj] = data[nxj];
        output[nxj + 1] = data[nxj + 1];
        output[nxj + 2] = data[nxj + 2];
        output[nxj + nx - 1] = data[nxj + nx - 1];
        output[nxj + nx - 2] = data[nxj + nx - 2];
        output[nxj + nx - 3] = data[nxj + nx - 3];
    }

    return;
}

/* Calculate the 9x9 separable median filter of an array data that has
 * dimensions nx x ny. The results are saved in the output array. The output
 * array should already be allocated as we work on it in place. The median
 * filter is not calculated for a 4 pixel border around the image. These pixel
 * values are copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory location
 * data[i + nx *j]. Note that the rows are median filtered first, followed by
 * the columns.
 */
void
PySepMedFilt9(float* data, float* output, int nx, int ny)
{
    PyDoc_STRVAR(PySepMedFilt9__doc__,
        "PySepMedFilt9(data, output, nx, ny) -> void\n\n"
            "Calculate the 9x9 separable median filter on an array data with "
            "dimensions nx x ny. The results are saved in the output array "
            "which should already be allocated as we work on it in place. The "
            "median filter is not calculated for a 4 pixel border which is "
            "copied from the input data. The data array should be striped in "
            "the x direction such that pixel i,j has memory location "
            "data[i + nx * j]. Note that the rows are median filtered first, "
            "followed by the columns.");

    /* Total number of pixels */
    int nxny = nx * ny;

    /* Output array for the median filter of the rows. We later median filter
     * the columns of this array. */
    float* rowmed = (float *) malloc(nxny * sizeof(float));

    /* Loop indices */
    int i, j, nxj;

    /* 9 element array to calculate the median and a counter index. Note that
     * this array needs to be unique for each thread so it needs to be
     * private and we wait to allocate memory until the pragma below. */
    float* medarr;

    /* Median filter the rows first */

    /* Each thread needs to access the data and rowmed so we make them
     * firstprivate. We make sure that our algorithm doesn't have multiple
     * threads read or write the same piece of memory. */
#pragma omp parallel firstprivate(data, rowmed, nx, ny) \
    private(i, j, nxj, medarr)
    {
        /*Each thread allocates its own array. */
        medarr = (float *) malloc(9 * sizeof(float));

        /* For each pixel excluding the border */
#pragma omp for nowait
        for (j = 0; j < ny; j++) {
            nxj = nx * j;
            for (i = 4; i < nx - 4; i++) {
                medarr[0] = data[nxj + i];
                medarr[1] = data[nxj + i - 1];
                medarr[2] = data[nxj + i + 1];
                medarr[3] = data[nxj + i - 2];
                medarr[4] = data[nxj + i + 2];
                medarr[5] = data[nxj + i - 3];
                medarr[6] = data[nxj + i + 3];
                medarr[7] = data[nxj + i - 4];
                medarr[8] = data[nxj + i + 4];
                /* Calculate the median in the fastest way possible */
                rowmed[nxj + i] = PyOptMed9(medarr);
            }
        }
        /* Each thread needs to free its own medarr */
        free(medarr);
    }

    /* Fill in the borders of rowmed with the original data values */
#pragma omp parallel for firstprivate(rowmed, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        rowmed[nxj] = data[nxj];
        rowmed[nxj + 1] = data[nxj + 1];
        rowmed[nxj + 2] = data[nxj + 2];
        rowmed[nxj + 3] = data[nxj + 3];
        rowmed[nxj + nx - 1] = data[nxj + nx - 1];
        rowmed[nxj + nx - 2] = data[nxj + nx - 2];
        rowmed[nxj + nx - 3] = data[nxj + nx - 3];
        rowmed[nxj + nx - 4] = data[nxj + nx - 4];
    }

    /* Median filter the columns */
#pragma omp parallel firstprivate(rowmed, output, nx, ny) \
    private(i, j, nxj, medarr)
    {
        /* Each thread needs to reallocate a new medarr */
        medarr = (float *) malloc(9 * sizeof(float));

        /* For each pixel excluding the border */
#pragma omp for nowait
        for (j = 4; j < ny - 4; j++) {
            nxj = nx * j;
            for (i = 4; i < nx - 4; i++) {
                medarr[0] = rowmed[i + nxj];
                medarr[1] = rowmed[i + nxj - nx];
                medarr[2] = rowmed[i + nxj + nx];
                medarr[3] = rowmed[i + nxj + nx + nx];
                medarr[4] = rowmed[i + nxj - nx - nx];
                medarr[5] = rowmed[i + nxj + nx + nx + nx];
                medarr[6] = rowmed[i + nxj - nx - nx - nx];
                medarr[7] = rowmed[i + nxj + nx + nx + nx + nx];
                medarr[8] = rowmed[i + nxj - nx - nx - nx - nx];
                /* Calculate the median in the fastest way possible */
                output[nxj + i] = PyOptMed9(medarr);
            }
        }
        /* Each thread needs to free its own medarr */
        free(medarr);
    }
    /* Clean up rowmed */
    free(rowmed);

    /* Copy the border pixels from the original data into the output array */
#pragma omp parallel for firstprivate(output, data, nx, nxny) private(i)
    for (i = 0; i < nx; i++) {
        output[i] = data[i];
        output[i + nx] = data[i + nx];
        output[i + nx + nx] = data[i + nx + nx];
        output[i + nx + nx + nx] = data[i + nx + nx + nx];
        output[nxny - nx + i] = data[nxny - nx + i];
        output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
        output[nxny - nx - nx - nx + i] = data[nxny - nx - nx - nx + i];
        output[nxny - nx - nx - nx - nx + i] = data[nxny - nx - nx - nx - nx
            + i];
    }
#pragma omp parallel for firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        output[nxj] = data[nxj];
        output[nxj + 1] = data[nxj + 1];
        output[nxj + 2] = data[nxj + 2];
        output[nxj + 3] = data[nxj + 3];
        output[nxj + nx - 1] = data[nxj + nx - 1];
        output[nxj + nx - 2] = data[nxj + nx - 2];
        output[nxj + nx - 3] = data[nxj + nx - 3];
        output[nxj + nx - 4] = data[nxj + nx - 4];
    }

    return;
}

/* Subsample an array 2x2 given an input array data with size nx x ny. Each
 * pixel is replicated into 4 pixels; no averaging is performed. The results
 * are saved in the output array. The output array should already be allocated
 * as we work on it in place. Data should be striped in the x direction such
 * that the memory location of pixel i,j is data[nx *j + i].
 */
void
PySubsample(float* data, float* output, int nx, int ny)
{
    PyDoc_STRVAR(PySubsample__doc__,
        "PySubample(data, output, nx, ny) -> void\n\n"
            "Subsample an array 2x2 given an input array data with size "
            "nx x ny.The results are saved in the output array. The output "
            "array should already be allocated as we work on it in place. Each"
            " pixel is replicated into 4 pixels; no averaging is performed. "
            "Data should be striped in the x direction such that the memory "
            "location of pixel i,j is data[nx *j + i].");

    /* Precalculate the new length; minor optimization */
    int padnx = 2 * nx;

    /* Loop indices */
    int i, j, nxj, padnxj;

    /* Loop over all pixels */
#pragma omp parallel for firstprivate(data, output, nx, ny, padnx) \
    private(i, j, nxj, padnxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        padnxj = 2 * padnx * j;
        for (i = 0; i < nx; i++) {
            /* Copy the pixel value into a 2x2 grid on the output image */
            output[2 * i + padnxj] = data[i + nxj];
            output[2 * i + padnxj + padnx] = data[i + nxj];
            output[2 * i + 1 + padnxj + padnx] = data[i + nxj];
            output[2 * i + 1 + padnxj] = data[i + nxj];
        }
    }

    return;
}

/* Rebin an array 2x2, with size (2 * nx) x (2 * ny). Rebin the array by block
 * averaging 4 pixels back into 1. This is effectively the opposite of
 * subsample (although subsample does not do an average). The results are saved
 * in the output array. The output array should already be allocated as we work
 * on it in place. Data should be striped in the x direction such that the
 * memory location of pixel i,j is data[nx *j + i].
 */
void
PyRebin(float* data, float* output, int nx, int ny)
{
    PyDoc_STRVAR(PyRebin__doc__,
        "PyRebin(data, output, nx, ny) -> void\n    \n"
            "Rebin an array 2x2, with size (2 * nx) x (2 * ny). Rebin the "
            "array by block averaging 4 pixels back into 1. This is "
            "effectively the opposite of subsample (although subsample does "
            "not do an average). The results are saved in the output array. "
            "The output array should already be allocated as we work on it in "
            "place. Data should be striped in the x direction such that the "
            "memory location of pixel i,j is data[nx *j + i].");

    /* Size of original array */
    int padnx = nx * 2;

    /* Loop variables */
    int i, j, nxj, padnxj;

    /* Pixel value p. Each thread needs its own copy of this variable so we
     * wait to initialize it until the pragma below */
    float p;
#pragma omp parallel for firstprivate(output, data, nx, ny, padnx) \
    private(i, j, nxj, padnxj, p)
    /*Loop over all of the pixels */
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        padnxj = 2 * padnx * j;
        for (i = 0; i < nx; i++) {
            p = data[2 * i + padnxj];
            p += data[2 * i + padnxj + padnx];
            p += data[2 * i + 1 + padnxj + padnx];
            p += data[2 * i + 1 + padnxj];
            p /= 4.0;
            output[i + nxj] = p;
        }
    }
    return;
}

/* Convolve an image of size nx x ny with a kernel of size  kernx x kerny. The
 * results are saved in the output array. The output array should already be
 * allocated as we work on it in place. Data and kernel should both be striped
 * in the x direction such that the memory location of pixel i,j is
 * data[nx *j + i].
 */
void
PyConvolve(float* data, float* kernel, float* output, int nx, int ny,
           int kernx, int kerny)
{
    PyDoc_STRVAR(PyConvolve__doc__,
        "PyConvolve(data, kernel, output, nx, ny, kernx, kerny) -> void\n\n"
            "Convolve an image of size nx x ny with a a kernel of size "
            "kernx x kerny. The results are saved in the output array. The "
            "output array should already be allocated as we work on it in "
            "place. Data and kernel should both be striped along the x "
            "direction such that the memory location of pixel i,j is "
            "data[nx *j + i].");

    /* Get the width of the borders that we will pad with zeros */
    int bnx = (kernx - 1) / 2;
    int bny = (kerny - 1) / 2;

    /* Calculate the dimensions of the array including padded border */
    int padnx = nx + kernx - 1;
    int padny = ny + kerny - 1;
    /* Get the total number of pixels in the padded array */
    int padnxny = padnx * padny;
    /*Get the total number of pixels in the output image */
    int nxny = nx * ny;

    /*Allocate the padded array */
    float* padarr = (float *) malloc(padnxny * sizeof(float));

    /* Loop variables. These should all be thread private. */
    int i, j;
    int nxj;
    int padnxj;
    /* Inner loop variables. Again thread private. */
    int k, l;
    int kernxl, padnxl;

    /* Define a sum variable to use in the convolution calculation. Each
     * thread needs its own copy of this so it should be thread private. */
    float sum;

    /* Precompute maximum good index in each dimension */
    int xmaxgood = nx + bnx;
    int ymaxgood = ny + bny;

    /* Set the borders of padarr = 0.0
     * Fill the rest of the padded array with the input data. */
#pragma omp parallel for \
    firstprivate(padarr, data, nx, padnx, padny, bnx, bny, xmaxgood, ymaxgood)\
    private(nxj, padnxj, i, j)
    for (j = 0; j < padny; j++) {
        padnxj = padnx * j;
        nxj = nx * (j - bny);
        for (i = 0; i < padnx; i++) {
            if (i < bnx || j < bny || j >= ymaxgood || i >= xmaxgood) {
                padarr[padnxj + i] = 0.0;
            }
            else {
                padarr[padnxj + i] = data[nxj + i - bnx];
            }
        }

    }

    /* Calculate the convolution  */
    /* Loop over all pixels */
#pragma omp parallel for \
    firstprivate(padarr, output, nx, ny, padnx, bnx, bny, kernx) \
    private(nxj, padnxj, kernxl, padnxl, i, j, k, l, sum)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        /* Note the + bvy in padnxj */
        padnxj = padnx * (j + bny);
        for (i = 0; i < nx; i++) {
            sum = 0.0;
            /* Note that the sums in the definition of the convolution go from
             * -border width to + border width */
            for (l = -bny; l <= bny; l++) {
                padnxl = padnx * (l + j + bny);
                kernxl = kernx * (-l + bny);
                for (k = -bnx; k <= bnx; k++) {
                    sum += kernel[bnx - k + kernxl]
                        * padarr[padnxl + k + i + bnx];
                }
            }
            output[nxj + i] = sum;
        }
    }

    free(padarr);

    return;
}

/* Convolve an image of size nx x ny the following kernel:
 *  0 -1  0
 * -1  4 -1
 *  0 -1  0
 * The results are saved in the output array. The output array should
 * already be allocated as we work on it in place.
 * This is a discrete version of the Laplacian operator.
 * Data should be striped in the x direction such that the memory location of
 * pixel i,j is data[nx *j + i].
 */
void
PyLaplaceConvolve(float* data, float* output, int nx, int ny)
{
    PyDoc_STRVAR(PyLaplaceConvolve__doc__,
        "PyLaplaceConvolve(data, output, nx, ny) -> void\n\n"
            "Convolve an image of size nx x ny the following kernel:\n"
            " 0 -1  0\n"
            "-1  4 -1\n"
            " 0 -1  0\n"
            "This is a discrete version of the Laplacian operator. The results"
            " are saved in the output array. The output array should already "
            "be allocated as we work on it in place.Data should be striped in "
            "the x direction such that the memory location of pixel i,j is "
            "data[nx *j + i].");

    /* Precompute the total number of pixels in the image */
    int nxny = nx * ny;

    /* Loop variables */
    int i, j, nxj;

    /* Pixel value p. Each thread will need its own copy of this so we need to
     * make it private*/
    float p;
    /* Because we know the form of the kernel, we can short circuit the
     * convolution and calculate the results with inner nest for loops. */

    /*Loop over all of the pixels except the edges which we will do explicitly
     * below */
#pragma omp parallel for firstprivate(nx, ny, output, data) \
    private(i, j, nxj, p)
    for (j = 1; j < ny - 1; j++) {
        nxj = nx * j;
        for (i = 1; i < nx - 1; i++) {
            p = 4.0 * data[nxj + i];
            p -= data[i + 1 + nxj];
            p -= data[i - 1 + nxj];
            p -= data[i + nxj + nx];
            p -= data[i + nxj - nx];

            output[nxj + i] = p;
        }
    }

    /* Leave the corners until the very end */

#pragma omp parallel firstprivate(output, data, nx, nxny) private(i)
    /* Top and Bottom Rows */
    for (i = 1; i < nx - 1; i++) {
        output[i] = 4.0 * data[i] - data[i + 1] - data[i - 1] - data[i + nx];

        p = 4.0 * data[i + nxny - nx];
        p -= data[i + 1 + nxny - nx];
        p -= data[i + nxny - nx - 1];
        p -= data[i - nx + nxny - nx];
        output[i + nxny - nx] = p;
    }

#pragma omp parallel firstprivate(output, data, nx, ny) private(j, nxj)
    /* First and Last Column */
    for (j = 1; j < ny - 1; j++) {
        nxj = nx * j;
        p = 4.0 * data[nxj];
        p -= data[nxj + 1];
        p -= data[nxj + nx];
        p -= data[nxj - nx];
        output[nxj] = p;

        p = 4.0 * data[nxj + nx - 1];
        p -= data[nxj + nx - 2];
        p -= data[nxj + nx + nx - 1];
        p -= data[nxj - 1];
        output[nxj + nx - 1] = p;
    }

    /* Bottom Left Corner */
    output[0] = 4.0 * data[0] - data[1] - data[nx];
    /* Bottom Right Corner */
    output[nx - 1] = 4.0 * data[nx - 1] - data[nx - 2] - data[nx + nx - 1];
    /* Top Left Corner */
    p = 4.0 * data[nxny - nx];
    p -= data[nxny - nx + 1];
    p -= data[nxny - nx - nx];
    output[nxny - nx] = p;
    /* Top Right Corner */
    p = 4.0 * data[nxny - 1];
    p -= data[nxny - 2];
    p -= data[nxny - 1 - nx];
    output[nxny - 1] = p;

    return;
}

/* Perform a boolean dilation on an array of size nx x ny. The results are
 * saved in the output array. The output array should already be allocated as
 * we work on it in place.
 * Dilation is the boolean equivalent of a convolution but using logical ors
 * instead of a sum.
 * We apply the following kernel:
 * 1 1 1
 * 1 1 1
 * 1 1 1
 * The binary dilation is not computed for a 1 pixel border around the image.
 * These pixels are copied from the input data. Data should be striped along
 * the x direction such that the memory location of pixel i,j is
 * data[i + nx * j].
 */
void
PyDilate3(bool* data, bool* output, int nx, int ny)
{
    PyDoc_STRVAR(PyDilate3__doc__,
        "PyDilate3(data, output, nx, ny) -> void\n\n"
            "Perform a boolean dilation on an array of size nx x ny. The "
            "results are saved in the output array which should already be "
            "allocated as we work on it in place. "
            "Dilation is the boolean equivalent of a convolution but using "
            "logical or instead of a sum. We apply a 3x3 kernel of all ones. "
            "Dilation is not computed for a 1 pixel border which is copied "
            "from the input data. Data should be striped along the x-axis "
            "such that the location of pixel i,j is data[i + nx * j].");

    /* Precompute the total number of pixels; minor optimization */
    int nxny = nx * ny;

    /* Loop variables */
    int i, j, nxj;

    /* Pixel value p. Each thread needs its own unique copy of this so we don't
     initialize this until the pragma below. */
    bool p;

#pragma omp parallel for firstprivate(output, data, nxny, nx, ny) \
    private(i, j, nxj, p)

    /* Loop through all of the pixels excluding the border */
    for (j = 1; j < ny - 1; j++) {
        nxj = nx * j;
        for (i = 1; i < nx - 1; i++) {
            /*Start in the middle and work out */
            p = data[i + nxj];
            /* Right 1 */
            p = p || data[i + 1 + nxj];
            /* Left 1 */
            p = p || data[i - 1 + nxj];
            /* Up 1 */
            p = p || data[i + nx + nxj];
            /* Down 1 */
            p = p || data[i - nx + nxj];
            /* Up 1 Right 1 */
            p = p || data[i + 1 + nx + nxj];
            /* Up 1 Left 1 */
            p = p || data[i - 1 + nx + nxj];
            /* Down 1 Right 1 */
            p = p || data[i + 1 - nx + nxj];
            /* Down 1 Left 1 */
            p = p || data[i - 1 - nx + nxj];

            output[i + nxj] = p;
        }
    }

#pragma omp parallel firstprivate(output, data, nx, nxny) private(i)
    /* For the borders, copy the data from the input array */
    for (i = 0; i < nx; i++) {
        output[i] = data[i];
        output[nxny - nx + i] = data[nxny - nx + i];
    }
#pragma omp parallel firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        output[nxj] = data[nxj];
        output[nxj - 1 + nx] = data[nxj - 1 + nx];
    }

    return;
}

/* Do niter iterations of boolean dilation on an array of size nx x ny. The
 * results are saved in the output array. The output array should already be
 * allocated as we work on it in place.
 * Dilation is the boolean equivalent of a convolution but using logical ors
 * instead of a sum.
 * We apply the following kernel:
 * 0 1 1 1 0
 * 1 1 1 1 1
 * 1 1 1 1 1
 * 1 1 1 1 1
 * 0 1 1 1 0
 * The edges are padded with zeros so that the dilation operator is defined for
 * all pixels. Data should be striped along the x direction such that the
 * memory location of pixel i,j is data[i + nx * j].
 */
void
PyDilate5(bool* data, bool* output, int niter, int nx, int ny)
{
    PyDoc_STRVAR(PyDilate5__doc__,
        "PyDilate5(data, output, nx, ny) -> void\n\n"
            "Do niter iterations of boolean dilation on an array of size "
            "nx x ny. The results are saved in the output array. The output "
            "array should already be allocated as we work on it in place. "
            "Dilation is the boolean equivalent of a convolution but using "
            "logical ors instead of a sum. We apply the following kernel:\n"
            "0 1 1 1 0\n"
            "1 1 1 1 1\n"
            "1 1 1 1 1\n"
            "1 1 1 1 1\n"
            "0 1 1 1 0\n"
            "Data should be striped along the x direction such that the "
            "location of pixel i,j is data[i + nx * j].");

    /* Pad the array with a border of zeros */
    int padnx = nx + 4;
    int padny = ny + 4;

    /* Precompute the total number of pixels; minor optimization */
    int padnxny = padnx * padny;
    int nxny = nx * ny;

    /* The padded array to work on */
    bool* padarr = (bool *) malloc(padnxny * sizeof(bool));

    /*Loop indices */
    int i, j, nxj, padnxj;
    int iter;

    /* Pixel value p. This needs to be unique for each thread so we initialize
     * it below inside the pragma. */
    bool p;

#pragma omp parallel firstprivate(padarr, padnx, padnxny) private(i)
    /* Initialize the borders of the padded array to zero */
    for (i = 0; i < padnx; i++) {
        padarr[i] = false;
        padarr[i + padnx] = false;
        padarr[padnxny - padnx + i] = false;
        padarr[padnxny - padnx - padnx + i] = false;
    }

#pragma omp parallel firstprivate(padarr, padnx, padny) private(j, padnxj)
    for (j = 0; j < padny; j++) {
        padnxj = padnx * j;
        padarr[padnxj] = false;
        padarr[padnxj + 1] = false;
        padarr[padnxj + padnx - 1] = false;
        padarr[padnxj + padnx - 2] = false;
    }

#pragma omp parallel firstprivate(output, data, nxny) private(i)
    /* Initialize the output array to the input data */
    for (i = 0; i < nxny; i++) {
        output[i] = data[i];
    }

    /* Outer iteration loop */
    for (iter = 0; iter < niter; iter++) {
#pragma omp parallel for firstprivate(padarr, output, nx, ny, padnx, iter) \
    private(nxj, padnxj, i, j)
        /* Initialize the padded array to the output from the latest
         * iteration*/
        for (j = 0; j < ny; j++) {
            padnxj = padnx * j;
            nxj = nx * j;
            for (i = 0; i < nx; i++) {
                padarr[i + 2 + padnx + padnx + padnxj] = output[i + nxj];
            }
        }

        /* Loop over all pixels */
#pragma omp parallel for firstprivate(padarr, output, nx, ny, padnx, iter) \
    private(nxj, padnxj, i, j, p)
        for (j = 0; j < ny; j++) {
            nxj = nx * j;
            /* Note the + 2 padding in padnxj */
            padnxj = padnx * (j + 2);
            for (i = 0; i < nx; i++) {
                /* Start with the middle pixel and work out */
                p = padarr[i + 2 + padnxj];
                /* Right 1 */
                p = p || padarr[i + 3 + padnxj];
                /* Left 1 */
                p = p || padarr[i + 1 + padnxj];
                /* Up 1 */
                p = p || padarr[i + 2 + padnx + padnxj];
                /* Down 1 */
                p = p || padarr[i + 2 - padnx + padnxj];
                /* Up 1 Right 1 */
                p = p || padarr[i + 3 + padnx + padnxj];
                /* Up 1 Left 1 */
                p = p || padarr[i + 1 + padnx + padnxj];
                /* Down 1 Right 1 */
                p = p || padarr[i + 3 - padnx + padnxj];
                /* Down 1 Left 1 */
                p = p || padarr[i + 1 - padnx + padnxj];
                /* Right 2 */
                p = p || padarr[i + 4 + padnxj];
                /* Left 2 */
                p = p || padarr[i + padnxj];
                /* Up 2 */
                p = p || padarr[i + 2 + padnx + padnx + padnxj];
                /* Down 2 */
                p = p || padarr[i + 2 - padnx - padnx + padnxj];
                /* Right 2 Up 1 */
                p = p || padarr[i + 4 + padnx + padnxj];
                /* Right 2 Down 1 */
                p = p || padarr[i + 4 - padnx + padnxj];
                /* Left 2 Up 1 */
                p = p || padarr[i + padnx + padnxj];
                /* Left 2 Down 1 */
                p = p || padarr[i - padnx + padnxj];
                /* Up 2 Right 1 */
                p = p || padarr[i + 3 + padnx + padnx + padnxj];
                /* Up 2 Left 1 */
                p = p || padarr[i + 1 + padnx + padnx + padnxj];
                /* Down 2 Right 1 */
                p = p || padarr[i + 3 - padnx - padnx + padnxj];
                /* Down 2 Left 1 */
                p = p || padarr[i + 1 - padnx - padnx + padnxj];

                output[i + nxj] = p;

            }
        }

    }
    free(padarr);

    return;
}
