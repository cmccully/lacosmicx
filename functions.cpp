/*
 * functions.c
 *
 *  Created on: May 27, 2011
 *      Author: cmccully
 */
using namespace std;
#include<malloc.h>
#include<stdlib.h>
#include<iostream>
#include<math.h>
#include"fitsio.h"
#include<string.h>
#include<stdio.h>
#include<omp.h>
#include "functions.h"

#define ELEM_SWAP(a,b) { float t=(a);(a)=(b);(b)=t; }

float median(float* a, int n) {
	/*
	 *  This Quickselect routine is based on the algorithm described in
	 *  "Numerical recipes in C", Second Edition,
	 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
	 *  This code by Nicolas Devillard - 1998. Public domain.
	 */

	//Make a copy of the array
	float* arr;
	arr = new float[n];
	float med;
	int i;
	for (i = 0; i < n; i++) {
		arr[i] = a[i];

	}
	int low, high;
	int median;
	int middle, ll, hh;

	low = 0;
	high = n - 1;
	median = (low + high) / 2;
	for (;;) {
		if (high <= low) { /* One element only */
			med = arr[median];
			delete[] arr;
			return med;
		}

		if (high == low + 1) { /* Two elements only */
			if (arr[low] > arr[high])
				ELEM_SWAP(arr[low], arr[high]);
			med = arr[median];
			delete[] arr;
			return med;
		}

		/* Find median of low, middle and high items; swap into position low */
		middle = (low + high) / 2;
		if (arr[middle] > arr[high])
			ELEM_SWAP(arr[middle], arr[high]);
		if (arr[low] > arr[high])
			ELEM_SWAP(arr[low], arr[high]);
		if (arr[middle] > arr[low])
			ELEM_SWAP(arr[middle], arr[low]);

		/* Swap low item (now in position middle) into position (low+1) */ELEM_SWAP(arr[middle], arr[low+1]);

		/* Nibble from each end towards middle, swapping items when stuck */
		ll = low + 1;
		hh = high;
		for (;;) {
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

		/* Swap middle item (in position low) back into correct position */ELEM_SWAP(arr[low], arr[hh]);

		/* Re-set active partition */
		if (hh <= median)
			low = ll;
		if (hh >= median)
			high = hh - 1;
	}

}

#undef ELEM_SWAP

/** All of the optimized median methods were written by Nicolas Devillard and are in public domain */
#define PIX_SORT(a,b) { if (a>b) PIX_SWAP(a,b); }
#define PIX_SWAP(a,b) { float temp=a;a=b;b=temp; }

/*----------------------------------------------------------------------------
 Function :   opt_med3()
 In       :   pointer to array of 3 pixel values
 Out      :   a pixelvalue
 Job      :   optimized search of the median of 3 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made
 on the nature of the input signal.
 ---------------------------------------------------------------------------*/

float optmed3(float* p) {
	PIX_SORT(p[0],p[1]);PIX_SORT(p[1],p[2]);PIX_SORT(p[0],p[1]);
	return (p[1]);
}

/*----------------------------------------------------------------------------
 Function :   opt_med5()
 In       :   pointer to array of 5 pixel values
 Out      :   a pixelvalue
 Job      :   optimized search of the median of 5 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made
 on the nature of the input signal.
 ---------------------------------------------------------------------------*/

float optmed5(float* p) {
	PIX_SORT(p[0],p[1]);PIX_SORT(p[3],p[4]);PIX_SORT(p[0],p[3]);
	PIX_SORT(p[1],p[4]);PIX_SORT(p[1],p[2]);PIX_SORT(p[2],p[3]);
	PIX_SORT(p[1],p[2]);
	return (p[2]);
}

/*----------------------------------------------------------------------------
 Function :   opt_med7()
 In       :   pointer to array of 7 pixel values
 Out      :   a pixelvalue
 Job      :   optimized search of the median of 7 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made
 on the nature of the input signal.
 ---------------------------------------------------------------------------*/

float optmed7(float* p) {
	PIX_SORT(p[0], p[5]);PIX_SORT(p[0], p[3]);PIX_SORT(p[1], p[6]);
	PIX_SORT(p[2], p[4]);PIX_SORT(p[0], p[1]);PIX_SORT(p[3], p[5]);
	PIX_SORT(p[2], p[6]);PIX_SORT(p[2], p[3]);PIX_SORT(p[3], p[6]);
	PIX_SORT(p[4], p[5]);PIX_SORT(p[1], p[4]);PIX_SORT(p[1], p[3]);
	PIX_SORT(p[3], p[4]);
	return (p[3]);
}
/*----------------------------------------------------------------------------
 Function :   opt_med9()
 In       :   pointer to an array of 9 pixelvalues
 Out      :   a pixelvalue
 Job      :   optimized search of the median of 9 pixelvalues
 Notice   :   in theory, cannot go faster without assumptions on the
 signal.
 Formula from:
 XILINX XCELL magazine, vol. 23 by John L. Smith

 The input array is modified in the process
 The result array is guaranteed to contain the median
 value
 in middle position, but other elements are NOT sorted.
 ---------------------------------------------------------------------------*/

float optmed9(float* p) {
	PIX_SORT(p[1], p[2]);PIX_SORT(p[4], p[5]);PIX_SORT(p[7], p[8]);
	PIX_SORT(p[0], p[1]);PIX_SORT(p[3], p[4]);PIX_SORT(p[6], p[7]);
	PIX_SORT(p[1], p[2]);PIX_SORT(p[4], p[5]);PIX_SORT(p[7], p[8]);
	PIX_SORT(p[0], p[3]);PIX_SORT(p[5], p[8]);PIX_SORT(p[4], p[7]);
	PIX_SORT(p[3], p[6]);PIX_SORT(p[1], p[4]);PIX_SORT(p[2], p[5]);
	PIX_SORT(p[4], p[7]);PIX_SORT(p[4], p[2]);PIX_SORT(p[6], p[4]);
	PIX_SORT(p[4], p[2]);
	return (p[4]);
}

/*----------------------------------------------------------------------------
 Function :   opt_med25()
 In       :   pointer to an array of 25 pixelvalues
 Out      :   a pixelvalue
 Job      :   optimized search of the median of 25 pixelvalues
 Notice   :   in theory, cannot go faster without assumptions on the
 signal.
 Code taken from Graphic Gems.
 ---------------------------------------------------------------------------*/

float optmed25(float* p) {

	PIX_SORT(p[0], p[1]);PIX_SORT(p[3], p[4]);PIX_SORT(p[2], p[4]);
	PIX_SORT(p[2], p[3]);PIX_SORT(p[6], p[7]);PIX_SORT(p[5], p[7]);
	PIX_SORT(p[5], p[6]);PIX_SORT(p[9], p[10]);PIX_SORT(p[8], p[10]);
	PIX_SORT(p[8], p[9]);PIX_SORT(p[12], p[13]);PIX_SORT(p[11], p[13]);
	PIX_SORT(p[11], p[12]);PIX_SORT(p[15], p[16]);PIX_SORT(p[14], p[16]);
	PIX_SORT(p[14], p[15]);PIX_SORT(p[18], p[19]);PIX_SORT(p[17], p[19]);
	PIX_SORT(p[17], p[18]);PIX_SORT(p[21], p[22]);PIX_SORT(p[20], p[22]);
	PIX_SORT(p[20], p[21]);PIX_SORT(p[23], p[24]);PIX_SORT(p[2], p[5]);
	PIX_SORT(p[3], p[6]);PIX_SORT(p[0], p[6]);PIX_SORT(p[0], p[3]);
	PIX_SORT(p[4], p[7]);PIX_SORT(p[1], p[7]);PIX_SORT(p[1], p[4]);
	PIX_SORT(p[11], p[14]);PIX_SORT(p[8], p[14]);PIX_SORT(p[8], p[11]);
	PIX_SORT(p[12], p[15]);PIX_SORT(p[9], p[15]);PIX_SORT(p[9], p[12]);
	PIX_SORT(p[13], p[16]);PIX_SORT(p[10], p[16]);PIX_SORT(p[10], p[13]);
	PIX_SORT(p[20], p[23]);PIX_SORT(p[17], p[23]);PIX_SORT(p[17], p[20]);
	PIX_SORT(p[21], p[24]);PIX_SORT(p[18], p[24]);PIX_SORT(p[18], p[21]);
	PIX_SORT(p[19], p[22]);PIX_SORT(p[8], p[17]);PIX_SORT(p[9], p[18]);
	PIX_SORT(p[0], p[18]);PIX_SORT(p[0], p[9]);PIX_SORT(p[10], p[19]);
	PIX_SORT(p[1], p[19]);PIX_SORT(p[1], p[10]);PIX_SORT(p[11], p[20]);
	PIX_SORT(p[2], p[20]);PIX_SORT(p[2], p[11]);PIX_SORT(p[12], p[21]);
	PIX_SORT(p[3], p[21]);PIX_SORT(p[3], p[12]);PIX_SORT(p[13], p[22]);
	PIX_SORT(p[4], p[22]);PIX_SORT(p[4], p[13]);PIX_SORT(p[14], p[23]);
	PIX_SORT(p[5], p[23]);PIX_SORT(p[5], p[14]);PIX_SORT(p[15], p[24]);
	PIX_SORT(p[6], p[24]);PIX_SORT(p[6], p[15]);PIX_SORT(p[7], p[16]);
	PIX_SORT(p[7], p[19]);PIX_SORT(p[13], p[21]);PIX_SORT(p[15], p[23]);
	PIX_SORT(p[7], p[13]);PIX_SORT(p[7], p[15]);PIX_SORT(p[1], p[9]);
	PIX_SORT(p[3], p[11]);PIX_SORT(p[5], p[17]);PIX_SORT(p[11], p[17]);
	PIX_SORT(p[9], p[17]);PIX_SORT(p[4], p[10]);PIX_SORT(p[6], p[12]);
	PIX_SORT(p[7], p[14]);PIX_SORT(p[4], p[6]);PIX_SORT(p[4], p[7]);
	PIX_SORT(p[12], p[14]);PIX_SORT(p[10], p[14]);PIX_SORT(p[6], p[7]);
	PIX_SORT(p[10], p[12]);PIX_SORT(p[6], p[10]);PIX_SORT(p[6], p[17]);
	PIX_SORT(p[12], p[17]);PIX_SORT(p[7], p[17]);PIX_SORT(p[7], p[10]);
	PIX_SORT(p[12], p[18]);PIX_SORT(p[7], p[12]);PIX_SORT(p[10], p[18]);
	PIX_SORT(p[12], p[20]);PIX_SORT(p[10], p[20]);PIX_SORT(p[10], p[12]);

	return (p[12]);
}

#undef PIX_SORT
#undef PIX_SWAP

/**
 * All of these median filters don't do anything to a border of pixels the size of the half width
 */
float* medfilt3(float* data, int nx, int ny) {
	/**
	 * To save on space and computation we just leave the border pixels alone. Most data has blank edges
	 * and funny things happen at the edges anyway so we don't worry too much about it.
	 *
	 */

	int i;
	int j;
	int nxj;
	int nxny = nx * ny;

	float* output;
	output = new float[nxny];
	int k, l, nxk;
	float* medarr;
	int counter;

#pragma omp parallel firstprivate(output,data,nx,ny) private(i,j,k,l,medarr,nxj,counter,nxk)
	{
		medarr = new float[9];

#pragma omp for nowait
		for (j = 1; j < ny - 1; j++) {
			nxj = nx * j;
			for (i = 1; i < nx - 1; i++) {

				counter = 0;
				for (k = -1; k < 2; k++) {
					nxk = nx * k;
					for (l = -1; l < 2; l++) {
						medarr[counter] = data[nxj + i + nxk + l];
						counter++;
					}
				}

				output[nxj + i] = optmed9(medarr);
			}
		}

		delete[] medarr;
	}

	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[nxny - nx + i] = data[nxny - nx + i];
	}
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + nx - 1] = data[nxj + nx - 1];
	}

	return output;
}

float* medfilt5(float* data, int nx, int ny) {
	/**
	 * To save on space and computation we just leave the border pixels alone. Most data has blank edges
	 * and funny things happen at the edges anyway so we don't worry too much about it.
	 *
	 */

	int i;
	int j;
	int nxj;
	int nxny = nx * ny;

	float* output;
	output = new float[nxny];
	int k, l, nxk;
	float* medarr;
	int counter;

#pragma omp parallel firstprivate(output,data,nx,ny) private(i,j,k,l,medarr,nxj,counter,nxk)
	{
		medarr = new float[25];

#pragma omp for nowait
		for (j = 2; j < ny - 2; j++) {
			nxj = nx * j;
			for (i = 2; i < nx - 2; i++) {

				counter = 0;
				for (k = -2; k < 3; k++) {
					nxk = nx * k;
					for (l = -2; l < 3; l++) {
						medarr[counter] = data[nxj + i + nxk + l];
						counter++;
					}
				}

				output[nxj + i] = optmed25(medarr);
			}
		}

		delete[] medarr;
	}

	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[i + nx] = data[i + nx];
		output[nxny - nx + i] = data[nxny - nx + i];
		output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
	}
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + 1] = data[nxj + 1];
		output[nxj + nx - 1] = data[nxj + nx - 1];
		output[nxj + nx - 2] = data[nxj + nx - 2];
	}

	return output;
}

float* medfilt7(float* data, int nx, int ny) {
	/**
	 * To save on space and computation we just leave the border pixels alone. Most data has blank edges
	 * and funny things happen at the edges anyway so we don't worry too much about it.
	 *
	 */

	int i;
	int j;
	int nxj;
	int nxny = nx * ny;

	float* output;
	output = new float[nxny];
	int k, l, nxk;
	float* medarr;
	int counter;

#pragma omp parallel firstprivate(output,data,nx,ny) private(i,j,k,l,medarr,nxj,counter,nxk)
	{
		medarr = new float[49];

#pragma omp for nowait
		for (j = 3; j < ny - 3; j++) {
			nxj = nx * j;
			for (i = 3; i < nx - 3; i++) {

				counter = 0;
				for (k = -3; k < 4; k++) {
					nxk = nx * k;
					for (l = -3; l < 4; l++) {
						medarr[counter] = data[nxj + i + nxk + l];
						counter++;
					}
				}

				output[nxj + i] = median(medarr, 49);
			}
		}

		delete[] medarr;
	}

	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[i + nx] = data[i + nx];
		output[i + nx + nx] = data[i + nx + nx];
		output[nxny - nx + i] = data[nxny - nx + i];
		output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
		output[nxny - nx - nx - nx + i] = data[nxny - nx - nx - nx + i];
	}
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + 1] = data[nxj + 1];
		output[nxj + 2] = data[nxj + 2];
		output[nxj + nx - 1] = data[nxj + nx - 1];
		output[nxj + nx - 2] = data[nxj + nx - 2];
		output[nxj + nx - 3] = data[nxj + nx - 3];
	}

	return output;
}



float* sepmedfilt3(float* data, int nx, int ny) {
	//Just ignore the borders, fill them with data as strange things happen along the edges anyway
	int nxny = nx * ny;

	float* rowmed;
	rowmed = new float[nxny];
	int i;
	int j;
	int nxj;

	//The median seperates so we can median the rows and then median the columns
	float* medarr;
#pragma omp parallel firstprivate(data,rowmed,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[3];

#pragma omp for nowait
		for (j = 0; j < ny; j++) {
			nxj = nx * j;
			for (i = 1; i < nx - 1; i++) {
				medarr[0] = data[nxj + i];
				medarr[1] = data[nxj + i - 1];
				medarr[2] = data[nxj + i + 1];
				rowmed[nxj + i] = optmed3(medarr);
			}
		}
		delete[] medarr;
	}

	float* output;
	output = new float[nxny];

#pragma omp parallel firstprivate(rowmed,output,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[3];

#pragma omp for nowait
		for (j = 1; j < ny - 1; j++) {
			nxj = nx * j;
			for (i = 1; i < nx - 1; i++) {

				medarr[0] = rowmed[i + nxj - nx];
				medarr[1] = rowmed[i + nxj + nx];
				medarr[2] = rowmed[i + nxj];
				output[nxj + i] = optmed3(medarr);
			}
		}
		delete medarr;
	}
	delete[] rowmed;
	//Fill up the skipped borders
#pragma omp parallel for firstprivate(output,nx,ny,nxny) private(i,j,nxj)
	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[nxny - nx + i] = data[nxny - nx + i];
	}
#pragma omp parallel for firstprivate(output,nx,ny) private(i,nxj)
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + nx - 1] = data[nxj + nx - 1];
	}

	return output;
}

float* sepmedfilt5(float* data, int nx, int ny) {
	//Just ignore the borders, fill them with data as strange things happen along the edges anyway
	int nxny = nx * ny;

	float* rowmed;
	rowmed = new float[nxny];
	int i;
	int j;
	int nxj;

	//The median seperates so we can median the rows and then median the columns
	float* medarr;
#pragma omp parallel firstprivate(data,rowmed,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[5];

#pragma omp for nowait
		for (j = 0; j < ny; j++) {
			nxj = nx * j;
			for (i = 2; i < nx - 2; i++) {
				medarr[0] = data[nxj + i];
				medarr[1] = data[nxj + i - 1];
				medarr[2] = data[nxj + i + 1];
				medarr[3] = data[nxj + i - 2];
				medarr[4] = data[nxj + i + 2];
				rowmed[nxj + i] = optmed5(medarr);
			}
		}
		delete[] medarr;
	}

	float* output;
	output = new float[nxny];

#pragma omp parallel firstprivate(rowmed,output,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[5];

#pragma omp for nowait
		for (j = 2; j < ny - 2; j++) {
			nxj = nx * j;
			for (i = 2; i < nx - 2; i++) {

				medarr[0] = rowmed[i + nxj - nx];
				medarr[1] = rowmed[i + nxj + nx];
				medarr[2] = rowmed[i + nxj + nx + nx];
				medarr[3] = rowmed[i + nxj - nx - nx];
				medarr[4] = rowmed[i + nxj];
				output[nxj + i] = optmed5(medarr);
			}
		}
		delete medarr;
	}
	delete[] rowmed;
	//Fill up the skipped borders
#pragma omp parallel for firstprivate(output,nx,ny,nxny) private(i,j,nxj)
	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[i + nx] = data[i + nx];
		output[nxny - nx + i] = data[nxny - nx + i];
		output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
	}
#pragma omp parallel for firstprivate(output,nx,ny) private(i,nxj)
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + 1] = data[nxj + 1];
		output[nxj + nx - 1] = data[nxj + nx - 1];
		output[nxj + nx - 2] = data[nxj + nx - 2];
	}

	return output;
}

float* sepmedfilt7(float* data, int nx, int ny) {
	//Just ignore the borders, fill them with data as strange things happen along the edges anyway
	int nxny = nx * ny;

	float* rowmed;
	rowmed = new float[nxny];
	int i;
	int j;
	int nxj;

	//The median separates so we can median the rows and then median the columns
	float* medarr;
#pragma omp parallel firstprivate(data,rowmed,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[7];

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
				rowmed[nxj + i] = optmed7(medarr);
			}
		}
		delete[] medarr;
	}

	float* output;
	output = new float[nxny];

#pragma omp parallel firstprivate(rowmed,output,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[9];

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
				output[nxj + i] = optmed7(medarr);
			}
		}
		delete medarr;
	}
	delete[] rowmed;
	//Fill up the skipped borders
#pragma omp parallel for firstprivate(output,nx,ny,nxny) private(i,j,nxj)
	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[i + nx] = data[i + nx];
		output[i + nx + nx] = data[i + nx + nx];
		output[nxny - nx + i] = data[nxny - nx + i];
		output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
		output[nxny - nx - nx - nx + i] = data[nxny - nx - nx - nx + i];
	}
#pragma omp parallel for firstprivate(output,nx,ny) private(i,nxj)
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + 1] = data[nxj + 1];
		output[nxj + 2] = data[nxj + 2];
		output[nxj + nx - 1] = data[nxj + nx - 1];
		output[nxj + nx - 2] = data[nxj + nx - 2];
		output[nxj + nx - 3] = data[nxj + nx - 3];
	}

	return output;
}

float* sepmedfilt9(float* data, int nx, int ny) {
	//Just ignore the borders, fill them with data as strange things happen along the edges anyway
	int nxny = nx * ny;

	float* rowmed;
	rowmed = new float[nxny];
	int i;
	int j;
	int nxj;

	//The median seperates so we can median the rows and then median the columns
	float* medarr;
#pragma omp parallel firstprivate(data,rowmed,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[9];

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
				rowmed[nxj + i] = optmed9(medarr);
			}
		}
		delete[] medarr;
	}

	float* output;
	output = new float[nxny];

#pragma omp parallel firstprivate(rowmed,output,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[9];

#pragma omp for nowait
		for (j = 4; j < ny - 4; j++) {
			nxj = nx * j;
			for (i = 4; i < nx - 4; i++) {

				medarr[0] = rowmed[i + nxj - nx];
				medarr[1] = rowmed[i + nxj + nx];
				medarr[2] = rowmed[i + nxj + nx + nx];
				medarr[3] = rowmed[i + nxj - nx - nx];
				medarr[4] = rowmed[i + nxj];
				medarr[5] = rowmed[i + nxj + nx + nx + nx];
				medarr[6] = rowmed[i + nxj - nx - nx - nx];
				medarr[7] = rowmed[i + nxj + nx + nx + nx + nx];
				medarr[8] = rowmed[i + nxj - nx - nx - nx - nx];
				output[nxj + i] = optmed9(medarr);
			}
		}
		delete medarr;
	}
	delete[] rowmed;
	//Fill up the skipped borders
#pragma omp parallel for firstprivate(output,nx,ny,nxny) private(i,j,nxj)
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
#pragma omp parallel for firstprivate(output,nx,ny) private(i,nxj)
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + 1] = data[nxj + 1];
		output[nxj + 2] = data[nxj + 2];
		output[nxj + 3] = data[nxj + 3];
		output[nxj + nx - 1] = data[nxj + nx - 1];
		output[nxj + nx - 2] = data[nxj + nx - 2];
		output[nxj + nx - 3] = data[nxj + nx - 3];
		output[nxj + nx - 4] = data[nxj + nx - 4];
	}

	return output;
}

float* subsample(float* data, int nx, int ny) {
	float* output;
	output = new float[4 * nx * ny];
	int padnx = 2 * nx;
	int i;
	int j;
	int nxj;
	int padnxj;
#pragma omp parallel for firstprivate(padnx,data,output,nx,ny) private(i,j,nxj,padnxj)
	for (j = 0; j < ny; j++) {
		nxj = nx * j;
		padnxj = 2 * padnx * j;
		for (i = 0; i < nx; i++) {
			output[2 * i + padnxj] = data[i + nxj];
			output[2 * i + padnxj + padnx] = data[i + nxj];
			output[2 * i + 1 + padnxj + padnx] = data[i + nxj];
			output[2 * i + 1 + padnxj] = data[i + nxj];
		}
	}

	return output;
}

bool* dilate(bool* data, int iter, int nx, int ny) {
	/**
	 * Here we do a boolean dilation of the image to connect the cosmic rays for the masks
	 * We use a kernel that looks like
	 * 0 1 1 1 0
	 * 1 1 1 1 1
	 * 1 1 1 1 1
	 * 1 1 1 1 1
	 * 0 1 1 1 0
	 *
	 * Since we have to do multiple iterations, this takes a little more memory.
	 * But it's bools so its ok.
	 */
	//Pad the array with a border of zeros

	int padnx = nx + 4;
	int padny = ny + 4;
	int padnxny = padnx * padny;
	int nxny = nx * ny;
	bool* padarr;
	padarr = new bool[padnxny];
	int i;
	for (i = 0; i < padnx; i++) {
		padarr[i] = false;
		padarr[i + padnx] = false;
		padarr[padnxny - padnx + i] = false;
		padarr[padnxny - padnx - padnx + i] = false;
	}
	for (i = 0; i < padny; i++) {

		padarr[padnx * i] = false;
		padarr[padnx * i + 1] = false;
		padarr[padnx * i + padnx - 1] = false;
		padarr[padnx * i + padnx - 2] = false;
	}

	bool* output;
	output = new bool[nxny];

	//Set the first iteration output array to the input data
	for (i = 0; i < nxny; i++) {
		output[i] = data[i];
	}

	int counter;
	int j;
	int nxj;
	int padnxj;
	for (counter = 0; counter < iter; counter++) {
#pragma omp parallel for firstprivate(padarr,output,nx,ny,padnx,padny,counter) private(nxj,padnxj,i,j)
		for (j = 0; j < ny; j++) {
			padnxj = padnx * j;
			nxj = nx * j;
			for (i = 0; i < nx; i++) {
				padarr[i + 2 + padnx + padnx + padnxj] = output[i + nxj];
			}
		}
#pragma omp parallel for firstprivate(padarr,output,nx,ny,padnx,padny,counter) private(nxj,padnxj,i,j)
		for (j = 0; j < ny; j++) {
			nxj = nx * j;
			padnxj = padnx * j;
			for (i = 0; i < nx; i++) {

				//Start in the middle and work out
				output[i + nxj] = padarr[i + 2 + padnx + padnx + padnxj] ||
				//right 1
						padarr[i + 3 + padnx + padnx + padnxj] ||
				//left 1
						padarr[i + 1 + padnx + padnx + padnxj] ||
				//up 1
						padarr[i + 2 + padnx + padnx + padnx + padnxj] ||
				//down 1
						padarr[i + 2 + padnx + padnxj] ||
				//up 1 right 1
						padarr[i + 3 + padnx + padnx + padnx + padnxj] ||
				//up 1 left 1
						padarr[i + 1 + padnx + padnx + padnx + padnxj] ||
				//down 1 right 1
						padarr[i + 3 + padnx + padnxj] ||
				//down 1 left 1
						padarr[i + 1 + padnx + padnxj] ||
				//right 2
						padarr[i + 4 + padnx + padnx + padnxj] ||
				//left 2
						padarr[i + padnx + padnx + padnxj] ||
				//up 2
						padarr[i + 2 + padnx + padnx + padnx + padnx + padnxj]
						||
						//down 2
						padarr[i + 2 + padnxj] ||
				//right 2 up 1
						padarr[i + 4 + padnx + padnx + padnx + padnxj] ||
				//right 2 down 1
						padarr[i + 4 + padnx + padnxj] ||
				//left 2 up 1
						padarr[i + padnx + padnx + padnx + padnxj] ||
				//left 2 down 1
						padarr[i + padnx + padnxj] ||
				//up 2 right 1
						padarr[i + 3 + padnx + padnx + padnx + padnx + padnxj]
						||
						//up 2 left 1
						padarr[i + 1 + padnx + padnx + padnx + padnx + padnxj]
						||
						//down 2 right 1
						padarr[i + 3 + padnxj] ||
				//down 2 left 1
						padarr[i + 1 + padnxj];

			}
		}

	}
	delete[] padarr;

	return output;
}
float* laplaceconvolve(float* data, int nx, int ny) {
	/*
	 * Here we do a short circuited convolution using the kernel
	 *  0 -1  0
	 * -1  4 -1
	 *  0 -1  0
	 */

	int nxny = nx * ny;
	float* output;
	output = new float[nxny];
	int i;
	int j;
	int nxj;
	//Do all but the edges that we will do explicitly to save memory.
#pragma omp parallel for firstprivate(nx,ny,nxny,output,data) private(i,j,nxj)
	for (j = 1; j < ny - 1; j++) {
		nxj = nx * j;
		for (i = 1; i < nx - 1; i++) {

			output[nxj + i] = 4.0 * data[nxj + i] - data[i + 1 + nxj] - data[i
					- 1 + nxj] - data[i + nxj + nx] - data[i + nxj - nx];
		}
	}

	//bottom row and top row
	for (i = 1; i < nx - 1; i++) {
		output[i] = 4.0 * data[i] - data[i + 1] - data[i - 1] - data[i + nx];
		output[i + nxny - nx] = 4.0 * data[i + nxny - nx] - data[i + 1 + nxny
				- nx] - data[i - 1] - data[i - nx + nxny - nx];
	}

	//first and last column
	for (j = 1; j < ny - 1; j++) {
		nxj = nx * j;
		output[nxj] = 4.0 * data[nxj] - data[nxj + 1] - data[nxj + nx]
				- data[nxj - nx];
		output[nxj + nx - 1] = 4.0 * data[nxj + nx - 1] - data[nxj + nx - 2]
				- data[nxj + nx + nx - 1] - data[nxj - 1];
	}

	//bottom left corner
	output[0] = 4.0 * data[0] - data[1] - data[nx];
	//bottom right corner
	output[nx - 1] = 4.0 * data[nx - 1] - data[nx - 2] - data[nx + nx - 1];
	//top left corner
	output[nxny - nx] = 4.0 * data[nxny - nx - 1] - data[nxny - nx] - data[nxny
			- 1 - nx - nx];
	//top right corner
	output[nxny - 1] = 4.0 * data[nxny - 1] - data[nxny - 2] - data[nxny - 1
			- nx];

	return output;
}

bool* growconvolve(bool* data, int nx, int ny) {
	/* This basically does a binary dilation with all ones in a 3x3 kernel:
	 * I have not decided if this is exactly equivalent or which is faster to calculate.
	 * In python this looks like
	 *  np.cast['bool'](signal.convolve2d(np.cast['float32'](cosmics), growkernel, mode="same", boundary="symm"))
	 * For speed and memory savings, I just convolve the whole image except the border. The border is just copied from the input image
	 * This is not technically correct, but it should be good enough.
	 */

	//Pad the array with a border of zeros
	int nxny = nx * ny;
	int i;
	int j;
	int nxj;
	bool* output;
	output = new bool[nxny];

#pragma omp parallel for firstprivate(output,data,nxny,nx,ny) private(i,j,nxj)
	for (j = 1; j < ny - 1; j++) {
		nxj = nx * j;

		for (i = 1; i < nx - 1; i++) {
			//Start in the middle and work out
			output[i + nxj] = data[i + nxj] ||
			//right 1
					data[i + 1 + nxj] ||
			//left 1
					data[i - 1 + nxj] ||
			//up 1
					data[i + nx + nxj] ||
			//down 1
					data[i - nx + nxj] ||
			//up 1 right 1
					data[i + 1 + nx + nxj] ||
			//up 1 left 1
					data[i - 1 + nx + nxj] ||
			//down 1 right 1
					data[i + 1 - nx + nxj] ||
			//down 1 left 1
					data[i - 1 - nx + nxj];

		}
	}

	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[nxny - nx + i] = data[nxny - nx + i];
	}
	for (j = 0; j < ny; j++) {
		nxj = nx * j;
		output[nxj] = data[nxj];
		output[nxj - 1 + nx] = data[nxj - 1 + nx];
	}

	return output;
}

float* rebin(float* data, int nx, int ny) {
	//Basically we want to do the opposite of subsample averaging the 4 pixels back down to 1
	//nx and ny are the final dimensions of the rebinned image
	float* output;
	output = new float[nx * ny];
	int padnx = nx * 2;
	int i;
	int j;
	int nxj;
	int padnxj;
#pragma omp parallel for firstprivate(output,data,nx,ny,padnx) private(i,j,nxj,padnxj)
	for (j = 0; j < ny; j++) {
		nxj = nx * j;
		padnxj = 2 * padnx * j;
		for (i = 0; i < nx; i++) {
			output[i + nxj] = (data[2 * i + padnxj] + data[2 * i + padnxj
					+ padnx] + data[2 * i + 1 + padnxj + padnx] + data[2 * i
					+ 1 + padnxj]) / 4.0;
		}
	}
	return output;
}

/*
 FITS import - export
 */
float* fromfits(char* filename, bool verbose) {
	/*
	 Reads a FITS file and returns a 1D array of floats.
	 Use hdu to specify which HDU you want (default = primary = 0)
	 **/
	int status = 0;
	fitsfile* infptr;
	if (fits_open_file(&infptr, filename, READONLY, &status)) {
		printfitserror(status);
	}
	long naxes[2] = { 1, 1 };
	int naxis = 2;
	int bitpix = -32;
	long fpixel[2] = { 1, 1 };
	if (fits_get_img_param(infptr, 2, &bitpix, &naxis, naxes, &status)) {
		printfitserror(status);
	}
	int nx = naxes[0];
	int ny = naxes[1];
	if (verbose) {
		cout << "FITS import shape : (" << nx << "," << ny << ")\n";
		cout << "FITS file BITPIX : " << bitpix << "\n";
	}

	float* data;
	data = new float[nx * ny];
	if (fits_read_pix(infptr, TFLOAT, fpixel, nx * ny, NULL, data, NULL,
			&status)) {
		printfitserror(status);
	}

	if (fits_close_file(infptr, &status)) {
		printfitserror(status);
	}
	return data;
}

bool* boolfromfits(char* filename, bool verbose) {
	/*
	 Reads a FITS file and returns a 1D array of floats.
	 Use hdu to specify which HDU you want (default = primary = 0)
	 **/
	int status = 0;
	fitsfile* infptr;
	if (fits_open_file(&infptr, filename, READONLY, &status)) {
		printfitserror(status);
	}
	long naxes[2] = { 1, 1 };
	int naxis = 2;
	int bitpix = -32;
	long fpixel[2] = { 1, 1 };
	if (fits_get_img_param(infptr, 2, &bitpix, &naxis, naxes, &status)) {
		printfitserror(status);
	}
	int nx = naxes[0];
	int ny = naxes[1];
	if (verbose) {
		cout << "FITS import shape : (" << nx << "," << ny << ")\n";
		cout << "FITS file BITPIX : " << bitpix << "\n";
	}

	bool* data;
	data = new bool[nx * ny];
	if (fits_read_pix(infptr, TBYTE, fpixel, nx * ny, NULL, data, NULL, &status)) {
		printfitserror(status);
	}

	if (fits_close_file(infptr, &status)) {
		printfitserror(status);
	}
	return data;
}
void tofits(char* filename, float *data, int nx, int ny, char* hdr,
		bool verbose) {
	/*
	 Takes a 1D  array and write it into a FITS file.
	 Pass the filename with an ! to clobber a file
	 If you specify a header file, the header will be copied from the header file to the new file
	 */

	if (verbose) {
		cout << "FITS export shape : (" << nx << "," << ny << ")\n";
	}
	int status = 0;
	long naxis = 2; //2-D image
	long naxes[2] = { nx, ny };
	int i, nkeys;
	char card[81];
	//Create the new file
	remove(filename);
	fitsfile* outfptr;
	if (fits_create_file(&outfptr, filename, &status)) {
		printfitserror(status);
	}

	if (fits_create_img(outfptr, FLOAT_IMG, naxis, naxes, &status)) {
		printfitserror(status);
	}
	if (strcmp(hdr, "") != 0) {
		if (verbose) {
			cout << "Copying header from " << hdr << "\n";
		}
		fitsfile* infptr;
		if (fits_open_file(&infptr, hdr, READONLY, &status)) {
			printfitserror(status);
		}
		//Copy the header keywords that are not the structure keywords
		if (fits_get_hdrspace(infptr, &nkeys, NULL, &status)) {
			printfitserror(status);
		}
		for (i = 1; i <= nkeys; i++) {
			fits_read_record(infptr, i, card, &status);
			if (fits_get_keyclass(card) > TYP_CMPRS_KEY) {
				fits_write_record(outfptr, card, &status);
			}
		}
		if (fits_close_file(infptr, &status)) {
			printfitserror(status);
		}
	}
	if (fits_write_img(outfptr, TFLOAT, 1, nx * ny, data, &status)) {
		printfitserror(status);
	}
	if (fits_close_file(outfptr, &status)) {
		printfitserror(status);
	}

	if (verbose) {
		cout << "Wrote " << filename << "\n";
	}
}
void booltofits(char* filename, bool* data, int nx, int ny, char* hdr,
		bool verbose) {
	/*
	 Takes a 1D  array and write it into a FITS file.
	 Pass the filename with an ! to clobber a file
	 If you specify a header file, the header will be copied from the header file to the new file
	 You can give me boolean arrays, I will convert them into shorts.
	 */

	int status = 0;
	long naxis = 2; //2-D image
	long naxes[2] = { nx, ny };
	int i, nkeys;
	char card[81];

	remove(filename);
	//Create the new file
	fitsfile* outfptr;

	if (fits_create_file(&outfptr, filename, &status)) {
		printfitserror(status);
	}

	if (fits_create_img(outfptr, SBYTE_IMG, naxis, naxes, &status)) {
		printfitserror(status);
	}

	if (strcmp(hdr, "") != 0) {
		if (verbose) {
			cout << "Copying header from " << hdr << "\n";
		}
		fitsfile* infptr;
		if (fits_open_file(&infptr, hdr, READONLY, &status)) {
			printfitserror(status);
		}
		//Copy the header keywords that are not the structure keywords

		if (fits_get_hdrspace(infptr, &nkeys, NULL, &status)) {
			printfitserror(status);
		}
		for (i = 1; i <= nkeys; i++) {
			fits_read_record(infptr, i, card, &status);
			if (fits_get_keyclass(card) > TYP_CMPRS_KEY) {
				fits_write_record(outfptr, card, &status);
			}
		}
		if (fits_close_file(infptr, &status)) {
			printfitserror(status);
		}
	}
	if (fits_write_img(outfptr, TBYTE, 1, nx * ny, data, &status)) {
		printfitserror(status);
	}

	if (fits_close_file(outfptr, &status)) {
		printfitserror(status);
	}

	if (verbose) {
		cout << "Wrote " << filename << "\n";
	}

}
void printfitserror(int status) {
	/*****************************************************/
	/* Print out cfitsio error messages and exit program */
	/*****************************************************/

	if (status) {
		fits_report_error(stderr, status); /* print error report */

		exit(status); /* terminate the program, returning error status */
	}
	return;
}

