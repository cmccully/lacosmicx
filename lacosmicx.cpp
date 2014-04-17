//============================================================================
// Name        : Lacosmicx.cpp
// Author      : Curtis McCully
// Version     :
// Copyright   : 
// Description : Lacosmic Written in C++
//============================================================================

using namespace std;
#include "lacosmicx.h"
#include<malloc.h>
#include<stdlib.h>
#include<iostream>
#include<math.h>
#include"fitsio.h"
#include<string.h>
#include<stdio.h>
#include "functions.h"

/*
 * Lacosmicx.cpp
 *
 *  Created on: Apr 19, 2011
 *      Author: cmccully
 */

//#include "Lacosmicx.h"

/*
 About
 =====

 lacosmicx is designed to detect and clean cosmic ray hits on images (numpy arrays or FITS), using scipy, and based on Pieter van Dokkum's L.A.Cosmic algorithm.

 Most of this code was directly adapted from cosmics.py written by Malte Tewes. I have removed some of the extras that he wrote, ported everything to c++, and optimized any places that I can.
 This is designed to be as fast as possible so some of the readability has been sacrificed.

 L.A.Cosmic = Laplacian cosmic ray detection

 U{http://www.astro.yale.edu/dokkum/lacosmic/}

 (article : U{http://arxiv.org/abs/astro-ph/0108003})


 Differences from original LA-cosmic
 ===================

 - Automatic recognition of saturated stars, including their full saturation trails.
 This avoids that such stars are treated as big cosmics.
 Indeed saturated stars tend to get even uglier when you try to clean them. Plus they
 keep L.A.Cosmic iterations going on forever.
 This feature is mainly for pretty-image production. It is optional, requires one more parameter (a CCD saturation level in ADU), and uses some
 nicely robust morphology operations and object extraction.


 -I have tried to optimize all of the code as much as possible while maintaining the integrity of the algorithm.

 -This implementation is much faster than the Python or Iraf versions by ~factor of 7.

 - In cfitsio, data are striped along x dimen, thus all loops
 are y outer, x inner.  or at least they should be...
 Usage
 =====


 Todo
 ====

 Curtis McCully April 2011


 __version__ = '1.0'


 sigclip : increase this if you detect cosmics where there are none. Default is 5.0, a good value for earth-bound images.
 objlim : increase this if normal stars are detected as cosmics. Default is 5.0, a good value for earth-bound images.

 Constructor of the cosmic class, takes a 2D numpy array of your image as main argument.
 sigclip : laplacian-to-noise limit for cosmic ray detection
 objlim : minimum contrast between laplacian image and fine structure image. Use 5.0 if your image is undersampled, HST, ...

 satlevel : if we find an agglomeration of pixels above this level, we consider it to be a saturated star and
 do not try to correct and pixels around it.This is given in electrons

 pssl is the previously subtracted sky level !

 real   gain    = 1.0         # gain (electrons/ADU)
 real   readn   = 6.5		      # read noise (electrons)
 real   skyval  = 0.           # sky level that has been subtracted (ADU)
 real   sigclip = 4.5          # detection limit for cosmic rays (sigma)
 real   sigfrac = 0.3          # fractional detection limit for neighboring pixels
 real   objlim  = 5.0           # contrast limit between CR and underlying object
 int    niter   = 4            # maximum number of iterations

 */
int main(int argc, char* argv[]) {

	int iarg;
	int i;
	char* infile;
	char* outfile;
	char* outmask;
	char* maskfile;
	infile = NULL;
	outfile = NULL;
	outmask = NULL;
	maskfile = NULL;
	float sigclip = 4.5;
	float sigfrac = 0.3;
	float objlim = 5.0;
	float pssl = 0.0;
	float gain = 1.0;
	float readnoise = 6.5;
	bool verbose = false;
	float satlevel = 50000.0;
	int niter = 4;
	bool robust = false;
	int nx;
	int ny;
	int nxny;
	float* indat;
	bool* maskdat;
	if (argc == 1) {
		cout << "Usage: Lacosmicx -options\n";
		cout << "-infile    : Input Image Filename (Required)\n";
		cout
				<< "-inmask    : Input Mask Image Filename: Bad Pixels; Saturated Stars are detected automatically\n";
		cout << "-outfile   : Output Image Filename\n";
		cout << "-outmask   : Output Cosmic Ray Mask Image Filename\n";
		cout
				<< "-sigclip   : Detection Limit for Cosmic Rays (sigma): Default(4.5)\n";
		cout
				<< "-sigfrac   : Fractional Detection Limit for Neighboring Pixels: Default(0.3)\n";
		cout
				<< "-objlim    : Contrast Level Between Cosmic Rays and Underlying Objects: Default(5.0)\n";
		cout << "-gain      : Gain: Electrons/ADU: Default(1.0)\n";
		cout
				<< "-pssl      : Previously Subtracted Level: ADU: Default(0.0) \n";
		cout << "-readnoise : Read Noise: Electrons: Default(6.5)\n";
		cout << "-satlevel  : Saturation Level: Electrons: Default(50000.0)\n";
		cout << "-niter     : Number of Lacosmic Iterations: Default(4)\n";
		cout
				<< "-robust	: Use the true median instead of the separable median filter: Default(False)\n";
		cout
				<< "Note that the true median is much slower to calculate (factor of ~3), but more robustly flags cosmic rays without flagging real stars\n";
		cout << "-verbose   : Verbose: Default(False)\n";

		exit(0);
	}
	/* read in command options. j counts # of required args given */
	for (iarg = 1; iarg < argc; iarg++) {
		if (argv[iarg][0] == '-') {
			if (strcasecmp(argv[iarg] + 1, "infile") == 0) {
				infile = argv[++iarg];
			} else if (strcasecmp(argv[iarg] + 1, "outfile") == 0) {
				outfile = argv[++iarg];
			} else if (strcasecmp(argv[iarg] + 1, "outmask") == 0) {
				outmask = argv[++iarg];
			} else if (strcasecmp(argv[iarg] + 1, "sigclip") == 0) {
				sigclip = atof(argv[++iarg]);
			} else if (strcasecmp(argv[iarg] + 1, "sigfrac") == 0) {
				sigfrac = atof(argv[++iarg]);
			} else if (strcasecmp(argv[iarg] + 1, "pssl") == 0) {
				pssl = atof(argv[++iarg]);
			} else if (strcasecmp(argv[iarg] + 1, "gain") == 0) {
				gain = atof(argv[++iarg]);
			} else if (strcasecmp(argv[iarg] + 1, "inmask") == 0) {
				maskfile = argv[++iarg];
			} else if (strcasecmp(argv[iarg] + 1, "objlim") == 0) {
				objlim = atof(argv[++iarg]);
			} else if (strcasecmp(argv[iarg] + 1, "readnoise") == 0) {
				readnoise = atof(argv[++iarg]);
			} else if (strcasecmp(argv[iarg] + 1, "satlevel") == 0) {
				satlevel = atof(argv[++iarg]);
			} else if (strcasecmp(argv[iarg] + 1, "verbose") == 0) {
				verbose = true;
			} else if (strcasecmp(argv[iarg] + 1, "robust") == 0) {
				robust = true;
			} else if (strcasecmp(argv[iarg] + 1, "niter") == 0) {
				niter = atoi(argv[++iarg]);
			} else {
				fprintf(stderr, "Unknown option : %s\n", argv[iarg]);
				exit(1);
			}
		} else {
			fprintf(stderr,
					"Unexpected string encountered on command line : %s\n",
					argv[iarg]);
			exit(1);
		}
	}
	if (infile == NULL) {
		fprintf(stderr, "You must include the input file!\n");
		exit(1);
	} else {
		//Get nx and ny
		int status = 0;
		fitsfile* infptr;
		if (fits_open_file(&infptr, infile, READONLY, &status)) {
			printfitserror(status);
		}
		long naxes[2] = { 1, 1 };
		int naxis = 2;
		int bitpix = -32;
		if (fits_get_img_param(infptr, 2, &bitpix, &naxis, naxes, &status)) {
			printfitserror(status);
		}
		nx = naxes[0];
		ny = naxes[1];
		nxny = nx * ny;
		if (fits_close_file(infptr, &status)) {
			printfitserror(status);
		}
		indat = fromfits(infile);
	}

	if (maskfile == NULL) {
		//By default don't mask anything
		maskdat = new bool[nxny];
		for (i = 0; i < nxny; i++) {
			maskdat[i] = false;
		}
	} else {
		maskdat = boolfromfits(maskfile);
	}

	lacosmicx* l;
	l = new lacosmicx(indat, maskdat, nx, ny, pssl, gain, readnoise, sigclip,
			sigfrac, objlim, satlevel, robust, verbose);
	l->run(niter);
	cout << l;
	if (outfile != NULL) {
		tofits(outfile, l->cleanarr, nx, ny);
	}

	if (outmask != NULL) {
		booltofits(outmask, l->crmask, nx, ny);
	}
	//explicitly delete here so we don't have to derefence the arrays in the python call
	delete[] l->data;
	delete[] l->mask;
	delete l;
}
lacosmicx::lacosmicx(float* data, bool* mask, int nx, int ny, float pssl,
		float gain, float readnoise, float sigclip, float sigfrac,
		float objlim, float satlevel, bool robust, bool verbose) {

	int i;
	this->nx = nx;
	this->ny = ny;
	int this_nxny = nx * ny;
	this->nxny = this_nxny;

#pragma omp parallel for firstprivate(this_nxny,data,gain,pssl) private(i)
	for (i = 0; i < this_nxny; i++) {
		// internally, we will always work "with sky" and in electrons, not ADU (gain=1)
		data[i] += pssl;
		data[i] *= gain;
	}

	//This data mask needs to be indexed with (i,j) -> (nx *j+i)
	this->data = data;
	this->gain = gain;
	this->readnoise = readnoise;
	this->sigclip = sigclip;
	this->objlim = objlim;
	this->sigcliplow = sigclip * sigfrac;
	this->satlevel = satlevel;

	this->verbose = verbose;
	this->robust = robust;
	this->pssl = pssl;

	//A mask of saturated stars and pixels with no data
	this->mask = mask;
	//Calculate a default background level, take into account the mask
	//This background level is used for large cosmics
	int ngoodpix = 0;
#pragma omp parallel for firstprivate(this_nxny,mask) private(i) reduction(+ : ngoodpix)
	for (i = 0; i < this_nxny; i++) {
		if (!mask[i]) {
			ngoodpix++;
		}
	}
	int goodcounter = 0;
	float *gooddata = new float[ngoodpix];
	for (i = 0; i < ngoodpix; i++) {
		if (!mask[i]) {
			gooddata[goodcounter] = data[i];
			goodcounter++;
		}
	}
	backgroundlevel = median(gooddata, ngoodpix);
	delete[] gooddata;

	cleanarr = new float[nxny];
	float* this_cleanarr;
	this_cleanarr = cleanarr;
#pragma omp parallel for firstprivate(this_cleanarr,this_nxny,data) private(i)
	for (i = 0; i < this_nxny; i++) {
		this_cleanarr[i] = data[i]; // In lacosmiciteration() we work on this guy
	}

	crmask = new bool[nxny];
	bool *this_crmask;
	this_crmask = crmask;
#pragma omp parallel for firstprivate(this_nxny,this_crmask) private(i)
	for (i = 0; i < this_nxny; i++) {
		// All False, no cosmics yet
		this_crmask[i] = false;
	}
}

lacosmicx::~lacosmicx() {
	//Don't delete mask and data so that we can use them in the python call
	//Deleting either of those messes things up.
	delete[] crmask;
	delete[] cleanarr;
}

ostream& operator<<(ostream& out, lacosmicx *l) {
	/*
	 Gives a summary of the current state, including the number of cosmic pixels in the mask etc.
	 */
	out << "Input array: (" << l->nx << "," << l->ny << ")\n";
	int i;
	int crsum = 0;
	int masksum = 0;
	int this_nxny = l->nx * l->ny;
	bool* this_crmask;
	this_crmask = l->crmask;
	bool* this_mask;
	this_mask = l->mask;
#pragma omp parallel for reduction(+ : crsum) reduction(+ : masksum) firstprivate(this_nxny,this_crmask,this_mask) private(i)
	for (i = 0; i < this_nxny; i++) {
		if (this_crmask[i]) {
			crsum++;
		}
		if (this_mask[i]) {
			masksum++;
		}
	}
	out << "Current cosmic ray mask : " << crsum << " pixels \n";

	out << "Using a previously subtracted sky level of " << l->pssl << "\n";

	out << "Median Sky Level: " << l->backgroundlevel << "\n";

	out << "Saturated Stars and Masked Data: " << masksum << " pixels \n";

	return out;
}

void lacosmicx::run(int maxiter) {
	/*
	 Full artillery :-)
	 - Find saturated stars
	 - Run maxiter L.A.Cosmic iterations (stops if no more cosmics are found)
	 */

	findsatstars();
	int i;
	cout << "Starting " << maxiter << " L.A.Cosmic iterations \n";
	for (i = 0; i < maxiter; i++) {
		cout << "Iteration " << i + 1 << "\n";

		//Detect the cosmic rays
		int ncrpix;
		ncrpix = lacosmiciteration();
		cout << ncrpix << " cosmic pixels\n";

		//If we didn't find anything, we're done.
		if (ncrpix == 0) {
			break;
		}
	}
	//Convert back to ADU and subtract the sky again
	int this_nxny = nxny;
	float this_gain = gain;
	float this_pssl = pssl;
	float* this_data;
	this_data = data;
	float* this_cleanarr;
	this_cleanarr = cleanarr;
#pragma omp parallel for firstprivate(this_nxny,this_gain,this_pssl,this_data,this_cleanarr)
	for (i = 0; i < this_nxny; i++) {
		this_data[i] /= this_gain;
		this_data[i] -= this_pssl;
		this_cleanarr[i] /= this_gain;
		this_cleanarr[i] -= this_pssl;
	}
	cout << "Finished!\n";
}

void lacosmicx::findsatstars() {
	/*
	 Uses the satlevel to find saturated stars (not cosmics !), and puts the result as a mask in self.satstars.
	 This can then be used to avoid these regions in cosmic detection and cleaning procedures.
	 */

	if (verbose) {
		cout << "Detecting saturated stars\n";
	}
	// DETECTION

	//Find all of the saturated pixels
	bool* satpixels;
	satpixels = new bool[nxny];

	int i;
	int this_nxny = nxny;
	float *this_data;
	this_data = data;
	float this_satlevel = satlevel;
#pragma omp parallel for firstprivate(this_nxny,this_data,this_satlevel,satpixels) private(i)
	for (i = 0; i < this_nxny; i++) {
		satpixels[i] = this_data[i] > this_satlevel;
	}

	//in an attempt to avoid saturated cosmic rays we try prune the saturated stars using the large scale structure
	float* m5;
	if (robust) {
		m5 = medfilt5(data, nx, ny);
	} else {
		m5 = sepmedfilt5(data, nx, ny);
	}
	//This mask will include saturated pixels and masked pixels


#pragma omp parallel for firstprivate(this_nxny,this_satlevel,m5,satpixels) private(i)
	for (i = 0; i < this_nxny; i++) {
		satpixels[i] = satpixels[i] && (m5[i] > this_satlevel / 10.0);
	}
	delete[] m5;

	if (verbose) {
		cout << "Building mask of saturated stars\n";
	}

	// BUILDING THE MASK

	//Combine the saturated pixels with the given input mask
	//Grow the input mask by one pixel to make sure we cover bad pixels
	bool* grow_mask;
	grow_mask = growconvolve(mask, nx, ny);

	//We want to dilate both the mask and the saturated stars to remove false detections along the edges of the mask
	bool* dilsatpixels;
	dilsatpixels = dilate(satpixels, 2, nx, ny);
	delete[] satpixels;
	bool* this_mask;
	this_mask = mask;
#pragma omp parallel for firstprivate(this_nxny,this_mask,dilsatpixels,grow_mask) private(i)
	for (i = 0; i < this_nxny; i++) {
		this_mask[i] = dilsatpixels[i] || grow_mask[i];
	}
	delete[] dilsatpixels;
	delete[] grow_mask;
	if (verbose) {
		cout << "Mask of saturated stars done\n";
	}

}

int lacosmicx::lacosmiciteration() {
	/*
	 Performs one iteration of the L.A.Cosmic algorithm.
	 It operates on cleanarray, and afterwards updates crmask by adding the newly detected
	 cosmics to the existing crmask. Cleaning is not done automatically ! You have to call
	 clean() after each iteration.
	 This way you can run it several times in a row to to L.A.Cosmic "iterations".
	 See function lacosmic, that mimics the full iterative L.A.Cosmic algorithm.

	 Returns numcr : the number of cosmic pixels detected in this iteration

	 */

	if (verbose) {
		cout << "Convolving image with Laplacian kernel\n";
	}

	// We subsample, convolve, clip negative values, and rebin to original size
	float* subsam;
	subsam = subsample(cleanarr, nx, ny);

	float* conved;
	conved = laplaceconvolve(subsam, 2 * nx, 2 * ny);
	delete[] subsam;
	int this_nxny = nxny;
	int i;
	int nxny4 = 4 * nxny;
#pragma omp parallel for firstprivate(nxny4,conved) private(i)
	for (i = 0; i < nxny4; i++) {
		if (conved[i] < 0.0) {
			conved[i] = 0.0;
		}
	}

	float* s;
	s = rebin(conved, nx, ny);
	delete[] conved;

	if (verbose) {
		cout << "Creating noise model\n";
	}

	// We build a custom noise map, to compare the laplacian to

	float* m5;
	if (robust) {
		m5 = medfilt5(cleanarr, nx, ny);
	} else {
		m5 = sepmedfilt7(cleanarr, nx, ny);
	}
	float* noise;
	noise = new float[nxny];

	float this_noise;
	float this_readnoise = readnoise;
#pragma omp parallel for firstprivate(this_nxny,this_readnoise,m5,noise) private(i,this_noise)
	for (i = 0; i < this_nxny; i++) {
		// We clip noise so that we can take a square root
		m5[i] < 0.0001 ? this_noise = 0.0001 : this_noise = m5[i];
		noise[i] = sqrt(this_noise + this_readnoise * this_readnoise);
	}
	delete[] m5;
	if (verbose) {
		cout << "Calculating Laplacian signal to noise ratio\n";
	}
	// Laplacian signal to noise ratio :

#pragma omp parallel for firstprivate(this_nxny,noise,s) private(i)
	for (i = 0; i < this_nxny; i++) {
		s[i] = s[i] / (2.0 * noise[i]);
		// the 2.0 is from the 2x2 subsampling
		// This s is called sigmap in the original lacosmic.cl
	}

	float* sp;
	if (robust) {
		sp = medfilt5(s, nx, ny);
	} else {
		sp = sepmedfilt7(s, nx, ny);
	}

	// We remove the large structures (s prime) :
#pragma omp parallel for firstprivate(this_nxny,s,sp) private(i)
	for (i = 0; i < this_nxny; i++) {
		sp[i] = s[i] - sp[i];
	}
	delete[] s;
	if (verbose) {
		cout
				<< "Selecting candidate cosmic rays, excluding saturated stars and bad pixels\n";
	}

	// We build the fine structure image :
	float* m3;
	if (robust) {
		m3 = medfilt3(cleanarr, nx, ny);
	} else {
		m3 = sepmedfilt5(cleanarr, nx, ny);
	}
	float* f;
	if (robust) {
		f = medfilt7(m3, nx, ny);
	} else {
		f = sepmedfilt9(m3, nx, ny);
	}

#pragma omp parallel for firstprivate(this_nxny,f,m3,noise) private(i)
	for (i = 0; i < this_nxny; i++) {
		f[i] = (m3[i] - f[i]) / noise[i];
		if (f[i] < 0.01) {
			// as we will divide by f. like in the iraf version.
			f[i] = 0.01;
		}
	}

	delete[] m3;
	delete[] noise;

	//Comments from Malte Tewes
	// In the article that's it, but in lacosmic.cl f is divided by the noise...
	// Ok I understand why, it depends on if you use sp/f or L+/f as criterion.
	// There are some differences between the article and the iraf implementation.
	// So we will stick to the iraf implementation.

	if (verbose) {
		cout << "Removing suspected compact bright objects\n";
	}
	// Now we have our better selection of cosmics :
	// Note the sp/f and not lplus/f ... due to the f = f/noise above.
	bool* cosmics;
	cosmics = new bool[nxny];
	float this_sigclip = sigclip;
	bool* this_mask;
	this_mask = mask;
	float this_objlim = objlim;
#pragma omp parallel for firstprivate(cosmics,sp,f,this_objlim,this_sigclip,this_nxny,this_mask) private(i)
	for (i = 0; i < this_nxny; i++) {
		cosmics[i] = (sp[i] > this_sigclip) && !this_mask[i] && ((sp[i] / f[i])
				> this_objlim);
	}

	delete[] f;

	// What follows is a special treatment for neighbors, with more relaxed constraints.
	// We grow these cosmics a first time to determine the immediate neighborhood  :
	bool* growcosmics;
	growcosmics = growconvolve(cosmics, nx, ny);

	delete[] cosmics;
	// From this grown set, we keep those that have sp > sigmalim
	// so obviously not requiring sp/f > objlim, otherwise it would be pointless
	//This step still feels pointless to me, but we leave it in because the iraf implementation has it
#pragma omp parallel for firstprivate(this_nxny,sp,growcosmics,this_mask,this_sigclip) private(i)
	for (i = 0; i < this_nxny; i++) {
		growcosmics[i] = (sp[i] > this_sigclip) && growcosmics[i]
				&& !this_mask[i];
	}

	// Now we repeat this procedure, but lower the detection limit to sigmalimlow :
	bool* finalsel;
	finalsel = growconvolve(growcosmics, nx, ny);
	delete[] growcosmics;

	//Our CR counter
	int numcr = 0;
	float this_sigcliplow = sigcliplow;
#pragma omp parallel for reduction(+ : numcr) firstprivate(finalsel,sp,this_sigcliplow,this_nxny,this_mask) private(i)
	for (i = 0; i < this_nxny; i++) {
		finalsel[i] = (sp[i] > this_sigcliplow) && finalsel[i]
				&& (!this_mask[i]);
		if (finalsel[i]) {
			numcr++;
		}
	}
	delete[] sp;
	if (verbose) {
		cout << numcr << " pixels detected as cosmics\n";
	}
	// We update the crmask with the cosmics we have found :
	bool* this_crmask;
	this_crmask = crmask;
#pragma omp parallel for firstprivate(this_crmask,finalsel,this_nxny) private(i)
	for (i = 0; i < this_nxny; i++) {
		this_crmask[i] = this_crmask[i] || finalsel[i];
	}

	// Now the replacement of the cosmics...
	// we outsource this to the function clean(), as for some purposes the cleaning might not even be needed.
	// Easy way without masking would be :
	//self.cleanarray[finalsel] = m5[finalsel]
	//Go through and clean the image using a masked mean filter, we outsource this to the clean method
	clean();

	delete[] finalsel;
	// We return the number of cr pixels
	// (used by function lacosmic)

	return numcr;
}

void lacosmicx::clean() {
	//Go through all of the pixels, ignore the borders
	int i;
	int j;
	int nxj;
	int idx;
	int k, l;
	int nxl;
	float sum;
	int numpix;

	int this_nx = nx;
	int this_ny = ny;
	bool* this_crmask;
	this_crmask = crmask;
	float* this_cleanarr;
	this_cleanarr = cleanarr;
	float this_backgroundlevel = backgroundlevel;
#pragma omp parallel for firstprivate(this_nx,this_ny,this_crmask,this_cleanarr,this_backgroundlevel) private(i,j,nxj,idx,numpix,sum,nxl,k,l)
	for (j = 2; j < this_ny - 2; j++) {
		nxj = this_nx * j;
		for (i = 2; i < this_nx - 2; i++) {
			//if the pixel is in the crmask
			idx = nxj + i;
			if (this_crmask[idx]) {
				numpix = 0;
				sum = 0.0;
				//sum the 25 pixels around the pixel ignoring any pixels that are masked

				for (l = -2; l < 3; l++) {
					nxl = this_nx * l;
					for (k = -2; k < 3; k++) {
						if (!this_crmask[idx + k + nxl]) {
							sum += this_cleanarr[idx + k + nxl];
							numpix++;
						}
					}

				}

				//if the pixels count is 0 then put in the background of the image
				if (numpix == 0) {
					sum = this_backgroundlevel;
				} else {
					//else take the mean
					sum /= float(numpix);
				}
				this_cleanarr[idx] = sum;
			}
		}
	}
}
