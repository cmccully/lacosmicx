/*
 * functions.h
 *
 *  Created on: May 27, 2011
 *      Author: cmccully
 */

#ifndef FUNCTION_H_
#define FUNCTION_H_

float median(float* a, int n);
float optmed25(float* a);
float optmed3(float* a);
float optmed7(float* a);
float optmed5(float* a);
float optmed9(float* a);
float* medfilt3(float* data, int nx, int ny);
float* medfilt5(float* data, int nx, int ny);
float* medfilt7(float* data, int nx, int ny);
float* sepmedfilt3(float* data, int nx, int ny);
float* sepmedfilt5(float* data, int nx, int ny);
float* sepmedfilt7(float* data, int nx, int ny);
float* sepmedfilt9(float* data, int nx, int ny);
bool* dilate(bool* data, int iter, int nx, int ny);
float* subsample(float* data, int nx, int ny);
float* laplaceconvolve(float* data, int nx, int ny);
float* rebin(float* data, int nx, int ny);
bool* growconvolve(bool* data, int nx, int ny);
float* fromfits(char* filename, bool verbose = true);
bool* boolfromfits(char* filename, bool verbose = true);
void tofits(char* filename, float* data, int nx, int ny,
		char* hdr = (char *) "", bool verbose = true);
void booltofits(char* filename, bool* data, int nx, int ny,
		char* hdr = (char *) "", bool verbose = true);
void printfitserror(int status);
#endif /* FUNCTION_H_ */
