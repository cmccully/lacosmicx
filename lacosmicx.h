/*
 * Lacosmicx.h
 *
 *  Created on: Apr 21, 2011
 *      Author: cmccully
 */

#ifndef LACOSMICX_H_
#define LACOSMICX_H_
#include<iostream>

class lacosmicx {

	/*
	 sigclip : increase this if you detect cosmics where there are none. Default is 5.0, a good value for earth-bound images.
	 objlim : increase this if normal stars are detected as cosmics. Default is 5.0, a good value for earth-bound images.

	 Constructor of the cosmic class, takes a 2D numpy array of your image as main argument.
	 sigclip : laplacian-to-noise limit for cosmic ray detection
	 objlim : minimum contrast between laplacian image and fine structure image. Use 5.0 if your image is undersampled, HST, ...

	 satlevel : if we find agglomerations of pixels above this level, we consider it to be a saturated star and
	 do not try to correct and pixels around it. A negative satlevel skips this feature.

	 pssl is the previously subtracted sky level !

	 real   gain    = 1.0          # gain (electrons/ADU)	(0=unknown)
	 real   readn   = 6.5		      # read noise (electrons) (0=unknown)
	 ##gain0  string statsec = "*,*"       # section to use for automatic computation of gain
	 real   skyval  = 0.           # sky level that has been subtracted (ADU)
	 real   sigclip = 4.5          # detection limit for cosmic rays (sigma)
	 real   sigfrac = 035          # fractional detection limit for neighbouring pixels
	 real   objlim  = 3.5           # contrast limit between CR and underlying object
	 int    niter   = 4            # maximum number of iterations

	 */

public:
	friend ostream &operator<<(ostream &out, lacosmicx *l);
	int nx;
	int ny;
	int nxny;
	float* data;
	float* cleanarr;
	bool* crmask;
	bool* mask;
	float pssl;
	float gain;
	float readnoise;
	float sigclip;
	float sigfrac;
	float objlim;
	float satlevel;
	float sigcliplow;
	float backgroundlevel;
	bool verbose;
	bool robust;
	lacosmicx(float* data, bool* mask, int nx, int ny, float pssl = 0.0,
			float gain = 1.0, float readnoise = 6.5, float sigclip = 4.5,
			float sigfrac = 0.3, float objlim = 5.0, float satlevel = 50000.0,
			bool robust = false, bool verbose = true);
	~lacosmicx();
	void run(int maxiter = 4);
	void findsatstars();
	int lacosmiciteration();
	void clean();
};

#endif /* LACOSMICX_H_ */
