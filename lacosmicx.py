'''
Created on May 26, 2011

@author: cmccully
'''
import _lacosmicx
import numpy
def run(inmat,inmask=None,outmaskfile="",sigclip=3.0,objlim=5.0,sigfrac=0.1,satlevel=50000.0,gain=1.0,pssl=0.0,readnoise=6.5,robust=False,verbose=False,niter=4):
    # .... Check arguments, double NumPy matrices?
    test=numpy.zeros((2,2)) # create a NumPy matrix as a test object to check matin
    typetest= type(test) # get its type  
    if type(inmat) != typetest:
        raise 'In inmat, matrix argument is not *NumPy* array'
    if inmask is None:
        inmask=numpy.zeros(inmat.shape)
    if type(inmask) != typetest:
        raise 'In inmask, matrix argument is not *NumPy* array'
    this_shape=inmat.shape
    nx=this_shape[1]
    ny=this_shape[0]
    inmat=numpy.cast["float32"](inmat.ravel())
    inmask=numpy.cast["bool"](inmask.ravel())
    outmat=_lacosmicx.pyrun(inmat,inmask,outmaskfile,nx,ny,sigclip,objlim,sigfrac,satlevel,gain,pssl,readnoise,robust,verbose,niter)
    return outmat.reshape(this_shape)