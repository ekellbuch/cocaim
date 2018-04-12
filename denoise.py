import spatial_filtering
import tool_grid
import time
import noise_estimator
import numpy as np
from trefide.utils.noise import estimate_noise

# ____________________________
# Wrapper to call denoisers
# see individual functions for more information
# ____________________________


def spatial(Y_new,
            gHalf=[2,2],
            sn=None):
    """
    Calls spatial wiener filter in pixel neighborhood
    """
    mov_wf,_ = spatial_filtering.spatial_filter_image(Y_new,
                                                gHalf=gHalf,
                                                sn=sn)
    return mov_wf


def temporal(W,
             nblocks=[10,10],
             dx=1,
             maxlag=3,
             confidence=0.99,
             greedy=False,
             fudge_factor=1,
             mean_th_factor=1.15,
             U_update=False,
             min_rank=1,
             verbose=False):
    """
    Calls greedy temporal denoiser in pixel neighborhood
    """
    #start = time.time()
    mov_d, ranks = tool_grid.denoise_dx_tiles(W,
                                              nblocks=nblocks,
                                              dx=dx,
                                              maxlag=maxlag,
                                              confidence=confidence,
                                              greedy=greedy,
                                              fudge_factor=fudge_factor,
                                              mean_th_factor=mean_th_factor,
                                              U_update=U_update,
                                              min_rank=min_rank,
                                              verbose=verbose)
    #print('Temporal denoiser run for %.3f sec'%(time.time()-start))
    return mov_d, ranks


def noise_level(mov_wf,
                range_ff =[0.25,0.5]):
    """
    Calculate noise level in movie pixels
    """
    ndim_ = np.ndim(mov_wf)
    if ndim_==3:
      dims_ = mov_wf.shape
      mov_wf = mov_wf.reshape((np.prod(dims_[:2]), dims_[2]),order='F')
    #noise_level = estimate_noise(mov_wf, summarize='mean')# ** 2
    noise_level = noise_estimator.get_noise_fft(mov_wf,
                                                      noise_range=range_ff)[0]

    #noise_level = estimate_noise(mov_wf, summarize='mean')#[0] #** 2
    #print(noise_level.shape)
    #noise_level = noise_estimator.noise_estimator(mov_wf,method='logmexp')#[0]

    if ndim_ ==3:
      noise_level = noise_level.reshape(dims_[:2], order='F')

    return noise_level
