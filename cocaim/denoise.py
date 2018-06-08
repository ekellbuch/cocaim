import numpy as np
import time

from . import noise_estimator, spatial_filtering, tool_grid

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
    mov_wf = spatial_filtering.spatial_filter_image(Y_new,
                                                    gHalf=gHalf,
                                                    sn=sn)[0]
    return mov_wf


def temporal_reconstructed(W,
                         dx=1,
                         fudge_factor=1,
                         greedy=False,
                         min_rank=0,
                         nblocks=[10,10],
                         plot_en=False,
                         U_update=False,
                         verbose=False):
    """
    Calls greedy denoiser in each tile
    outputs reconstructed movie
    """
    mov_d, ranks = tool_grid.denoise_dx_reconstructed(W,
                                                    dx=dx,
                                                    fudge_factor=fudge_factor,
                                                    greedy=greedy,
                                                    min_rank=min_rank,
                                                    nblocks=nblocks,
                                                    plot_en=plot_en,
                                                    U_update=U_update,
                                                    verbose=verbose)
    return mov_d, ranks


def temporal_components(W,
                         dx=1,
                         fudge_factor=1,
                         greedy=False,
                         min_rank=0,
                         nblocks=[10,10],
                         U_update=False,
                         verbose=False):
    """
    Calls greedy denoiser in each tile
    outputs spatial and temporal components
    """
    spatial_components, \
    temporal_components, \
    tiling_ranks, \
    tiling_dims = tool_grid.denoise_dx_components(W,
                                                dx=dx,
                                                fudge_factor=fudge_factor,
                                                greedy=greedy,
                                                min_rank=min_rank,
                                                nblocks=nblocks,
                                                U_update=U_update,
                                                verbose=verbose)
    return spatial_components, temporal_components, tiling_ranks, tiling_dims


def noise_level(mov_wf,
                range_ff =[0.25,0.5],
                method='median'):
    """
    Calculate noise level in movie pixels
    """
    ndim_ = np.ndim(mov_wf)

    if ndim_==3:
        dims_ = mov_wf.shape
        mov_wf = mov_wf.reshape((np.prod(dims_[:2]), dims_[2]))
    mov_wf = mov_wf - mov_wf.mean(1,keepdims=True)

    noise_level = noise_estimator.get_noise_fft(mov_wf,
                                                noise_range=range_ff,
                                                noise_method=method)[0]
    if ndim_ ==3:
        noise_level = noise_level.reshape(dims_[:2])

    return noise_level
