import sys
sys.path.append('../')

import numpy as np
import scipy.io as io
# Preprocessing Dependencies
from trefide.utils import psd_noise_estimate
from functools import partial


import multiprocessing

# CPP Wrapper Dependencies
from trefide.pmd import batch_decompose, batch_recompose

import multiprocessing
# Experimental Implementation Dependencies
from trefide.temporal import TrendFilter
from trefide.extras.greedyPCA import choose_rank

#import greedyPCA_SV1 as gpca
#import greedyPCA_SVS as gpca
import greedyPCA_SV as gpca
# Plotting & Video Dependencies
#import cv2
#from cv2 import VideoWriter, VideoWriter_fourcc, imshow

def eval_tv(image):
    return np.sum(np.abs(image[1:,:] - image[:-1,:])) + np.sum(np.abs(image[:,1:] - image[:,:-1]))

def play(movie, gain=3, fr=120, offset=0, magnification=3,
    frame_range=[350,1000]):
    maxmov = np.max(movie)
    looping=True
    terminated=False
    while looping:
        for t in range(frame_range[0], frame_range[1]):
            if magnification != 1:
                frame = cv2.resize(movie[:,:,t],
                                   None,
                                   fx=magnification,
                                   fy=magnification,
                                   interpolation=cv2.INTER_LINEAR)
            imshow('frame', (frame - offset) / maxmov*gain)
            if cv2.waitKey(int(1. / fr * 1000)) & 0xFF == ord('q'):
                looping = False
                terminated = True
                break
        if terminated:
            break

    cv2.waitKey(100)
    cv2.destroyAllWindows()
    for i in range(10):
        cv2.waitKey(100)
    return


def denoise_gpca_single(X,
                bheight=20,
                bwidth=20,
                confidence=0.99,
                greedy=True,
                max_components=20,
                maxlag=10,
                mean_th=1.0,
                mean_th_factor=1.0,
                 min_rank=1,
                 plot_en=False,
                snr_threshold=0,
                U_update=False,
                verbose=False
                ):
    """
    """
    d1, d2, T = X.shape
    nbi = int(d1 / bheight)
    nbj = int(d2 / bwidth)
    denoised_movie = np.zeros(X.shape)
    all_ranks = []
    for j in range(nbj):
        for i in range(nbi):
            #print('\n')
            #print(i,j)
            a = X[i*bheight:(i+1)*bheight, j*bwidth:(j+1)*bwidth, :]
            if a.max() <=0:
                continue
            #print('\n')
            #print(a.shape)
            denoised_block, ranks = gpca.denoise_patch(a,
                                                    confidence=confidence,
                                                    greedy=greedy,
                                                    maxlag=maxlag,
                                                    max_num_components=max_components,
                                                    mean_th=mean_th,
                                                    mean_th_factor=mean_th_factor,
                                                    min_rank=min_rank,
                                                    snr_threshold=snr_threshold,
                                                    plot_en=plot_en,
                                                    U_update=U_update,
                                                    verbose=verbose)

            denoised_movie[i*bheight:(i+1)*bheight, j*bwidth:(j+1)*bwidth,:] = denoised_block
            all_ranks.append(ranks)
    return denoised_movie, np.asarray(all_ranks)#.astype('int64')


def denoise_gpca(X,
                bheight=20,
                bwidth=20,
                confidence=0.99,
                greedy=True,
                max_components=20,
                maxlag=10,
                mean_th=1.0,
                mean_th_factor=1.0,
                 min_rank=1,
                 plot_en=False,
                snr_threshold=0,
                U_update=False,
                verbose=False
                ):
    """
    """
    d1, d2, T = X.shape
    nbi = int(d1 / bheight)
    nbj = int(d2 / bwidth)
    all_ranks = []

    cpu_count = max(1, multiprocessing.cpu_count()-2)
    args = []
    for j in range(nbj):
        for i in range(nbi):
            a = X[i*bheight:(i+1)*bheight, j*bwidth:(j+1)*bwidth, :]
            args.append([a])
    #start = time.time()
    pool = multiprocessing.Pool(cpu_count)
            #if a.max() <=0:
            #    continue
            #print('\n')
            #print(a.shape)
    c_outs = pool.starmap(partial(gpca.denoise_patch,
                      confidence=confidence,
                      greedy=greedy,
                      maxlag=maxlag,
                      max_num_components=max_components,
                      mean_th=mean_th,
                      mean_th_factor=mean_th_factor,
                      min_rank=min_rank,
                      plot_en=plot_en,
                      snr_threshold=snr_threshold,
                      U_update=U_update,
                      verbose=verbose),
                      args)

    pool.close()
    pool.join()

    denoised_block = [out_[0] for out_ in c_outs]
    all_ranks = [out_[1] for out_ in c_outs]
    denoised_movie = np.zeros(X.shape)
    count = 0
    for j in range(nbj):
        for i in range(nbi):
            #a = X[i*bheight:(i+1)*bheight, j*bwidth:(j+1)*bwidth, :]
            denoised_movie[i*bheight:(i+1)*bheight, j*bwidth:(j+1)*bwidth,:] = denoised_block[count]
            count+=1
    #all_ranks.append(ranks)
    return denoised_movie, np.asarray(all_ranks)#.astype('int64')


def denoise_pmd(X,
                bheight=20,
                bwidth=20,
                max_components=20,
                #maxlag=10,
                #confidence=0.99,
                #mean_th_factor=1,
                ):
    """
    Most up to date version as of 04/23/18
    """
    # Constants
    w =.0025
    maxiter = 50
    tol = 5e-3

    # Parameters
    d1, d2, T = X.shape
    spatial_cutoff = (bheight*bwidth / ((bheight*(bwidth-1) + bwidth*(bheight-1))))


    U, V, K, indices = batch_decompose(d1, d2, T, X, bheight, bwidth, w, spatial_cutoff, max_components, maxiter, tol)

    reconstructed = batch_recompose(U,V,K, indices)

    #keep = []
    #ranks_pmd = []
    #for b in range(int(d1/bheight)*int(d2/bwidth)):
    #    temporal_failed = np.argwhere(np.isnan(gpca.choose_rank(V[b,:K[b],:],
    #                                                            maxlag=maxlag,
    #                                                            confidence=confidence,
    #                                                            mean_th_factor=mean_th_factor)[0,:])).flatten()
    #    keep.append(np.setdiff1d(np.arange(K[b]).astype(np.int64), temporal_failed[temporal_failed < K[b]]))
    #       ranks_pmd.append(len(keep[b]))

    # Reconstruct Movie
    #reconstructed = np.zeros(np.prod(X.shape)).reshape(X.shape)
    #for b_idx, num_components in enumerate(K):
    #    for k in range(num_components):
    #        idx, jdx = indices[b_idx].astype(np.int64)
    #        idx *= bheight
    #        jdx *= bwidth
    #        reconstructed[idx:idx+bheight, jdx:jdx+bwidth] += \
    #            U[b_idx,:,:,k][:,:,None].dot(V[b_idx,k,:][None,:]).reshape(
    #               (bheight,bwidth,T), order='F')
    return np.asarray(reconstructed), np.asarray(K)#.astype('int64')

