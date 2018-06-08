import sys
import numpy as np
import itertools
import multiprocessing
import time
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from math import ceil

from functools import partial
from itertools import product


from . import greedy as gpca

def reconstruct_st_component(spatial_components,
                            temporal_components,
                            dims,
                            nblocks=[10,10],
                            weighted=False,
                            plot_en=False,):

    # reconstruct from each offset case
    offset_case=[None,'r','c','rc']
    matrix_reconstructed = []
    for ii, offset in enumerate(offset_case):
        print('reconstructing offset %s'%offset)
        dW_ = reconstruct_from_components(spatial_components[ii],
                                    temporal_components[ii],
                                    dims,
                                    nblocks=nblocks,
                                    offset_case=offset)
        matrix_reconstructed.append(dW_)
        del dW_

    # once all matrices are reconstructed call combine
    W_four = combine_4xd(nblocks,
                         matrix_reconstructed[0],
                         matrix_reconstructed[1],
                         matrix_reconstructed[2],
                         matrix_reconstructed[3],
                         weighted=weighted,
                         plot_en=plot_en)
    return W_four



def reconstruct_from_components(spatial,
                                temporal,
                                dims,
                                nblocks=[10,10],
                                offset_case=None,
                                list_order='C'):
    dims_cut, dims_idv = offset_tiling_dims(dims,
                                            nblocks=nblocks,
                                            offset_case=offset_case)
    n_tiles = len(spatial)
    Yd = []
    for tile in range(n_tiles):
        u = spatial[tile]
        v = temporal[tile]
        if np.ndim(u)==1:
            u = u[:,np.newaxis]
        if np.ndim(v)==1:
            v = v[np.newaxis,:]
        Yd.append(u.dot(v))

    dW_ = combine_blocks(dims_cut,
                        Yd,
                        dimsMc=dims_idv,
                        list_order=list_order)
    return dW_

def denoise_tiles(W,
                offset_case=None,
                fudge_factor=1.,
                greedy=False,
                min_rank=0,
                nblocks=[10,10],
                reconstruct=True,
                U_update=False,
                verbose=False):
    """
    Given matrix W, denoise it according
    Input:
    ------
    Output:
    ------
    """
    dims = W.shape
    W_, dimsW_ = offset_tiling(W,
                             nblocks=nblocks,
                             offset_case=offset_case)

    dims_ = list(map(np.shape,W_))

    #########################
    # No offset tiling
    #########################
    if verbose:
        print('Running for %s offset'%str(offset_case))

    dW_, rank_W_ = run_single_tiles(W_,
                            fudge_factor=fudge_factor,
                            greedy=greedy,
                            min_rank=min_rank,
                            U_update=U_update,
                            reconstruct=reconstruct,
                            verbose=verbose)
    del W_
    dW_ = combine_blocks(dimsW_,
                        dW_,
                        dimsMc=dims_,
                        list_order='C')
    return dW_, rank_W_


def denoise_tile_components(W,
                    offset_case=None,
                    fudge_factor=1.,
                    greedy=False,
                    min_rank=1,
                    nblocks=[10,10],
                    reconstruct=False,
                    U_update=False,
                    verbose=False):
    """
    Given matrix W, denoise it according
    Input:
    ------
    Output:
    ------
    """
    W_, dimsW_ = offset_tiling(W,
                             nblocks=nblocks,
                             offset_case=offset_case)
    #########################
    # No offset tiling
    #########################
    if verbose:
        print('Running for %s offset'%str(offset_case))


    UW_, VW_, rank_W_ = run_single_components(W_,
                            fudge_factor=fudge_factor,
                            greedy=greedy,
                            min_rank=min_rank,
                            U_update=U_update,
                            reconstruct=reconstruct,
                            verbose=verbose)
    return UW_, VW_, rank_W_, dimsW_


def denoise_dx_components(W,
                    dx=1,
                    fudge_factor=1.,
                    greedy=False,
                    min_rank=1,
                    nblocks=[10,10],
                    reconstruct=False,
                    U_update=False,
                    verbose=False):
    """
    Given matrix W, denoise it according
    Input:
    ------
    Output:
    ------
    """
    dims = W.shape

    #########################
    # No offset tiling
    #########################
    UW_, VW_, rank_W_, dimsW_ = denoise_tile_components(W,
                                                offset_case=None,
                                                fudge_factor=fudge_factor,
                                                greedy=greedy,
                                                min_rank=min_rank,
                                                nblocks=nblocks,
                                                reconstruct=reconstruct,
                                                U_update=U_update,
                                                verbose=verbose)
    if dx ==1:
        return UW_, VW_, rank_W_, dimsW_

    #########################
    # Row wise offset tiling
    #########################

    UW_rs, VW_rs, rank_W_rs, dimsW_rs = denoise_tile_components(W,
                                                offset_case='r',
                                                fudge_factor=fudge_factor,
                                                greedy=greedy,
                                                min_rank=min_rank,
                                                nblocks=nblocks,
                                                reconstruct=reconstruct,
                                                U_update=U_update,
                                                verbose=verbose)

    #########################
    # Col wise offset tiling
    #########################

    UW_cs, VW_cs, rank_W_cs, dimsW_cs = denoise_tile_components(W,
                                                offset_case='c',
                                                fudge_factor=fudge_factor,
                                                greedy=greedy,
                                                min_rank=min_rank,
                                                nblocks=nblocks,
                                                reconstruct=reconstruct,
                                                U_update=U_update,
                                                verbose=verbose)
    #########################
    # Row/Col wise offset tiling
    #########################

    UW_rcs, VW_rcs, rank_W_rcs, dimsW_rcs = denoise_tile_components(W,
                                                offset_case='rc',
                                                fudge_factor=fudge_factor,
                                                greedy=greedy,
                                                min_rank=min_rank,
                                                nblocks=nblocks,
                                                reconstruct=reconstruct,
                                                U_update=U_update,
                                                verbose=verbose)

    spatial_components = [UW_, UW_rs, UW_cs, UW_rcs]
    del UW_, UW_rs, UW_cs, UW_rcs
    temporal_components = [VW_, VW_rs, VW_cs, VW_rcs]
    del VW_, VW_rs, VW_cs, VW_rcs
    tiling_ranks = [rank_W_, rank_W_rs, rank_W_cs, rank_W_rcs]
    del rank_W_, rank_W_rs, rank_W_cs, rank_W_rcs
    tiling_dims = [dimsW_, dimsW_rs, dimsW_cs, dimsW_rcs]
    return spatial_components, temporal_components, tiling_ranks, tiling_dims


def denoise_dx_reconstructed(W,
                            dx=1,
                            fudge_factor=1.,
                            greedy=False,
                            min_rank=1,
                            nblocks=[10,10],
                            plot_en=False,
                            reconstruct=True,
                            U_update=False,
                            verbose=False):
    """
    Given matrix W, denoise it according
    Input:
    ------
    Output:
    ------
    """
    dims = W.shape


    dW_, rank_W_ = denoise_tiles(W,
                                offset_case=None,
                                fudge_factor=fudge_factor,
                                greedy=greedy,
                                min_rank=min_rank,
                                nblocks=nblocks,
                                reconstruct=reconstruct,
                                U_update=U_update,
                                verbose=verbose)

    if dx ==1:
        return dW_, rank_W_

    #########################
    # Row wise offset tiling
    #########################

    dW_rs, rank_W_rs = denoise_tiles(W,
                                offset_case='r',
                                fudge_factor=fudge_factor,
                                greedy=greedy,
                                min_rank=min_rank,
                                nblocks=nblocks,
                                reconstruct=reconstruct,
                                U_update=U_update,
                                verbose=verbose)

    #########################
    # Col wise offset tiling
    #########################
    dW_cs, rank_W_cs = denoise_tiles(W,
                                offset_case='c',
                                fudge_factor=fudge_factor,
                                greedy=greedy,
                                min_rank=min_rank,
                                nblocks=nblocks,
                                reconstruct=reconstruct,
                                U_update=U_update,
                                verbose=verbose)

    #########################
    # Row/Col wise offset tiling
    #########################
    dW_rcs, rank_W_rcs = denoise_tiles(W,
                                offset_case='rc',
                                fudge_factor=fudge_factor,
                                greedy=greedy,
                                min_rank=min_rank,
                                nblocks=nblocks,
                                reconstruct=reconstruct,
                                U_update=U_update,
                                verbose=verbose)

    W_four = combine_4xd(nblocks,
                         dW_,
                         dW_rs,
                         dW_cs,
                         dW_rcs,
                         plot_en=plot_en)

    return W_four , [rank_W_,rank_W_rs,rank_W_cs,rank_W_rcs]


def combine_4xd(nblocks,dW_,dW_rs,dW_cs,dW_rcs, weighted=False,plot_en=False):
    """
    Inputs:
    -------


    Output:
    -------
    """

    dims = dW_.shape

    ## Get offset dims
    _, dims_ = offset_tiling_dims(dims,nblocks=nblocks,offset_case=None)#[1]
    drs, dims_rs = offset_tiling_dims(dims,nblocks=nblocks,offset_case='r')#[1]
    dcs, dims_cs = offset_tiling_dims(dims,nblocks=nblocks,offset_case='c')#[1]
    drcs, dims_rcs = offset_tiling_dims(dims,nblocks=nblocks,offset_case='rc')#[1]
    ##
    row_array, col_array = tile_grids(dims,
                                    nblocks=nblocks)

    r_offset = vector_offset(row_array)
    c_offset = vector_offset(col_array)

    r1, r2 = (row_array[1:]-r_offset)[[0,-1]]
    c1, c2 = (col_array[1:]-c_offset)[[0,-1]]

    # Get pyramid functions for each grid
    ak1 = np.zeros(dims[:2])
    ak2 = np.zeros(dims[:2])
    ak3 = np.zeros(dims[:2])

    ak0 = pyramid_tiles(dims,dims_,list_order='C')

    ak1[r1:r2,:] = pyramid_tiles(drs,dims_rs,list_order='C')

    ak2[:,c1:c2] = pyramid_tiles(dcs,dims_cs,list_order='C')


    ak3[r1:r2,c1:c2] = pyramid_tiles(drcs,dims_rcs,list_order='C')

    # Force outer most border = 1
    ak0[[0,-1],:]=1
    ak0[:,[0,-1]]=1

    # Force corner components to be equal to 1

    #return ak0,ak1,ak2,ak3,patches,W_rs,W_cs,W_rcs
    if False:
        print('427 -- debug')
        return ak0,ak1,ak2,ak3

    W1 = np.zeros(dims)
    W2 = np.zeros(dims)
    W3 = np.zeros(dims)
    W1[r1:r2,:,:] = dW_rs
    W2[:,c1:c2,:] = dW_cs
    W3[r1:r2,c1:c2,:] = dW_rcs


    if plot_en:
        for ak_ in [ak0,ak1,ak2,ak3]:
            plt.figure(figsize=(10,10))
            plt.imshow(ak_[:,:])
            plt.show()

    if plot_en:
        plt.figure(figsize=(10,10))
        plt.imshow((ak0+ak1+ak2+ak3)[:,:])
        plt.colorbar()

    if weighted:
        W_hat = dW_+W1+W2+W3
    else:
        W_hat = ak0[:,:,np.newaxis]*dW_
        W_hat += ak1[:,:,np.newaxis]*W1
        W_hat += ak2[:,:,np.newaxis]*W2
        W_hat += ak3[:,:,np.newaxis]*W3

    W_hat /= (ak0+ak1+ak2+ak3)[:,:,np.newaxis]
    return W_hat


def run_single_components(Y,
                          debug = False,
                          fudge_factor=1,
                          greedy=False,
                          min_rank=1,
                          parallel=True,
                          reconstruct=False,
                          U_update=False,
                          verbose=False
                          ):
    """
    Run denoiser in each movie in the list Y.
    Inputs:
    ------
    Y:      list (number_movies,)
            list of 3D movies, each of dimensions (d1,d2,T)
            Each element in the list can be of different size.
    Outputs:
    --------
    Yds:    list (number_movies,)
            list of denoised 3D movies, each of same dimensions
            as the corresponding input movie.input
    vtids:  list (number_movies,)
            rank or final number of components stored for each movie.
    ------
    """
    if sys.platform == 'darwin':
        print('parallel version not for Darwin')
        parallel = False

    start = time.time()

    if parallel:
        cpu_count = max(1, multiprocessing.cpu_count()-2)
        args=[[patch] for patch in Y]
        start=time.time()
        pool = multiprocessing.Pool(cpu_count)
        print('Running %d blocks in %d cpus'%(len(Y),
                                              cpu_count))#if verbose else 0
        # define params in function
        c_outs = pool.starmap(partial(gpca.denoise_patch,
                              fudge_factor=fudge_factor,
                              greedy=greedy,
                              min_rank=min_rank,
                              reconstruct=reconstruct,
                              U_update=U_update,
                              verbose=verbose),
                              args)
        pool.close()
        pool.join()

        Uds = [out_[0] for out_ in c_outs]
        Vds = [out_[1] for out_ in c_outs]
        vtids = [out_[2] for out_ in c_outs]
    else:
        print('Running in series')
        Uds = [None]*len(Y)
        Vds = [None]*len(Y)
        vtids = [None]*len(Y)
        for ii, patch in enumerate(Y):
            print('Tile %d'%ii)
            #if not debug:
            u_ , v_, vt_ = gpca.denoise_patch(patch,
                            fudge_factor=fudge_factor,
                            greedy=greedy,
                            min_rank=min_rank,
                            reconstruct=reconstruct,
                            U_update=U_update,
                            verbose=verbose)
            Uds[ii] = u_
            Vds[ii] = v_
            vtids[ii] = vt_
    vtids = np.asarray(vtids).astype('int')

    print('Blocks(=%d) run time: %f'%(len(Y),time.time()-start))
    return Uds, Vds, vtids


def run_single_tiles(Y,
              debug = False,
              fudge_factor=1,
              greedy=False,
              min_rank=1,
              parallel=True,
              reconstruct=False,
              U_update=False,
              verbose=False
              ):
    """
    Run denoiser in each movie in the list Y.
    Inputs:
    ------
    Y:      list (number_movies,)
            list of 3D movies, each of dimensions (d1,d2,T)
            Each element in the list can be of different size.
    Outputs:
    --------
    Yds:    list (number_movies,)
            list of denoised 3D movies, each of same dimensions
            as the corresponding input movie.input
    vtids:  list (number_movies,)
            rank or final number of components stored for each movie.
    ------
    """
    if debug:
        print('485-debug')
        vtids = np.zeros((len(Y),))
        return Y, vtids

    if sys.platform == 'darwin':
        print('parallel version not for Darwin')
        parallel = False

    start = time.time()

    if parallel:
        cpu_count = max(1, multiprocessing.cpu_count()-2)
        args=[[patch] for patch in Y]
        start=time.time()
        pool = multiprocessing.Pool(cpu_count)
        print('Running %d blocks in %d cpus'%(len(Y),
                                              cpu_count))#if verbose else 0
        # define params in function
        c_outs = pool.starmap(partial(gpca.denoise_patch,
                              fudge_factor=fudge_factor,
                              greedy=greedy,
                              min_rank=min_rank,
                              reconstruct=reconstruct,
                              U_update=U_update,
                              verbose=verbose),
                              args)
        pool.close()
        pool.join()

        Yds = [out_[0] for out_ in c_outs]
        vtids = [out_[1] for out_ in c_outs]
    else:
        print('Running in series')
        Yds = [None]*len(Y)
        vtids = [None]*len(Y)
        for ii, patch in enumerate(Y):
            print('Tile %d'%ii)
            #if not debug:
            y_ , vt_ = gpca.denoise_patch(patch,
                            fudge_factor=fudge_factor,
                            greedy=greedy,
                            min_rank=min_rank,
                            reconstruct=reconstruct,
                            U_update=U_update,
                            verbose=verbose)
            Yds[ii] = y_
            vtids[ii] = vt_
    vtids = np.asarray(vtids).astype('int')

    print('Blocks(=%d) run time: %f'%(len(Y),time.time()-start))
    return Yds, vtids

# ----------------------------------
# Helper Functions
# ----------------------------------
def pyramid_matrix(dims, plot_en=False):
    bheight, bwidth = dims[:2]
    hbheight = bheight//2
    hbwidth = bwidth//2
    # Generate Single Quadrant Weighting matrix
    ul_weights = np.zeros((hbheight, hbwidth), dtype=np.float64)
    for i in range(hbheight):
        for j in range(hbwidth):
            ul_weights[i,j] = min(i, j)

    # Compute Cumulative Overlapped Weights (Normalizing Factor)
    cum_weights = np.asarray(ul_weights) +\
            np.fliplr(ul_weights) + np.flipud(ul_weights) +\
            np.fliplr(np.flipud(ul_weights))

    # Normalize By Cumulative Weights
    for i in range(hbheight):
        for j in range(hbwidth):
            ul_weights[i,j] = ul_weights[i,j] / cum_weights[i,j]

    # Construct Full Weighting Matrix From Normalize Quadrant
    W = np.hstack([np.vstack([ul_weights,
                              np.flipud(ul_weights)]),
                   np.vstack([np.fliplr(ul_weights),
                              np.fliplr(np.flipud(ul_weights))])])

    if plot_en:
        plt.figure(figsize=(10,10))
        plt.imshow(W)
        plt.xticks(np.arange(dims[1]))
        plt.yticks(np.arange(dims[0]))
        plt.colorbar()
        plt.show()
    return W

def pyramid_matrix_v0(dims,plot_en=False):
    """
    Compute a 2D pyramid function of size dims.

    Parameters:
    ----------
    dims:       tuple (d1,d2)
                size of pyramid function

    Outputs:
    -------
    a_k:        np.array (dims)
                 Pyramid function ranges [0,1],
                 where 0 indicates the boundary
                 and 1 the center.
    """
    a_k = np.zeros(dims[:2])
    xc, yc = ceil(dims[0]/2),ceil(dims[1]/2)

    for ii in range(xc):
        for jj in range(yc):
            a_k[ii,jj]=max(dims)-min(ii,jj)
            a_k[-ii-1,-jj-1]=a_k[ii,jj]
    for ii in range(xc,dims[0]):
        for jj in range(yc):
            a_k[ii,jj]=a_k[ii,-jj-1]
    for ii in range(xc):
        for jj in range(yc,dims[1]):
            a_k[ii,jj]=a_k[-ii-1,jj]
    a_k = a_k.max() - a_k
    a_k /=a_k.max()

    if plot_en:
        plt.figure(figsize=(10,10))
        plt.imshow(a_k)
        plt.xticks(np.arange(dims[1]))
        plt.yticks(np.arange(dims[0]))
        plt.colorbar()
        plt.show()
    #if len(dims)>2:
        #a_k = np.array([a_k,]*dims[2]).transpose([1,2,0])
    return a_k


def pyramid_tiles(dims_rs,
                  dims_,
                  list_order='C',
                  plot_en=False):
    """
    Calculate 2D array of size dims_rs,
    composed of pyramid matrices, each of which has the same
    dimensions as an element in W_rs.
    Inputs:
    -------
    dims_rs:    tuple (d1,d2)
                dimension of array
    W_rs:       list
                list of pacthes which indicate dimensions
                of each pyramid function
    list_order: order in which the

    Outputs:
    --------

    """
    #dims_ = np.asarray(list(map(np.shape,W_rs)))
    a_ks = []
    for dim_ in dims_:
        a_k = pyramid_matrix(dim_)
        a_ks.append(a_k)
    # given W_rs and a_ks reconstruct array
    a_k = combine_blocks(dims_rs[:2],
                        a_ks,
                        dims_,
                        list_order=list_order)

    if plot_en:
        plt.figure(figsize=(10,10))
        plt.imshow(a_k)
        plt.colorbar()
    return a_k


def cn_ranks(dim_block, ranks, dims, list_order='C'):
    """
    """
    Crank = np.zeros(shape=dims)*np.nan
    d1,d2  = Crank.shape
    i,j = 0,0
    for ii in range(0,len(ranks)):
        d1c , d2c  = dim_block[ii][:2]
        Crank[i:i+d1c,j:j+d2c].fill(int(ranks[ii]))
        if list_order=='F':
            i += d1c
            if i == d1:
                j += d2c
                i = 0
        else:
            j+= d2c
            if j == d2:
                i+= d1c
                j = 0
    return Crank


def combine_blocks(dimsM,
                  Mc,
                  dimsMc=None,
                  list_order='C',
                  array_order='F'):
    """
    Combine blocks given by compress_blocks

    Parameters:
    ----------
    dimsM:          tuple (d1,d2,T)
                    dimensions of original array
    Mc:             np.array or list
                    contains (padded) tiles from array.
    dimsMc:         np.array of tuples (d1,d2,T)
                    (original) dimensions of each tile in array
    list_order:     string {'F','C'}
                    determine order to reshape tiles in array
                    array order if dxT instead of d1xd2xT assumes always array_order='F'
                    NOTE: if dimsMC is NONE then MC must be a d1 x d2 x T array
    array_order:    string{'F','C'}
                    array order to concatenate tiles
                    if Mc is (dxT), the outputs is converted to (d1xd2xT)
    Outputs:
    --------
    M_all:          np.array (dimsM)
                    reconstruction of array from Mc
    """

    ndims = len(dimsM)

    if ndims ==3:
        d1, d2, T = dimsM
        Mall = np.zeros(shape=(d1, d2, T))*np.nan
    elif ndims ==2:
        d1,d2 = dimsM[:2]
        Mall = np.zeros(shape=(d1, d2))*np.nan

    if type(Mc)==list:
        k = len(Mc)
    elif type(Mc)==np.ndarray:
        k = Mc.shape[0]
    else:
        print('error= must be np.array or list')
    if dimsMc is None:
        dimsMc = np.asarray(list(map(np.shape,Mc)))
    #else:
        #print('dimsMC given')
    i, j = 0, 0
    for ii, Mn in enumerate(Mc):
        # shape of current block
        d1c, d2c = dimsMc[ii][:2]
        if (np.isnan(Mn).any()):
            Mn = unpad(Mn)
        if Mn.ndim < 3 and ndims ==3:
            Mn = Mn.reshape((d1c, d2c)+(T,), order=array_order)
        if ndims ==3:
            Mall[i:i+d1c, j:j+d2c, :] = Mn
        elif ndims ==2:
            Mall[i:i+d1c, j:j+d2c] = Mn
        if list_order=='F':
            i += d1c
            if i == d1:
                j += d2c
                i = 0
        else:
            j += d2c
            if j == d2:
                i += d1c
                j = 0
    return Mall

def block_split_size(l, n):
    """
    For an array of length l that should be split into n sections,
    calculate the dimension of each section:
    l%n sub-arrays of size l//n +1 and the rest of size l//n
    Input:
    ------
    l:      int
            length of array
    n:      int
            number of section in which an array of size l
            will be partitioned
    Output:
    ------
    d:      np.array (n,)
            length of each partitioned array.
    """
    d = np.zeros((n,)).astype('int')
    cut = l%n
    d[:cut] = l//n+1
    d[cut:] = l//n
    return d


def split_image_into_blocks(image,
                          nblocks=[10,10]):
    """
    Split an image into blocks.

    Parameters:
    ----------
    image:          np.array (d1 x d2 x T)
                    array to be split into nblocks
                    along first two dimensions
    nblocks:        list (2,)
                    parameters to split image across
                    the first two dimensions, respectively

    Outputs
    -------
    blocks:         list,
                    contains nblocks[0]*nblocks[1] number of tiles
                    each of dimensions (d1' x d2' x T)
                    in fortran 'F' order.
    """

    if all(isinstance(n, int) for n in nblocks):
        number_of_blocks = np.prod(nblocks)
    else:
        number_of_blocks = (len(nblocks[0])+1)*(len(nblocks[1])+1)
    blocks = []
    if number_of_blocks != (image.shape[0] * image.shape[1]):
        block_divided_image = np.array_split(image,nblocks[0],axis=0)
        for row in block_divided_image:
            blocks_ = np.array_split(row,nblocks[1],axis=1)
            for block in blocks_:
                blocks.append(np.array(block))
    else:
        blocks = image.flatten()
    return blocks


def vector_offset(array, offset_factor=2):
    """
    -------
    """
    #x,y = np.meshgrid(row_array[:],col_array[:])
    array_offset = np.ceil(np.divide(np.diff(array),
                                     offset_factor)).astype('int')
    return array_offset


def tile_grids(dims,
               nblocks=[10,10],
               offset_case=None,
               indiv_grids=True):
    """
    Input:
    ------

    Output:
    ------
    """
    if all(isinstance(n, int) for n in nblocks):
        d_row = block_split_size(dims[0], int(nblocks[0]))
        d_col = block_split_size(dims[1], int(nblocks[1]))
    else:
        d_row, d_col=nblocks

    if indiv_grids:
        d_row = np.insert(d_row,0,0)
        d_col = np.insert(d_col,0,0)
        return d_row.cumsum(), d_col.cumsum()

    # list(product(d_row,d_col))
    d_row = np.append(d_row,dims[0])
    d_col = np.append(d_col,dims[1])
    d_row = np.diff(np.insert(d_row,0,0))
    d_col = np.diff(np.insert(d_col,0,0))

    number_blocks = (len(d_row))*(len(d_col))

    #row_array = np.zeros((number_blocks,))
    #col_array = np.zeros((number_blocks,))
    array = np.zeros((number_blocks,2))

    for ii,row in enumerate(product(d_row,d_col)):
        array[ii] = row

    return array.astype('int')


def offset_tiling_dims(dims,
                      nblocks,
                      offset_case=None):
    """
    """
    row_array, col_array = tile_grids(dims,
                                    nblocks=nblocks)
    r_offset = vector_offset(row_array)
    c_offset = vector_offset(col_array)

    rc0, rc1 = (row_array[1:]-r_offset)[[0,-1]]
    cc0, cc1 = (col_array[1:]-c_offset)[[0,-1]]

    if offset_case is None:
        dims2 = dims
        row_array = row_array[1:-1]
        col_array = col_array[1:-1]

    elif offset_case == 'r':
        dims2 = rc1-rc0,dims[1],dims[2]
        row_array=row_array[1:-2]
        col_array=col_array[1:-1]

    elif offset_case == 'c':
        dims2 = dims[0],cc1-cc0,dims[2]
        row_array=row_array[1:-1]
        col_array=col_array[1:-2]

    elif offset_case == 'rc':
        dims2 = rc1-rc0,cc1-cc0,dims[2]
        row_array=row_array[1:-2]
        col_array=col_array[1:-2]

    else:
        print('Invalid option')

    #import pdb; pdb.set_trace()
    indiv_dim = tile_grids(dims2,nblocks=[row_array,col_array],indiv_grids=False)


    return dims2, indiv_dim


def offset_tiling(W,
                  nblocks=[10,10],
                  offset_case=None):
    """
    Given a matrix W, which was split row and column wise
    given row_cut,col_cut, calculate three off-grid splits
    of the same matrix. Each offgrid will be only row-,
    only column-, and row and column-wise.
    Inputs:
    -------
    W:          np.array (d1 x d2 x T)
    r_offset:
    c_offset:
    row_cut:
    col_cut:

    Outputs:
    --------
    W_rs:       list
    W_cs:       list
    W_rcs:      list
    """

    #col_array,row_array = tile_grids(dims,nblocks)

    #r_offset,c_offset = extract_4dx_grid(dims,row_array,col_array)
    dims=W.shape
    row_array,col_array = tile_grids(dims,
                                    nblocks=nblocks)

    r_offset = vector_offset(row_array)
    c_offset = vector_offset(col_array)

    rc0, rc1 = (row_array[1:]-r_offset)[[0,-1]]
    cc0, cc1 = (col_array[1:]-c_offset)[[0,-1]]

    if offset_case is None:
        W_off = split_image_into_blocks(W,
                                        nblocks=nblocks)

    elif offset_case == 'r':
        W = W[rc0:rc1,:,:]
        W_off = split_image_into_blocks(W,
                                        nblocks=[row_array[1:-2],
                                                 col_array[1:-1]])
    elif offset_case == 'c':
        W = W[:,cc0:cc1,:]
        W_off = split_image_into_blocks(W,
                                        nblocks=[row_array[1:-1],
                                                 col_array[1:-2]])
    elif offset_case == 'rc':
        W = W[rc0:rc1,cc0:cc1,:]
        W_off = split_image_into_blocks(W,
                                        nblocks=[row_array[1:-2],
                                                 col_array[1:-2]])
    else:
        print('Invalid option')
        W_off = W

    return W_off, W.shape

## legacy code

def denoise_dx_reconstructed_deprecated(W,
                    dx=1,
                    fudge_factor=1.,
                    greedy=False,
                    min_rank=1,
                    nblocks=[10,10],
                    plot_en=False,
                    reconstruct=True,
                    U_update=False,
                    verbose=False):
    """
    Given matrix W, denoise it according
    Input:
    ------
    Output:
    ------
    """
    dims = W.shape

    W_ = split_image_into_blocks(W, nblocks=nblocks)

    #########################
    # No offset tiling
    #########################
    if verbose:
        print('Running individual tiles')


    dW_, rank_W_ = run_single_tiles(W_,
                            fudge_factor=fudge_factor,
                            greedy=greedy,
                            min_rank=min_rank,
                            U_update=U_update,
                            reconstruct=reconstruct,
                            verbose=verbose)
    dims_ = list(map(np.shape,W_))
    del W_
    dW_ = combine_blocks(dims,
                        dW_,
                        dimsMc=dims_,
                        list_order='C')

    if dx ==1:
        return dW_, rank_W_

    #########################
    # Row wise offset tiling
    #########################
    if verbose:
        print('Row wise tiling')

    W_rs, drs = offset_tiling(W,
                             nblocks=nblocks,
                             offset_case='r')

    dims_rs = list(map(np.shape,W_rs))

    dW_rs, rank_W_rs = run_single_tiles(W_rs,
                                fudge_factor=fudge_factor,
                                min_rank=min_rank,
                                U_update=U_update,
                                reconstruct=reconstruct,
                                verbose=verbose)
    del W_rs

    dW_rs = combine_blocks(drs,
                        dW_rs,
                        dimsMc=dims_rs,
                        list_order='C')

    #########################
    # Col wise offset tiling
    #########################
    if verbose:
        print('Col wise tiling')

    W_cs, dcs = offset_tiling(W,
                            nblocks=nblocks,
                            offset_case='c')
    dims_cs = list(map(np.shape,W_cs))

    dW_cs, rank_W_cs = run_single_tiles(W_cs,
                                fudge_factor=fudge_factor,
                                greedy=greedy,
                                min_rank=min_rank,
                                U_update=U_update,
                                reconstruct=reconstruct,
                                verbose=verbose)

    del W_cs
    dW_cs = combine_blocks(dcs,
                        dW_cs,
                        dimsMc=dims_cs,
                        list_order='C')

    #########################
    # Row/Col wise offset tiling
    #########################
    if verbose:
        print('Row/Col wise tiling')


    W_rcs, drcs = offset_tiling(W,
                      nblocks=nblocks,
                      offset_case='rc')

    dims_rcs = list(map(np.shape,W_rcs))
    dW_rcs,rank_W_rcs = run_single_tiles(W_rcs,
                                  fudge_factor=fudge_factor,
                                  greedy=greedy,
                                  min_rank=min_rank,
                                  U_update=U_update,
                                  reconstruct=reconstruct,
                                  verbose=verbose)
    del W_rcs

    dW_rcs = combine_blocks(drcs,
                            dW_rcs,
                            dimsMc=dims_rcs,
                            list_order='C')

    if False: # debug
        return nblocks, dW_, dW_rs, dW_cs, dW_rcs, dims_, dims_rs, dims_cs, dims_rcs

    W_four = combine_4xd(nblocks,
                         dW_,
                         dW_rs,
                         dW_cs,
                         dW_rcs,
                         #dims_,
                         #dims_rs,
                         #dims_cs,
                         #dims_rcs,
                         plot_en=plot_en)

    return W_four , [rank_W_,rank_W_rs,rank_W_cs,rank_W_rcs]