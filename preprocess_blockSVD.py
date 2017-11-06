
import numpy as np
import scipy as sp
from scipy.stats import norm

from mpl_toolkits.axes_grid1 import make_axes_locatable


import matplotlib.pyplot as plt



def nextpow2(value):
    """
    Extracted from 
    from caiman.source_extraction.cnmf.deconvolution import axcov    

    Find exponent such that 2^exponent is equal to or greater than abs(value).

    Parameters:
    ----------
    value : int

    Returns:
    -------
    exponent : int
    """

    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent


def axcov(data, maxlag=10):
    """
    Edited from cnmf.deconvolution
    Compute the autocovariance of data at lag = -maxlag:0:maxlag

    Parameters:
    ----------
    data : array
        Array containing fluorescence data

    maxlag : int
        Number of lags to use in autocovariance calculation

    Returns:
    -------
    axcov : array
        Autocovariances computed from -maxlag:0:maxlag
    """

    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    return np.real(np.divide(xcov, T))



def svd_patch(M,k=1,tsub=1,noise_norm=True,iterate=True,confidence=0.90,corr=True,kurto=True,tfilt=True):
    """
    Apply svd to k patches of M
    tsub: temporal decimation
    noise_norm: Noise normalization
    iterate: find 
    confidence: CI level
    corr: correlation flag
    kurto: kurto: kurto flag
    tfilt: temporal filter
    """
    dimsM = M.shape
    if k > 1:
        patches = split_image_into_blocks(M, k)
        dimsMc = map(np.shape,patches)
        Yds, vtids = compress_patches(patches,tsub=tsub,noise_norm=noise_norm,iterate=iterate,confidence=confidence,corr=corr,kurto=kurto,tfilt=tfilt)
        Yd = combine_blocks(dimsM, Yds, dimsMc)
    # Plot ranks
        ranks = np.logical_not(np.isnan(vtids[:,:2,:])).any(axis=1).sum(axis=1,dtype='int')
        Cn = cn_ranks_plot(dimsMc, ranks, dimsM[:2])
        print('M rank {}'.format(sum(ranks)))
        rlen = sum(ranks)
    else:
        print('Single patch')
        Yd, vtids = compress_dblocks(M,tsub=tsub,noise_norm=noise_norm,iterate=iterate,confidence=confidence,corr=corr,kurto=kurto,tfilt=tfilt)
        Yd = Yd.reshape((dimsM[0],dimsM[1])+(dimsM[2],),order='F')
        ranks = np.where(np.logical_or(vtids[0, :] == 1, vtids[1, :] == 1))[0]
        
        if not np.any(ranks):
            print('M rank Empty')
            rlen = 0
        else:
            print('M rank {}'.format(len(ranks))) 
            rlen = len(ranks)
    return Yd, rlen


def split_image_into_blocks(image, number_of_blocks=16):
    """
    Image d1 x d2 (x T)
    is split into number_of_blocks
    """
    blocks = None
    if number_of_blocks != (image.shape[0] * image.shape[1]):
        blocks = []
        block_divided_image = map(lambda sub_array: np.array_split(sub_array,
        np.sqrt(number_of_blocks), axis=1), np.array_split(image, np.sqrt(number_of_blocks)))
        for row in block_divided_image:
            for block in row:
                blocks.append(np.array(block))
    else:
        blocks = image.flatten()
    return blocks


def compress_patches(patches,tsub=1,noise_norm=True,iterate=True,confidence=0.90,corr=True,kurto=True,tfilt=True):
    """
    Compress each patch
    """
    # For any given patch
    k = len(patches)
    M = patches[0]  # first block with max size
    dx, dy, T = M.shape  # max cuts
    dxy = dx*dy  # max num pixels in block

    # Initialize as array for quick retrieval
    Yds = np.zeros(shape=(k,dxy,T))*np.nan
    vtids = np.zeros(shape=(k,3,dxy))*np.nan
    
    # Apply function to each patch
    for cpatch in np.arange(k):
        data_in = patches[cpatch]
        print('Tile {}'.format(cpatch))
        Yd_patch, keep1 = compress_dblocks(data_in,tsub=tsub,noise_norm=noise_norm,iterate=iterate,confidence=confidence,corr=corr,kurto=kurto,tfilt=tfilt)
        Yds[cpatch] = pad(Yd_patch, (dxy, T), (0, 0))
        if np.any(keep1):
            vtids[cpatch] = pad(keep1, (3, dxy), (0, 0))
    return Yds, vtids



def compress_dblocks(data_in, tsub=1, noise_norm=False, iterate=False, confidence=0.90,corr=True,kurto=True,tfilt=True):
    """
    Compress a single block
    """
    dims = data_in.shape
    data_all = data_in.reshape((dims[0]*dims[1],dims[2]), order='F').T


    # in a 2d matrix, we get rid of any broke (nan) pixels
    # we assume fixed broken across time
    broken_idx = np.isinf(data_all[0,:])
    # Work only on good pixels
    data =  data_all[:, ~broken_idx]

    # data mean is
    mu = data.mean(0, keepdims=True)

    # temporally filter the data
    if tfilt == True:
        print('Applying exponential filter')
        T, num_pxls = data.shape
        data_filt = np.zeros(shape=(T,num_pxls))
        for pixel in range(num_pxls):
            # Estimate tau for exponential
            fluor = data[:,pixel]
            tau = cnmf.deconvolution.estimate_time_constant(fluor,p=1,sn=None,lags=5,fudge_factor=1.)
            window = tau **range(0,100)
            s_out = np.convolve(fluor,window,mode='full')[:T]/np.sum(window)
            data_filt[:,pixel]=s_out
    else:
        data_filt = data

    # temporally decimate the data
    if tsub > 1:
        data0 = temporal_decimation(data_filt, tsub)
        data0 = data0 - data0.mean(0, keepdims=True)
    else:
        data0 = data_filt - mu

    # run regular non randomized svd
    U, s, Vt = compute_svd(data0.T, method='vanilla')

    if tfilt == True or tsub > 1:
        Vt1 = U.T.dot(data.T - mu.T)
    else:
        Vt1 = np.copy(Vt)

    # Select components
    ctid = choose_rank(Vt1, iterate=iterate, confidence=confidence,corr=corr,kurto=kurto)
    keep1 = np.where(np.logical_or(ctid[0, :] == 1, ctid[1, :] == 1))[0]
    # Crude noise-variance normalization
    # Compute the total variance of the noise
    # divide the temporal SVs in each path by the sqrt of these values 
    if not np.any(keep1):
        print('Length of stored components is {}'.format(len(keep1)))
        #keep1 = np.where([True])[0]
        #keep1 =[]
    if noise_norm:
        loose = np.setdiff1d(np.arange(Vt1.shape[0]),keep1)
        Vt1[loose,:]=Vt1[loose,:]/np.sqrt(s[loose,None])

    if tfilt == True or tsub > 1:
        Yd_patch = U[:,keep1].dot(Vt1[keep1,:]) + mu.T#mu0.T#+ mu.T
    else:
        if not np.any(keep1):
            Yd_patch = np.tile(mu.T,dims[2])
        else:
            S = np.eye(len(keep1))*s[keep1.astype(int)]
            Yd_patch = U[:,keep1].dot(S.dot(Vt1[keep1,:])) + mu.T

    # return original matrix
    if broken_idx.sum() > 0:
        print('There are {} broken pixels.'.format(broken_idx.sum()))
        Yd_out = np.ones(shape=data_all.shape).T*np.inf
        Yd_out[~broken_idx,:] = Yd_patch
    else:
        Yd_out =  Yd_patch

    plot_en = True

    if plot_en:
        maxlag=20
        fig, axarr = plt.subplots(1,2,sharey=True)
        loose = np.setdiff1d(np.arange(Vt1.shape[0]),keep1)
        c1 = 1
        for keep in keep1:
            vi = Vt1[keep,:]
            vi = (vi-vi.mean())/vi.std()
            metric = axcov(vi,maxlag)[maxlag:]/vi.var()
            #axarr[0].plot(c1, metric.mean(),'.')
            axarr[0].plot(metric,':')
            #c1 = c1+1

        c2 = 0
        for lost in loose:
            vi = Vt1[lost,:]
            vi = (vi-vi.mean())/vi.std()
            metric = axcov(vi,maxlag)[maxlag:]/vi.var()
            axarr[1].plot(metric,':')

        axarr[0].set_xscale('symlog')
        axarr[1].set_xscale('symlog')
        axarr[0].set_ylabel('ACF')
        axarr[0].set_xlabel('lag')
        axarr[1].set_xlabel('lag')
        axarr[1].set_yticks(())
        axarr[0].set_title('Selected components: {}'.format(len(keep1)))
        axarr[1].set_title('Discarded components: {}'.format(len(loose)))
        plt.show()
    return Yd_out, ctid


def compute_svd(M, method='randomized', n_components=400):
    """
    Compute svd on centered M
    M d x T
    """

    if method == 'vanilla':
        # print('vanilla')
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
    elif method == 'randomized':
        U, s, Vt = randomized_svd(M, n_components=n_components,
                                  n_iter=7, random_state=None)
    return U, s, Vt


def choose_rank(Vt,iterate=False,confidence=0.90,corr=True,kurto=False):
    """
    Select rank wrt axcov and kurtosis
    """
    maxlag =  10

    n, L = Vt.shape
    vtid = np.zeros(shape=(3, n)) * np.nan
    # Reject null hypothesis comes from white noise
    #print ('Correlation lag is {}'.format(maxlag))
    if corr is True:
        mean_th = covCI_wnoise(L,confidence=confidence,maxlag=maxlag)
        keep1 = cov_one(Vt, mean_th, maxlag=maxlag,iterate=iterate)
    else:
        keep1 = []
    if kurto is True:
        keep2 = kurto_one(Vt)
    else:
        keep2 = []

    keep = list(set(keep1 + keep2))
    loose = np.setdiff1d(np.arange(n), keep)
    loose = list(loose)
    if np.any(keep1):
        vtid[0, keep1] = 1  # due to cov
    vtid[1, keep2] = 1  # due to kurto
    vtid[2, loose] = 1  # extra
    # print('rank cov {} and rank kurto {}'.format(len(keep1),len(keep)-len(keep1)))
    return vtid


def mean_confidence_interval(data, confidence=0.90):
    """
    Compute mean confidence interval (CI)
    for a normally distributed population
    Input:
        data: input
        confidence: confidence level
    Output:
        th: threshold at CI
    """
    _, th = sp.stats.norm.interval(confidence,loc =np.mean(data),scale=data.std())
    return th


def covCI_wnoise(L, confidence=0.90, maxlag=10, n=2000):
    """
    Compute mean_th for auto covariance of white noise
    """
    # th1 = 0
    print 'confidence is {}'.format(confidence)
    covs_ht = np.zeros(shape=(n,))
    for sample in np.arange(n):
        ht_data = np.random.randn(L)
        covdata = axcov(ht_data, maxlag)[maxlag:]/ht_data.var()
        covs_ht[sample] = covdata.mean()
    #hist, _,_=plt.hist(covs_ht)
    #plt.show()
    mean_th = mean_confidence_interval(covs_ht, confidence)
    #print('th is {}'.format(mean_th))
    return mean_th


def temporal_decimation(data, mb):
    """
    Decimate data by mb, new frame is mean of decimated frames
    """
    data0 = data[:len(data)/mb*mb].reshape((-1, mb) + data.shape[1:]).mean(1).astype('float32')
    return data0


def cov_one(Vt, mean_th=0.10, maxlag=10, iterate=True,extra=1):
    """
    Compute only auto covariance, 
    calculate iteratively: until mean reaches mean_th
    """
    keep = []
    num_components = Vt.shape[0]
    print('mean_th is %s'%mean_th)
    for vector in range(0, num_components):
        # standarize and normalize
        vi = Vt[vector, :]
        vi =(vi - vi.mean())/vi.std()
        vi_cov = axcov(vi, maxlag)[maxlag:]/vi.var()
        if vi_cov.mean() < mean_th:
            if iterate is True:
                break
        else:
            keep.append(vector)
    if not np.any(keep):
        print('from cov Empty')
    else:
        print('from cov {}'.format(len(keep)))
    # Store extra components
    if vector < num_components and extra != 1:
        extra = min(vector*extra,Vt.shape[0])
        for addv in range(1, extra-vector+ 1):
            keep.append(vector + addv)
    return keep


def pad(array, reference_shape, offsets, array_type=np.nan):
    """
    array: Array to be padded
    reference_shape: tuple of size of narray to create
    offsets: list of offsets (number of elements must be equal to the dimension
    of the array) will throw a ValueError if offsets is too big and the
    reference_shape cannot handle the offsets
    """

    # Create an array of zeros with the reference shape
    result = np.ones(reference_shape) * array_type
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim])
                  for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result


def unpad(x):
    """
    Given padded matrix with nan
    Get rid of all nan in order (row, col)
    does not retain dimension
    """
    x = x[:, ~np.isnan(x).all(0)]
    x = x[~np.isnan(x).all(1)]
    return x


def combine_blocks(dimsM, Mc, dimsMc):
    """
    Reshape blocks given by compress_blocks
    """
    d1, d2, T = dimsM
    k = Mc.shape[0]
    Mall = np.zeros(shape=(d1, d2, T))*np.nan
    dx, dy = Mc[0].shape
    i, j = 0, 0
    for ii in range(0, k):
        # shape of current block
        d1c, d2c, Tc = dimsMc[ii]
        Mn = unpad(Mc[ii])
        Mn = Mn.reshape((d1c, d2c)+(T,), order='F')
        Mall[i:i+d1c, j:j+d2c, :] = Mn
        j += d2c
        if j == d2:
            i += d1c
            j = 0
    return Mall


def cn_ranks(dim_block, ranks, dims):
    Crank = np.zeros(shape=dims)*np.nan
    d1,d2  = Crank.shape
    i,j = 0,0
    for ii in range(0,len(ranks)):
        d1c , d2c , _= dim_block[ii]
        Crank[i:i+d1c,j:j+d2c].fill(int(ranks[ii]))
        j+= d2c
        if j == d2:
            i+= d1c
            j = 0
    return Crank


def cn_ranks_plot(dim_block, ranks, dims):
    """
        Calls function cn_ranks and plots rank matrix
    """
    Cplot3 = cn_ranks(dim_block, ranks, dims[:2])

    fig = plt.figure(num=None, figsize=(5, 5))
    ax3 = fig.add_subplot(1, 1, 1)
    ax3.set_title('Ranks in each tile')
    im3 = ax3.imshow(Cplot3, vmin=max(0,Cplot3.min()-5),
                     vmax=Cplot3.max()+10, cmap='Reds',
                     interpolation='nearest', aspect='equal')

    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im3, cax=cax3)


    dim_block = np.asarray(dim_block)
    cols, rows = dim_block.T[0] , dim_block.T[1]

    K = int(np.sqrt(len(dim_block)))
    row_array = np.insert(rows[::K+1], 0, 0).cumsum()
    col_array = np.insert(cols[::K+1], 0, 0).cumsum()
    x, y = np.meshgrid(row_array[:-1], col_array[:-1])

    ax3.set_yticks(col_array[:-1])
    ax3.set_xticks(row_array[:-1])

    for ii , (row_val, col_val) in enumerate(zip(x.flatten(), y.flatten())):
        c = str(int(Cplot3[int(col_val), int(row_val)]))
        ax3.text(row_val + rows[ii]/2, col_val+cols[ii]/2, c, va='center', ha='center')

    plt.tight_layout()
    plt.show()
    return Cplot3
