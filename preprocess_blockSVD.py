import numpy as np
import scipy as sp
from scipy.stats import norm

from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

import trefide as tfd
import spatial_filtering as sp_filters
import tools as tools


from sklearn import preprocessing
import cvxpy as cp
import time

# compare parallel
import concurrent
import multiprocessing
import itertools
import time

def old_div(a,b):
    return np.divide(a,b)


def get_noise_fft(Y, noise_range = [0.25,0.5],
        noise_method = 'logmexp', max_num_samples_fft=3072):
    """
    Extracted from caiman
    Estimate the noise level for each pixel by averaging
    the power spectral density.

    Inputs:
    -------

    Y: np.ndarray

    Input movie data with time in the last axis

    noise_range: np.ndarray [2 x 1] between 0 and 0.5
        Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
        default: [0.25,0.5]

    noise method: string
        method of averaging the noise.
        Choices:
            'mean': Mean
            'median': Median
            'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Output:
    ------
    sn: np.ndarray
        Noise level for each pixel
    """
    T = np.shape(Y)[-1]
    Y = np.array(Y).astype('float64')

    if T > max_num_samples_fft:
        Y=np.concatenate((Y[...,1:np.int(old_div(max_num_samples_fft,3))+1],        
                         Y[...,np.int(old_div(T,2)-max_num_samples_fft/3/2):np.int(old_div(T,2)+max_num_samples_fft/3/2)],
                         Y[...,-np.int(old_div(max_num_samples_fft,3)):]),axis=-1)
        T = np.shape(Y)[-1]

    dims = len(np.shape(Y))
    #we create a map of what is the noise on the FFT space
    ff = np.arange(0,0.5+old_div(1.,T),old_div(1.,T))
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1,ind2)
    #we compute the mean of the noise spectral density s
    if dims > 1:
        xdft = np.fft.rfft(Y,axis=-1)
        psdx = (old_div(1.,T))*abs(xdft)**2
        psdx[...,1:] *= 2
        sn = mean_psd(psdx[...,ind[:psdx.shape[-1]]], method = noise_method)

    else:
        xdft = np.fliplr(np.fft.rfft(Y))
        psdx = (old_div(1.,T))*(xdft**2)
        psdx[1:] *=2
        sn = mean_psd(psdx[ind[:psdx.shape[0]]], method = noise_method)

    return sn, psdx


def mean_psd(y, method = 'logmexp'):
    """
    Averaging the PSD

    Parameters:
    ----------

        y: np.ndarray
             PSD values

        method: string
            method of averaging the noise.
            Choices:
             'mean': Mean
             'median': Median
             'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
    -------
        mp: array
            mean psd
    """

    if method == 'mean':
        mp = np.sqrt(np.mean(old_div(y,2),axis=-1))
    elif method == 'median':
        mp = np.sqrt(np.median(old_div(y,2),axis=-1))
    else:
        mp = np.log(old_div((y+1e-10),2))
        mp = np.mean(mp,axis=-1)
        mp = np.exp(mp)
        mp = np.sqrt(mp)

    return mp

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


def svd_patch(M, k=1, maxlag=10, tsub=1, noise_norm=False, iterate=False,
        confidence=0.90, corr=True, kurto=False, tfilt=False, tfide=False,
        share_th=True, plot_en=False,greedy=False,fudge_factor=1.,mean_th=None,
        mean_th_factor=1.,U_update=True,min_rank=0,verbose=False):
    """
    Apply svd to k patches of M
    tsub: temporal decimation
    noise_norm: Noise normalization
    iterate: find xcovs iteratively
    confidence: CI level
    corr: correlation flag
    kurto: kurto: kurto flag
    tfilt: temporal filter
    greedy: include greedy updates of Vt and U
    """
    dimsM = M.shape
    if k > 1:
        patches = split_image_into_blocks(M, k)
        dimsMc = list(map(np.shape,patches))
        Yds, vtids = compress_patches(patches, maxlag=maxlag, tsub=tsub,
                noise_norm=noise_norm, iterate=iterate,confidence=confidence,
                corr=corr, kurto=kurto, tfilt=tfilt, tfide=tfide, share_th=share_th,
                greedy=greedy,fudge_factor=fudge_factor,mean_th_factor=mean_th_factor,
                U_update=U_update,plot_en=plot_en,
                min_rank=min_rank,verbose=verbose)
        Yd = combine_blocks(dimsM, Yds, dimsMc)
        ranks = np.logical_not(np.isnan(vtids[:,:2,:])).any(axis=1).sum(axis=1).astype('int')
        # Plot ranks Box
        plot_en = True #debug
        if plot_en:
            Cn = cn_ranks_plot(dimsMc, ranks, dimsM[:2])
        print('M rank {}'.format(sum(ranks)))
        rlen = sum(ranks)
    else:
        print('Single patch')
        Yd, vtids = compress_dblocks(M.reshape((np.prod(dimsM[:2]),dimsM[2]),order='F'),
                maxlag=maxlag,tsub=tsub, noise_norm=noise_norm,iterate=iterate,
                confidence=confidence, corr=corr,kurto=kurto,tfilt=tfilt,tfide=tfide,
                mean_th=mean_th, greedy=greedy,fudge_factor=fudge_factor,
                mean_th_factor=mean_th_factor, U_update=U_update,min_rank=min_rank,
                plot_en=plot_en,verbose=verbose)
        Yd = Yd.reshape(dimsM, order='F')
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


def compress_patches(patches,maxlag=10,tsub=1,noise_norm=False,
        iterate=False, confidence=0.90,corr=True,kurto=False,tfilt=False,
        tfide=False, share_th=True,greedy=False,fudge_factor=1.,
        mean_th_factor=1.,U_update=True,plot_en=False,
        min_rank=0,verbose=False):
    """
    Compress each patch
    patches is a list of d1xd2xT arrays
    assume a patch is a list of patches
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
    if corr==True and share_th==True:
        mean_th = covCI_wnoise(T,confidence=confidence,maxlag=maxlag)
        mean_th*=mean_th_factor
    else:
        mean_th = None
    for cpatch, data_in in enumerate(patches):
        if cpatch %1 ==0:
            print('Patch %d'%cpatch)
        start = time.time()
        try:
            dims_ = data_in.shape
            data_in = data_in.reshape((np.prod(dims_[:2]),dims_[2]), order='F')
            Yd_patch, keep1 = compress_dblocks(data_in,dims=dims_,maxlag=maxlag,tsub=tsub,
                    noise_norm=noise_norm, iterate=iterate,confidence=confidence,
                    corr=corr,kurto=kurto,tfilt=tfilt,tfide=tfide,mean_th=mean_th,
                    greedy=greedy,fudge_factor=fudge_factor,mean_th_factor=1.,
                    U_update=U_update,plot_en=plot_en,
                    min_rank=min_rank)
            Yds[cpatch] = pad(Yd_patch, (dxy, T), (0, 0))
            #print('Rank is %d'%len(keep1))
        except:
            print('Could not run patch %d'%cpatch)
        print('\tPatch %d run for %.f'%(cpatch,time.time()-start))
        if np.any(keep1):
            vtids[cpatch] = pad(keep1, (3, dxy), (0, 0))
    return Yds, vtids


def iterative_update_V(Y, U_hat,V_TF_,lambdas_=None,
        verbose=False,plot_en=False):
    """
    Run update for each vi wrt given lambdas_
    """
    U_hat_ = U_hat.copy()
    num_components, T = V_TF_.shape
    # difference operator
    diff = (np.diag(2*np.ones(T),0)+np.diag(-1*np.ones(T-1),1)+
            np.diag(-1*np.ones(T-1),-1))[1:T-1]

    # normalize each U st each component has unit L2 norm
    U_hat_n = preprocessing.normalize(U_hat_, norm='l2', axis=0)
    V_TF_2 = V_TF_.copy()
    for ii in range(num_components):
        # get idx of other components
        idx_ = np.setdiff1d(np.arange(num_components),ii)
        # subtract off the other components from Y
        R_ = Y - U_hat_[:,idx_].dot(V_TF_[idx_,:])
        # project U_i onto the residual
        V_ = U_hat_n[:,ii].T.dot(R_)
        #V_ = preprocessing.normalize(V_[np.newaxis,:], norm='l2')[0]
        if lambdas_ is None:
            # Estimate sigma_i
            #V_ = preprocessing.normalize(V_[np.newaxis,:], norm='l2')[0]
            noise_std_ = sp_filters.noise_estimator(V_[np.newaxis,:],
                    method='logmexp')
            print('V_i = argmin_V ||D^2 V_||_1 st ||V_i-V_||_2<sigma_i*sqrt(T)') if verbose else 0
            V_2 = c_l1tf_v_hat(V_,diff,noise_std_)[0]
        else: #if lambdas_[ii] is not None:
            #print(np.sum(V_**2))
            #V_ = preprocessing.normalize(V_[np.newaxis,:], norm='l2')[0]
            #print(np.sum(V_**2))
            V_2 = c_update_V(V_, diff, lambdas_[ii])
        V_TF_2[ii,:] = V_2.copy()#
        #V_TF_2[ii,:] = preprocessing.normalize(V_2[np.newaxis,:], norm='l2')[0]
        if plot_en:
            plt.figure(figsize=(10,5))
            plt.plot(V_,':')
            plt.plot(V_2,':')
            plt.show()
    # normalize each V to have unit L2 norm.
    V_TF_2 = preprocessing.normalize(V_TF_2, norm='l2')
    return V_TF_2


def denoise_dblocks(Y, U_hat, V_hat, dims=None, fudge_factor=1,
        maxlag=10, confidence=0.95, corr=True, mean_th=None,
        kurto=False, verbose=False,plot_en=False,U_update=True):
    """
    Greedy approach for denoising
    Inputs
    ------
    Y:              video dxT
    U:              spatial components (d x rank)
    V_hat:          temporal components (rank x T)
    fudge_factor:   fudge factor to scale noise_std \sigma by
    maxlag:         autocorrelation ftcn lag
    confidence:     confidence interval for null hypothesis
    corr
    kurto
    mean_th
    U_update: enforce L1 constraint on the spatial components
    verbose
    plot_en

    Outputs
    -------
    U_hat
    V_hat
    """
    # D^2 = discrete 2nd order difference operator
    T = V_hat.shape[1]
    diff = (np.diag(2*np.ones(T),0)+np.diag(-1*np.ones(T-1),1)+
            np.diag(-1*np.ones(T-1),-1))[1:T-1]
    # Counting
    rerun_1 = 1 # flag to run part (1)
    iteration = 0 # iteration # for part(1)

    if plot_en:
        for idx, Vt_ in enumerate(V_hat):
            plt.title('Temporal component %d'%idx)
            plt.plot(Vt_)
            plt.show()
     # Plot U
    if plot_en and (not dims==None):
        tmp_u= Y.dot(V_hat.T).T#V_hat.dot(Y.T)
        for ii in range(U_hat.shape[1]):
            plot_comp(U_hat[:,ii],tmp_u[ii,:],'Spatial component U,Y*Vt,R'+str(ii), dims[:2])

    #####################################################
    while rerun_1:
        ##############
        ### Initialize lambda_i and update V_i component(s):
        num_components = V_hat.shape[0]
        print('*Running Part (1) iter %d with %d components'
                %(iteration, num_components)) if verbose else 0
        #estimate noise coefficient sigma_i for temporal component
        #noise_std_ = np.asarray([sp_filters.noise_estimator(V_hat_[np.newaxis,:],
        #    method='logmexp',range_ff=[0.25,0.5]) for V_hat_ in V_hat])
        noise_std_ = sp_filters.noise_estimator(V_hat,method='logmexp')
        noise_std_ *= fudge_factor
        #print(np.sum(V_hat**2,1))
        print('solve V(i) = argmin_W ||D^2 W||_1 \n'
                +'\t st ||V_i-W||_2<fudge_factor*sigma_i*sqrt(T)')if verbose else 0
        outs_ = [c_l1tf_v_hat(V_hat[idx,:], diff, stdv)
                 for idx, stdv in enumerate(noise_std_)]
        V_TF, lambdas_ = map(np.asarray, zip(*np.asarray(outs_)))
        if plot_en:
            for idx, Vt_ in enumerate(V_TF):
                plt.title('Temporal component %d'%idx)
                plt.plot(V_hat[idx,:])
                plt.plot(Vt_)
                plt.show()
        # normalize each V to have unit L2 norm.
        #V_TF = preprocessing.normalize(V_TF, norm='l2')
        ####################
        ### Initialize nu_j and update U_j for pixel j:
        if U_update:
            print('solve U(j) = argmin_W ||W||_1 st ||Y_j-W\'V_TF(j)||_2^2<T*fudge^2') if verbose else 0
            outs_2 = [c_l1_u_hat(y, V_TF,fudge_factor) for y in Y]
            #outs_2 = update_U_parallel(Y,V_TF,fudge_factor)
            U_hat, nus_ = map(np.asarray,zip(*np.asarray(outs_2)))
        else:
            nus_= np.zeros((U_hat.shape[0],))
        # Plot U
        if plot_en and (not dims==None):
            tmp_u = Y.dot(V_TF.T).T#V_TF.dot(Y.T)
            for ii in range(U_hat.shape[1]):
                plot_comp(tmp_u[ii,:],U_hat[:,ii],'Spatial component: Y*V_TF, U,R '+str(ii), dims[:2])

        #################### Iterations
        num_min_iter = 5
        print('Iterate until F(U,V) stops decreasing significantly') if verbose else 0
        print('For now fixed iterations as %d'%num_min_iter) if verbose else 0
        F_UVs = []

        for k in range(num_min_iter):
            print('*Running Part (1) of iter %d with %d components'%(iteration, num_components)) if verbose else 0

            print('\tupdate V_i : min ||Y-UV||^2_2 + sum_i lambda_i ||D^2 V_i||_1') if verbose else 0
            V_TF = iterative_update_V(Y, U_hat,V_TF,lambdas_,plot_en=plot_en,verbose=verbose)

            if U_update:
                print('\tupdate U_j: min ||Y-UV||^2_2 + sum_j nu_j ||U_j||_1') if verbose else 0
                #U_hat = np.asarray([c_update_U(y,V_TF,nus_[idx]) for idx, y in enumerate(Y)])
                U_hat = np.asarray(c_update_U_parallel(Y,V_TF,nus_))
            # Plot U
            if plot_en and (not dims==None):
                tmp_u= Y.dot(V_TF.T).T#V_hat.dot(Y.T)
                for ii in range(U_hat.shape[1]):
                    plot_comp(tmp_u[ii,:],U_hat[:,ii],'Spatial component Y*V_TF, U, R'+str(ii), dims[:2])


            # F(U,V)=||Y-UV||^2_2 + sum_i lambda_i ||D^2 V_i||_1 + sum_j nu_j ||U_j||_1
            # due to normalization F(U,V) may not decrease monotonically. problem?
            F_uv1 = np.linalg.norm(Y - U_hat.dot(V_TF),2)**2
            F_uv2  = np.sum(lambdas_*np.sum(np.abs(diff.dot(V_TF.T)),0))
            F_uv3  = np.sum(nus_*np.sum(np.abs(U_hat),1)) if U_update else 0
            F_uv = F_uv1 + F_uv2 + F_uv3
            F_UVs.append(F_uv)
            print('\tIter %d errors (%d+%d+%d)=%d'%(k,F_uv1,F_uv2,F_uv3,F_uv)) if verbose else 0

            #plot_comp(Y,U_hat.dot(V_TF),'Y-Yd-RPixel variance', dims) if (plot_en and(not dims ==None)) else 0

        if plot_en:
            plt.title('Error F(u,v)')
            plt.plot(F_UVs)
            plt.show()

        print('*Running Part (2) of iter %d with %d components'%(iteration, V_TF.shape[0])) if verbose else 0

        ### (2) Compute PCA on residual R  and check for correlated components
        U_r, s_r, Vt_r = compute_svd((Y-U_hat.dot(V_TF)).astype('float32'), method='vanilla')
        # For greedy approach, only keep big highly correlated components
        ctid = choose_rank(Vt_r, maxlag=maxlag, confidence=confidence,
                       corr=corr, kurto=kurto, mean_th=mean_th)
        keep1_r = np.where(np.logical_or(ctid[0, :] == 1, ctid[1, :] == 1))[0]
        plot_vt_cov(Vt_r,keep1_r,maxlag) if plot_en else 0
        if len(keep1_r)==0:
            print('Final number of components %d'%V_TF.shape[0]) if verbose else 0
            rerun_1 = 0
        else:
            print('Iterate (1) since adding %d components'%(len(keep1_r))) if verbose else 0
            rerun_1 = 1
            V_hat = np.vstack((V_TF, Vt_r[keep1_r,:]))
            U_hat = np.hstack((U_hat,U_r[:,keep1_r].dot(np.diag(s_r[keep1_r]))))
            iteration +=1
    ##################
    ### Final update
    print('Running final update') if verbose else 0
    #################
    V_TF = iterative_update_V(Y, U_hat,V_TF,lambdas_=None,plot_en=plot_en)
    if U_update:
        print('U(j) = argmin_W ||W||_1 st ||Y_j-W\'V_TF(j)||_2^2<T') if verbose else 0
        outs_ = [c_l1_u_hat(y,V_TF,1) for y in Y]
        U_hat, _ = map(np.asarray,zip(*np.asarray(outs_)))
    else:
        U_hat = Y.dot(np.linalg.pinv(V_TF))

    #plot_comp(Y,U_hat.dot(V_TF),'Frame 0', dims, idx_=0) if (plot_en and(not dims ==None)) else 0
    # Plot U
    if plot_en and (not dims==None):
        tmp_u= Y.dot(V_TF.T).T#V_TF.dot(Y.T)
        for ii in range(U_hat.shape[1]):
            plot_comp(tmp_u[ii,:],U_hat[:,ii],'Spatial component Y*V_TF, U_hat, R'+str(ii), dims[:2])

    # this needs to be updated to reflect any new rank due to new numb of iterations
    return U_hat , V_TF


def compress_dblocks(data_all, dims=None, maxlag=10, tsub=1, ds=1,
        noise_norm=False, iterate=False, confidence=0.90,corr=True,
        kurto=False, tfilt=False, tfide=False, mean_th=None,
        greedy=False, mean_th_factor=1.,p=1.,fudge_factor=1.,
        plot_en=False,verbose=False,U_update=False,
        min_rank=0):
    """
    Compress a single block
    Inputs
    ------
    data_all:    video (d x T) or (d1 xd2 x T)
    maxlag:     max_corr lag
    tsub:       temporal downsample
    ds:         spatial downsample
    noise_norm: placeholder
    iterate:    Find corr components iteratively
    confidence: CI for corr null hypothesis
    corr:       Identify components via corr
    kurto:      Identify components via kurtosis
    tfilt:      Temporally filter traces with AR estimate of order p.
    p:          order of AR fc, used if tfilt is True
    tfide:      Denoise temporal traces with TF
    mean_th:    threshold employed to reject components wrt corr
    greedy:     Calculate spatial and temporal estimated with a greedy approach
    mean_th_factor: factor to scale mean_th with if goal is to keep certain components
    Outputs
    -------
    Yd_out:     Denoised video (dxT)
    ctids:      component matrix (corr-kurto-reject)
    """
    if data_all.ndim ==3:
        dims = data_all.shape
        data_all = data_all.reshape((np.prod(dims[:2]),dims[2]), order='F')
    data_all = data_all.T.astype('float32')
    # In a 2d matrix, we get rid of any broke (inf) pixels
    # we assume fixed broken across time
    broken_idx = np.isinf(data_all[0,:])
    # Work only on good pixels
    if np.any(broken_idx):
        print('broken pixels') if verbose else 0
        data = data_all[:, ~broken_idx]
    else:
        data = data_all.copy()

    # Remove the mean
    mu = data.mean(0, keepdims=True)
    data = data - mu

    # temporally filter the data
    if tfilt:
        print('Apply exponential filter') if verbose else 0
        data0 = np.zeros(data.shape)
        #T, num_pxls = data.shape
        for ii, trace in enumerate(data.T):
            # Estimate tau for exponential
            tau = cnmf.deconvolution.estimate_time_constant(
                    trace,p=p,sn=None,lags=5,fudge_factor=1.)
            window = tau **range(0,100)
            data0[:,ii] = np.convolve(fluor,window,mode='full')[:T]/np.sum(window)
    else:
        data0 = data.copy()

    # temporally decimate the data
    if tsub > 1:
        print('Temporal decimation %d'%tsub) if verbose else 0
        data0 = temporal_decimation(data0, tsub)

    # spatially decimate the data
    if ds > 1:
        print('Spatial decimation %d'%ds) if verbose else 0
        #D = len(dims)
        #ds = np.ones(D-1).astype('uint8')
        #data0 = spatial_decimation(data0, ds, dims)
        data0 = data0.copy()
    # Run svd
    U, s, Vt = compute_svd(data0.T, method='vanilla')

    # Project back if temporally filtered or downsampled
    if tfilt or tsub > 1:
        Vt = U.T.dot(data.T)

    # if greedy Force x2 mean_th (store only big components)
    if greedy and (mean_th_factor <= 1.):
        mean_th_factor = 2.

    # Select components
    if mean_th is None:
        mean_th = covCI_wnoise(Vt.shape[1],confidence=confidence,maxlag=maxlag)
        mean_th *= mean_th_factor

    ctid = choose_rank(Vt, maxlag=maxlag, iterate=iterate,
            confidence=confidence, corr=corr, kurto=kurto,
            mean_th=mean_th,min_rank=min_rank)
    keep1 = np.where(np.logical_or(ctid[0, :] == 1, ctid[1, :] == 1))[0]

    # Plot temporal correlations
    plot_vt_cov(Vt,keep1,maxlag) if plot_en else 0

    # If no components to store, return block as it is
    if np.all(keep1 == np.nan):
        Yd = np.zeros(data.T.shape)
        Yd += mu.T
        return Yd, ctid

    Vt = Vt[keep1,:]

    # Denoise each temporal component
    if tfide:
        noise_levels = sp_filters.noise_estimator(Vt)
        Vt = tfd.denoise(Vt, stdvs = noise_levels)

    if tfide and (tfilt or tsub > 1):
        U = data.T.dot(np.linalg.pinv(Vt).T)
    else:
        U = U[:,keep1].dot(np.eye(len(keep1))*s[keep1.astype(int)])

    # call greedy
    if greedy:
        start = time.time()
        try:
            U, Vt = denoise_dblocks(data.T, U, Vt, dims=dims,
                    fudge_factor=fudge_factor, maxlag=maxlag,
                    confidence=confidence, corr=corr,
                    kurto=kurto, mean_th=mean_th,U_update=U_update,
                    plot_en=plot_en,verbose=verbose)
            ctid[0,np.arange(Vt.shape[0])]=1
        except:
            print('ERROR: Greedy solving failed, using default parameters')

        #print('\t\tGreedy run for %.f'%(time.time()-start))
    # Reconstuct matrix and add mean
    Yd = U.dot(Vt) + mu.T
    # return original matrix and add any broken pixels
    if broken_idx.sum() > 0:
        print('ERROR: There are {} broken pixels.'.format(broken_idx.sum()))
        Yd_out = np.ones(shape=data_all.shape).T*np.inf
        Yd_out[~broken_idx,:] = Yd
    else:
        Yd_out =  Yd

    return Yd_out, ctid


def plot_vt_cov(Vt1,keep1, maxlag):
    fig, axarr = plt.subplots(1,2,sharey=True)
    loose = np.setdiff1d(np.arange(Vt1.shape[0]),keep1)
    for keep in keep1:
        vi = Vt1[keep,:]
        vi = (vi-vi.mean())/vi.std()
        metric = axcov(vi,maxlag)[maxlag:]/vi.var()
        axarr[0].plot(metric,':k')

    for lost in loose:
        vi = Vt1[lost,:]
        vi = (vi-vi.mean())/vi.std()
        metric = axcov(vi,maxlag)[maxlag:]/vi.var()
        axarr[1].plot(metric,':k')

    ttext =['Selected components: %d'%(len(keep1)),
        'Discarded components: %d'%(len(loose))]
    for ii, ax in enumerate(axarr):
        ax.set_xscale('symlog')
        ax.set_ylabel('ACF')
        ax.set_xlabel('lag')
        ax.set_yticks(())
        ax.set_title(ttext[ii])
    plt.show()
    return


def compute_svd(M, method='randomized', n_components=400):
    """
    Compute svd on centered M
    M d x T
    """

    if method == 'vanilla':
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
    elif method == 'randomized':
        U, s, Vt = randomized_svd(M, n_components=n_components,
                n_iter=7, random_state=None)
    return U, s, Vt


def choose_rank(Vt,maxlag=10,iterate=False,confidence=0.90,
        corr=True,kurto=False,mean_th=None,mean_th_factor=1.,min_rank=0):
    """
    Select rank wrt axcov and kurtosis
    """
    n, L = Vt.shape
    vtid = np.zeros(shape=(3, n)) * np.nan

    # Null hypothesis: white noise ACF
    if corr is True:
        if mean_th is None:
            mean_th = covCI_wnoise(L,confidence=confidence,maxlag=maxlag)
        mean_th*= mean_th_factor
        keep1 = cov_one(Vt, mean_th = mean_th, maxlag=maxlag, iterate=iterate,min_rank=min_rank)
    else:
        keep1 = []
    if kurto is True:
        keep2 = kurto_one(Vt)
    else:
        keep2 = []

    keep = list(set(keep1 + keep2))
    loose = np.setdiff1d(np.arange(n),keep)
    loose = list(loose)
    vtid[0, keep1] = 1  # components stored due to cov
    vtid[1, keep2] = 1  # components stored due to kurto
    vtid[2, loose] = 1  # extra components ignored
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


def covCI_wnoise(L, confidence=0.90, maxlag=10, n=3000):
    """
    Compute mean_th for auto covariance of white noise
    """
    # th1 = 0
    #print 'confidence is {}'.format(confidence)
    covs_ht = np.zeros(shape=(n,))
    for sample in np.arange(n):
        ht_data = np.random.randn(L)
        covdata = axcov(ht_data, maxlag)[maxlag:]/ht_data.var()
        covs_ht[sample] = covdata.mean()
        #covs_ht[sample] = np.abs(covdata[1:]).mean()
    #hist, _,_=plt.hist(covs_ht)
    #plt.show()
    mean_th = mean_confidence_interval(covs_ht, confidence)
    #print('th is {}'.format(mean_th))
    return mean_th


def temporal_decimation(data, mb):
    """
    Decimate data by mb, new frame is mean of decimated frames
    """
    data0 = data[:int(len(data)/mb*mb)].reshape((-1, mb) + data.shape[1:]).mean(1).astype('float64')
    return data0


def spatial_decimation(data,ds,dims):
    """
    Decimate data by ds, smaller frame is mean of decimated frames
    """
    #data0 = data.reshape(len(data0), dims[1] / ds[0], ds[0], dims[2] / ds[1], ds[1]).mean(2).mean(3)
    data0  = data.copy()
    return data0


def cov_one(Vt, mean_th=0.10, maxlag=10, iterate=False,
        extra=1,min_rank=0,verbose=False):
    """
    Compute only auto covariance,
    calculate iteratively: until mean reaches mean_th
    """
    keep = []
    num_components = Vt.shape[0]
    #print('mean_th is %s'%mean_th)
    for vector in range(0, num_components):
        # standarize and normalize
        vi = Vt[vector, :]
        vi =(vi - vi.mean())/vi.std()
        vi_cov = axcov(vi, maxlag)[maxlag:]/vi.var()
        #print(vi_cov.mean())
        if vi_cov.mean() < mean_th:
            if iterate is True:
                break
        else:
            keep.append(vector)
    #print(keep)
    #if not keep:
    #    print('from cov Empty')
    #else:
        # verbose
        #print('from cov {}'.format(len(keep)))
    # Store extra components
    if vector < num_components and extra != 1:
        extra = min(vector*extra,Vt.shape[0])
        for addv in range(1, extra-vector+ 1):
            keep.append(vector + addv)
    # Forcing one components
    # delete
    #if not np.any(keep) and min_rank >0:
    if not keep and min_rank>0:
        # vector is empty for once min
        keep.append(0)
        print('Forcing one component')
    #if not keep:
    #    print('from cov still empty')
    #    print(keep)
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


def combine_blocks(dimsM, Mc, dimsMc=None,
        list_order='C', array_order='F'):
    """
    Reshape blocks given by compress_blocks
    list order 'F' or 'C' : if Mc as list, assumes order is in such format
    array order if dxT instead of d1xd2xT assumes always array_order='F'
    NOTE: if dimsMC is NONE then MC must be a d1 x d2 x T matrix
    """

    d1, d2, T = dimsM
    if type(Mc)==list:
        k = len(Mc)
    elif type(Mc)==np.ndarray:# or type(Mc)==np.array:
        k = Mc.shape[0]
    else:
        print(type(Mc))
        print(Mc.shape)
        print('error= must be np.array or list')
    Mall = np.zeros(shape=(d1, d2, T))*np.nan
    if dimsMc is None:
        dimsMc = np.asarray(list(map(np.shape,Mc)))
    i, j = 0, 0
    for ii, Mn in enumerate(Mc):
        # shape of current block
        d1c, d2c, Tc = dimsMc[ii]
        if (np.isnan(Mn).any()):
            Mn = unpad(Mn)
        if Mn.ndim < 3:
            Mn = Mn.reshape((d1c, d2c)+(T,), order=array_order)
        if list_order=='F':
            Mall[i:i+d1c, j:j+d2c, :] = Mn
            i += d1c
            if i == d1:
                j += d2c
                i = 0
        else:
            Mall[i:i+d1c, j:j+d2c, :] = Mn
            j += d2c
            if j == d2:
                i += d1c
                j = 0
    return Mall


def cn_ranks(dim_block, ranks, dims):
    """
    """
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

    d1, d2 = dims[:2]//np.min(dims[:2])
    fig, ax3 = plt.subplots(1,1,figsize=(d1*5,d2*5))
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

### Additional Functions for greedy denoised (called in denoise_dblocks)


def c_l1tf_v_hat(v,diff,sigma):
    """
    V(i) = argmin_W ||D^2 W||_1
    st ||V_i-W||_2<sigma_i*sqrt(T)
    Include optimal lagrande multiplier for constraint
    """
    T = len(v)
    v_hat = cp.Variable(T)
    objective = cp.Minimize(cp.norm(cp.matmul(diff,v_hat),1))
    constraints = [cp.norm(v-v_hat,2)<=sigma*np.sqrt(T)]
    cp.Problem(objective, constraints).solve(solver='CVXOPT')#verbose=False)
    return v_hat.value, constraints[0].dual_value

def c_l1_u_hat(y,V_TF,fudge_factor,regression=True):
    """
    U(j) = argmin_W ||W||_1
    st ||Y_j-W'V_TF(j)||_2^2 < T
    if problem infeasible:
        set U = regression Vt onto Y and \nu = 0
    """
    num_components = V_TF.shape[0]
    u_hat = cp.Variable((1,num_components))
    objective = cp.Minimize(cp.norm(u_hat,1))
    constraints = [cp.norm(
        y[np.newaxis,:] - cp.matmul(u_hat,V_TF),2) <= np.sqrt(len(y))*fudge_factor]
    problem = cp.Problem(objective,constraints)
    problem.solve(solver='CVXOPT')
    if problem.status in ["infeasible", "unbounded"]:
        return y[np.newaxis,:].dot(np.linalg.pinv(V_TF)).flatten(), 0
    else:
        return u_hat.value.flatten(), constraints[0].dual_value


def c_update_V(v,diff,lambda_):
    """
    min ||Y-UV||_2^2 + sum_i lambda_i||D^2V_i||_1
    # Fixing U we have
    min ||v-v_hat||_2^2 + lambda_i||D^2V_i||_1
    """
    v_hat = cp.Variable(len(v))
    if lambda_ == 0:
        objective = cp.Minimize(
            cp.norm(v-v_hat,2)**2)
    else:
        objective = cp.Minimize(
            cp.norm(v-v_hat,2)**2
            + lambda_*cp.norm(cp.matmul(diff,v_hat),1))
    cp.Problem(objective).solve(solver='CVXOPT')
    return v_hat.value


def c_update_U(y,V_TF,nu_):
    """
    min ||Y-UV||^2_2 + sum_j nu_j ||U_j||_1.
    for each pixel
    min  ||y_j-u_j*v||_2^2 + nu_j ||u_j||_1.
    """
    num_components = V_TF.shape[0]
    u_hat = cp.Variable((1,num_components))
    if nu_ == 0:
        objective = cp.Minimize(
                cp.norm(y[np.newaxis,:]-cp.matmul(u_hat,V_TF),2)**2)
    else:
        objective = cp.Minimize(
                cp.norm(y[np.newaxis,:]-cp.matmul(u_hat,V_TF),2)**2+ nu_*cp.norm(u_hat,1))
    problem = cp.Problem(objective)
    problem.solve(solver='CVXOPT')
    return u_hat.value.flatten()


def plot_comp(Y,Y_hat,title_,dims,idx_=0):
    """
    Plot comparison for frame idx_ in Y, Y_hat.
    assume Y is in dxT to be reshaped to dims=(d1,d2,T)
    """
    R = Y - Y_hat
    fig, ax = plt.subplots(1,3,figsize=(15,6))
    ax[0].set_title(title_)

    for ax_ , arr in zip (ax,[Y,Y_hat,R]):
        if len(dims)>2:
            ims = ax_.imshow(arr.reshape(dims,order='F')[:,:,idx_])
        else:
            ims = ax_.imshow(arr.reshape(dims,order='F'))
        #ims = ax_.imshow(arr.reshape(dims,order='F').var(2))
        d = make_axes_locatable(ax_)
        cax0 = d.append_axes("bottom", size="5%", pad=0.5)
        cbar0 = plt.colorbar(ims, cax=cax0, orientation='horizontal',format='%.0e')
    plt.tight_layout()
    plt.show()
    return

###################
# Additional Functions for 4 offgrid denoisers
###################


def pyramid_function(dims):
    """
    Base on dims create respective pyramid function
    pyramid function is 0 at boundary and 1 at center
    """
    a_k = np.zeros(dims[:2])
    xc,yc = dims[0]//2,dims[1]//2

    for ii in range(dims[0]):
        for jj in range(dims[1]):
            a_k[ii,jj] = max(np.abs(xc-ii),np.abs(yc-jj))
    a_k = 1-a_k/a_k.max()

    if False:
        plt.figure(figsize=(10,10))
        plt.imshow(a_k)
        plt.colorbar()
    a_k = np.array([a_k,]*dims[2]).transpose([1,2,0])
    return a_k


def compute_ak(dims_rs,W_rs,list_order='C'):
    """
    Get ak pyramid matrix wrt center
    dims_rs: dimension of matrix
    W_rs: list of pacthes in order F

    """
    dims_ = np.asarray(list(map(np.shape,W_rs)))
    a_ks = []
    for dim_ in dims_:
        a_k = pyramid_function(dim_)
        a_ks.append(a_k)
    # given W_rs and a_ks reconstruct matrix
    a_k = combine_blocks(dims_rs,a_ks,dims_,list_order=list_order)
    if False:
        plt.imshow(a_k[:,:,0])
        plt.colorbar()
    return a_k


def denoisers_off_grid(W,k,maxlag=10,tsub=1,noise_norm=False,
        iterate=False, confidence=0.95,corr=True,kurto=False,
        tfilt=False,tfide=False, share_th=True,greedy=False,
        fudge_factor=1., mean_th_factor=1.,U_update=True):
    """
    WORK IN PROGRESS --- vanilla implementation
    Calculate four denoisers st each denoiser is in a new grid,
    (Given original tiling grid
    each additional grid has a 1/2 offset vertically, horizontally or both.
    this to minimize the number of block artifacts
    downside: x4 redundancy.
    vanilla implementation

    Inputs
    ------
    """
    # Given an video W d1 x d2 x T
    # split into patches
    patches = split_image_into_blocks(W,k)

    # calculate dimensionality
    dim_block = np.asarray(list(map(np.shape,patches)))
    K = int(np.sqrt(len(dim_block)))
    cols, rows = dim_block.T[0],dim_block.T[1]
    row_array = np.insert(rows[::K+1],0,0).cumsum()
    col_array = np.insert(cols[::K+1],0,0).cumsum()
    x,y = np.meshgrid(row_array[:-1],col_array[:-1])

    r_offset = np.diff(row_array)//2
    c_offset = np.diff(col_array)//2
    row_cut = row_array[:-1] + r_offset
    col_cut = col_array[:-1] + c_offset

    # Get three additional grids to denoise
    W_rows = np.array_split(W[:,row_cut[0]:row_cut[-1],:],(row_cut+r_offset)[:-2],axis=1)
    func_c = lambda x: (np.array_split(x,(col_cut+c_offset)[:-1],axis=0))
    W_r_off = list(map(func_c,W_rows))

    W_cols = np.array_split(W[col_cut[0]:col_cut[-1],:,:],row_array[1:-1],axis=1)
    func_c = lambda x: (np.array_split(x,(col_cut+c_offset)[:-2],axis=0))
    W_c_off = list(map(func_c,W_cols))

    Wrc_col = np.array_split(W[col_cut[0]:col_cut[-1],row_cut[0]:row_cut[-1],:],(row_cut+r_offset)[:-2],axis=1)
    func_c = lambda x: (np.array_split(x,(col_cut+c_offset)[:-2],axis=0))
    W_rc_off = list(map(func_c,Wrc_col))

    W_rs = [y for x in W_r_off for y in x]
    W_cs = [y for x in W_c_off for y in x]
    W_rcs = [y for x in W_rc_off for y in x]

    dims_rs = W[:,row_cut[0]:row_cut[-1],:].shape
    dims_cs = W[col_cut[0]:col_cut[-1],:,:].shape
    dims_rcs = W[col_cut[0]:col_cut[-1],row_cut[0]:row_cut[-1],:].shape

    # Get pyramid functions for each grid
    ak0,ak1,ak2,ak3 = [np.zeros(W.shape)]*4
    ak0 = compute_ak(W.shape,patches,list_order='C')
    ak1[:,row_cut[0]:row_cut[-1],:] = compute_ak(dims_rs,W_rs,list_order='F')
    ak2[col_cut[0]:col_cut[-1],:,:] = compute_ak(dims_cs,W_cs,list_order='F')
    ak3[col_cut[0]:col_cut[-1],row_cut[0]:row_cut[-1],:] = compute_ak(dims_rcs,W_rcs,list_order='F')

    # Force outer most border = 1
    #ak0[[0,-1],[0,-1],:]=1
    ak0[0,:,:]=1
    ak0[-1,:,:]=1
    ak0[:,0,:]=1
    ak0[:,-1,:]=1
    # if output a bunch of lists this will take forever
    #return ak0,ak1,ak2,ak3,patches,W_rs,W_cs,W_rcs

    # Here are one by one
    # here we need maxlag and confidence

    #dpatch = []
    #for ii, patch in enumerate(patches):
    #    Yd,_ = pre_svd.svd_patch(patch,k=1,
    #            maxlag=maxlag,confidence=confidence,greedy=greedy,
    #            fudge_factor=fudge_factor, mean_th_factor=mean_th_factor)
    #    dpatch.append(Yd)

    Yd ,_ = svd_patch(W,k=k,maxlag=maxlag,confidence=confidence,
            greedy=greedy,fudge_factor=fudge_factor, 
            mean_th_factor=mean_th_factor,U_update=U_update,
            min_rank=min_rank)
    ##
    dW_rs = []
    for ii, patch in enumerate(W_rs):
        print('running %d out of %d'%(ii,len(W_rs)))
        start=time.time()
        Yd1,_ = svd_patch(patch,k=1,maxlag=maxlag,
                                 confidence=confidence,greedy=greedy,
                                 fudge_factor=fudge_factor,
                                 mean_th_factor=mean_th_factor,
                                 U_update=U_update,min_rank=min_rank)
        dW_rs.append(Yd1)
        print('Run for %.f'%(time.time()-start))
    ##
    dW_cs = []
    for ii, patch in enumerate(W_cs):
        print('running %d '%ii)
        start=time.time()
        Yd2,_ = svd_patch(patch,k=1,maxlag=maxlag,
                                 confidence=confidence,greedy=greedy,
                                 fudge_factor=fudge_factor,
                                 mean_th_factor=mean_th_factor,
                                 U_update=U_update,min_rank=min_rank)
        dW_cs.append(Yd2)
        print('Run for %.f'%(time.time()-start))
        ##

    dw_rcs = []
    for ii, patch in enumerate(w_rcs):
        print('running %d out of %d'%(ii,len(w_rcs)))
        start=time.time()
        Yd3,_ = svd_patch(patch,k=1,maxlag=maxlag,
                                 confidence=confidence,greedy=greedy,
                                 fudge_factor=fudge_factor,
                                 mean_th_factor=mean_th_factor,
                                 U_update=U_update,min_rank=min_rank)
        dw_rcs.append(Yd3)
        print('Run for %.f'%(time.time()-start))
    ##
    #W0,W1,W2, W3 = [np.zeros(W.shape)]*4
    W0 = np.zeros(W.shape)
    W1 = np.zeros(W.shape)
    W2 = np.zeros(W.shape)
    W3 = np.zeros(W.shape)

    W0 = Yd.copy()#combine_blocks(W.shape,patches,list_order='C')
    W1[:,row_cut[0]:row_cut[-1],:] = combine_blocks(dims_rs,dW_rs,list_order='F')
    W2[col_cut[0]:col_cut[-1],:,:] =combine_blocks(dims_cs,dW_cs,list_order='F')
    W3[col_cut[0]:col_cut[-1],row_cut[0]:row_cut[-1],:] = combine_blocks(dims_rcs,dW_rcs,list_order='F')


    if False:
        for ak_ in [ak0,ak1,ak2,ak3]:
            plt.imshow(ak[:,:,0])
            plt.show()

    W_hat = (ak0*W0+ak1*W1+ak2*W2+ak3*W3)/(ak0+ak1+ak2+ak3)

    return W_hat


#### trial for parallel implementation

def update_u_parallel(Y,V_TF,fudge_factor):
    pool = multiprocessing.Pool(processes=20)
    c_outs = pool.starmap(c_l1_u_hat, itertools.product(y, V_TF, fudge_factor))
    pool.close()
    pool.join()
    return c_outs


def c_update_U_parallel(Y,V_TF,nus_):
    """
    call c_update_U as queue
    """
    pool = multiprocessing.Pool()#processes=20)
    args = [(y,V_TF,nus_[idx]) for idx, y in enumerate(Y)]
    c_outs = pool.starmap(c_update_U, args)

    pool.close()
    pool.join()
    return c_outs

