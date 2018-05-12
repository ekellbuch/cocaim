import numpy as np
import scipy as sp

import sklearn
from scipy.stats import norm
from sklearn.utils.extmath import randomized_svd
#from sklearn import preprocessing
from skimage.transform import downscale_local_mean

import concurrent
import cvxpy as cp
import multiprocessing
import itertools
import time

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# see how this changes with get_noise_fft
#import spatial_filtering as sp_filters

import trefide
from trefide.temporal import TrendFilter
import denoise
#import trefide_old
import util_plot as uplot
import tools as tools_
import noise_estimator
from l1_trend_filter.l1_tf_C.c_l1_tf import l1_tf# cython


def difference_operator(len_signal):
    # Gen Diff matrix
    diff_mat = (np.diag(2 * np.ones(len_signal), 0) +
                np.diag(-1 * np.ones(len_signal - 1), 1) +
                np.diag(-1 * np.ones(len_signal - 1), -1)
                )[1:len_signal - 1]
    return diff_mat


def stimulus_segmentation(T,
                          stim_knots=None,
                          stim_delta=10):
    """
    Returns boolean areas to ignore in frame
    """
    if stim_knots is None:
        return np.zeros(T).astype('bool')

    stim_knots = np.where(np.abs(np.diff(stim_knots))>1)[0]

    ignore_frames =np.zeros(T)

    for knot in stim_knots:
        arg_np = (np.ones(stim_delta).cumsum()-stim_delta//2+knot).astype('int')
        ignore_frames[arg_np]=1
    ignore_frames=ignore_frames.astype('bool')

    return ignore_frames


def mean_confidence_interval(data,
                            confidence=0.99,
                            one_sided=False):
    """
    Compute mean confidence interval (CI)
    for a normally distributed population
    _____________

    Parameters:
    ___________
    data:       np.array  (L,)
                input vector from which to calculate mean CI
                assumes gaussian-like distribution
    confidence: float
                confidence level for test statistic
    one_sided:  boolean
                enforce a one-sided test

    Outputs:
    _______
    th:         float
                threshold for mean value at CI
    """
    if one_sided:
        confidence = 1 - 2*(1-confidence)

    _, th = sp.stats.norm.interval(confidence,
                                loc =data.mean(),
                                scale=data.std())
    return th


def choose_rank(Vt,
                maxlag=10,
                iterate=False,
                confidence=0.90,
                corr=True,
                kurto=False,
                mean_th=None,
                mean_th_factor=1.,
                min_rank=0,
               enforce_both=False):
    """
    Select rank vectors in Vt which pass test statistic(s) enabled
    (e.g. axcov and/or kurtosis)

    __________
    Parameters:

    Vt:         np.array (k x T)
                array of k temporal components lasting T samples
    maxlag:     int
                max correlation lag for correlation null hypothesis in samples
                (e.g. indicator decay in samples)
    iterate:    boolean
                flag to include correlated components iteratively
    confidence: float
                confidence interval (CI) for correlation null hypothesis
    corr:       boolean
                flag to include components which pass correlation null hypothesis
    mean_th:    float
                threshold employed to reject components according to
                correlation null hypothesis
    mean_th_factor: float
                factor to scale mean_th
                typically set to 2 if greedy=True and mean_th=None or
                if mean_th has not been scaled yet.
    min_rank:   int
                minimum number of components to include in output
                even if no components of Vt pass any test

    Outputs:
    -------

    vtid:       np.array (3,d)
                indicator 3D matrix (corr-kurto-reject) which points which statistic
                a given component passed and thus it is included.
                can vary according to min_rank

    """
    if enforce_both:
        corr= True
        #kurto = True

    n, L = Vt.shape
    vtid = np.zeros(shape=(3, n)) * np.nan

    # Null hypothesis: white noise ACF
    if corr is True:
        if mean_th is None:
            mean_th = wnoise_acov_CI(L,
                                    confidence=confidence,
                                    maxlag=maxlag)
        mean_th*= mean_th_factor
        keep1 = vector_acov(Vt,
                            mean_th = mean_th,
                            maxlag=maxlag,
                            iterate=iterate,
                            min_rank=min_rank)
    else:
        keep1 = []
    if kurto is True:
        keep2 = kurto_one(Vt)
    else:
        keep2 = []

    keep = list(set(keep1 + keep2))
    loose = np.setdiff1d(np.arange(n),keep)
    loose = list(loose)

    if enforce_both:
        keep1 = np.intersect1d(keep1,keep2)
        keep2 = keep1
        print(len(keep1))


    vtid[0, keep1] = 1  # components stored due to cov
    vtid[1, keep2] = 1  # components stored due to kurto
    vtid[2, loose] = 1  # extra components ignored
    # print('rank cov {} and rank kurto {}'.format(len(keep1),len(keep)-len(keep1)))
    return vtid


def wnoise_acov_CI(L,
                    confidence=0.99,
                    maxlag=10,
                    n=3000,
                    plot_en=False):
    """
    Generate n AWGN vectors lasting L samples.
    Calculate the mean of the ACF of each vector for 0:maxlag
    Return the mean threshold with specified confidence.

    Parameters:
    ----------

    L:          int
                length of vector
    confidence: float
                confidence level for test statistic
    maxlag:     int
                max correlation lag for correlation null hypothesis
                in samples (e.g. indicator decay in samples)
    n:          int
                number of standard normal vectors to generate

    plot_en:    boolean
                plot histogram of pd
    Outputs:
    -------

    mean_th:    float
                value of mean of ACFs of each standard normal vector at CI.
    """

    # th1 = 0
    #print 'confidence is {}'.format(confidence)
    covs_ht = np.zeros(shape=(n,))
    for sample in np.arange(n):
        ht_data = np.random.randn(L)
        covdata = tools_.axcov(ht_data,
                            maxlag)[maxlag:]/ht_data.var()

        covs_ht[sample] = covdata.mean()
        #covs_ht[sample] = np.abs(covdata[1:]).mean()
    #hist, _,_=plt.hist(covs_ht)
    #plt.show()
    mean_th = mean_confidence_interval(covs_ht,confidence)
    return mean_th


def vector_acov(Vt,
                mean_th=0.10,
                maxlag=10,
                iterate=False,
                extra=1,
                min_rank=0,
                verbose=False):
    """
    Calculate auto covariance of row vectors in Vt
    and output indices of vectors which pass correlation null hypothesis.

    Parameters:
    ----------
    Vt:         np.array(k x T)
                row array of compoenents on which to test correlation null hypothesis
    mean_th:    float
                threshold employed to reject components according to correlation null hypothesis
    maxlag:     int
                determined lag until which to average ACF of row-vectors for null hypothesis
    iterate:    boolean
                flag to include components which pass null hypothesis iteratively
                (i.e. if the next row fails, no additional components are added)
    extra:      int
                number of components to add as extra to components which pass null hypothesis
                components are added in ascending order corresponding to order in mean_th
    min_rank:   int
                minimum number of components that should be included
                add additional components given components that (not) passed null hypothesis
    verbose:    boolean
                flag to enable verbose

    Outputs:
    -------
    keep:       list
                includes indices of components which passed the null hypothesis
                and/or additional components added given parameters
    """

    keep = []
    num_components = Vt.shape[0]
    print('mean_th is %s'%mean_th) if verbose else 0
    for vector in range(0, num_components):
        # standarize and normalize
        vi = Vt[vector, :]
        vi =(vi - vi.mean())/vi.std()

        print('vi ~ (mean: %.3f,var:%.3f)'%(vi.mean(),
                                            vi.var())) if verbose else 0

        vi_cov = tools_.axcov(vi,
                            maxlag)[maxlag:]/vi.var()

        if vi_cov.mean() < mean_th:
            if iterate is True:
                break
        else:
            keep.append(vector)
    # Store extra components
    if vector < num_components and extra != 1:
        extra = min(vector*extra,
                    Vt.shape[0])

        for addv in range(1, extra-vector+ 1):
            keep.append(vector + addv)
    # Forcing one components
    if not keep and min_rank>0:
        # vector is empty for once min
        keep.append(0)
        print('Forcing one component') if verbose else 0
    return keep


def compute_svd(M,
                method='vanilla',
                n_components=2,
                n_iter=7,
                min_value_ = 1e-6,
                random_state=None,
               reconstruct=False):
    """
    Decompose array M given parameters.
    asumme M has been mean_subtracted

    Parameters:
    ----------

    M:          np.array (d xT)
                input array to decompose
    method:     string
                method to decompose M
                ['vanilla','randomized']
    n_components: int
                number of components to extract from M
                if method='randomized'

    Outputs:
    -------

    U:          np.array (d x k)
                left singular vectors of M
                k = n_components if method='randomized'
    Vt:         np.array (k x T)
                right singular vectors of M
                k = n_components if method='randomized'
    s:          np.array (k x 1)
                variance of components
                k = n_components if method='randomized'
    """
    if M.max() < 1e-3:
        print('Not feasibe for small numbers ')

    if method == 'vanilla':
        U, s, Vt = np.linalg.svd(M, full_matrices=False)

    elif method == 'randomized':
        U, s, Vt = randomized_svd(M,
                                n_components=n_components,
                                n_iter=n_iter,
                                random_state=random_state)

    if reconstruct:
        return U.dot(np.diag(s).dot(Vt))
    return U, s, Vt


def temporal_decimation(data,
                        mb=1):
    """
    Decimate data by mb
    new frame is mean of decimated frames

    Parameters:
    ----------
    data:       np.array (T x d)
                array to be decimated wrt first axis

    mb:         int
                contant by which to decimate data
                (i.e. number of frames to average for decimation)

    Outputs:
    -------
    data0:      np.array (T//mb x d)
                temporally decimated array given data
    """
    data0 = data[:int(len(data)/mb*mb)].reshape(
        (-1, mb) + data.shape[1:]).mean(1).astype('float32')
    return data0


def spatial_decimation(data,
                       ds=1,
                       dims=None):
    """

    Decimate data spatially by ds
    pixel is avg of ds pixels
    Parameters:
    ----------
    data:       np.array (T x d) or (d1x d2 x T)
                array to be decimated wrt second axis

    ds:         int
                contant by which to decimate data
                (i.e. number of pixels to average for decimation)

    dims:       np.array or tupe (d1,d2,T)
                dimensions of data

    Outputs:
    -------
    data0:      np.array (T x d//ds)
                spatially decimated array given data
    """

    #data0 = data.reshape(len(data0), dims[1]
    # / ds[0], ds[0], dims[2] / ds[1], ds[1]).mean(2).mean(3)
    #data0  = data.copy()
    if ds ==1:
        return data

    ndim = np.ndim(data)
    if ndim <3:
        data0 = (data.T).reshape(dims,order='F') # d1 x d2 x T
    data_0 = downscale_local_mean(data0,
                                (ds,ds,1)) # d1//ds x d2//ds x T
    if ndim<3:
        dims_ = data_0.shape
        data_0 = data_0.reshape((np.prod(dims_[:2]),dims_[2]), order='F').T
    return data_0


def denoise_patch(M,
                  maxlag=5,
                  tsub=1,
                  ds=1,
                  noise_norm=False,
                  iterate=False,
                  confidence=0.99,
                  corr=True,
                  kurto=False,
                  tfilt=False,
                  tfide=False,
                  share_th=True,
                  plot_en=False,
                  greedy=True,
                  fudge_factor=0.99,
                  mean_th=None,
                  mean_th_factor=1.15,
                  U_update=False,
                  min_rank=1,
                  verbose=False,
                  pca_method='vanilla',
                  stim_knots=None,
                  stim_delta=200):
    """
    Given single patch, denoise it as outlined by parameters

    Parameters:
    ----------

    M:          np.array (d1xd2xT)
                array to be denoised
    maxlag:     int
                max correlation lag for correlation null hypothesis in samples
                (e.g. indicator decay in samples)
    tsub:       int
                temporal downsample constant
    ds:         int
                spatial downsample constant
    noise_norm: placeholder
    iterate:    boolean
                flag to include correlated components iteratively
    confidence: float
                confidence interval (CI) for correlation null hypothesis
    corr:       boolean
                flag to include components which pass correlation null hypothesis
    kurto:      boolean
                will be removed
                flag to include components which pass kurtosis null hypothesis
    tfilt:      boolean
                flag to temporally filter traces with AR estimate of order p.
    tfide:      boolean
                flag to denoise temporal traces with Trend Filtering
    min_rank:   int
                minimum rank of denoised/compressed matrix
                typically set to 1 to avoid empty output (array of zeros)
                if input array is mostly noise.
    greedy:     boolean
                flag to greedily update spatial and temporal components
                (estimated with PCA) greedily by denoising
                temporal and spatial components
    mean_th_factor: float
                factor to scale mean_th
                typically set to 2 if greedy=True and mean_th=None
                or if mean_th has not been scaled yet.
    share_th:   boolean
                flag to compute a unique thredhold for correlation null hypothesis
                to be used in each tile.
                If false: each tile calculates its own mean_th value.
    fudge_factor: float
                constant to scale estimated noise std st denoising is less
                (lower factor) or more (higher factor) restrictive.
    U_update:   boolean
                flag to (not) update spatial components by imposing L1- constraint.
                True for "round" neurons in 2p.
                False for dendritic data.
    plot_en:    boolean
                flag to enable plots
    verbose:    boolean
                flag to enable verbose
    pca_method: string
                method for matrix decomposition (e.g. PCA, sPCA, rPCA, etc).
                see compute_svd for options

    Outputs:
    -------

    Yd:         np.array (d1 x d2 x T)
                compressed/denoised array given input M
    rlen:       int
                sum of the ranks of all tiles
    """
    dimsM = M.shape
    M = M.reshape((np.prod(dimsM[:2]), dimsM[2]), order='F')
    M = M.astype('float32')
    #print('greedy here 505')
    start = time.time()
    Yd, vtids = denoise_components(M,
                                   maxlag=maxlag,
                                   tsub=tsub,
                                   confidence=confidence,
                                   corr=corr,
                                   mean_th=mean_th,
                                   greedy=greedy,
                                   fudge_factor=fudge_factor,
                                   mean_th_factor=mean_th_factor,
                                   U_update=U_update,
                                   min_rank=min_rank,
                                   plot_en=plot_en,
                                   verbose=verbose,
                                   dims=dimsM
                                  )

    Yd = Yd.reshape(dimsM, order='F')
    # determine individual rank
    rlen = total_rank(vtids)

    print('\tY rank:%d\trun_time: %f'%(rlen,time.time()-start))

    return Yd, rlen


def total_rank(vtids,
                verbose=False):
    # determine individual rank
    case1 = ~np.isnan(vtids[0,:])
    if vtids[0,case1].sum()>0:
        ranks = case1
    else:
        ranks = np.nan
    #ranks = np.where(np.logical_or(vtids[0, :] >= 1, vtids[1, :] == 1))[0]
    if np.all(ranks == np.nan):
        rlen = 0
    else:
        rlen = vtids[0,ranks].sum() #len(ranks)
    return rlen


def greedy_spatial_denoiser(Y,
                            V_TF,
                            fudge_factor=1,
                            U_update=False,
                            nus_=None):
                            #dims=None,
                            #plot_en=False):
    """
    Update U wrt V
    """
    if U_update:
        print('You should not be here')
        pass
        #outs_2 = [c_l1_u_hat(y, V_TF,fudge_factor) for y in Y]
        #outs_2 = update_U_parallel(Y,V_TF,fudge_factor)
        #U_hat, nus_ = map(np.asarray,zip(*np.asarray(outs_2)))
    else:
        nus_ = None#np.zeros((Y.shape[0],))
        try:
            #U_hat = np.matmul(Y, np.matmul(V_TF.T, np.linalg.inv(np.matmul(V_TF, V_TF.T))))
            U_hat = Y.dot((V_TF.T.dot(np.linalg.inv(V_TF.dot(V_TF.T)))))
        except Exception as e:
            print(e)
    #uplot.plot_spatial_component(U_hat,dims) if plot_en and (not dims==None) else 0
    return U_hat, nus_


def component_norm(V_TF_,U_hat_,verbose=False,title=''):
    norm_U = np.sqrt(np.sum(U_hat_**2,0))#[:,np.newaxis]
    norm_V = np.sqrt(np.sum(V_TF_**2,1))[:,np.newaxis]
    if verbose:
        print('\nnorm V '+ title)
        print(norm_V.flatten())
        print('norm U '+ title)
        print(norm_U)

    return norm_V, norm_U



def l1tf_lagrangian(V_,
                    lambda_,
                    solver='trefide',
                    solver_obj=None):

    if solver == 'trefide':
        try:
            if solver_obj is None:
                solver_obj = TrendFilter(len(V_))
            solver_obj.lambda_ = lambda_
            V_TF = solver_obj.denoise(np.double(V_),
                                        refit=False)

        except:
            print('AS failed. try PD')
            V_TF = l1_tf(V_,
                        lambda_,
                        False,
                        1000,
                        0)
            print('solved w cython')

    elif solver == 'cvxpy':
        V_TF = c_update_V(V_,
                        lambda_)

    return np.asarray(V_TF)


def l1tf_constrained(V_hat,
                   solver='trefide',
                   verbose=False,
                   ):

    if np.ndim(V_hat)==1:
        V_hat = V_hat[np.newaxis,:]

    if solver == 'trefide':
        num_components, len_signal = V_hat.shape
        denoise_filters = [TrendFilter(len_signal)]*num_components
        #print('started trefide constrained')
        V_TF = np.asarray([filt.denoise(signal)
                    for signal, filt in zip(np.double(V_hat), denoise_filters)])
        #print('ended trefide constrained')
        lambdas_ = [filt.lambda_ for filt in denoise_filters]
        #print(lambdas_)
        if False:
            lambdas_2 =[]
            for lambda1 in lambdas_:
                if lambda1 !=0:
                    lambda1=1./lambda1
                lambdas_2.append(lambda1)
            lambdas_=lambdas_2

        noise_std_ = [filt.delta for filt in denoise_filters]

    elif solver == 'cvxpy':
        noise_std_ = denoise.noise_level(V_hat)

        outs_ = [c_l1tf_v_hat(V_hat[idx,:],
                              stdv,
                              solver='SCS')
                 for idx, stdv in enumerate(noise_std_)]

        V_TF, lambdas_ = map(np.asarray,
                        zip(*np.asarray(outs_)))

        denoise_filters=[None]*len(lambdas_)

    if verbose:
        print('Noise range is %.3e %.3e'%(
            min(noise_std_), max(noise_std_)))

    return V_TF, lambdas_ , denoise_filters


def iteration_error(Y,
                    U_hat,
                    V_TF,
                    region_indices=None,
                    lambdas_=None,
                    nus_=None):
    """
    F(U,V)=||Y-UV||^2_2 + sum_i lambda_i ||D^2 V_i||_1 + sum_j nu_j ||U_j||_1
    # due to normalization F(U,V) may not decrease monotonically. problem?
    """
    #if np.ndim(V_TF) ==1:
    #    V_TF = V_TF[np.newaxis,:]
    num_components, T = V_TF.shape
    F_uv1 = np.linalg.norm(Y - U_hat.dot(V_TF))**2

    lambdas_ =(np.sum(U_hat**2, 0)*lambdas_)
    if region_indices is None:
        diff = difference_operator(T)
        norm1_Dv = np.abs(diff.dot(V_TF.T))
        F_uv2  = np.sum(lambdas_*np.sum(norm1_Dv, 0))
    else:
        pass
    if nus_ is None:
        F_uv3 = 0
    else:
        pass #F_uv3  = np.sum(nus_*np.sum(np.abs(U_hat),1)) #if U_update else 0

    return F_uv1 , F_uv2, F_uv3


def iteration_error2(V_init,
                    #U_hat,
                    V_TF,
                    region_indices=None,
                    lambdas_=None,
                    nus_=None):
    """
    F(U,V)=||Y-UV||^2_2 + sum_i lambda_i ||D^2 V_i||_1 + sum_j nu_j ||U_j||_1
    # due to normalization F(U,V) may not decrease monotonically. problem?
    """
    #if np.ndim(V_TF) ==1:
    #    V_TF = V_TF[np.newaxis,:]
    num_components, T = V_TF.shape
    F_uv1 = np.linalg.norm(V_init -V_TF,2)**2

    #lambdas_ =(np.sum(U_hat**2, 0)*lambdas_)
    if region_indices is None:
        diff = difference_operator(T)
        norm1_Dv = np.abs(diff.dot(V_TF.T))
        F_uv2  = np.sum(lambdas_*np.sum(norm1_Dv, 0))
    else:
        pass
    if nus_ is None:
        F_uv3 = 0
    else:
        pass #F_uv3  = np.sum(nus_*np.sum(np.abs(U_hat),1)) #if U_update else 0

    return F_uv1 , F_uv2, F_uv3

def greedy_temporal_denoiser(Y,
                            U_hat_,
                            V_TF_,
                            lambda_=None,
                            plot_en=False,
                            solver='trefide',
                            solver_obj=None,
                            ):

    #if plot_en:
    V_TF_2 = V_TF_.copy()

    num_components, T = V_TF_2.shape
    norm_U2 = np.sum(U_hat_**2,0)

    lambdas_all = []
    solvers_obji = []
    for ii in range(num_components):
        idx_ = np.setdiff1d(np.arange(num_components), ii)
        R_ = Y - U_hat_[:,idx_].dot(V_TF_2[idx_,:])
        V_ = U_hat_[:,ii].T.dot(R_)/norm_U2[ii]

        print(np.sum(V_**2,0))
        #V_=V_/np.sqrt(np.sum(V_**2))
        #print('Force lambda_')
        lambda_=None
        if lambda_ is None:
            V_2 , lambda_i, solv_obji =l1tf_constrained(V_,
                                solver=solver,
                                verbose=False
                                )#[0]
        else:
            V_2 = l1tf_lagrangian(V_,
                                lambda_=lambda_[ii],#/norm_U2[ii],
                                solver=solver,
                                solver_obj=solver_obj[ii]
                                )
        #print('634 - norm V2')
        #print(np.sqrt((V_2**2).sum()))
        V_2 = V_2.flatten()
        V_TF_2[ii,:] = V_2#/np.sqrt((V_2**2).sum())
        lambdas_all.append(lambda_i[0])
        solvers_obji.append(solv_obji[0])

    if plot_en:
        uplot.plot_temporal_traces(V_TF_, V_hat=V_TF_2)

    return V_TF_2 ,lambdas_all,solvers_obji


def greedy_component_denoiser(Y,
                              U_hat,
                              V_TF,
                              confidence=0.99,
                              corr=True,
                              dims=None,
                              extra_iterations=5,
                              final_regression=False,
                              fudge_factor=1.,
                              maxlag=5,
                              max_num_components=50,
                              max_num_iters=10,
                              mean_th=None,
                              plot_en=False,
                              solver='cvxpy',
                              U_update=False,
                              verbose=False
                              ):
    """
    """
    V_init =V_TF.copy()
    num_components, T = V_TF.shape
    Y2 = Y.copy()
    #####################################################
    print('Original error ||Y-UV||_F^2')
    print(np.linalg.norm(Y-U_hat.dot(V_TF))**2)
    print('\n')
    # Running iterations
    print('Initialization with %d components'%(num_components)) if verbose else 0
    print('Max # of greedy loops: %d (relative convergence)'%max_num_iters) if verbose else 0

    ################################ Rerun
    rerun_1 = 1 # flag to run part (1)
    run_count = 0 # run count

    while rerun_1:
        num_components, len_signal = V_TF.shape
        if verbose:
            print('*Run %d: Initialization with %d components'\
                    %(run_count, num_components))

        ####################################
        ### Initial temporal updates
        ####################################
        print('Temporal update - constrained') if verbose else 0

        if plot_en:
            V_hat_orig = V_TF.copy()

        V_TF, lambdas_ , solver_obj = l1tf_constrained(V_TF,
                                        solver=solver,
                                        verbose=verbose)
        ######################
        ## normalize V by norm 2
        ######################
        normV_init, _ = component_norm(V_TF,
                                    U_hat,
                                    verbose=verbose,
                                    title=' after temporal update')
        if verbose:
            print('\n Normalize V_TF after temporal update\n')
        #V_TF = V_TF/normV_init

        if plot_en:
            uplot.plot_temporal_traces(V_hat_orig, V_TF)
        ###################
        ### Initial spatial updates
        ####################
        if verbose:
            print('\n Spatial Update by regression\n')
        if plot_en and (not dims==None):
            U_orig = U_hat.copy()
        U_hat, nus_ = greedy_spatial_denoiser(Y,
                                              V_TF,
                                              fudge_factor=fudge_factor,
                                              U_update=U_update)

        if verbose:
            print('Initialization error ||Y-UV||_F^2')
            print(np.linalg.norm(Y-U_hat.dot(V_TF))**2)

        if plot_en and (not dims==None) :
            uplot.plot_spatial_component(U_orig,
                                        Y_hat=U_hat,
                                        dims=dims)

        norm_Vinit, norm_Uinit = component_norm(V_TF,
                                                U_hat,
                                                verbose=verbose,
                                                title=' after spatial update')

        # Scale lambda_ by the norm_U^2
        if verbose:
            pass
            #print('lambdas before scaling by norm U2')
        print('lambdas_')
        print(lambdas_)

        #print('\nTHIS IS AN ERROR\n')
        #lambdas_ = lambdas_ * (norm_Uinit)
        #print('not scaling lambda')
        #lambdas_ = lambdas_ * (norm_Uinit**2)

        if verbose:
            #print('lambdas after scaling by norm U2')
            #print(lambdas_)
            pass
        ##############################################
        ############# Begin loop
        ##############################################

        #################################
        ########## Errors ###############
        F_UVs = np.zeros((max_num_iters,))
        F_UV1 = np.zeros((max_num_iters,))
        F_UV2 = np.zeros((max_num_iters,))
        F_UV3 = np.zeros((max_num_iters,))
        norm_U = np.zeros((max_num_iters, num_components))
        norm_V = np.zeros((max_num_iters, num_components))

        ########## Errors ###############
        print('\nRun %d: begin greedy loops\n'%(run_count)) if verbose else 0

        remaining_extra_iterations = max(extra_iterations,1)

        for loop_ in range(max_num_iters):

            if remaining_extra_iterations == 0:
                print('remaining_extra_iterations %d'%remaining_extra_iterations)
                break

            print('\t Run %d iteration %d with %d components'%(run_count,
                        loop_, num_components)) if verbose else 0

            ###################
            ### Temporal updates
            ####################
            print('\nTemporal update - TF lagrangian') if verbose else 0
            V_TF, lambdas_, solver_obj= greedy_temporal_denoiser(Y,
                                            U_hat,
                                            V_TF,
                                            lambda_=lambdas_,
                                            plot_en=plot_en,
                                            solver=solver,
                                            solver_obj=solver_obj,
                                            )

            norm_Vinit, norm_Uinit = component_norm(V_TF,
                                                    U_hat,
                                                    verbose=verbose,
                                                    title=' after temp update')

            ######################
            ## normalize V by norm 2
            ######################
            if verbose:
                print('Normalization of V_TF')
            V_TF = V_TF/norm_Vinit

            ##################################################
            ### Spatial updates
            #################################################
            print('\nSpatial update - LS regression') if verbose else 0
            if plot_en and (not dims==None) :
                U_orig = U_hat.copy()

            U_hat, _ = greedy_spatial_denoiser(Y,
                                               V_TF,
                                               nus_=nus_,
                                               fudge_factor=fudge_factor)

            norm_Vinit, norm_Uinit = component_norm(V_TF,
                                                    U_hat,
                                                    verbose=verbose,
                                                    title=' after spatial update')

            np.testing.assert_array_equal(Y,Y2,err_msg='change in Y')

            if plot_en and (not dims==None) :
                uplot.plot_spatial_component(U_hat,
                                            Y_hat=U_orig,
                                            dims=dims)

            norm_U[loop_] = norm_Uinit
            norm_V[loop_] = norm_Vinit.flatten()

            ###################
            ### Calculate error in current iteration
            ####################
            print('lambda')
            print(lambdas_)

            if np.any(lambdas_)==np.nan:
                print('NAN lambda_')
                remaining_extra_iterations=0
            if False:
                F_uv1, F_uv2, F_uv3 = iteration_error(Y,
                                                      U_hat,
                                                      V_TF,
                                                      lambdas_=lambdas_,
                                                      nus_=nus_)
            else:
                F_uv1, F_uv2, F_uv3 = iteration_error2(V_init,
                                                      V_TF,
                                                      lambdas_=lambdas_,
                                                      nus_=nus_)


            np.testing.assert_array_equal(Y,Y2, err_msg='change in Y')

            F_uv = F_uv1 + F_uv2 + F_uv3

            if verbose:
                print('\n\tIteration %d loop %d error(s):'%(run_count, loop_))
                print('(%.3e + %.3e + %.3e)= %.3e\n'%(F_uv1,F_uv2,F_uv3,F_uv))

            F_UVs[loop_] = F_uv
            F_UV1[loop_] = F_uv1
            F_UV2[loop_] = F_uv2
            F_UV3[loop_] = F_uv3

            if loop_ >=1:
                no_change = np.isclose(F_uv,
                                        F_UVs[loop_-1],
                                        rtol=1e-04,
                                        atol=1e-08)

                bad_iter = (F_uv >= F_UVs[loop_-1])

                if no_change or bad_iter:
                    print('\tIteration %d loop %d end - no significant updates\n'%(
                                            run_count,loop_)) if verbose else 0
                    if bad_iter:
                        print('***diverged****') if verbose else 0

                    if (remaining_extra_iterations == extra_iterations):
                        print('\n\n***Begin %d extra iterations\n\n'%extra_iterations) if verbose else 0
                    else:
                        if verbose:
                            print('Remaining iterations %d'%remaining_extra_iterations)

                    if remaining_extra_iterations <= 0:
                        print('remaining_extra_iterations ==0') if verbose else 0
                        break
                    else:
                        print('Extra iterations minus 1') if verbose else 0
                        remaining_extra_iterations = remaining_extra_iterations-1
                        #continue
                else:
                    if verbose:
                        print('Did not converge in iteration %d\n'%loop_)

        if True:#plot_en:
            errors_loop=[F_UVs, F_UV1, F_UV2, F_UV3]
            error_names=['F_UVs','F_UV1','F_UV2','F_UV3']
            for eerr, error_ in enumerate(errors_loop):
                plt.title('Error '+ error_names[eerr] +' after %d loops'%loop_)
                plt.plot(error_[:loop_],'o-')
                plt.show()
                print(error_[:loop_])

            for comp_ in range(U_hat.shape[1]):
                fig, ax_ =plt.subplots(1,2,figsize=(10,5))
                ax_[0].set_title('Change in U_hat norm %d'%comp_)
                ax_[1].set_title('Change in V_TF norm %d'%comp_)
                ax_[0].plot(norm_U[:loop_,comp_],'o-')
                ax_[1].plot(norm_V[:loop_,comp_],'o-')
                plt.show()

        #return
        ###########################
        ### Begin search in residual
        ###########################
        print('\n Begin search in residual\n') if verbose else 0
        print('*****Iteration %d residual search with %d components'%(
                                run_count, V_TF.shape[0])) if verbose else 0

        ### (2) Compute PCA on residual R  and check for correlated components
        residual = Y - U_hat.dot(V_TF)
        residual_min_threshold = max(np.abs(Y.min()),np.abs(Y.mean()-3*Y.std()))
        keep1_r =[]
        # update according to movie dynamic range
        if residual.max() >= residual_min_threshold:
            U_r, s_r, Vt_r = compute_svd(residual)

            if np.abs(s_r.max()) <= residual_min_threshold:
                if verbose:
                    print('did not make the cut based on component variance') #if verbose
                keep1_r=[]
            else:
                ctid_r, _ = find_temporal_component(Vt_r,
                                                confidence=confidence,
                                                corr=corr,
                                                maxlag=maxlag,
                                                mean_th=mean_th,
                                                plot_en=plot_en)

                keep1_r = np.where(np.logical_or(ctid_r[0, :] == 1,
                                                ctid_r[1, :] == 1))[0]

        else:
            print('Residual <= %.3e'%(residual_min_threshold)) #if verbose else 0
            keep1_r = []

        if len(keep1_r)==0:
            print('\nFinal number of components %d'%num_components) if verbose else 0
            rerun_1 = 0
        else:

            if np.abs(s_r[keep1_r].max()) <=residual_min_threshold:
                print('did not make the cut based on component variance')
                keep1_r=[]

            print('Rerun Iterations - adding %d components\n'%(len(keep1_r))) if verbose else 0
            num_components = num_components + len(keep1_r)
            if max_num_components <= num_components:
                print('Number of components %d > max allowed %d\n'%(num_components,max_num_components))
                rerun_1 = 0
            else:
                rerun_1 = 1
                run_count +=1
                V_TF = np.vstack((V_TF, Vt_r[keep1_r,:]))
                U_hat = np.hstack((U_hat, U_r[:,keep1_r].dot(np.diag(s_r[keep1_r]))))

        #return
        print('1109')
        uplot.plot_temporal_traces(V_init,V_hat=V_TF)
        if len(keep1_r)>0:
            print('Extra')
            print(s[keep1_r])
            uplot.plot_temporal_traces(Vt_r[keep1_r,:])
        print('Goodbye')
        return
    ##################
    ### Final update
    ##################
    print('set en_true 1051')
    plot_en = True
    print('*Final update after %d iterations'%run_count) if verbose else 0
    print('\tFinal update of temporal components') if verbose else 0

    V_TF,_,_= greedy_temporal_denoiser(Y,
                                    U_hat,
                                    V_TF,
                                    solver=solver
                                    )
    print('1062')
    #print(V_init.shape)
    #print(V_TF.shape)
    uplot.plot_temporal_traces(V_init,V_hat=V_TF)

    print('\tFinal update of spatial components') if verbose else 0

    if plot_en:
        U_orig = U_hat.copy()

    U_hat, _ = greedy_spatial_denoiser(Y,
                                       V_TF,
                                       fudge_factor=fudge_factor,
                                       U_update=U_update)

    if plot_en and (not dims==None):
        uplot.plot_spatial_component(U_orig,
                                    Y_hat=U_hat,
                                    dims=dims)

    if final_regression:
        print('\tFinal regression for V(j)') if verbose else 0
        if plot_en:
            V_TF_i = V_TF.copy()

        V_TF = np.matmul(np.matmul(np.linalg.inv(np.matmul(U_hat.T, U_hat)), U_hat.T), Y)

        if plot_en:
            uplot.plot_temporal_traces(V_TF,V_hat=V_TF_i)

    # this needs to be updated to reflect any new rank due to new numb of iterations
    return U_hat , V_TF


def find_temporal_component(Vt,
                            confidence=0.99,
                            corr=True,
                            iterate=False,
                            kurto=False,
                            maxlag=5,
                            mean_th=None,
                            mean_th_factor=1.0,
                            plot_en=False,
                            stim_knots=None,
                            stim_delta=200):
    """
    """
    if mean_th is None:
        mean_th = wnoise_acov_CI(Vt.shape[1],
                                 confidence=confidence,
                                 maxlag=maxlag)
    mean_th *= mean_th_factor

    ignore_segments =stimulus_segmentation(Vt.shape[1],
                                           stim_knots=stim_knots,
                                           stim_delta=stim_delta
                                          )
    #print('th is {}'.format(mean_th))
    ctid = choose_rank(Vt[:,~ignore_segments],
                       maxlag=maxlag,
                       iterate=iterate,
                       confidence=confidence,
                       corr=corr,
                       kurto=kurto,
                       mean_th=mean_th)

    #print('1110')
    #print(mean_th)
    if plot_en:    # Plot temporal correlations
        keep1 = np.where(np.logical_or(ctid[0, :] == 1,
                        ctid[1, :] == 1))[0]
        uplot.plot_vt_cov(Vt, keep1, maxlag)
    return ctid, mean_th


def decimation_interpolation(data,
                            dims=None,
                            ds=1,
                            tsub=1,
                            rank=2,
                            verbose=False
                            ):
    """
    data = d1 x d2 x T
    this data has already been spatially decimated
    ds is to upsample up
    ds: spatial decimation
    tsub: temporal decimation
    """
    # data = data0.T (pxT)
    # Run rank-k svd on spatially and temporall decimated Y
    # spatially decimate
    # temporally decimate
    # run rank k
    print('Decimation interpolation') if verbose else 0
    #print(data.shape)
    #print(dims)
    data_tsd = temporal_decimation(data.T, tsub)

    U, s, Vt = compute_svd(data_tsd.T,
                            n_components=rank,
                            method='randomized')
    U = U.dot(np.diag(s))
    ndims_=dims[0]//ds, dims[1]//ds, dims[2]

    # Then upsample the resulting decimated U and V to initialize U and V
    # upsample temporal
    x_interp = np.linspace(0,dims[2],dims[2])
    xp_ = x_interp[::tsub]
    Vt_interp = np.zeros((rank,dims[2]))
    for comp_ in range(rank):
        Vt_interp[comp_,:] = np.interp(x_interp,xp_,Vt[comp_,:])

    # upsample spatial
    U_ds = U.reshape(ndims_[:2] + (-1,), order = 'F')

    U_ds = sp.ndimage.zoom(U_ds, (ds,ds, 1 ))
    U_ds = U_ds.reshape((np.prod(dims[:2]), rank), order='F')
    #print('1106')
    print(dims)
    print(ndims_)
    print(U_ds.shape)
    print(Vt_interp.shape)
    #return
    return U_ds, Vt_interp


def temporal_filter_ar(data,
                        p=1):
    """
    """
    data0 = np.zeros(data.shape)
    #T, num_pxls = data.shape
    for ii, trace in enumerate(data.T):
        # Estimate tau for exponential
        tau = cnmf.deconvolution.estimate_time_constant(
                trace,p=p,sn=None,lags=5,fudge_factor=1.)
        window = tau **range(0,100)
        data0[:,ii] = np.convolve(trace,window,mode='full')[:T]/np.sum(window)

    return data0


def denoise_components(data,
                        confidence=0.99,
                        corr=True,
                        decimation_flag=False,
                        dims=None,
                        ds=1,
                        fudge_factor=1.,
                        greedy=True,
                        maxlag=5,
                        max_num_iters=20,
                        mean_th=None,
                        mean_th_factor=1.,
                        mean_th_factor2=1.15,
                        min_rank=0,
                        plot_en=False,
                        solver='trefide',
                        tsub=1,
                        U_update=False,
                        verbose=False):
    """
    Compress array data_all as determined by parameters.

    Parameters:
    ----------

    data_all:   np.array (d x T) or (d1 x d2 xT)
                2D or 3D video array (pixels x Time) or (pixel x pixel x Time)
    dims:       tuple (d1 x d2 x T)
                dimensions of video array used for plotting
    maxlag:     int
                max correlation lag for correlation null hypothesis in samples
                (e.g. indicator decay in samples)
    tsub:       int
                temporal downsample constant
    ds:         int
                spatial downsample constant
    confidence: float
                confidence interval (CI) for correlation null hypothesis
    corr:       boolean
                flag to include components which pass correlation null hypothesis
    mean_th:    float
                threshold employed to reject components according to correlation null hypothesis
    min_rank:   int
                minimum rank of denoised/compressed matrix
                typically set to 1 to avoid empty output (array of zeros)
                if input array is mostly noise.
    greedy:     boolean
                flag to greedily update spatial and temporal components (estimated with PCA)
                greedyly by denoising temporal and spatial components
    mean_th_factor: float
                factor to scale mean_th
                typically set to 2 if greedy=True and mean_th=None or if mean_th has not been scaled yet.
    fudge_factor: float
                constant to scale estimated noise std st denoising st denoising is less
                (lower factor) or more (higher factor) restrictive.
    U_update:   boolean
                flag to (not) update spatial components by imposing L1- constraint.
                True for "round" neurons in 2p.
                False for dendritic data.
    plot_en:    boolean
                flag to enable plots
    verbose:    boolean
                flag to enable verbose
    Outputs:
    -------
    Yd_out:     np.array (d x T)
                compressed/denoised array (dxT)
    ctids:      np.array (3,d)
                indicator 3D matrix (corr-kurto-reject) which points which statistic
                a given component passed and thus it is included.
                If greedy=True, all components added are included as corr components.
    """
    if (ds !=1) or (tsub !=1):
        decimation_flag = True
        print('Reset flag') if verbose else 0

    mu = data.mean(1, keepdims=True)
    #std = data.std(1,keepdims=True)
    data = (data - mu)#/std

    # spatially decimate the data
    if ds > 1:
        print('Spatial decimation by %d'%ds) if verbose else 0
        data = spatial_decimation(data.T,
                                ds=ds,
                                dims=dims).T

    U, s, Vt = compute_svd(data)

    # if greedy Force x2 mean_th (store only big components)
    if greedy and (mean_th_factor <= 1.):
        mean_th_factor = 2.

    # Select components
    ctid, mean_th = find_temporal_component(Vt,
                                    confidence=confidence,
                                    corr=corr,
                                    maxlag=maxlag,
                                    mean_th=mean_th,
                                    mean_th_factor=mean_th_factor,
                                    plot_en=plot_en
                                    )

    keep1 = np.where(np.logical_or(ctid[0, :] == 1, ctid[1, :] == 1))[0]


    # If no components to store, change to lower confidence level
    if np.all(keep1 == np.nan):
        print("Change to lower confidence level") if verbose else 0
        mean_th /= mean_th_factor;
        mean_th_factor = mean_th_factor2;
        mean_th *= mean_th_factor
        ctid, mean_th = find_temporal_component(Vt,
                                confidence=confidence,
                                corr=corr,
                                maxlag=maxlag,
                                mean_th=mean_th,
                                plot_en=plot_en
                                )
        keep1 = np.where(np.logical_or(ctid[0, :] == 1, ctid[1, :] == 1))[0]
        #uplot.plot_vt_cov(Vt,keep1,maxlag) if plot_en else 0

    # If no components to store, exit & return min rank
    if np.all(keep1 == np.nan):
        if min_rank == 0:
            Yd = np.zeros(data.shape)
        else:
            min_rank = min_rank+1
            print('Forcing %d component(s)'%min_rank) if verbose else 0
            ctid[0, :min_rank]=1
            U = U[:,:min_rank].dot(np.eye(min_rank)*s[:min_rank])
            Yd = (U.dot(Vt[:min_rank,:]))
        Yd += mu
        #Yd*= std
        return Yd, ctid

    if decimation_flag:
        U, Vt = decimation_interpolation(data,
                                        dims=dims,
                                        ds=ds,
                                        tsub=tsub,
                                        rank=len(keep1)
                                        )
    else:
        # Select components
        Vt = Vt[keep1,:]
        U = U[:,keep1].dot(np.eye(len(keep1))*s[keep1.astype(int)])

    # call greedy
    if greedy:
        try:
            mean_th = mean_th*mean_th_factor2/mean_th_factor
            U, Vt = greedy_component_denoiser(data,
                                            U,
                                            Vt,
                                            confidence=confidence,
                                            corr=corr,
                                            dims=dims,
                                            fudge_factor=fudge_factor,
                                            maxlag=maxlag,
                                            max_num_iters=max_num_iters,
                                            mean_th=mean_th,
                                            plot_en=plot_en,
                                            solver=solver,
                                            U_update=U_update,
                                            verbose=verbose)

            ctid[0,np.arange(Vt.shape[0])]=1

        except:
            print('\tERROR: Greedy solving failed, keeping %d parameters'%
                    (len(keep1)))
            ctid[0,0] = 100

    Yd = U.dot(Vt) + mu
    #Yd = Yd*std
    return Yd, ctid


#########################################################################################
## DURING MERGING PHASE
#########################################################################################
def c_l1tf_v_hat(v,
                 sigma,
                 verbose=False,
                 solver='SCS',
                 abstol=1e-4,
                 max_iters=1000):
    """
    Update vector v according to difference fctn diff
    with noise_std(v) = sigma

    V(i) = argmin_W ||D^2 W||_1
    st ||V_i-W||_2<sigma_i*sqrt(T)
    Include optimal lagrande multiplier for constraint

    """
    #print('c_l1tf_v_hat') if verbose else 0
    if np.abs(sigma)<=1e-3:
        print('Do not denoise (high SNR: noise_level=%.3e)'%
                sigma) if verbose else 0
        return v , 0

    T = len(v)
    v_hat = cp.Variable(T)
    #print(sigma*np.sqrt(T)) if verbose else 0
    diff = difference_operator(T)
    objective = cp.Minimize(cp.norm(diff*v_hat,1))
    constraints = [ cp.norm(v-v_hat,2)**2<=((sigma)**2)*T ]

    cp.Problem(objective, constraints).solve(solver=solver,
                                             max_iters=max_iters,
                                             #eps=abstol,
                                             verbose=False)
    lambda_= constraints[0].dual_value
    if lambda_ !=0:
        lambda_=1./lambda_

    return np.asarray(v_hat.value).flatten(), lambda_


def c_update_V(v,
            lambda_,
            cvxpy_solver='SCS',
            max_iters=1000
            ):

    T = len(v)
    v_hat = cp.Variable(T)
    diff = difference_operator(T)

    cte2 = lambda_*cp.norm(diff*v_hat,1)
    objective = cp.Minimize(cp.norm(v-v_hat, 2)**2 + cte2)

    cp.Problem(objective).solve(solver=cvxpy_solver,
                                max_iters=max_iters,
                                #abstol=abstol,
                                verbose=False)

    return np.asarray(v_hat.value).flatten()