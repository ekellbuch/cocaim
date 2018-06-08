#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp

import sklearn
from scipy.stats import norm
from sklearn.utils.extmath import randomized_svd
from skimage.transform import downscale_local_mean

import concurrent
import cvxpy as cp
import multiprocessing
import itertools
import time
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

# see how this changes with get_noise_fft

import trefide
from trefide.temporal import TrendFilter


from . import denoise
from .plots import util_plot as uplot
from .utils import tools as tools_


# from . import greedy_spatial

# update comments
# reformat spatial
# reformat decimation
# reruns should not denoised bottom components
# better handle components to get order params
# update descriptions


def find_spatial_component(U, tv_th=2.5):
    """
    """
    if np.ndim(U) == 1:
        U = U[:, np.newaxis]
    Gr2 = tools_.grad(U)
    metric = (np.sqrt(np.sum(Gr2**2, axis=2)).sum(0)/np.sum(np.abs(U), 0))

    keep = np.nonzero(metric <= tv_th)[0]
    return keep


def find_temporal_component(Vt,
                            D=None,
                            s=None,
                            kurto=False,
                            l1tf_th=2.25,
                            plot_en=False,
                            sk_thres=2,
                            tolerance=1e-3):
    """
    """
    if np.ndim(Vt) == 1:
        Vt = Vt[np.newaxis, :]

    num_components, L = Vt.shape

    # Include components based on test statistic
    # TF_norm / L1_nom
    if D is None:
        D = discrete_diff_operator(L, order=2)
    D1 = np.sum(np.abs(D.dot(Vt.T)), 0)
    n1 = np.sum(np.abs(Vt), 1)
    keep = np.argwhere((D1/n1) < l1tf_th).flatten()

    # Reject components based on test statistic
    if True:
        zs, ps = sp.stats.skewtest(Vt, axis=1)
        reject = np.where(np.abs(zs) <= sk_thres)[0]
        keep = np.setdiff1d(keep, reject)

    # account for low tolerance
    if s is not None:
        S = np.diag(s)
        Vt = S.dot(Vt)
        keep = keep[(np.abs(Vt[keep, :])).max(1) > tolerance]

    # Plot temporal correlations
    if plot_en:
        uplot.plot_vt_cov(Vt, keep, maxlag)
    return keep


def synchronization_test(v_, maxlag=100):
    v_ = (v_-v_.mean())/v_.std()
    a = tools_.axcov(v_[:1000], maxlag)[maxlag:]
    a = a/a.max()
    return a


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

    stim_knots = np.where(np.abs(np.diff(stim_knots)) > 1)[0]

    ignore_frames = np.zeros(T)

    for knot in stim_knots:
        arg_np = int(np.ones(stim_delta).cumsum() - stim_delta//2+knot)
        ignore_frames[arg_np] = 1
    ignore_frames = ignore_frames.astype('bool')

    return ignore_frames


def discrete_diff_operator(L, order=2):
    """
    Returns discrete difference operator
    of order k+1
    where D(k+1) [n-k-1] x n
    """
    if order == 1:
        D = (np.diag(np.ones(L-1)*1, 1) + np.diag(np.ones(L)*-1))[:(L-1), :]
    elif order == 2:
        D = (np.diag(np.ones(L-1)*-2, 1) + np.diag(np.ones(L)*1) + np.diag(
            np.ones(L-2), 2))[:(L-2), :]
    return D


def norm_1(x):
    return np.sum(np.abs(x))


def compute_svd(M,
                method='vanilla',
                min_value_=1e-6,
                n_components=2,
                n_iter=7,
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
        U, s, Vt = randomized_svd(M, n_components=n_components, n_iter=n_iter,
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
        (-1, mb) + data.shape[1:]).mean(1)
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
    if ds == 1:
        return data

    ndim = np.ndim(data)
    if ndim < 3:
        data0 = (data.T).reshape(dims, order='F')  # d1 x d2 x T
    data_0 = downscale_local_mean(data0,
                                  (ds, ds, 1))  # d1//ds x d2//ds x T
    if ndim < 3:
        dims_ = data_0.shape
        data_0 = data_0.reshape((np.prod(dims_[:2]), dims_[2]), order='F').T

    return data_0


def denoise_patch(M,
                  tsub=1,
                  ds=1,
                  extra_iterations=1,
                  fudge_factor=1.0,
                  greedy=True,
                  max_num_components=30,
                  max_num_iters=10,
                  min_rank=1,
                  plot_en=False,
                  reconstruct=True,
                  tfilt=False,
                  U_update=False,
                  verbose=False):
    """
    Given single patch, denoise it as outlined by parameters

    Parameters:
    ----------

    M:          np.array (d1xd2xT)
                array to be denoised
    tsub:       int
                temporal downsample constant
    ds:         int
                spatial downsample constant
    tfilt:      boolean
                flag to temporally filter traces with AR estimate of order p.
    min_rank:   int
                minimum rank of denoised/compressed matrix
                typically set to 1 to avoid empty output (array of zeros)
                if input array is mostly noise.
    greedy:     boolean
                flag to greedily update spatial and temporal components
                (estimated with PCA) greedily by denoising
                temporal and spatial components
    U_update:   boolean
                flag to (not) update spatial components
                by imposing L1-constraint.
                True for "round" neurons in 2p.
                False for dendritic data.
    plot_en:    boolean
                flag to enable plots
    verbose:    boolean
                flag to enable verbose
    Outputs:
    -------
    Yd:         np.array (d1 x d2 x T)
                compressed/denoised array given input M
    rlen:       int
                sum of the ranks of all tiles
    """

    if M.min() == 0:
        if reconstruct:
            return np.zeros(M.shape), 0
        else:
            return 0, 0, 0
    ndim = np.ndim(M)

    if np.ndim(M) == 3:
        dimsM = M.shape
        M = M.reshape((np.prod(dimsM[:2]), dimsM[2]), order='F')
    else:
        dimsM = None
    U, Vt, vtids = denoise_components(M,
                                      dims=dimsM,
                                      extra_iterations=extra_iterations,
                                      fudge_factor=fudge_factor,
                                      greedy=greedy,
                                      max_num_components=max_num_components,
                                      max_num_iters=max_num_iters,
                                      min_rank=min_rank,
                                      plot_en=plot_en,
                                      tsub=tsub,
                                      U_update=U_update,
                                      verbose=verbose
                                      )

    rlen = len(vtids)

    if reconstruct:
        Y = U.dot(Vt)
        if ndim == 3:
            Y.reshape(dimsM, order='F')
            return Y, rlen

    return U, Vt, rlen


def greedy_spatial_denoiser(Y,
                            V_TF,
                            U=None,
                            fudge_factor=1,
                            U_update=False,
                            nus_=None):
    """
    Update U wrt V
    """
    U_update = False

    if U_update:
        pass
        # if nus_ is None:
        #     U_hat, nus_ = greedy_spatial.greedy_sp_constrained(Y, V_TF, U=U)
        # else:
        #     U_hat = greedy_spatial.greedy_sp_dual(Y, V_TF, nus_, U=U)

    else:
        nus_ = None
        try:
            U_hat = np.matmul(Y, np.matmul(V_TF.T, np.linalg.inv(
                np.matmul(V_TF, V_TF.T))))
        except Exception as e:
            print(e)
    return U_hat, nus_


def norm_l2(V_TF_, verbose=False):
    norm_V = np.sqrt(np.sum(V_TF_**2, 1))[:, np.newaxis]
    return norm_V


def component_norm(V_TF_,
                   U_hat_,
                   verbose=False,
                   title=''):
    norm_U = np.sqrt(np.sum(U_hat_**2, 0))
    norm_V = np.sqrt(np.sum(V_TF_**2, 1))[:, np.newaxis]
    if verbose:
        print('\nnorm V ' + title)
        print(norm_V.flatten())
        print('norm U ' + title)
        print(norm_U)

    return norm_V, norm_U


def l1tf_lagrangian(V_,
                    lambda_,
                    solver='trefide',
                    solver_obj=None,
                    verbose=False):

    if solver == 'trefide':
        try:
            if solver_obj is None:
                solver_obj = TrendFilter(len(V_))
            solver_obj.lambda_ = lambda_
            v_ = np.double(V_).copy(order='C')
            V_TF = solver_obj.denoise(v_,
                                      refit=False)

        except:
            print('PDAS failed -- not denoising')
            pass

    elif solver == 'cvxpy':
        V_TF = c_update_V(V_, lambda_)

    return np.asarray(V_TF)


def l1tf_constrained(V_hat,
                     solver='trefide',
                     verbose=False,
                     ):

    if np.ndim(V_hat) == 1:
        V_hat = V_hat[np.newaxis, :]

    if solver == 'trefide':
        num_components, len_signal = V_hat.shape
        V_TF = V_hat.copy()

        lambdas_ = []
        noise_std_ = []
        denoise_filters = []
        for ii in range(num_components):
            filt = []
            filt = TrendFilter(len_signal)
            v_ = np.double(V_hat[ii, :]).copy(order='C')
            V_TF[ii, :] = np.asarray(filt.denoise(v_))
            noise_std_.append(filt.delta)
            if np.sqrt(filt.delta) <= 1e-3:
                lambdas_.append(0)
            else:
                lambdas_.append(filt.lambda_)
            denoise_filters.append(filt)
            filt = []
    elif solver == 'cvxpy':
        noise_std_ = denoise.noise_level(V_hat)

        outs_ = [c_l1tf_v_hat(V_hat[idx, :],
                              stdv,
                              solver='SCS')
                 for idx, stdv in enumerate(noise_std_)]

        V_TF, lambdas_ = map(np.asarray, zip(*np.asarray(outs_)))

        denoise_filters = [None]*len(lambdas_)

    else:
        print('not a solver')

    return V_TF, lambdas_, denoise_filters


def iteration_error(Y,
                    U_hat,
                    V_TF,
                    region_indices=None,
                    lambdas_=None,
                    nus_=None):
    """
    """

    num_components, T = V_TF.shape
    F_uv1 = np.linalg.norm(Y - U_hat.dot(V_TF))**2

    lambdas_2 = lambdas_
    if region_indices is None:
        diff = difference_operator(T)
        F_uv2 = np.sum(lambdas_2*np.sum(
            np.abs(diff.dot(V_TF.T)), axis=0), axis=0)
    else:
        pass
    if nus_ is None:
        F_uv3 = 0
    else:
        pass  #
        # print('815')
        if len(nus_) == num_components:
            F_uv3 = np.sum(nus_*np.sum(np.abs(U_hat), 0))
        else:
            F_uv3 = np.sum(nus_*np.sum(np.abs(U_hat), 1))  # if U_update else 0

    return F_uv1, F_uv2, F_uv3


def greedy_temporal_denoiser(Y,
                             U_hat_,
                             V_TF_,
                             lambda_=None,
                             scale_lambda_=None,
                             plot_en=False,
                             solver='trefide',
                             solver_obj=None,
                             ):
    V_TF_2 = V_TF_.copy()

    num_components, T = V_TF_2.shape
    norm_U2 = np.sum(U_hat_**2, 0)

    for ii in range(num_components):
        idx_ = np.setdiff1d(np.arange(num_components), ii)
        R_ = Y - U_hat_[:, idx_].dot(V_TF_2[idx_, :])
        V_ = U_hat_[:, ii].T.dot(R_)
        norm_Vnew = np.linalg.norm(V_, 2)
        norm_Vdiff = np.linalg.norm(V_TF_2[ii, :]-V_, 2)

        if norm_Vdiff/norm_Vnew >= 1:
            pass
            # continue

        if V_.var() <= 1e-3:
            pass
            # continue

        if lambda_ is None:
            V_2 = l1tf_constrained(V_,
                                   solver=solver,
                                   verbose=False
                                   )[0]

        else:
            clambda = lambda_[ii]

            if clambda == 0:
                continue

            V_2 = l1tf_lagrangian(V_,
                                  lambda_=clambda,
                                  solver=solver,
                                  solver_obj=solver_obj[ii]
                                  )
        V_TF_2[ii, :] = V_2
    if plot_en:
        uplot.plot_temporal_traces(V_TF_, V_hat=V_TF_2)

    return V_TF_2


def greedy_component_denoiser(Y,
                              U_hat,
                              V_TF,
                              dims=None,
                              extra_iterations=1,
                              final_regression=True,
                              fudge_factor=1.,
                              max_num_components=20,
                              max_num_iters=10,
                              plot_en=False,
                              solver='trefide',
                              U_update=False,
                              verbose=False
                              ):
    """
    """
    num_components, T = V_TF.shape

    rerun_1 = 1  # flag to run part (1)
    run_count = 0  # run count

    while rerun_1:
        num_components, len_signal = V_TF.shape

        V_TF, lambdas_, solver_obj = l1tf_constrained(V_TF,
                                                      solver=solver,
                                                      verbose=verbose)

        normV_init = np.sqrt(np.sum(V_TF**2, 1))[:, np.newaxis]
        # -- Initial spatial updates
        U_hat, nus_ = greedy_spatial_denoiser(Y,
                                              V_TF/normV_init,
                                              U=U_hat,
                                              fudge_factor=fudge_factor,
                                              U_update=U_update)

        norm_Uinit = np.sqrt(np.sum(U_hat**2, 0))
        U_hat = U_hat/norm_Uinit
        # -------- Begin loop
        F_UVs = np.zeros((max_num_iters,))

        remaining_extra_iterations = max(extra_iterations, 1)

        for loop_ in range(max_num_iters):

            if remaining_extra_iterations == 0:
                break

            # ---- Temporal updates
            V_TF = greedy_temporal_denoiser(Y,
                                            U_hat,
                                            V_TF,
                                            lambda_=lambdas_,
                                            plot_en=plot_en,
                                            solver=solver,
                                            solver_obj=solver_obj,
                                            )

            normV_init = np.sqrt(np.sum(V_TF**2, 1))[:, np.newaxis]
            # ---- Spatial updates

            U_hat, _ = greedy_spatial_denoiser(Y,
                                               V_TF/normV_init,
                                               nus_=nus_,
                                               U=U_hat,
                                               fudge_factor=fudge_factor,
                                               U_update=U_update)

            norm_Uinit = np.sqrt(np.sum(U_hat**2, 0))

            # ----  normalize U by norm 2
            U_hat = U_hat/norm_Uinit

            # np.testing.assert_array_equal(Y, Y2, err_msg='change in Y')

            # --  Calculate error in current iteration

            if np.any([math.isnan(lambda_) for lambda_ in lambdas_]):
                print('NAN lambda_')
                remaining_extra_iterations = 0
                print('\n')

            F_uv1, F_uv2, F_uv3 = iteration_error(Y,
                                                  U_hat,
                                                  V_TF,
                                                  lambdas_=lambdas_,
                                                  nus_=nus_)

            # np.testing.assert_array_equal(Y,Y2, err_msg='change in Y')

            F_uv = F_uv1 + F_uv2 + F_uv3
            F_UVs[loop_] = F_uv

            if loop_ >= 1:
                no_change = np.isclose(F_uv,
                                       F_UVs[loop_-1],
                                       rtol=1e-04,
                                       atol=1e-08)

                bad_iter = (F_uv >= F_UVs[loop_-1])

                if no_change or bad_iter:
                    if (remaining_extra_iterations == extra_iterations):
                        remaining_extra_iterations -= 1
                    elif remaining_extra_iterations == 0:
                        break
                    else:
                        remaining_extra_iterations -= 1

        # --- Begin search in residual
        residual = Y - U_hat.dot(V_TF)
        residual_min_threshold = max(np.abs(Y.min()),
                                     np.abs(Y.mean()-3*Y.std()))
        keep1_r = []

        # update according to movie dynamic range
        if residual.max() >= residual_min_threshold:
            U_r, s_r, Vt_r = compute_svd(residual,
                                         method='randomized',
                                         n_components=5)

            if np.abs(s_r.max()) <= residual_min_threshold:
                keep1_r = []
            else:
                keep1_r = find_temporal_component(Vt_r,
                                                  s=s_r,
                                                  plot_en=plot_en)
        else:
            keep1_r = []

        if len(keep1_r) == 0:
            rerun_1 = 0
        else:
            signals_ = np.diag(s_r[keep1_r]).dot(Vt_r[keep1_r, :])
            noise_level_res = denoise.noise_level(signals_)
            if np.abs(s_r[keep1_r].max()) <= residual_min_threshold \
               or np.any(noise_level_res >= Y.std()/3):
                keep1_r = []
                rerun_1 = 0
            else:
                num_components = num_components + len(keep1_r)
                if max_num_components <= num_components:
                    rerun_1 = 0
                else:
                    rerun_1 = 1
                    run_count += 1
                    S = np.diag(s_r[keep1_r])
                    Vt_r = S.dot(Vt_r[keep1_r, :])
                    V_TF = np.vstack((V_TF, Vt_r))
                    U_hat = np.hstack((U_hat, U_r[:, keep1_r]))

            if len(keep1_r) > 0:
                print('Extra')
                print(s_r[keep1_r])
                # uplot.plot_temporal_traces(Vt_r[keep1_r,:])

    # --- Final update

    V_TF = greedy_temporal_denoiser(Y,
                                    U_hat,
                                    V_TF,
                                    solver=solver
                                    )

    normV_init = np.sqrt(np.sum(V_TF**2, 1))[:, np.newaxis]

    U_hat, _ = greedy_spatial_denoiser(Y,
                                       V_TF/normV_init,
                                       U=U_hat,
                                       fudge_factor=fudge_factor,
                                       U_update=U_update)

    # --- normalize U by norm 2
    U_hat = U_hat/np.sqrt(np.sum(U_hat**2, 0))

    if final_regression:

        V_TF = np.matmul(np.matmul(np.linalg.inv(
            np.matmul(U_hat.T, U_hat)), U_hat.T), Y)

        # normV_init = np.sqrt(np.sum(V_TF**2,1))[:,np.newaxis]
    return U_hat, V_TF


def decimation_interpolation(data,
                             dims=None,
                             ds=1,
                             rank=2,
                             tsub=1,
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
    # run rank k SVD
    # print('Decimation interpolation') if verbose else 0
    data_tsd = temporal_decimation(data.T, tsub).T
    # print(data_tsd.shape)
    U, s, Vt = compute_svd(data_tsd,
                           n_components=rank,
                           method='randomized')

    U = U.dot(np.diag(s))
    ndims_ = dims[0]//ds, dims[1]//ds, dims[2]

    # Then upsample the resulting decimated U and V to initialize U and V
    # upsample temporal
    x_interp = np.linspace(0, dims[2], dims[2])
    xp_ = x_interp[::tsub]
    Vt_interp = np.zeros((rank, dims[2]))

    for comp_ in range(rank):
        Vt_interp[comp_, :] = np.interp(x_interp, xp_, Vt[comp_, :])

    # upsample spatial
    U_ds = U.reshape(ndims_[:2] + (-1, ), order='F')
    U_ds = sp.ndimage.zoom(U_ds, (ds, ds, 1))
    U_ds = U_ds.reshape((np.prod(dims[:2]), rank), order='F')
    return U_ds, Vt_interp


def temporal_filter_ar(data, p=1):
    """
    """
    data0 = np.zeros(data.shape)
    # T, num_pxls = data.shape
    for ii, trace in enumerate(data.T):
        # Estimate tau for exponential
        tau = cnmf.deconvolution.estimate_time_constant(
                trace, p=p, sn=None, lags=5, fudge_factor=1.)
        window = tau**range(0, 100)
        data0[:, ii] = np.convolve(trace, window,
                                   mode='full')[:T]/np.sum(window)

    return data0


def denoise_components(data,
                       decimation_flag=False,
                       dims=None,
                       ds=1,
                       extra_iterations=1,
                       fudge_factor=1.,
                       greedy=True,
                       max_num_components=20,
                       max_num_iters=10,
                       min_rank=1,
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
                flag to include components which pass correlation
                null hypothesis
    mean_th:    float
                threshold employed to reject components
                according to correlation null hypothesis
    min_rank:   int
                minimum rank of denoised/compressed matrix
                typically set to 1 to avoid empty output (array of zeros)
                if input array is mostly noise.
    greedy:     boolean
                flag to greedily update spatial
                and temporal components (estimated with PCA)
                greedyly by denoising temporal and spatial components
    mean_th_factor: float
                factor to scale mean_th
                typically set to 2 if greedy=True and mean_th=None or
                if mean_th has not been scaled yet.
    fudge_factor: float
                constant to scale estimated noise std
                st denoising st denoising is less
                (lower factor) or more (higher factor) restrictive.
    U_update:   boolean
                flag to (not) update spatial components
                by imposing L1- constraint.
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
                indicator 3D matrix (corr-kurto-reject) which points
                which statistic
                a given component passed and thus it is included.
                If greedy=True, all components added are included
    """
    if (ds != 1) or (tsub != 1):
        print('here')
        decimation_flag = True
        print('Reset flag') if verbose else 0

    # assume data is centered already

    # spatially decimate the data
    if ds > 1:
        print('Spatial decimation by %d' % ds) if verbose else 0
        data = spatial_decimation(data.T, ds=ds, dims=dims).T

    U, s, Vt = compute_svd(data,
                           method='randomized',
                           n_components=max_num_components)

    # if greedy Force x2 mean_th (store only big components)
    if verbose:
        print('Finding components')
    keep1 = find_temporal_component(Vt,
                                    s=s,
                                    plot_en=plot_en
                                    )

    # If no components to store, exit & return min rank
    if len(keep1) == 0:
        if min_rank == 0:
            U = np.zeros(shape=(data.shape[0], 1))
            Vt = np.zeros(shape=(1, data.shape[-1]))
        else:
            min_rank = min_rank
            print('Forcing %d component(s)' % min_rank) if verbose else 0
            # ctid[0, :min_rank]=1
            keep1 = np.arange(min_rank)
            S = np.eye(min_rank)*s[:min_rank]
            U = U[:, :min_rank]
            Vt = S.dot(Vt[:min_rank, :])
            # Yd = U.dot(Vt)
        # Yd += mu
        # Yd*= std
        return U, Vt, keep1

    # Select components
    if decimation_flag:
        if verbose:
            print('Decimation')

        U, Vt = decimation_interpolation(data,
                                         dims=dims,
                                         ds=ds,
                                         rank=len(keep1),
                                         tsub=tsub
                                         )
    else:
        S = np.diag(s[keep1])
        Vt = S.dot(Vt[keep1, :])
        U = U[:, keep1]

    if not greedy:
        return U, Vt, keep1
    if verbose:
        print('Call greedy')
    try:
        U, Vt = greedy_component_denoiser(data,
                                          U,
                                          Vt,
                                          dims=dims,
                                          extra_iterations=extra_iterations,
                                          fudge_factor=fudge_factor,
                                          max_num_iters=max_num_iters,
                                          plot_en=plot_en,
                                          solver=solver,
                                          U_update=U_update,
                                          verbose=verbose)

        # ctid[0,np.arange(Vt.shape[0])] = 1

    except:
        print('\tERROR: Greedy failed, %d components' % (len(keep1)))
        # ctid[0, 0] = 100
    # Yd = U.dot(Vt)
    # n_comp, T = Vt.shape
    return U, Vt, keep1

# -- merge


def c_l1tf_v_hat(v,
                 sigma,
                 abstol=1e-4,
                 solver='SCS',
                 max_iters=1000,
                 verbose=False):
    """
    Update vector v according to difference fctn diff
    with noise_std(v) = sigma

    V(i) = argmin_W ||D^2 W||_1
    st ||V_i-W||_2<sigma_i*sqrt(T)
    Include optimal lagrande multiplier for constraint

    """

    if np.abs(sigma) <= 1e-3:
        return v, 0

    T = len(v)
    v_hat = cp.Variable(T)
    # print(sigma*np.sqrt(T)) if verbose else 0
    diff = difference_operator(T)
    objective = cp.Minimize(cp.norm(diff*v_hat, 1))

    constraints = [cp.norm(v-v_hat, 2)**2 <= (sigma**2)*T]

    cp.Problem(objective, constraints).solve(solver=solver,
                                             max_iters=max_iters,
                                             verbose=False)
    lambda_ = constraints[0].dual_value
    if lambda_ != 0:
        lambda_ = 1./lambda_

    return np.asarray(v_hat.value).flatten(), lambda_


def c_update_V(v, lambda_, cvxpy_solver='SCS', max_iters=1000):
    """
    Peform updates to temporal components
    """
    T = len(v)
    v_hat = cp.Variable(T)
    diff = difference_operator(T)

    cte2 = lambda_*cp.norm(diff*v_hat, 1)
    objective = cp.Minimize(cp.norm(v-v_hat, 2)**2 + cte2)

    cp.Problem(objective).solve(solver=cvxpy_solver,
                                max_iters=max_iters,
                                verbose=False)

    return np.asarray(v_hat.value).flatten()
