import numpy as np
from sklearn import linear_model

from scipy.optimize import root
import operator
import cvxpy as cp
from . import denoise

# import admm
# LARS, SP, ADMM
# CP, AIC, BIC
# Deprecated

def soft_threshold_ls(betas_, lambdas_):
    # LS estimate beta_
    ndim = np.ndim(betas_)

    if ndim ==1:
        betas_=betas_[:,np.newaxis]
    betas_lasso = np.zeros(betas_.shape)
    for kk, beta_ in enumerate(betas_.T):
        #print(kk)
        lambda_ = lambdas_[kk]
        #print('lambdas')
        #print(lambdas_[kk])
        additive_factors = np.zeros(len(beta_),)
        beta_lasso = np.sign(beta_)*(np.abs(beta_)-lambda_)
        
        for ii, cbeta_ in enumerate(beta_):
            if lambda_ < np.abs(cbeta_):
                if cbeta_ > 0:
                    #print('a')
                    additive_factor = cbeta_- lambda_
                elif cbeta_ < 0:
                    #print('b')
                    additive_factor = cbeta_ + lambda_
            elif lambda_ >= np.abs(cbeta_):
                #print('c')
                additive_factor = 0
            additive_factors[ii] = additive_factor
        betas_lasso[:,kk] = beta_lasso + additive_factors

    return betas_lasso


def greedy_sp_constrained(M,
                          V1_hat,
                          dimsM=None,
                          mus=None,
                          U=None,
                          method='lars'):
    """
    Solve via crossvalidation
    slow-pow
    """
    # assert norm
    if np.ndim(V1_hat)==1:
        V1_hat = V1_hat[np.newaxis,:]
    V1_norm = np.sqrt(np.sum(V1_hat**2,1))[:,np.newaxis]
    V1n = (V1_hat/V1_norm)
    if method == 'lars1':
        num_pixels  = M.shape[0]
        num_components = V1_hat.shape[0]

        U_hat = np.zeros((num_pixels,num_components))
        mus = np.zeros(num_pixels)

        for pix in range(num_pixels):
            y = M[pix,:]
            u_hat, mu = l1_spatial_constrained(y,V1n)
            U_hat[pix,:] = u_hat
            mus[pix] = mu
    elif method == 'sp':
        mus = lagrangian_hat_spatial(U)
        U_hat = iterative_solution(M, V1n, mus)
    elif method == 'admm':
        pass
    elif method == 'soft':
        U_hat = np.matmul(M, np.matmul(V1n.T, np.linalg.inv(np.matmul(V1n, V1n.T))))
        #U_hat = U_hat/V1_norm.flatten()
        #nU = np.sqrt(np.sum(U_hat**2,0))
        #U_hat = U_hat/nU
        if mus is None:
            #U_hat2 = U_hat.copy()
            mus = lagrangian_hat_spatial(U_hat)
            #print(np.array_equiv(U_hat2,U_hat))

        #return U_hat,mus
        U_hat = soft_threshold_ls(U_hat,mus)

    return U_hat, mus

def greedy_sp_dual(M,V1_hat,mus_,U=None,method='lars'):
    #print('Calling sp dual')
    #print(len(mus_))
    V1_norm = np.sqrt(np.sum(V1_hat**2,1))[:,np.newaxis]
    V1n = (V1_hat/V1_norm)

    if method == 'lars':

        num_pixels  = M.shape[0]
        num_components = V1n.shape[0]

        U_hat = np.zeros((num_pixels,num_components))
        #print('Called sp dual')
        for pix in range(num_pixels):
            y = M[pix,:]
            calph = mus_[pix]
            u_hat = l1_spatial_dual(y,V1n,calph)
            #print('out from spdual')
            U_hat[pix,:] = u_hat

    elif method =='sp':
        print('35')
        print(U.shape)
        mus_ = lagrangian_hat_spatial(U)
        print('36')
        U_hat = iterative_solution(M, V1n, mus_)

    elif method == 'admm':
        pass

    return U_hat


def l1_spatial_dual(y,v,alpha_):
    """
    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    """
    #print('sp dual each')
    #print('here verbose is True')
    #print(alpha_)
    reglars = linear_model.LassoLars(alpha=alpha_)
    #print('sp 46')
    #print(reglars.__dict__.keys())
    #print('init reg')
    reglars.fit(v.T, y, Xy=None)#[np.newaxis,:])
    #print('sp 50 out')
    u_hat = reglars.coef_
    return u_hat


def lagrangian_hat_spatial(U,
                noise_method='logmexp'):
    n_components = U.shape[1]
    mus_ = np.zeros(n_components)
    for ii, u in enumerate(U.T):
        #print('noise in u ')
        try:
            #print(u.shape)
            mu = denoise.noise_level(u[np.newaxis,:])[0]
        except:
            #print('failed')
            mu = 0
        mus_[ii]=mu
    return mus_


def iterative_solution(M,V_TF_n,sigma2_S):
    #print('sp 105')
    n_pixels = M.shape[0]
    n_components = V_TF_n.shape[0]
    R = M.copy()
    U_hat_p = np.zeros((n_pixels,n_components))
    #print('sp 2016')
    for ii in range(n_components):
        vi_ii = V_TF_n[ii,:]
        cu_chosen = lambda_u(Y=R, v=vi_ii, sigma2=sigma2_S[ii])
        ui = update_u(Y=M,v=vi_ii,c_u=cu_chosen)
        U_hat_p[:,ii]=ui
        R -= np.outer(ui,vi_ii)

    return U_hat_p




def l1_spatial_constrained(y,v,n_alphas=100,
                    eps=1e-3):
    """
    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    """
    #print('greedy_spatial 79')
    ls = linear_model.LassoCV(n_alphas=n_alphas,
                            eps=eps,
                            normalize=True,
                            tol=1e-3)
    ls.fit(v.T, y)
    #print('greedy_spatial 83')
    alpha_ = ls.alpha_ # amount of penalization
    u_hat = ls.coef_
    #print('greedy_spatial 86')
    return u_hat , alpha_


def dual_search_l1_pixel(y,v,u,num_lambda=10):
    """
    """

    # Defina parameters
    T = len(y)          # vector length
    # num_lambda = 10   # number of lambdas over which to solve for

    # Define range of lambdas over which to search
    lambda_max = np.sqrt(T)
    lambda_vals = np.linspace(1, lambda_max, num_lambda)

    # calculate CP for each lambda
    # (1) try by calculating CP
    # (2) try by calculating lasso Largs regression?
    CPS = np.zeros(num_lambda)
    for ii, lambda_ in enumerate(lambda_vals):
        # number of nonzero coordinates
        u_n0 = np.linalg.norm(u, ord=0)
        # calculate current value
        #CPS[ii]= cp_lasso_dual(y,u,v,lambda_,u_n0)
        u_hat = lasso_lars_regression(y,v,lambda_)
        CPS[ii] = cp_lasso_dual(y,u_hat,v,lambda_,u_n0)

    return CPS , lambda_vals


def cp_lasso_dual(y,u,v,lambda_,df):
    """
    CP (lambda) = min || y- uv||_2 ^2 + lambda_||u||_1  + 2*df
    y           y \in R^(1,T)
    u           u \in R^(1,k)
    v           v \in R^(k,T)
    lambda_     lambda_ \in R
    df          df \in R
    """
    CP = np.linalg.norm( y-u.dot(v),2)**2 + lambda_*np.linalg.norm(u,1) + 2*df
    return CP


def lasso_lars_regression(y,
                        v,
                        lambda_lasso):
    """
    Given lambda, we solve the following problem
    min_u || y- uv||_2 ^2 + lambda_||u||_1
    by using the LARS which says
    """
    #if np.size(cct_) index only neurons
    #lambda_lasso = 0 if np.size(cct_) == 0 else \
    #lambda_lasso = .5 * noise_sn[px] * np.sqrt(np.max(cct_)) / T
    clf = linear_model.LassoLars(alpha=lambda_lasso, positive=True)
    a_lrs = clf.fit(np.array(v.T), np.ravel(y))
    a = a_lrs.coef_
    return a


def lambda_u(Y,v,sigma2):
    Yv = Y.dot(v)
#    def fu(x):
#        return sum(xplus(abs(Yv)-x))/(sum((xplus(abs(Yv)-x))**2))**0.5
    c_max = len(Yv)**0.5 #fu(0)
    can_cu0 = np.arange(1,c_max,c_max/10)
    can_cu = can_cu0[1:-1]
    uCP = np.zeros(len(can_cu))
    #can_cv = np.arange(1,T**0.5,T**0.5/10)
    #print('sp 204')
    for i in range(len(can_cu)):
        cui = can_cu[i]
        uhat = update_u(Y,v,c_u=cui)
        df = np.linalg.norm(uhat,ord=0)
        uCP[i] = CP(y=Y.dot(v),yhat=uhat,df=df,sigma2=sigma2)
    index, value = min(enumerate(uCP), key=operator.itemgetter(1))
    cu_chosen = can_cu[index]
    return cu_chosen

#CP criterion
def CP(y,yhat,df,sigma2):
    return (np.linalg.norm(y-yhat,ord=2))**2+2*df*sigma2

def xplus(vector):
    vectorplus = vector.copy()
    for i in range(len(vector)):
        vectorplus[i] = max(vector[i],0)
    return vectorplus


def dist(a,c):
    """
    a is vector; c is a constant c>0
    """
    dist = []
    for ai in range(len(a)):
        dist_element = np.sign(a[ai])*max((abs(a[ai])-c),0)
        dist.append(dist_element)
    return dist

def update_u(Y,v,c_u):
    """
    update u given Y and v
    """
    Yv = Y.dot(v)
    def fu(x):
        return sum(xplus(abs(Yv)-x))/(sum((xplus(abs(Yv)-x))**2))**0.5

    delta = 0
    u0no = dist(Yv,delta)
    u0_l2 = np.linalg.norm(u0no,ord=2)
#    u0 = u0no.copy
    u0 = u0no/u0_l2
    u0_l1 = np.linalg.norm(u0,ord=1) #sum(xplus(abs(Yv)-delta))

    if u0_l1 <= c_u:
        u = u0.copy()
    else:
        vec = Yv.copy()
        def gu(x):
            return sum(xplus(abs(vec)-x))/(sum((xplus(abs(vec)-x))**2))**0.5-c_u
        delta_root = root(gu,x0=2,tol=1e-5)
        delta = delta_root.x[0]
        u = dist(Yv,delta)
        u = u/np.linalg.norm(u,ord=2)

    return u



# serving the function "get_noise_fft"
def mean_psd(y, method = 'logmexp'):
    """
    Averaging the PSD

    Parameters:
    ----------

    y:  np.ndarray
        PSD values

    method: string
        method of averaging the noise.
        Choices:
         'mean': Mean
         'median': Median
         'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Outputs:
    -------
        mp: array
            mean psd
    """

    if method == 'mean':
        mp = np.sqrt(np.mean(np.divide(y,2),axis=-1))
    elif method == 'median':
        mp = np.sqrt(np.median(np.divide(y,2),axis=-1))
    else:
        mp = np.log(old_div((y+1e-10),2))
        mp = np.mean(mp,axis=-1)
        mp = np.exp(mp)
        mp = np.sqrt(mp)

    return mp


#get the noise via fft
def get_noise_fft(Y, noise_range = [0.25,0.5],
        noise_method = 'logmexp', max_num_samples_fft=3072):
    """
    Extracted from caiman
    Estimate the noise level for each pixel by averaging
    the power spectral density.

    Parameters:
    ----------

    Y: np.ndarray

    Input movie data with time in the last axis

    noise_range: np.ndarray [2 x 1] between 0 and 0.5
        Range of frequencies compared to Nyquist rate over
        which the power spectrum is averaged
        default: [0.25,0.5]

    noise method: string
        method of averaging the noise.
        Choices:
            'mean': Mean
            'median': Median
            'logmexp': Exponential of the mean of the log of PSD

    Outputs:
    -------

    sn: np.ndarray
        Noise level for each pixel
    """
    T = np.shape(Y)[-1]
    Y = np.array(Y).astype('float64')

    if T > max_num_samples_fft:
        Y=np.concatenate((Y[...,1:np.int(old_div(max_num_samples_fft,3))+1],        
                         Y[...,np.int(np.divide(T,2)-max_num_samples_fft/3/2):np.int(np.divide(T,2)+max_num_samples_fft/3/2)],
                         Y[...,-np.int(old_div(max_num_samples_fft,3)):]),axis=-1)
        T = np.shape(Y)[-1]

    dims = len(np.shape(Y))
    #we create a map of what is the noise on the FFT space
    ff = np.arange(0,0.5+np.divide(1.,T),np.divide(1.,T))
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1,ind2)
    #we compute the mean of the noise spectral density s
    if dims > 1:
        xdft = np.fft.rfft(Y,axis=-1)
        psdx = (np.divide(1.,T))*abs(xdft)**2
        psdx[...,1:] *= 2
        sn = mean_psd(psdx[...,ind[:psdx.shape[-1]]], method = noise_method)

    else:
        xdft = np.fliplr(np.fft.rfft(Y))
        psdx = (np.divide(1.,T))*(xdft**2)
        psdx[1:] *=2
        sn = mean_psd(psdx[ind[:psdx.shape[0]]], method = noise_method)

    return sn, psdx

def local_spatial_estimate(Y_new, gHalf=[2,2], sn=None):
    """
    Apply a wiener filter to image Y_new d1 x d2 x T
    """
    mean_ = Y_new.mean(axis=2,keepdims=True)
    if sn is None:
        sn,_= get_noise_fft(Y_new - mean_,noise_method='median')
        if 0:
            plt.title('Noise level per pixel')
            plt.imshow(sn)
            plt.colorbar()
            plt.show()
    else:
        print('sn given')
    #Cnb, _ = util_plot.correlation_pnr(Y_new) #
    #maps = [Cnb.min(), Cnb.max()]

    Y_new2 = Y_new.copy()
    Y_new3 = np.zeros(Y_new.shape)#Y_new.copy()

    d = np.shape(Y_new)
    n_pixels = np.prod(d[:-1])

    center = np.zeros((n_pixels,2)) #2D arrays

    k_hats=[]
    for pixel in np.arange(n_pixels):
        if pixel % 1e3==0:
            print('first %d/%d pixels'%(pixel,n_pixels))
        ij = np.unravel_index(pixel,d[:2])
        for c, i in enumerate(ij):
            center[pixel, c] = i
        # Get surrounding area
        ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, d[c])]
                for c in range(len(ij))]

        Y_curr = np.array(Y_new[[slice(*a) for a in ijSig]].copy(),dtype=np.float32)
        sn_curr = np.array(sn[[slice(*a) for a in ijSig]].copy(),dtype=np.float32)
        cc1 = ij[0]-ijSig[0][0]
        cc2 = ij[1]-ijSig[1][0]
        #neuron_indx = int(np.ravel_multi_index((cc1,cc2),Y_curr.shape[:2],order='F'))
        #Y_out , k_hat = spatial_filter_block(Y_curr, sn=sn_curr,
        #        maps=maps, neuron_indx=neuron_indx)
        #Y_new3[ij[0],ij[1],:] = Y_out[cc1,cc2,:]
        #Y_curr
        #k_hats.append(k_hat)
        mean_img = np.median(Y_curr)#np.mean(sn_curr)/np.prod(sn_curr.shape)
        error = Y_curr -  mean_img
        #k_hats.append(error**2)
       # k_hat = np.median(sn_curr)
        k_hat=np.sum(error[cc1,cc2]**2)/np.prod(d)#/d[2]
        #print(d[2])
        k_hats.append(k_hat)
    return np.asarray(k_hats)#.asarray()
