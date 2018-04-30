import numpy as np
from sklearn import linear_model

from scipy.optimize import root
import operator
import cvxpy as cp
import denoise
# import admm

# LARS, SP, ADMM
# CP, AIC, BIC


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


def greedy_sp_constrained(M,
                          V1_hat,
                          U=None,
                          method='lars'):
    """
    Solve via crossvalidation
    slow-pow
    """
    # assert norm
    V1_norm = np.sqrt(np.sum(V1_hat**2,1))[:,np.newaxis]
    V1n = (V1_hat/V1_norm)

    if method == 'lars':

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

    return U_hat, mus


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
    print('103')
    print(U.shape)
    n_components = U.shape[1]
    mus_ = np.zeros(n_components)
    for ii, u in enumerate(U.T):
        print(u.shape)
        try:
            mu = denoise.noise_level(u)
        except:
            mu = 0
        mus_[ii]=mu
    #print('Len')
    #print(len(mus_))
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
    #util_plot.plot_spatial_component(UT[:,ii],
    #                             Y_hat=ui,
    #                             dims=dimsM)
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