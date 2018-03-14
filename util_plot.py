from mpl_toolkits.axes_grid1 import make_axes_locatable
import tool_grid as tgrid
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import caiman as cm
from caiman.summary_images import local_correlations_fft
import tools as tools_
import tool_grid as tgrid


def cn_ranks_dx_plot(ranks,
                  dims,
                  nblocks=[10, 10],
                    figsize=15,
                    fontsize=20):
    rtype=[None,'r','c','rc']
    for ii,rank in enumerate(ranks):
        cn_ranks_plot(rank,
                  dims,
                  nblocks=nblocks,
                  offset_case=rtype[ii],
                     figsize=figsize,
                     fontsize=fontsize)
    return
        
        

def cn_ranks_plot(ranks,
                  dims,
                  nblocks=[10, 10],
                  offset_case=None,
                  list_order='C',
                  exclude_max=True,
                  max_rank=100,
                 fontsize=20,
                 figsize=15):
    """
    Plot rank array given ranks of individual tiles,
    and tile coordinates.

    Parameters:
    ----------
    dim_block:
    ranks:
    dims:

    Outputs:
    -------
    Cplot3:         np.array
                    array of ranks per tile
    """
    
     #offset_tiling_dims(dims,nblocks,offset_case=None):

    dims, dim_block = tgrid.offset_tiling_dims(dims,
                                               nblocks,
                                               offset_case=offset_case)
    
    
    K1 = nblocks[0] - 1
    K2 = nblocks[1] - 1
    
    K1 = K1-1 if offset_case =='r' else K1
    K2 =K2-1 if offset_case =='c' else K2
    K1 =K1-1 if offset_case =='rc' else K1
    K2 =K2-1 if offset_case =='rc' else K2
        
    
    Cplot3 = tgrid.cn_ranks(dim_block, ranks,
                            dims[:2], list_order=list_order)
    d1, d2 = dims[:2] // np.min(dims[:2])
    fig, ax3 = plt.subplots(1, 1, figsize=(d1 * figsize, d2 * figsize))

    ranks_ = ranks.copy()
    if exclude_max:
        ranks_[ranks > max_rank] = ranks[ranks > max_rank] % max_rank

    ranks_std = np.std(ranks_)
    vmin_ = max(0, ranks_.min() - ranks_std)
    vmax_ = ranks_.max() + ranks_std

    ax3.set_title('Ranks in each tile %d' % (
        np.sum(np.asarray(ranks_))))
    im3 = ax3.imshow(Cplot3, vmin=vmin_,
                     vmax=vmax_, cmap='Reds',
                     interpolation='nearest', aspect='equal')

    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im3, cax=cax3, format='%d',
                 ticks=np.linspace(vmin_, vmax_, 5))

    dim_block = np.asarray(dim_block)
    cols, rows = dim_block.T[0], dim_block.T[1]
    

    row_array = np.insert(rows[::K1 + 1], 0, 0).cumsum()
    col_array = np.insert(cols[::K2 + 1], 0, 0).cumsum()

    x, y = np.meshgrid(row_array[:-1], col_array[:-1])
    ax3.set_yticks(col_array[:-1])
    ax3.set_xticks(row_array[:-1])

    for ii, (row_val, col_val) in enumerate(zip(x.flatten(order=list_order), 
                                                y.flatten(order=list_order))):
        c = str(int(Cplot3[int(col_val + 1), int(row_val + 1)]) % max_rank)
        ax3.text(row_val + rows[ii] / 2, col_val +
                 cols[ii] / 2, c, va='center', ha='center',fontsize=fontsize)
    plt.tight_layout()
    plt.show()
    return Cplot3


def plot_comp(Y, Y_hat=None, title_=None, dims=None, idx_=0):
    """
    Plot comparison for frame idx_ in Y, Y_hat.
    assume Y is in dxT to be reshaped to dims=(d1,d2,T)
    """
    if Y_hat is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        ax.set_title(title_)
        plots_ = zip([ax], [Y])
    else:
        R = Y - Y_hat
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        ax[0].set_title(title_)
        plots_ = zip(ax, [Y, Y_hat, R])

    for ax_, arr in plots_:
        if len(dims) > 2:
            ims = ax_.imshow(arr.reshape(dims, order='F')[:, :, idx_])
        else:
            ims = ax_.imshow(arr.reshape(dims, order='F'))
        #ims = ax_.imshow(arr.reshape(dims,order='F').var(2))
        d = make_axes_locatable(ax_)
        cax0 = d.append_axes("bottom", size="5%", pad=0.5)
        cbar0 = plt.colorbar(
            ims, cax=cax0, orientation='horizontal', format='%.0e')
    plt.tight_layout()
    plt.show()
    return


def plot_temporal_traces(V_TF, V_hat=None):
    """
    """
    for idx, vt in enumerate(np.asarray(V_TF)):
        plt.figure(figsize=(15, 5))
        plt.title('Temporal component %d' % idx)
        if V_hat is not None:
            if np.ndim(V_hat) <= 1:
                plt.plot(V_hat, 'b')
            else:
                plt.plot(V_hat[idx, :], 'b')
        plt.plot(vt, 'r')
        plt.show()
    return


def plot_spatial_component(U_hat, dims):
    """
    """
    for ii in range(U_hat.shape[1]):
        plot_comp(U_hat[:, ii], title_='Spatial component U' +
                  str(ii), dims=dims[:2])
    return


def plot_vt_cov(Vt1, keep1, maxlag):
    """
    Plot figures of ACF of vectors in Vt1 until maxlag
    (right pannel) keep1 and (left pannel) other components

    Parameters:
    ----------

    Vt1:    np.array (k xT)
            array of k temporal components lasting T samples
    keep1: np.array (k1 x 1)
            array of temporal components which passed a given hypothesis
    maxlag: int
            determined lag until which to plot ACF function of row-vectors
            in Vt1

    Outputs:
    -------

    None:   displays a figure

    """
    fig, axarr = plt.subplots(1, 2, sharey=True,figsize=(10,5))
    loose = np.setdiff1d(np.arange(Vt1.shape[0]), keep1)
    for keep in keep1:
        vi = Vt1[keep, :]
        vi = (vi - vi.mean()) / vi.std()
        metric = tools_.axcov(vi, maxlag)[maxlag:] / vi.var()
        axarr[0].plot(metric, '-r',linewidth=1.0)

    for lost in loose:
        vi = Vt1[lost, :]
        vi = (vi - vi.mean()) / vi.std()
        metric = tools_.axcov(vi, maxlag)[maxlag:] / vi.var()
        axarr[1].plot(metric, ':b',linewidth=2.0)

    ttext = ['Selected components: %d' % (len(keep1)),
             'Discarded components: %d' % (len(loose))]
    for ii, ax in enumerate(axarr):
        ax.set_xscale('symlog')
        ax.set_ylabel('ACF')
        ax.set_xlabel('lag')
        ax.set_yticks([0,0.5,1])
        ax.set_title(ttext[ii])
    #plt.savefig('cosyne_Vt_keep.svg')
    plt.show()
    return


def show_img(ax,
             img,
             vmin=None,
             vmax=None,
             cbar_orientation='horizontal',
             plot_colormap='jet',
             cbar_ticks_number=None,
             cbar_ticks=None,
             cbar_enable=True):
    """
    Visualize image
    """

    vmin= img.min() if vmin is None else vmin
    vmax= img.max() if vmax is None else vmax


    if np.abs(img.min()) <= 1:
        if np.abs(img.min()) <= -1e-1:
            format_tile = '%.1e'
        else:
            format_tile = '%.2f'
    else:
        format_tile = '%5d'

    if cbar_ticks_number is not None:
        cbar_ticks= np.linspace(vmin,
                                vmax,
                                cbar_ticks_number,
                                endpoint=True)
        cbar_ticks=np.round(cbar_ticks,4)   
        cbar_ticks_labels= [format_tile%(cbar_) 
                            for cbar_ in cbar_ticks]
        vmin, vmax = cbar_ticks[0], cbar_ticks[-1]

    #######################
    # Build Plot
    #######################

    d1,d2= img.shape
    im = ax.imshow(img,
                   vmin=vmin,
                   vmax=vmax,
                   cmap=plot_colormap,
                   extent=[0,d2,0,d1])

    divider = make_axes_locatable(ax)
    if cbar_orientation == 'horizontal':
        cbar_direction ='bottom'
    elif cbar_orientation == 'vertical':
        cbar_direction ='right'

    if cbar_enable is False:
        return
    cax = divider.append_axes(cbar_direction,
                          size="5%",
                          pad=0.3)

    cbar = plt.colorbar(im,
                        cax=cax,
                        orientation=cbar_orientation,
                        spacing='uniform',
                        format=format_tile,
                        ticks=cbar_ticks)

    return


def comparison_plot(cn_see,
                    option='corr',
                    plot_orientation='horizontal',
                    cbar_orientation='vertical',
                    cbar_indiv_range=None,
                    title=True,
                    title_suffix='',
                    titles_='',
                    share_colorbar=False,
                    plot_colormap='jet',
                    plot_num_samples=1000,
                    cbar_ticks_number=None):
    """
    """
    if share_colorbar:
        min_dim=4
    else:
        min_dim=3
    if titles_=='':
        titles_=['original ','denoised ','residual ']

    if len(cn_see)==2:
        cn_see.append(cn_see[0]-cn_see[1])
        titles_.append('residual ')

    if plot_orientation == 'horizontal':
        d1, d2 = min_dim,1
        sharex = True
        sharey = False

    elif plot_orientation =='vertical':
        d1, d2 = 1,min_dim
        sharex = False
        sharey = True

    Cn_all =[]

    #######################
    # Calculate Cn to plot
    #######################
    for ii, array in enumerate(cn_see):
        print(array.shape)        
        if option =='corr': # Correlation
            Cn, _ = cm.summary_images.correlation_pnr(array,
                                                      gSig=None,
                                                      center_psf=False,
                                                      swap_dim=True) # 10 no ds
            title_prefix = 'Local correlation: '
        elif option =='var': #Variance
            Cn = array.var(2)/array.shape[2]
            title_prefix = 'Pixel variance: '
            print(Cn.min())
            print(Cn.max())
        elif option =='pnr': # PNR
            _, Cn = cm.summary_images.correlation_pnr(array,
                                                      gSig=None,
                                                      center_psf=False,
                                                      swap_dim=True)
        elif option=='input':
            Cn =array - array.min()
            Cn = Cn/Cn.max()
            title_prefix = 'Single Frame: '
        print ('%s range [%.1e %.1e]'%(title_prefix,
                                   Cn.min(),
                                   Cn.max()))
        Cn_all.append(Cn)

    #######################
    # Plot configuration
    #######################

    vmax_ = list(map(np.max,Cn_all))
    vmin_ = list(map(np.min,Cn_all))

    if share_colorbar:
        vmax_ = [max(vmax_)]*3
        vmin_ = [min(vmin_)]*3

    if cbar_indiv_range is not None:
        for ii,range_ in enumerate(cbar_indiv_range):
            vmin_[ii]=range_[0]
            vmax_[ii]=range_[1]

    dim2, dim1 = Cn.shape
    x_ticks= np.linspace(0,dim1,5).astype('int')
    y_ticks= np.linspace(0,dim2,5).astype('int')

    fig, axarr = plt.subplots(d1,d2,
                              figsize=(d1*12,d2*12),
                              sharex=sharex,
                              sharey=sharey)

    #cbar_enable= [False,False,True]
    #if share_colorbar:
    cbar_enable= not share_colorbar
    

    for ii, Cn in enumerate(Cn_all):
        show_img(axarr[ii],
                 Cn,
                 cbar_orientation=cbar_orientation,
                 vmin=vmin_[ii],
                 vmax=vmax_[ii],
                 plot_colormap=plot_colormap,
                 cbar_ticks_number=cbar_ticks_number,
                 cbar_enable=cbar_enable)

        axarr[ii].set_xticks(x_ticks)
        axarr[ii].set_yticks(y_ticks)
        axarr[ii].set_xticklabels([])
        axarr[ii].set_yticklabels([])

        if title:
            axarr[ii].set_title(title_prefix
                                + titles_[ii]
                                + title_suffix)


    plt.tight_layout()
    #plt.savefig('cosyne_comparison_single_frame_vertical.svg')
    plt.show()
    return


def intialization_plot(data_highpass,
                       patch_radius=20,
                       min_pnr=0,
                       min_corr=0,
                       stdv_pixel=None,
                       noise_thresh=3,
                       orientation='horizontal'):  # down,side
    """
    """

    # Create plot obj according to specifications
    if orientation == 'horizontal':
        d1, d2 = 2, 1
    elif orientation == 'vertical':
        d1, d2 = 1, 2
    fig, axarr = plt.subplots(d1, d2, figsize=(14, 7), sharex=True)

    # Compute pixel-wise noise stdv
    if not stdv_pixel:
        stdv_pixel = np.sqrt(np.var(data_highpass, axis=-1))

    # Compute & plot corr image
    data_spikes = data_highpass - \
        np.median(data_highpass, axis=-1)[:, :, np.newaxis]
    data_spikes[data_spikes < noise_thresh * stdv_pixel[:, :, np.newaxis]] = 0
    corr_image = local_correlations_fft(
        data_spikes.transpose([2, 0, 1]), swap_dim=False)

    if min_corr:
        corr_image[corr_image < min_corr] = 0
    show_img(axarr[0], corr_image, orientation=orientation)
    axarr[0].set_title('Thresholded Corr Image')

    # Compute & plot pnr image
    pnr_image = np.divide(np.max(data_highpass, axis=-1),
                          stdv_pixel)
    pnr_image[np.logical_or(corr_image < min_corr, pnr_image < min_pnr)] = 0
    pnr_image = filters.median_filter(pnr_image,
                                      size=(int(round(patch_radius / 4)),) * 2,
                                      mode='constant')
    show_img(axarr[1], pnr_image, orientation=orientation)
    axarr[1].set_title('Thresholded & Filtered PNR Image')

    # Display PLot
    plt.tight_layout()
    plt.show()
    return


def tiling_grid_plot(W,
                     nblocks=[10, 10],
                    plot_option='var'):
    """
    """
    dims = W.shape

    col_array, row_array = tgrid.tile_grids(dims, nblocks)

    x, y = np.meshgrid(row_array, col_array)
    if plot_option == 'var':
        Cn1 = W.var(2)
    elif plot_option =='same':
        Cn1 = W
    plt.figure(figsize=(15, 5))
    plt.yticks(col_array)
    plt.xticks(row_array)
    plt.plot(x.T, y.T)
    plt.plot(x, y)
    plt.imshow(Cn1)
    plt.show()
    return