
# data for each plot

# trace plot Y, Yd row, colspan 1, colspan2
# trace plot R row, colspan 1, colspan2

import util_plot
import denoise
import numpy as np
# trace plots
# neigh_pixels = 20

#col_right = min(b-neigh_pixels,0)
#col_left= max(b+neigh_pixels,d2)

def trace_extract(Y,Yd,R,a,b,trace_seg=[0,1000]):
    trace_seg_a, trace_seg_b = trace_seg
    t1 = Y[a,b,trace_seg_a:trace_seg_b]
    t2 = Yd[a,b,trace_seg_a:trace_seg_b]
    t3 = R[a,b,trace_seg_a:trace_seg_b]
    offset = min(t2.min(),t1.min())#,t3.min())
    scale = max((t2 -offset).max(), (t1 -offset).max())#,(t3 -offset).max())
    trace1 = (t1 - offset)/scale
    trace2 = (t2 - offset)/scale
    #trace3 = (t3 - offset)/scale #trace1 - trace2#(t3 - offset)/scale
    trace3 = trace1-trace2
    return trace1, trace2, trace3


def extract_frame (x,al,au,bl,bu,frame_idx):
    a = x[al:au,bl:bu,frame_idx]
    #a -= a.min()
    return a#a/a.max()

def box_lim(a1, b1, dims, zoom_box=15):
    al1 = max(0,a1-zoom_box)
    au1 = min(dims[0],a1+zoom_box)

    bl1 = max(0 , b1-zoom_box)
    bu1 = min(dims[1],b1+zoom_box)
    return al1, au1, bl1, bu1

def plot_datain(ax,page_count,cplot_row,Y,Yd,nblocks=None,ranks=None,
                frame_idx=1000,pixel_coor1=[10,10],pixel_coor2=[11,11],
                trace_seg=[0,1000],zoom_box=20,plot_colormap='jet',
                list_order='C'):

    cbar_ticks_number = 4
    trace_offset = 0.02

    #---- Superpixel parameters
    sup_cut_off_point1 = 0.2
    sup_cut_off_point2 = 0.7
    sup_length_cut = 10
    sup_min_threshold = 2
    sup_residual_cut = 0.2
    sup_background = False
    sup_lowrank = False
    sup_hals = False
    # ----------------------
    dims = Y.shape
    R = Y-Yd
    #CnYd, _ = util_plot.correlation_pnr(Yd)
    a1, b1 = pixel_coor1
    a2, b2 = pixel_coor2

    trace1, trace2,trace3 = trace_extract(Y,Yd,R,a1,b1,trace_seg=trace_seg)
    trace4, trace5,trace6 = trace_extract(Y,Yd,R,a2,b2,trace_seg=trace_seg)

    #al1, au1, bl1, bu1 = box_lim(a1, b1, dims, zoom_box=zoom_box)
    al, au, bl, bu = box_lim(a1, b1, dims, zoom_box=zoom_box)
    #al2, au2, bl2, bu2 = box_lim(a2, b2, dims, zoom_box=zoom_box)

    #al = min(al1, al2)
    #au = max(au1, au2)
    #bl = min(bl1, bl2)
    #bu = max(bu1, bu2)

    g1trace_ub = max(trace1.max(), trace2.max(), trace3.max()) + trace_offset
    g1trace_lb = min(trace1.min(), trace2.min(), trace3.min()) - trace_offset
    g2trace_ub = max(trace4.max(), trace5.max(), trace6.max()) + trace_offset
    g2trace_lb = min(trace4.min(), trace5.min(), trace6.min()) - trace_offset

    trace_ub = max(g1trace_ub, g2trace_ub)
    trace_lb = min(g1trace_lb, g2trace_lb)
    #verts = list(zip([-1., 1., 1., -1.], [-1., -1., 1., -1.]))


    # Plot variables

    #################### PAGE 1
    if page_count == 1:

        if cplot_row==1:
            # single frame
            cin =[extract_frame(Y,al,au,bl,bu,frame_idx),
                extract_frame(Yd,al,au,bl,bu,frame_idx),
                extract_frame(R,al,au,bl,bu,frame_idx)]

            d1,d2= cin[0].shape
            for myax in ax[:3]:
                myax.plot(b1-bl,#d2-(a1-bl),
                          a1-al,#d1-(a1-al),
                          c='red',
                          marker='o',
                          mfc="None",
                         markeredgewidth=2)
                myax.plot(b2-bl,#d2-(b1-bl)-(b1-b2),
                          a2-al,#d1-(a1-al)+(a1-a2),
                          c='red',
                          marker='o',
                          mfc="None",
                         markeredgewidth=2)
            
            util_plot.comparison_plot(cin,
                                      plot_show=False,
                                      option='input',
                                      axarr=ax,
                                      cbar_enable=True,
                                      cbar_share=True,
                                      plot_aspect='auto')
  
            for myax in ax[:3]:
                myax.set_xticks([])
                myax.set_yticks([])

        elif cplot_row==2:
            # Trace point 1
            ax.plot(trace1, c='dimgray',ls='-')
            ax.plot(trace2, c='navy',ls='-')
            ax.set_xticks([])

        elif cplot_row ==3:
            ax.plot(trace3, c='dimgray',ls='-')
            ax.set_xticks([])
            ax.set_yticklabels([])

        elif cplot_row ==4:
            ax.plot(trace4, c='dimgray',ls='-')
            ax.plot(trace5, c='navy',ls='-')
            ax.set_xticks([])
            ax.set_yticklabels([])

        elif cplot_row ==5:
            ax.plot(trace6, c='dimgray',ls='-')
            ax.set_xlabel('frames')
            ax.set_yticklabels([])
        return
    ####################
    #################### PAGE 2
    ####################
    if page_count == 2:
        # SNR
        if cplot_row == 1:
            cin =[Y[al:au,bl:bu,:],
                Yd[al:au,bl:bu,:],
                R[al:au,bl:bu,:]]
            util_plot.comparison_plot(cin,
                                      plot_show=False,
                                      option='snr',
                                      axarr=ax,
                                      cbar_enable=True,
                                      cbar_share=True,
                                      #cbar_ticks_number=cbar_ticks_number,
                                      plot_colormap=plot_colormap,
                                      #cbar_orientation='vertical',
                                      plot_aspect='auto')
            for myax in ax[:3]:
                myax.set_xticks([])
                myax.set_yticks([])

        if cplot_row == 2:
            _=util_plot.cn_ranks_plot(ranks,
                                    dims=dims,
                                    nblocks=nblocks,
                                    ax3=ax,
                                    grid_cut=[al,au,bl,bu],
                                    cbar_orientation='vertical',
                                    plot_aspect='auto',
                                    fig_cmap='YlGnBu',
                                    text_en=False)
            ax.set_xticks([])
            ax.set_yticks([])


        elif cplot_row == 3:
            Y1=Y[al:au,bl:bu]
            R1=R[al:au,bl:bu]

            util_plot.comparison_plot([Y1,R1],
                               #vmin=vmin,
                               #vmax=vmax,
                                plot_show=False,
                                plot_aspect='auto',
                                plot_add_residual=False,
                                plot_colormap=plot_colormap,
                                cbar_orientation='vertical',
                                cbar_ticks_number=cbar_ticks_number,
                                cbar_share=True,
                                titles_=['Raw','Residual'],
                                axarr=ax)
            
            for myax in ax[:2]:
                myax.set_xticks([])
                myax.set_yticks([])

            

        elif cplot_row ==4:
            Y1=Y[al:au,bl:bu,:]
            connect_mat_1, unique_pix,  pure_pix = util_plot.extract_superpixels(Y1,
                                                    cut_off_point=sup_cut_off_point1,
                                                    length_cut=sup_length_cut,
                                                    th=sup_min_threshold,
                                                    bg=sup_background,
                                                    residual_cut =sup_residual_cut,
                                                    low_rank=sup_lowrank,
                                                    hals=sup_hals)

            util_plot.superpixel_plotpixel(connect_mat_1,
                                            unique_pix,
                                            pure_pix,
                                            plot_aspect='auto',
                                            plot_colormap=plot_colormap,
                                            ax1=ax,
                                            text=False)

            ax.set_title('Raw\n(Cut %.1f,Len %d)'%(sup_cut_off_point1,
                                                sup_length_cut))

        elif cplot_row ==5:
            Yd1=Yd[al:au,bl:bu,:]
            connect_mat_1, unique_pix, pure_pix = util_plot.extract_superpixels(Yd1,
                                                    cut_off_point=sup_cut_off_point2,
                                                    length_cut=sup_length_cut,
                                                    th=sup_min_threshold,
                                                    bg=sup_background,
                                                    residual_cut =sup_residual_cut,
                                                    low_rank=sup_lowrank,
                                                    hals=sup_hals)

            util_plot.superpixel_plotpixel(connect_mat_1,
                                            unique_pix,
                                            pure_pix,
                                            ax1=ax,
                                            plot_aspect='auto',
                                            plot_colormap=plot_colormap,
                                            text=False)
            ax.set_title('Denoised\n(Cut %.1f,Len %d)'%(sup_cut_off_point2,
                                                sup_length_cut))
        else:
            pass
        return