from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import util_plot
import mpdf_data
import numpy as np
#######################################
# Figure 1

def pdf_write(Y,Yd,
            frame_idx=1000,
            nblocks=None,
            list_order='C',
            pdf_name='single_pdf.pdf',
            pixel_coor1=[10,10],
            pixel_coor2=[11,11],
            plot_colormap='jet',
            ranks=None,
            trace_seg=[0,1000],
            zoom_box=20):
    R = Y-Yd
    pdf_pages = PdfPages(pdf_name)


    ##################### Figure 1
    fig_number = 1
    subplot_number = 1

    f = plt.figure(figsize=(8.27, 11.69))
    fig_rows = 2
    fig_cols = 1
    height_ratios=[0.6,0.8]
    ##### Figure divisions
    gs0 = gridspec.GridSpec(fig_rows,
                            fig_cols,
                            height_ratios=height_ratios,
                            hspace=0.1)
    ###### subfigure 1
    subfig_rows, subfig_cols = 1,4
    width_ratios  = [1,1,1]
    if subfig_cols==4:
        width_ratios.append(0.05)

    gs00 = gridspec.GridSpecFromSubplotSpec(subfig_rows,
                                            subfig_cols,
                                            subplot_spec=gs0[0],
                                            width_ratios=width_ratios)
    axs=[]
    for gs_idx in range(subfig_cols):
        ax1 = plt.Subplot(f, gs00[0, gs_idx])
        f.add_subplot(ax1)
        axs.append(ax1)

    mpdf_data.plot_datain(axs,fig_number,subplot_number,
                            Y,Yd,
                            nblocks=nblocks,
                            ranks=ranks,
                            list_order=list_order,
                            frame_idx=frame_idx,
                            pixel_coor1=pixel_coor1,
                            pixel_coor2=pixel_coor2,
                            trace_seg=trace_seg,
                            zoom_box=zoom_box,
                            plot_colormap=plot_colormap)

    subplot_number+=1
    ###### subfigure 2


    subfig_rows,subfig_cols = 4,1
    gs01 = gridspec.GridSpecFromSubplotSpec(subfig_rows,
                                            subfig_cols,
                                            subplot_spec=gs0[1],
                                            hspace=0)
    # For each row
    for gs_idx in range(subfig_rows):
        ax4 = plt.Subplot(f, gs01[gs_idx, :])
        f.add_subplot(ax4)
        ax4.set_yticks([])

        mpdf_data.plot_datain(ax4,fig_number,subplot_number,
                    Y,Yd,
                    nblocks=nblocks,
                    ranks=ranks,
                    list_order=list_order,
                    frame_idx=frame_idx,
                    pixel_coor1=pixel_coor1,
                    pixel_coor2=pixel_coor2,
                    trace_seg=trace_seg,
                    zoom_box=zoom_box,
                    plot_colormap=plot_colormap)

        if gs_idx<fig_rows-1:
            ax4.set_xticks([])
        subplot_number+=1

    ##################################
    #-------------------------------
    # Store figure
    pdf_pages.savefig(f)
    # ------------------------------
    # Figure 2
    # ------------------------------
    fig_number = 2
    subplot_number = 1

    f = plt.figure(figsize=(8.27, 11.69))
    fig_rows, fig_cols = 3, 1
    height_ratios = [1.1,1,1]#*fig_rows
    gs0 = gridspec.GridSpec(fig_rows,
                            fig_cols,
                            height_ratios=height_ratios)

    ###### subfigures
    subfig_rows, subfig_cols = 1, 4

    width_ratios  = [1,1,1]
    if subfig_cols==4:
        width_ratios.append(0.05)
    
    #############################
    # First Row
    #############################
    # SNR
    gs0_idx = 0
    gs00 = gridspec.GridSpecFromSubplotSpec(subfig_rows,
                                            subfig_cols,
                                            subplot_spec=gs0[gs0_idx],
                                            width_ratios=width_ratios)
    ax_arr=[]
    for gs_idx in range(subfig_cols):
        ax1 = plt.Subplot(f, gs00[0, gs_idx])
        f.add_subplot(ax1)
        #ax1.yaxis.set_ticks_position('right')
        #ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax_arr.append(ax1)

    mpdf_data.plot_datain(ax_arr,
                          fig_number,
                          subplot_number,
                            Y,Yd,
                            nblocks=nblocks,
                            ranks=ranks,
                            list_order=list_order,
                            frame_idx=frame_idx,
                            pixel_coor1=pixel_coor1,
                            pixel_coor2=pixel_coor2,
                            trace_seg=trace_seg,
                            zoom_box=zoom_box,
                            plot_colormap=plot_colormap)

    subplot_number += 1
    
    #############################
    # Second Row
    #############################
    # Compression, Corr Y, Corr R
    #for gs0_idx in [0]:#range(fig_rows):

    gs0_idx = 1
    
    gs00 = gridspec.GridSpecFromSubplotSpec(subfig_rows,
                                            subfig_cols,
                                            subplot_spec=gs0[gs0_idx],
                                            width_ratios=width_ratios)
    
        
    # for each subfig_cols
    for gs_idx in [0,[1,2,3]]:#range(subfig_cols-1):
        if gs_idx==0:
            ax1 = plt.Subplot(f, gs00[0, gs_idx])
            f.add_subplot(ax1)
            #ax1.yaxis.set_ticks_position('right')
            #ax1.set_xticklabels([])
        else:
            ax1=[]
            for gx in gs_idx:
                ax10 = plt.Subplot(f, gs00[0, gx])
                f.add_subplot(ax10)
                ax1.append(ax10)
                #ax1.yaxis.set_ticks_position('right')
                #ax1.set_xticklabels([])          

        mpdf_data.plot_datain(ax1,
                              fig_number,subplot_number,
                                Y,Yd,
                                nblocks=nblocks,
                                ranks=ranks,
                                list_order=list_order,
                                frame_idx=frame_idx,
                                pixel_coor1=pixel_coor1,
                                pixel_coor2=pixel_coor2,
                                trace_seg=trace_seg,
                                zoom_box=zoom_box,
                                plot_colormap=plot_colormap)
        #if gs_idx < subfig_cols-1:
        #    ax1.set_yticklabels([])
        #f.add_subplot(ax1)
        subplot_number +=1


    
    # superpixels
    if False:
        gs0_idx = 2
        gs00 = gridspec.GridSpecFromSubplotSpec(subfig_rows,
                                                subfig_cols,
                                                subplot_spec=gs0[gs0_idx],
                                                width_ratios=width_ratios)
        for gs_idx in range(2):
            ax1 = plt.Subplot(f, gs00[0, gs_idx])
            f.add_subplot(ax1)
            ax1.yaxis.set_ticks_position('right')
            ax1.set_xticklabels([])

            mpdf_data.plot_datain(ax1,fig_number,subplot_number,
                        Y,Yd,
                        nblocks=nblocks,
                        ranks=ranks,
                        list_order=list_order,
                        frame_idx=frame_idx,
                        pixel_coor1=pixel_coor1,
                        pixel_coor2=pixel_coor2,
                        trace_seg=trace_seg,
                        zoom_box=zoom_box,
                        plot_colormap=plot_colormap)
            ax1.set_yticklabels([])

            subplot_number += 1
    #-------------------------------
    # Store figure
    pdf_pages.savefig(f)
    # ------------------------------
    # --------------- Close
    # -------------------------------------
    pdf_pages.close()
    return