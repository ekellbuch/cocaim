# Define all desired plots given two inputs

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
import matplotlib.pyplot as plt

import util_plot

# Y-Yd
# where Y d1 x d2 x T
# where Yd d1 x d2 x T
# T max_nume_times to consider =

def call_figure(Y,Yd,gpca_ranks,nblocks,dx):

    d1,d2,T = Yd.shape
    dims = Yd.shape
    #nblocks = [d1/bwidth,d2/bheight]
    #dx = 1
    sample_frame = 700#
    figure_size = 14
    font_size = 10
    titles_ = ['Raw', 'Denoised']
    if d1 < d2:
        plot_orientation='horizontal'
        cbar_orientation='horizontal'

    with PdfPages('multipage_pdf.pdf') as pdf:
        # Figure 1: rank-by-tile plots for dx = 4 or dx =1
        # If dx === 1 plot individual ranks
        # 1 page
        print('Writing Rank plots')
        if dx ==1:
            util_plot.cn_ranks_plot(gpca_ranks,
                                        dims,
                                        nblocks=nblocks,
                                        figsize=figure_size,
                                        fontsize=font_size,
                                        save_fig=True,
                                        fig_pdf_var=pdf)
        # for dx =4,
        # page 1 plot individual ranks in a 4x4 grid
        elif dx ==4:
            util_plot.cn_ranks_dx_plot(gpca_ranks,
                                        dims,
                                        nblocks=nblocks,
                                        figsize=figure_size,
                                        fontsize=font_size,
                                        save_fig=True,
                                        fig_pdf_var=pdf)

            # Figure 2
            # page 2 plot a sum rank 2 plot in a single page
            util_plot.cn_ranks_sum_plot(gpca_ranks,
                                dims,
                                nblocks=nblocks,
                                figsize=figure_size,
                                #fontsize=20,
                                #tile_err=100,
                                #include_err=True,
                                fig_pdf_var=pdf,
                                #plot_pdf_var=None,
                                #save_fig=False,
                                #save_fig_name=''):
                                )

        # Figure 3
        # sample frame from Y,Yd,R video
        # page 3 subplot 1
        #print('Writing single frame plots')
        #if plot_orientation == 'horizontal':
        #    d1, d2 = 3,1

        #elif plot_orientation =='vertical':
        #    d1, d2 = 1,3

        #fig, axall = plt.subplots(2,3,fig_size=)
        util_plot.comparison_plot([Y[:,:,sample_frame],Yd[:,:,sample_frame]],
                                  option='input',
                                  titles_=titles_,
                                  plot_orientation=plot_orientation,
                                  cbar_orientation=cbar_orientation,
                                  save_fig=True,
                                  fig_pdf_var=pdf
                                 )

        #canvas = FigureCanvasPdf(fig_2a)
        #canvas.print_figure(pdf)
        # Figure 4
        # page 3 subplot 2 Local correlation plot
        print('Writing corr plots')
        util_plot.comparison_plot([Y,Yd],
                                  option='corr',
                                  titles_=titles_,
                                  plot_orientation=plot_orientation,
                                  save_fig=True,
                                  cbar_orientation=cbar_orientation,
                                  fig_pdf_var=pdf
                                 )
        #canvas = FigureCanvasPdf(fig_2b)
        #canvas.print_figure(pdf)

        # SNR-per-pixel image of Y, Yd, R
        # page 4 subplot 1
        #if False:
        print('Writing SNR per pixel plots')
        util_plot.comparison_plot([Y,Yd],
                                  option='snr',
                                  titles_=titles_,
                                  plot_orientation=plot_orientation,
                                  save_fig=True,
                                  cbar_orientation=cbar_orientation,
                                  fig_pdf_var=pdf
                                 )
        # page 4 subplot 2
        # SNR-per-frame trace of Y, Yd, R
        print('Writing SNR per frame plots')
        util_plot.correlation_traces(Y,Yd,
                                    #fig_size=(figure_size,figure_size),
                                    fig_pdf_var=pdf)

        # plot showing a comparison against vanilla PCA baseline
        # superpixel plots (based on Y vs based on Yd)?
    return
