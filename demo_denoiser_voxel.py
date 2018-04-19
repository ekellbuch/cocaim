import os

import sys
import h5py


#sys.path.append('../CaImAn/')

import time
import numpy as np

import skimage.io
import skvideo.io

import denoise
import util_plot
import util_movie
import voxel_denoiser

#import greedyPCA as gpca
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
    return

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")
    return

def store_output_movie(movie,
            file_name):
    np.savez(file_name+'.npz',
      data = movie)
    return

def main(m_orig1,m_orig2,m_orig3,
    den_spatial=False,
    den_temporal=False,
    extract_rank=False,
    store_outputs=False,
    make_movie=False,
    make_plots=False):

    gHalf = [2,2]
    range_ff=[0.25,0.5]
    nblocks = [7,7]
    greedy = True
    dx = 4
    movie_length = m_orig1.shape[2]#1000
    verbose = False
    rank_k = 4
    # File name outs
    fname_out_rk = data_file_out+ '_rk'
    fname_out_wf = data_file_out +'_wf'
    fname_out_pca = data_file_out +'_gpca'
    fname_out_movie = data_file_out 


    print('Calculate/scale wrt noise level')
    tic
    m_orig1=m_orig1[:,:,100:-100]
    m_orig2=m_orig2[:,:,100:-100]
    m_orig3=m_orig3[:,:,100:-100]

    m_orig1=m_orig1[:,:,:movie_length]
    m_orig2=m_orig2[:,:,:movie_length]
    m_orig3=m_orig3[:,:,:movie_length]

    print('Calculate noise level')
    tic()
    noise_level1 = denoise.noise_level(m_orig1,
                                      range_ff=[0.25,0.5])

    mov_nn1 = m_orig1/noise_level1[:,:,np.newaxis]
    noise_level2 = denoise.noise_level(m_orig2,
                                      range_ff=[0.25,0.5])

    mov_nn2 = m_orig2/noise_level2[:,:,np.newaxis]

    noise_level3 = denoise.noise_level(m_orig3,
                                      range_ff=[0.25,0.5])
    mov_nn3 = m_orig3/noise_level3[:,:,np.newaxis]
    toc()

    interleave=False
    maxlag = 3
    confidence = 0.99
    fudge_factor = 1
    mean_th_factor=1.15
    U_update = False
    min_rank = 1

    print('Running temporal denoiser')
    tic()
    W_four_1,W_four_2,W_four_3,ranks_ = voxel_denoiser.denoise_dx_voxel_tiling(mov_nn1,mov_nn2,mov_nn3,dx=dx,
                                                                         nblocks=nblocks,
                                                                         interleave=interleave,
                                                                         maxlag=maxlag,
                                                                         confidence=confidence,
                                                                         greedy = greedy,
                                                                         fudge_factor=fudge_factor,
                                                                         mean_th_factor=mean_th_factor,
                                                                         U_update=U_update,
                                                                         min_rank=min_rank
                                                )
    toc()

    if make_plots:
        util_plot.comparison_plot([mov_nn1, W_four_1],
                                  option='corr',
                                  titles_=['Raw in plane 1', 'gPCA output'],
                                 plot_orientation='horizontal',
                                 cbar_orientation='horizontal',
                                save_fig=True,
                                save_fig_name=data_file_out+'_p1_gpca')

        util_plot.comparison_plot([mov_nn2, W_four_2],
                                  option='corr',
                                  titles_=['Raw in plane 2', 'gPCA output'],
                                 plot_orientation='horizontal',
                                 cbar_orientation='horizontal',
                                save_fig=True,
                                save_fig_name=data_file_out+'_p2_gpca')

        util_plot.comparison_plot([mov_nn3, W_four_3],
                                  option='corr',
                                  titles_=['Raw in plane 3', 'gPCA output'],
                                 plot_orientation='horizontal',
                                 cbar_orientation='horizontal',
                                save_fig=True,
                                save_fig_name=data_file_out+'_p3_gpca')

        dims=mov_nn3.shape
        if dx ==1:
            _ = util_plot.cn_ranks_plot(ranks_,
                                           dims,
                                            nblocks=nblocks,
                                            figsize=15,
                                            fontsize=20,
                                            save_fig=True,
                                            save_fig_name=data_file_out)
        else:
            _ = util_plot.cn_ranks_dx_plot(ranks_,
                               dims,
                                nblocks=nblocks,
                                figsize=15,
                                fontsize=20,
                                save_fig=True,
                                save_fig_name=data_file_out)
            
    W1d=W_four_1*noise_level1[:,:,np.newaxis]
    W2d=W_four_2*noise_level2[:,:,np.newaxis]
    W3d=W_four_3*noise_level3[:,:,np.newaxis]


    if store_outputs:
            store_output_movie(W1d,
                                fname_out_pca+'_p1')
            store_output_movie(W2d,
                                fname_out_pca+'_p2')
            store_output_movie(W3d,
                                fname_out_pca+'_p3')

    if make_movie:
            print('Write movie 1')
            tic()
            cname_out =fname_out_movie+ 'Plane1.mp4'
            
            util_movie.movie_writer(m_orig1,
                                    W1d,
                                    cname_out,
                                    movie_length)
            toc()
            print('Write movie 2')
            tic()
            cname_out =fname_out_movie+ 'Plane2.mp4'
            
            util_movie.movie_writer(m_orig2,
                                    W2d,
                                    cname_out,
                                    movie_length)
            toc()
            print('Write movie 3')
            tic()
            cname_out =fname_out_movie+ 'Plane3.mp4'
            
            util_movie.movie_writer(m_orig3,
                                    W3d,
                                    cname_out,
                                    movie_length)
            toc()
    return

if __name__ == "__main__":
    ## Input movie
    data_file_path ='/data/paninski_lab/Tolias_lab/cropped/scan4_block2.mat'
    data_dir_out = 'denoising_results/IARPA3D'
    data_file_name = os.path.split(data_file_path)[-1]
    data_base_name = os.path.splitext(data_file_name)[0]
    data_file_out = os.path.join(data_dir_out,
                                data_base_name)

    print('outputs will write to:')
    print('\t'+data_file_out)
    #mov = skimage.io.imread(data_file_path).transpose([1,2,0])
    #print(mov.shape)
    m_origs = []
    with h5py.File(data_file_path) as f:
        m_origs.append(f['data'].value)
    m_orig1,m_orig2,m_orig3=m_origs[0].transpose([1,3,2,0])
    del m_origs


    store_outputs = True
    extract_rank = True
    den_spatial = False
    den_temporal = True
    make_movie = True
    make_plots = True

    total_start = time.time()
    main(m_orig1,m_orig2,m_orig3,
            extract_rank=extract_rank,
            den_spatial=den_spatial,
            den_temporal=den_temporal,
            store_outputs=store_outputs,
            make_movie = make_movie,
            make_plots=make_plots)
    print('Total run time %.e sec'%(time.time()-total_start))
    #main(mov,data_file_out)