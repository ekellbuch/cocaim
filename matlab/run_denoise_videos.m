%% run_denoise_videos
%%
% Wrapper for denoising widefield gcamp data, and saving reduced sized files after denoising.
% Saves movies of the comparison as well.
%
% The denoising first divides up the video into 'numblocks' blocks, and then
% performs an SVD on each block. It only keeps components that have autocorrelation
% significantly different than white noise, and an SNR higher than
% 'snr_threshold'. It saves the lower rank representation of each block.
%
% INPUTS
% matfiles given in 'matfile_root'/'matfiles' should contain a [HxWxT] array
% called 'gcamp', and if available, a mask called 'brainmaskbg' with 1's at
% relevant values and 0's for masked out regions.
%
% OUTPUTS
% (1) A .mat file is saved in 'results_root'/'results_file', with the arrays
% necessary to rebuild the dataset, using 'rebuild_denoised_data.m'
% (2) A comparison .avi movie is saved if specified, in 'movie_root'
%
% Author: Shreya Saxena (Columbia University 2018), adapted from Kelly Buchanan et al's paper https://arxiv.org/abs/1807.06203
% Please cite https://arxiv.org/abs/1807.06203 if you use this.

%% Paths, Setting values
% Data Folder and files
matfile_root = '../Data/raw'; % path to data
matfiles = {'stim1_gcamp.mat','stim2_gcamp.mat'}; 

% Results Folder and files.
results_root = '../Data/denoised'; % path to results folder
results_file = ''; % empty for same file name. No file extension. Saves a .mat file.

% all functions are in the folder called utils/
addpath('utils/');

%% Denoising Params
denoise_data = 1; % reduces the size of the results file.
numblocks = 16; % should be the square of a number, to have the same number of horizontal and vertical tiles
SVD_method = 'vanilla'; % 'vanilla' or 'randomized'
maxlag = 10; % lag for the autocorrelation computation
confidence = 0.95; % confidence value for the autocorrelation test
mean_threshold_factor=1; % multiplicative factor for the confidence test
snr_threshold = 1.05; % SNR threshold - signals below this SNR will be discarded

%% Movie Params
save_movies = 0; % only if denoising data as well.
movie_root = '../Data/denoised/movies/'; % path to movies folder
filename_movie = ''; % empty for same file name. No file extension. Saves an .avi file.
frame_indices = 1:200; %frame indices to consider, if a subset. if empty, takes all frames.
framerate = 10; % frames per second
quality = 75; % maximum is 100

%% More paths
% all functions are in the folder called utils/
addpath('utils/');

% Get folder names
if ~isdir(results_root), mkdir(results_root); end
filename_results = fullfile(results_root,results_file);

fprintf('Running %s SVD\n',SVD_method);
for i=1:length(matfiles)
    dataset = load_matlab_data(matfile_root,matfiles{i}); % struct dataset has gcamp matrix and brainmask
    gcampmat = trim_data(dataset); % applies brainmask and deletes empty rows and columns
    clear dataset;
    
    % clips and normalizes data
    gcampmat = clipnorm(gcampmat,0.01,99.99);
    
    if ~isdir(results_root), mkdir(results_root); end
    if isempty(results_file), results_file=matfiles{i}(1:end-4); end
    % Denoise data
    if denoise_data
        fprintf('Denoising file %s\n',matfiles{i});
        filename_results = strcat(results_root,results_file,'_denoised.mat');

        [Ureds,Vtreds,means,keeprowids,dim_block,dimsM,totrank,gcampmat_lowerd]=denoise_videos(gcampmat, numblocks,SVD_method,maxlag,confidence,mean_threshold_factor,snr_threshold);
        
        save(filename_results,'Ureds','Vtreds','means','keeprowids','dimsM','dim_block','-v7.3');
        fprintf('###########################  Total Rank of Video : %d  ###########################\n',totrank);
        if save_movies
            if ~isdir(movie_root), mkdir(movie_root); end
            if isempty(filename_movie), filename_movie=matfiles{i}(1:end-4); end
            movie_file=strcat(movie_root,filename_movie , '_k',num2str(numblocks),'_c',num2str(confidence*100),'_lag',num2str(maxlag),'.avi');
            if ~isempty(frame_indices)
                gcampmat=gcampmat(:,:,frame_indices);
                gcampmat_lowerd=gcampmat_lowerd(:,:,frame_indices);
            end
            % calculate residual as absolute value of difference. Clip lower-d matrix.
            gcampmat_resid=abs(gcampmat-gcampmat_lowerd);
            gcampmat_lowerd(gcampmat_lowerd>1)=1;
            gcampmat_lowerd(gcampmat_lowerd<0)=0;
            % save movie of the three matrices, side by side.
            save_redmovies(movie_file,gcampmat,gcampmat_lowerd,gcampmat_resid,framerate,quality);
        end
    else
        filename_results = strcat(results_root,results_file ,'.mat');
        save(filename_results,'gcampmat','-v7.3'); %#ok<*UNRCH>
    end
    clear gcampmat gcampmat_lowerd;
end