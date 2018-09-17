function [Ureds,Vtreds,means,keeprowids,dim_block,dimsM,totrank,Yd]=denoise_videos(Y, k,SVD_method,maxlag,confidence,mean_threshold_factor,snr_threshold)
%%%%%%%%%%%%%%%%%%
%     Split Y in k tiles d1 x d2 x T, and denoise each of them.
%%%%%%%%%%%%%%%%%%
% Defaults
if nargin<1, error('Please provide video to denoise'); end
if nargin<2, k=16; end
if nargin<3, maxlag=10; end
if nargin<4, confidence=0.95; end

% Dimensions of original video
dimsM = size(Y); T=dimsM(3);
% Split image into blocks
blocks = split_image_into_blocks(Y, k);
% Find dims of each block
dim_block=zeros(length(blocks),3);
for i=1:length(blocks)
    dim_block(i,:) = size(blocks{i});
end
% Compress/denoise each block
dx=dim_block(end,1); dy=dim_block(end,2); % last block with max size
dxy = dx*dy;  % max num pixels in block

%       Initialize as array for quick retrieval
Mc = nan*ones(k,dxy,T);
Ureds=cell(k,1);Vtreds=cell(k,1);
means=cell(k,1);keeprowids=cell(k,1);
totrank=0;
%       Call for method on each block
for block =1:k
    fprintf('Evaluating block %d\n',block);
    M = blocks{block};
    [Ured,Vtred,mean_vec,keeprowid,Mlowr] = compress_svd(M,SVD_method,maxlag,confidence,mean_threshold_factor,snr_threshold);
    totrank=totrank+size(Vtred,1); 
    Mc(block,:,:) = pad(Mlowr, [dxy, T], nan);
    Ureds{block}=Ured;
    Vtreds{block}=Vtred;
    means{block}=mean_vec;
    keeprowids{block}=keeprowid;
end
if nargout>7
    %     Recapture lower rank matrix
    Yd = combine_blocks(dimsM, Mc, dim_block);
end
end