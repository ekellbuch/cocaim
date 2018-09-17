function fulldata=rebuild_denoised_data(compressed_file_path)
%%%%%%%%%%%%%%%%%%
% Rebuilds full-sized matrix from the contents of the reduced-size files as produced in run_denoise_videos.m
%%%%%%%%%%%%%%%%%%
load(compressed_file_path,'Ureds','Vtreds','means','keeprowids','dimsM','dim_block')
addpath('utils/');
k=length(Ureds);

dx=dim_block(end,1); dy=dim_block(end,2); T=dim_block(end,3);  % last block with max size
dxy = dx*dy;  % max num pixels in block
Mlowrfull=nan*ones(k,dxy,T);

for tile=1:k
    d1=dim_block(tile,1); d2=dim_block(tile,2);
    Mlowr=nan*ones(d1*d2,T);
    Mlowr(keeprowids{tile},:)=Ureds{tile}*(Vtreds{tile});
    Mlowr = Mlowr + means{tile};
    Mlowrfull(tile,:,:) = pad(Mlowr, [dxy, T]);
end
fulldata=combine_blocks(dimsM, Mlowrfull, dim_block);
end
