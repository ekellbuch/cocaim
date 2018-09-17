function save_redmovies(movie_file,mat,redmat,resmat,framerate,quality)
%%%%%%%%%%%%%%%%%%
%     Takes the original, reduced and residual matrix as inputs.
%     Makes one movie for the three videos, after rescaling each matrix.
%%%%%%%%%%%%%%%%%%
if nargin<5,framerate=10; end % frames per second
if nargin<6, quality=75; end % highest is 100

min_over_time=min(redmat,[],3);
mat=mat-min_over_time;
redmat=redmat-min_over_time;

mat=reshape((mat-min(mat(:)))/(max(mat(:))-min(mat(:))),size(mat,1),size(mat,2),1,size(mat,3));
redmat=reshape((redmat-min(redmat(:)))/(max(redmat(:))-min(redmat(:))),size(redmat,1),size(redmat,2),1,size(redmat,3));
resmat=reshape((resmat-min(resmat(:)))/(max(resmat(:))-min(resmat(:))),size(resmat,1),size(resmat,2),1,size(resmat,3));
border=zeros(size(mat,1),floor(size(mat,2)/20),1,size(mat,4));

moviemat = cat(2,mat,border,redmat,border,resmat);

fprintf('Making the movie...\n');
myVideo = VideoWriter(movie_file);
myVideo.FrameRate = framerate;  
myVideo.Quality = quality;
open(myVideo);
writeVideo(myVideo, moviemat);
close(myVideo);
end