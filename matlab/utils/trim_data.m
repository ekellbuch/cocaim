function gcampmat=trim_data(dataset)
%%%%%%%%%%%%%%%%%%
% Any entire row or column with brainmask=0 is cut out of the gcamp images.
% gcampmat is of size Height x Width x Time
% if brainmask is not provided, take nans in the first frame as brainmask
%%%%%%%%%%%%%%%%%%
if ~isfield(dataset,'brainmask'), brainmask=~isnan(dataset.gcamp(:,:,1)); 
else, brainmask=dataset.brainmask; 
end
fprintf('Data Shape : [%d,%d,%d]\n',size(dataset.gcamp));
gcampmat=dataset.gcamp; clear dataset
delrowids=[];
delcolids=[];
for i=1:size(brainmask,1)
    if ~any(brainmask(i,:)), delrowids=[delrowids;i]; end %#ok<AGROW>
end
for j=1:size(brainmask,2)
    if ~any(brainmask(:,j)), delcolids=[delcolids;j]; end %#ok<AGROW>
end
gcampmat(delrowids,:,:)=[];
gcampmat(:,delcolids,:)=[];
fprintf('Trimmed Data Shape : [%d,%d,%d]\n',size(gcampmat));
end