function data=load_matlab_data(matfile_root,matfile)
%%%%%%%%%%%%%%%%%%
% Loads matlab (.mat) data
%%%%%%%%%%%%%%%%%%
fprintf('Loading .mat datafile: %s\n',matfile);
data_path = [matfile_root,filesep,matfile];

mat_contents = load(data_path);
data.gcamp=mat_contents.gcamp;
if isfield(mat_contents,'brainmaskbg')
    data.brainmask=mat_contents.brainmaskbg;
end
end