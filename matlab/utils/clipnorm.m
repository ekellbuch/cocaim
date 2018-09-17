function gcampmat=clipnorm(gcampmat,lowerprctile,upperprctile)
if nargin<2, lowerprctile=0.1; end
if nargin<3, upperprctile=99.9; end

gcampvalues=gcampmat(~isnan(gcampmat));
min_gcamp=prctile(gcampvalues,lowerprctile);max_gcamp=prctile(gcampvalues,upperprctile);
gcampmat(gcampmat<min_gcamp)=min_gcamp;
gcampmat(gcampmat>max_gcamp)=max_gcamp;
gcampmat=(gcampmat-min_gcamp)/(max_gcamp-min_gcamp);

end