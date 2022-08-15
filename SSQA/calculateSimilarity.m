% This function is to compute S_data in eq.(4) of the following paper:
%
% J. Wang, N. Tabassum, T.T. Toma, Y. Wang, A. Gahlmann, and S.T. Acton,
% "3D GAN image aynthesis and dataset quality assessment for bacterial
% biofilm", 2022
%
% inputs are two images, and parameters
% outputs: S_data, similarity scores that commpares two images
%          P: S_data in probability
% Jie Wang, VIVA lab
% Last update: Apr. 17, 2022

function [S_data,P] = calculateSimilarity(V,J,N,W,edges,FIG)
%% stochatistic evaluation: compare mean and variance
%N = 10000; % num of patches to compare in each image
%W = 4; %window/patch size
CONSTANT = 0;% C1 and C2 in the paper to aviod zero denominator. In our experiment, it's ok to keep it as zero.
%% calculate MSCN 2D
if size(V,3)== 1 
    S_data = zeros(N,1);
    for t = 1: N
        coordsxi = randperm(size(V,1)-W+1,1); % (xi, yi, xj, yj)
        coordsyi = randperm(size(V,2)-W+1,1);
        coordsxj = randperm(size(J,1)-W+1,1); % (xi, yi, xj, yj)
        coordsyj = randperm(size(J,2)-W+1,1);
        patchI = V(coordsxi(1):coordsxi(1)+W-1,coordsyi(1):coordsyi(1)+W-1);
        patchJ = J(coordsxj(1):coordsxj(1)+W-1,coordsyj(1):coordsyj(1)+W-1);
        mu_i = mean(patchI(:));
        mu_j = mean(patchJ(:));
        sigma_i = std(patchI(:));
        sigma_j = std(patchJ(:));
        l = (2*mu_i*mu_j+0)/(mu_i^2+mu_j^2+CONSTANT); % luminance
        c = (2*sigma_i*sigma_j+0.1)/(sigma_i^2+sigma_j^2+CONSTANT); % contrast
        %S = 
        S_data(t) = l*c;
        %LCscore(t) = (mu_i-mu_j)/sqrt(sigma_i^2+sigma_j^2);
        %LCscore(t) = ssimscore;
        %CQscore(t) = CQ(patchI,patchJ);
        %CTQscore(t) = CTQ(patchI,patchJ);
    end
else % calculate 3D
    S_data = zeros(N,1);
    for t = 1: N
        coordsi = randperm(size(V,1)-W+1,2); % (xi, yi, xj, yj)
        coordszi = randperm(size(V,3)-W+1,1); % (zi,zj)
        coordsj = randperm(size(J,1)-W+1,2); % (xi, yi, xj, yj)
        coordszj = randperm(size(J,3)-W+1,1); % (zi,zj)
        patchI = V(coordsi(1):coordsi(1)+W-1,coordsi(2):coordsi(2)+W-1,coordszi(1):coordszi(1)+W-1);
        patchJ = J(coordsj(1):coordsj(1)+W-1,coordsj(2):coordsj(2)+W-1,coordszj(1):coordszj(1)+W-1);
        %ssimscore = ssim(patchI,patchJ);
        mu_i = mean(patchI(:));
        mu_j = mean(patchJ(:));
        sigma_i = std(patchI(:));
        sigma_j = std(patchJ(:));
        l = (2*mu_i*mu_j+CONSTANT)/(mu_i^2+mu_j^2+CONSTANT);
        c = (2*sigma_i*sigma_j+CONSTANT)/(sigma_i^2+sigma_j^2+CONSTANT);
        S_data(t) = l*c;
    end
end

%edges = linspace(0,1,51);
if FIG == 1
    % this is to plot histogram of Similarity in current comparison
    h = histogram(S_data,'BinEdges',edges,'Normalization','probability');
    axisHandle = gca;                         %handle to the axis that contains the histogram
    histHandle = axisHandle.Children;         %handle to the histogram
    histData = histHandle.Data;               %The first input to histogram()
    binEdges = histHandle.BinEdges; 
    barHeight = histHandle.Values;
else
    barHeight = histcounts(S_data,edges,'Normalization','probability');
end
P = barHeight./sum(barHeight(:)); % probability 