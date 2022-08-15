% This code is the demo code to run SSQA
% Stochastic Synthetic dataset Quality Assessment
%
% Please refer to section 2.3 in the following paper:
% J. Wang, N. Tabassum, T.T. Toma, Y. Wang, A. Gahlmann, and S.T. Acton,
% "3D GAN image aynthesis and dataset quality assessment for bacterial
% biofilm", 2022
%
% I: Synthetic/fake images, J: Real images
%
% Jie Wang, VIVA lab
% Last update: Apr. 17, 2022
% -------------------------------------------------------------------------
%% load the datasets
% the numbers of fake images and real images can be different
[dataFilef, datapathf] = uigetfile({'*fake*';'*.*'},'Load fake data','MultiSelect', 'on');
datanumf = size(dataFilef,2);
% 
[dataFiler, datapathr] = uigetfile({'*.*';'*.*'},'Load real data','MultiSelect', 'on');
datanumr = size(dataFiler,2);

%% run SSQA
N = 10000; % num of patches to compare in each image, in the paper N = 10000
W = 4; % window/patch size, % 8 cell diameter -- reveals similar SSQA freq.
edges = linspace(0,1,101); % edges for SSQA freq. ranges
FIG = 0; % if 1: show the plots
D = 3; % Dimension of images

TESTNUM = 600; % number of stochastic comparisons 
SSQA_k = zeros(TESTNUM,1);

for k = 1:TESTNUM
    % ------------ randomly choose a fake image ---------------------------
    Iidx = randperm(datanumf,1);
    if datanumf == 1
        filenamef = fullfile(datapathf, dataFilef);
    else
        filenamef = fullfile(datapathf, dataFilef{1,Iidx});
    end
    if D == 3
        V = tiff2mat_3D(filenamef,1);
        if FIG == 1
            figure;subplot(2,2,1);imagesc(max(V,[],3));colormap gray; title('fake image');hold on;
        end
    else %2D
        V = double(rgb2gray(imread(filenamef)));
        if FIG == 1
            figure;subplot(2,2,1);imagesc(V);colormap gray; title('fake image');hold on;
        end
    end
    % ------------ randomly choose two real images ------------------------
    Jidx = randperm(datanumr,1);
    if datanumr == 1
        filenamer = fullfile(datapathr, dataFiler);
    else
        filenamer = fullfile(datapathr, dataFiler{1,Jidx});
    end
    if D == 3
        J = tiff2mat_3D(filenamer,1);
        if FIG == 1
            subplot(2,2,2);imagesc(max(J,[],3));colormap gray; title('real image');hold on;
            subplot(2,2,3);title('Similarity distribution between real and synthetic');hold on;
        end
    else %2D
        J = double(rgb2gray(imread(filenamer)));
        if FIG == 1
            subplot(2,2,2);imagesc(J);colormap gray; title('real image');hold on;
            subplot(2,2,3);title('Similarity distribution between real and synthetic');hold on;

        end
    end
    
    [~,Q] = calculateSimilarity(V,J,N,W,edges,FIG); % Q: inter-dataset similarity 
    
%     % -----------------reference real image-----------------------------
    J0idx = randperm(datanumr,1);
    if datanumr == 1
        filenamer2 = fullfile(datapathr, dataFiler);
    else
        filenamer2 = fullfile(datapathr, dataFiler{1,J0idx});
    end
    if D == 3
        J0 = tiff2mat_3D(filenamer2,1);
        if FIG == 1
            subplot(2,2,4);title('Similarity distribution between real and real');hold on;
        end
    else %2D
        J0 = double(rgb2gray(imread(filenamer2)));
        if FIG == 1
            subplot(2,2,4);title('Similarity distribution between real and real');hold on;
        end

    end
%     % --------------------------------------------------
    [~,P] = calculateSimilarity(J,J0,N,W,edges,FIG); % can also try J vs. J 
    SSQA_k(k)= calculateBD(P,Q); % calculate Bhattacharyya distance
    %BD(t)
end
%%
figure;histogram(SSQA_k);
meanBD = mean(abs(SSQA_k));
stdBD = std(abs(SSQA_k));
