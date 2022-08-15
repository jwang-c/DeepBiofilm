% Supporting function for SSQA: load 3D image to matrix
%
% Jie Wang, VIVA lab
% Last update: Apr. 17, 2022
%
% Inputs: filename: location and filename to load
%         NORMALIZE: 1 for normalize the image to 0-1
% Output: V: 3D matrix
% -------------------------------------------------------------------------
function V = tiff2mat_3D(filename,NORMALIZE)

    V = imread(filename, 1) ;
    info = imfinfo(filename);
    for ii = 2 : size(info, 1)
        temp = imread(filename, ii);
        V = cat(3 , V, temp);
    end 
    V = double(V);
    if NORMALIZE ==1 
        V = (V-min(V(:)))/(max(V(:))-min(V(:))); % normalize the original image to range 0-1
    end
