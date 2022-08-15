% Function to calculate Bhattacharyya distance
% Eq.(5) in the following paper:
%
% J. Wang, N. Tabassum, T.T. Toma, Y. Wang, A. Gahlmann, and S.T. Acton,
% "3D GAN image aynthesis and dataset quality assessment for bacterial
% biofilm", 2022
%
% I: Synthetic/fake images, J: Real images
%
% Jie Wang, VIVA lab
% Last update: Apr. 17, 2022
% -------------------------------------------------------------------------

function BD = calculateBD(P,Q)
    BD_x = zeros(size(P));
    for x = 1:size(P,2) 
        s = [P(x),Q(x)];
        BD_x(x) = sqrt(s(1)*s(2));
    end
    BD = -log(sum(BD_x(:)));
end
        