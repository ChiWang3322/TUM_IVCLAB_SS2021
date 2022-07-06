% general script, e.g. image loading, function calls, etc.
epsilon          = 0.001;

imageLena_small  = double(imread('lena_small.tif'));
[qImageLena_small clusters_small] = LloydMax(imageLena_small, 3, epsilon);
recImage_small   = InvLloydMax(qImageLena_small, clusters_small);
PSNR_small       = calcPSNR(imageLena_small, recImage_small);

imageLena  = double(imread('lena.tif'));
[qImageLena clusters] = LloydMax(imageLena, 3, epsilon);
recImageLena   = InvLloydMax(qImageLena, clusters);
PSNR       = calcPSNR(imageLena, recImageLena);

 ...

% define your functions, e.g. calcPSNR, LloydMax, InvLloydMax
function [qImage, clusters] = LloydMax(image, bits, epsilon)
%  Input         : image (Original RGB Image)
%                  bits (bits for quantization)
%                  epsilon (Stop Condition)
%  Output        : qImage (Quantized Image)
%                  clusters (Quantization Table)
    clusters = 0 : 2^bits - 1;
    clusters = reshape( (clusters + 0.5) * floor(256 / 2^bits), [], 1);
    J = inf;
    temp = 0;
    count = 0
    while true
        [D, I] = pdist2(clusters, image(:), 'euclidean', 'Smallest', 1);
        for i = 1:2^bits
            index = (I == i);
            clusters(i) = mean( image(index) );
        end
        temp = J;
        J = mean(D.^2);
        count = count + 1;
        if (abs(temp - J) / J) < epsilon
            break
        end
    end
    clusters = round(clusters);
    qImage = reshape(I, size(image));
end

function image = InvLloydMax(qImage, clusters)
%  Input         : qImage   (Quantized Image)
%                  clusters (Quantization Table)
%  Output        : image    (Recovered Image)
    image = zeros(size(qImage));
    for i = 1:length(clusters)
        image(qImage == i) = clusters(i);
    end
end

function PSNR = calcPSNR(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
%
% Output        : PSNR     (Peak Signal to Noise Ratio)
% YOUR CODE HERE
% call calcMSE to calculate MSE
PSNR=10*log10((2^8-1).^2/calcMSE(Image, recImage));
end

function MSE = calcMSE(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
% Output        : MSE      (Mean Squared Error)
% YOUR CODE HERE
    [m, n, c] = size(Image);
    Image = double(Image);
    recImage = double(recImage);
    MSE = 1/(m * n * c) * sum((Image - recImage).^2, 'all');
end