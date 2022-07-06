% general script, e.g. image loading, function calls, etc.
imageLena_small = double(imread('lena_small.tif'));
imageLena = double(imread('lena.tif'));
bits_small      = [1 2 3 5 7];
bits = [3 5];
PSNR_small = [];
for bit = bits_small
    qImageLena_small = UniQuant(imageLena_small, bit);
    recImage_small   = InvUniQuant(qImageLena_small, bit);
    PSNR_small = [PSNR_small calcPSNR(imageLena_small, recImage_small)];
end

PSNR = [];
for bit = bits
    qImageLena = UniQuant(imageLena, bit);
    recImage   = InvUniQuant(qImageLena, bit);
    PSNR = [PSNR calcPSNR(imageLena, recImage)];
end
save('PSNR_Lena.mat', 'PSNR', 'bits');
save('PSNR_Lena_small.mat', 'PSNR_small', 'bits_small');

%E3-1-d
figure;
slope_Lena = (PSNR(end) - PSNR(1))/(bits(end) *3  - bits(1) * 3);
slope_Lena_small = (PSNR_small(end) - PSNR_small(1))/...,
                               (bits_small(end) * 3 - bits_small(1) * 3);
plot(bits_small * 3, PSNR_small, '-xr');
hold on;
plot(bits * 3, PSNR, '-ob');
legend('Lena small: 1.89', 'Lena: 2.00', 'Location', 'best');
xlabel('Rate in bits/pixel');
ylabel('PSNR in dB');
title('R-D Curve');
% define your functions, e.g. calcPSNR, UniQuant, InvUniQuant
function qImage = UniQuant(image, bits)
    %  Input         : image (Original Image)
    %                : bits (bits available for representatives)
    %
    %  Output        : qImage (Quantized Image)
    image = image / 256;
    qImage = floor( image * (2^bits));
end

function image = InvUniQuant(qImage, bits)
    %  Input         : qImage (Quantized Image)
    %                : bits (bits available for representatives)
    %
    %  Output        : image (Mid-rise de-quantized Image)
    qImage=qImage+0.5;
    image = floor(256 / (2 ^ bits) * qImage);
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