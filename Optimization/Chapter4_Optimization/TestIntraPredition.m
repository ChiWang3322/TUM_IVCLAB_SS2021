%% DO NOT MODIFY THIS CODE
%% This code will call your function and should work without any changes
clc,clear
%% Read image

image_rgb = double(imread('foreman0030.bmp'));
image = ictRGB2YCbCr(image_rgb);
%% Initialization
ResImage8x8 = zeros(size(image));
[M, N, C] = size(image);
modeArray8x8 = ones(M / 8, N / 8, 3) * 10;
TempImage = zeros(M + 1, N + 1, 3);
%% Prediction
for c = 1:C
    TempImage(:, :, c) = padarray(image(:, :, c), [1 1], 'pre', 'replicate');
end
BorderArray = TempImage;



[ResImage8x8, modeArray8x8] = IntraPrediction8x8(image);


pmf_ori = stats_marg(TempImage, -1000:1000);
H_ori = calc_entropy(pmf_ori);

pmf_res8x8 = stats_marg(ResImage8x8, -1000 : 1000);
H_res8x8 = calc_entropy(pmf_res8x8);

fprintf('Entropy of original image: %.3f\n', H_ori);
fprintf('Entropy of Resisual image(4x4 block): %.3f\n', H_res8x8);

%% Reconstruction
rec_image = IntraPredictionDecode8x8(ResImage8x8, BorderArray, modeArray8x8);
rec_image_rgb = ictYCbCr2RGB(rec_image);
figure
imshow(uint8(image_rgb));
title('Original image');
figure
imshow(uint8(rec_image_rgb));
title('Reconstructed image');
PSNR_rec = calcPSNR(image_rgb, rec_image_rgb);
fprintf('PSNR of Reconstructed image(8x8 block): %.3f dB\n', PSNR_rec);
fprintf('MSE of Reconstructed image(8x8 block): %.3f \n', calcMSE(image_rgb, rec_image_rgb));
%% Sub-functions
function pmf = stats_marg(image, range)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[counts,~]=hist(image(:),range);
pmf=counts/sum(counts);
end

function H = calc_entropy(pmf)
    pmf = pmf(:);
    H = -sum(nonzeros(pmf).*log2(nonzeros(pmf)));
end