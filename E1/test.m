%% P 1.1
I1 = ones(512, 512);
I2 = ones(512, 512);
I2(1, 1) = 0;
calcPSNR(I1, I2)
%% P 1.2
I1 = ones(256, 256, 3) * 255;
I2 = ones(256, 256, 3) * 255;
I2(1, 1, 1) = 0;
I2(1, 1, 2) = 0;
calcPSNR(I1, I2)
%% P 1.3
clc, clear all;
x = linspace(0, 2*pi, 9);
y = sin(x);
y_subsampled = y(1:2:end);
stem(y)
figure;
stem(y_subsampled);
hold on;

%% E1
%% This code will call your function and should work without any changes
Image = imread('smandril.tif');
recImage = imread('smandril_rec.tif');
MSE = calcMSE(Image, recImage);
fprintf('MSE is %.3f\n', MSE)
subplot(121), imshow(Image), title('Original Image')
subplot(122), imshow(recImage), title('Reconstructed Image')
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");
%% E2
%% This code will call your function and should work without any changes
Image = imread('smandril.tif');
recImage = imread('smandril_rec.tif');
PSNR = calcPSNR(Image, recImage);
fprintf('PSNR is %.3f dB\n', PSNR);
subplot(121), imshow(Image), title('Original Image')
subplot(122), imshow(recImage), title('Reconstructed Image')
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");
%% E 1-2 e
clc,clear,close all;
% Read Image
I = double(imread('satpic1.bmp'));
%Define Kernel
kernel = [1, 2, 1; 2, 4, 2; 1, 2, 1];
kernel = kernel ./ sum(kernel(:));  
% without prefiltering
% YOUR CODE HERE
I_downsampled_notpre = [];
for i = 1:3 
    temp = I(:, :, i);
    temp = downsample(temp, 2);
    temp = downsample(temp', 2);
    I_downsampled_notpre(:, :, i) = temp;
end
I_rec_notpre = [];
for i = 1:3 
    temp = I_downsampled_notpre(:, :, i);
    temp = upsample(temp, 2);
    temp = upsample(temp', 2);
    I_rec_notpre(:, :, i) = temp;
    I_rec_notpre(:, :, i)=4*prefilterlowpass2d(I_rec_notpre(:, :, i), kernel);
end

% Evaluation without prefiltering
% I_rec_notpre is the reconstructed image WITHOUT prefiltering
PSNR_notpre = psnr(I, I_rec_notpre);
fprintf('Reconstructed image, not prefiltered, PSNR = %.2f dB\n', PSNR_notpre);
%% E 1-3
Image_lena = imread('lena.tif');
Image_lena_compressed = imread('lena_compressed.tif');
Image_monarch = imread('monarch.tif');
Image_monarch_compressed = imread('monarch_compressed.tif');

% YOUR CODE HERE
% do NOT change the name of variables (PSNR_lena, PSNR_monarch), the assessment code will check these values with our reference answers, same for all the script assignment.
PSNR_lena = calcPSNR(Image_lena, Image_lena_compressed);
PSNR_monarch = calcPSNR(Image_monarch, Image_monarch_compressed);

fprintf('PSNR of lena.tif is %.3f dB\n', PSNR_lena)
fprintf('PSNR of monarch.tif is %.3f dB\n', PSNR_monarch)

subplot(221), imshow(Image_lena), title('Original Image Lena')
subplot(222), imshow(Image_lena_compressed), title('Compressed Image Lena')
subplot(223), imshow(Image_monarch), title('Original Image Monarch')
subplot(224), imshow(Image_monarch_compressed), title('Compressed Image Monarch')
%% E 1-4
% image read
I_lena = double(imread('lena.tif'));
I_sail = double(imread('sail.tif'));

% Wrap Round
% YOUR CODE HERE
I_lena_pad = padarray(I_lena, [4 4], 'replicate', 'both')
I_sail_pad = padarray(I_sail, [4 4], 'replicate', 'both')
% Resample(subsample)
% YOUR CODE HERE

% Crop Back
% YOUR CODE HERE

% Wrap Round
% YOUR CODE HERE

% Resample (upsample)
% YOUR CODE HERE

% Crop back
% YOUR CODE HERE

% Distortion Analysis
PSNR_lena        = calcPSNR(I_lena, I_rec_lena);
PSNR_sail        = calcPSNR(I_sail, I_rec_sail);
fprintf('PSNR lena subsampling = %.3f dB\n', PSNR_lena)
fprintf('PSNR sail subsampling = %.3f dB\n', PSNR_sail)
%% E 1-6
% read original RGB image 
I_ori = imread('sail.tif');

% YOUR CODE HERE for chroma subsampling 

% Evaluation
% I_rec is the reconstructed image in RGB color space
PSNR = calcPSNR(I_ori, I_rec)
fprintf('PSNR is %.2f dB\n', PSNR);

% put all the sub-functions called in your script here
function rgb = ictYCbCr2RGB(yuv)
end

function yuv = ictRGB2YCbCr(rgb)
end

function MSE = calcMSE(Image, recImage)
end

function PSNR = calcPSNR(Image, recImage)
end