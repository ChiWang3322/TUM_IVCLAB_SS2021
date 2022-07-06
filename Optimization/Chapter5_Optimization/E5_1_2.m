%% DO NOT MODIFY THIS CODE
%% This code will call your function and should work without any changes
%% Load images
clc,clear;
image_rgb = double(imread('foreman0026.bmp'));
[M, N, C] = size(image_rgb);
ref_image4 = ictRGB2YCbCr(double(imread('foreman0022.bmp')));
ref_image3 = ictRGB2YCbCr(double(imread('foreman0023.bmp')));
ref_image2 = ictRGB2YCbCr(double(imread('foreman0024.bmp')));
ref_image1 = ictRGB2YCbCr(double(imread('foreman0025.bmp')));
reference_image{1} = ref_image1;
reference_image{2} = ref_image2;
reference_image{3} = ref_image3;
reference_image{4} = ref_image4;
%% evaluate multiple reference image SSD
image = ictRGB2YCbCr(image_rgb);
tic
[mv_indices_mul, mv_index] = SSD_frac_mul_ref(reference_image, image(:, :, 1));


rec_image = SSD_rec_frac_mul_ref(reference_image, mv_indices_mul, mv_index);
rec_image_rgb = ictYCbCr2RGB(rec_image);
PSNR_mul = calcPSNR(rec_image_rgb, image_rgb);
toc
%% evaluate fractional SSD
tic 
mv_indices_optimized = SSD_frac(ref_image1(:,:,1), image(:,:,1));
rec_image = SSD_rec_frac(ref_image1, mv_indices_optimized);
rec_image_rgb = ictYCbCr2RGB(rec_image);
PSNR_frac = calcPSNR(rec_image_rgb, image_rgb);
toc
%% Evaluate normal SSD
tic
mv_indices = SSD(ref_image1(:,:,1), image(:,:,1));
rec_image = SSD_rec(ref_image1, mv_indices);
rec_image_rgb = ictYCbCr2RGB(rec_image);
PSNR_normal= calcPSNR(rec_image_rgb, image_rgb);
% fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness\n");
toc
%%

fprintf("PSNR of normal SSD: %.2f dB\n", PSNR_normal);
fprintf("PSNR of fractional SSD: %.2f dB\n", PSNR_frac);
fprintf("PSNR of multiple ref SSD: %.2f dB\n", PSNR_mul);


