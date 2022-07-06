%% DO NOT MODIFY THIS CODE
%% This code will call your function and should work without any changes
clc,clear
image_rgb = double(imread('foreman0026.bmp'));
[M, N, C] = size(image_rgb);
ref_image4 = ictRGB2YCbCr(double(imread('foreman0022.bmp')));
ref_image3 = ictRGB2YCbCr(double(imread('foreman0023.bmp')));
ref_image2 = ictRGB2YCbCr(double(imread('foreman0024.bmp')));
ref_image1 = ictRGB2YCbCr(double(imread('foreman0025.bmp')));
%% Test section
tic
reference_image(:, :, 1) = ref_image1(:, :, 1);
reference_image(:, :, 2) = ref_image2(:, :, 1);
reference_image(:, :, 3) = ref_image3(:, :, 1);
reference_image(:, :, 4) = ref_image4(:, :, 1);
mv_indices = zeros(M / 8, N / 8, 4);
SAD = zeros(M / 8, N / 8, 4);
image = ictRGB2YCbCr(image_rgb);

for i = 1:4
    [mv_indices(:, :, i), SAD(:, :, i)] = SSD_frac(reference_image(:, :, i), image(:,:,1));
end

[W, H, ~] = size(SAD);
final_mv = zeros(W, H);
mv_index = zeros(W, H);
for w = 1:W
    for h = 1:H
        [~, min_SAD_index] = min(SAD(w, h, :));
        mv_index(w, h) = min_SAD_index;
        final_mv(w, h) = mv_indices(w, h, min_SAD_index);
    end
end


toc