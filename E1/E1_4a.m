% image read
I_lena = double(imread('lena.tif'));
I_sail = double(imread('sail.tif'));
%Define parameter N
N = 10;
% Wrap Round
% YOUR CODE HERE
I_lena_pad=padarray(I_lena,[4 4],'replicate','both');
I_sail_pad=padarray(I_sail,[4 4],'replicate','both');
% Resample(subsample)
% YOUR CODE HERE
for i = 1:3
    temp = resample(I_lena_pad(:,:,i),1,2,N);
    I_lena_subsampled(:, :, i) = resample(temp',1,2,N);
    temp = resample(I_sail_pad(:,:,i),1,2,N);
    I_sail_subsampled(:, :, i) = resample(temp',1,2,N);
end
% Crop Back
% YOUR CODE HERE
size_lena = size(I_lena_subsampled);
size_sail = size(I_sail_subsampled);
I_lena_subsampled = I_lena_subsampled(3:size_lena(1) - 2, 3:size_lena(2) - 2, :);
I_sail_subsampled = I_sail_subsampled(3:size_sail(1) - 2, 3:size_sail(2) - 2, :);
% Wrap Round
% YOUR CODE HERE
I_lena_subsampled = padarray(I_lena_subsampled, [2 2], 'replicate','both');
I_sail_subsampled = padarray(I_sail_subsampled, [2 2], 'replicate','both');
% Resample (upsample)
% YOUR CODE HERE
for i = 1:3
    temp = resample(I_lena_subsampled(:,:,i), 2, 1, N);
    I_lena_upsampled(:, :, i) = resample(temp', 2, 1, N);
    temp = resample(I_sail_subsampled(:,:,i), 2, 1, N);
    I_sail_upsampled(:, :, i) = resample(temp', 2, 1, N);
end
% Crop back
% YOUR CODE HERE
size_lena_upsampled=size(I_lena_upsampled);
size_sail_upsampled=size(I_sail_upsampled);
I_rec_lena=I_lena_upsampled(5:size_lena_upsampled(1)-4,5:size_lena_upsampled(2)-4,:);
I_rec_sail=I_sail_upsampled(5:size_sail_upsampled(1)-4,5:size_sail_upsampled(2)-4,:);
% Distortion Analysis
PSNR_lena        = calcPSNR(I_lena, I_rec_lena);
PSNR_sail        = calcPSNR(I_sail, I_rec_sail);
fprintf('PSNR lena subsampling = %.3f dB\n', PSNR_lena)
fprintf('PSNR sail subsampling = %.3f dB\n', PSNR_sail)

% put all the sub-functions called in your script here
function MSE = calcMSE(Image, recImage)
% YOUR CODE HERE
    [m, n, c] = size(Image);
    Image = double(Image);
    recImage = double(recImage);
    MSE = 1/(m * n * c) * sum((Image - recImage).^2, 'all');
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