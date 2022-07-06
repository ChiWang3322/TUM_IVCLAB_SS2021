% read original RGB image 
I_ori = imread('sail.tif');

% YOUR CODE HERE for chroma subsampling 
%Transform from rgb to yuv format
I_yuv=ictRGB2YCbCr(double(I_ori));
%Wrap around
I_y = I_yuv(:, :, 1);
I_yuv=padarray(I_yuv, [4 4], 'replicate', 'both');
%Resample(subsample)
%Define parameter N
N = 10;
%Subsample
for i = 2:3
    temp = resample(I_yuv(:,:,i),1, 2, N);
    I_chroma(:, :, i - 1) = resample(temp', 1, 2, N);
end
%Crop back
size_chroma=size(I_chroma);
I_chroma_subsampled=I_chroma(3:size_chroma(1)-2,3:size_chroma(2)-2,:);
% Wrap Round
I_chroma_subsampled_pad = padarray(I_chroma_subsampled, [2 2], 'replicate', 'both');
% Resample(upsample)
for i = 1:2
    temp = resample(I_chroma_subsampled_pad(:,:,i), 2, 1, N);
    I_chroma_upsampled(:, :, i) = resample(temp', 2, 1, N);
end
% Crop back
size_chroma = size(I_chroma_upsampled);
I_chroma_rec=I_chroma_upsampled(5:size_chroma(1)-4,5:size_chroma(2)-4,:);
I_yuv_rec(:, :, 1)=I_y;
I_yuv_rec(:, :, 2:3) = I_chroma_rec;
%Transform back to RGB
I_rec = ictYCbCr2RGB(I_yuv_rec);
% Evaluation
% I_rec is the reconstructed image in RGB color space
PSNR = calcPSNR(I_ori, I_rec)
fprintf('PSNR is %.2f dB\n', PSNR);

% put all the sub-functions called in your script here
function rgb = ictYCbCr2RGB(yuv)
% Input         : yuv (Original YCbCr image)
% Output        : rgb (RGB Image after transformation)
% YOUR CODE HERE
    y = yuv(:, :, 1);
    Cb = yuv(:, :, 2);
    Cr = yuv(:, :, 3);
    rgb(:, :, 1) = y + 1.402 * Cr;
    rgb(:, :, 2) = y - 0.344 * Cb - 0.714 * Cr;
    rgb(:, :, 3) = y + 1.772 * Cb;
end

function yuv = ictRGB2YCbCr(rgb)
% Input         : rgb (Original RGB Image)
% Output        : yuv (YCbCr image after transformation)
% YOUR CODE HERE
    r = rgb(:, :, 1);
    g = rgb(:, :, 2);
    b = rgb(:, :, 3);
    yuv(:, :, 1) = 0.299 * r + 0.587 * g + 0.114 * b;
    yuv(:, :, 2) = -0.169 * r - 0.331 * g + 0.5 * b;
    yuv(:, :, 3) = 0.5 * r - 0.419 * g - 0.081 * b;
end

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

