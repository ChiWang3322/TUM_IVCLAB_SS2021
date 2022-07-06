%% read image
clc;
path(path,'data/images');     
path(path,'encoder');
path(path,'decoder'); 
imageLena = double(imread('lena.tif'));
imageLena_small = double(imread('lena_small.tif'));
% To YUV domain
imageLena_yuv       = ictRGB2YCbCr(imageLena);
imageLena_small_yuv = ictRGB2YCbCr(imageLena_small);
%Downsampling CbCr component
imageLena_yuv_subsampled = subsample_yuv(imageLena_yuv);
% residual calculation
coeffs_y = [7/8, -1/2, 5/8];
coeffs_c = [3/8, -1/4, 7/8];
resLenaY = get_res_image(imageLena_yuv_subsampled.Y, coeffs_y);
resLenaCb = get_res_image(imageLena_yuv_subsampled.Cb, coeffs_c);
resLenaCr = get_res_image(imageLena_yuv_subsampled.Cr, coeffs_c);
%% Coding
% codebook construction
[a,b,c]=size(imageLena_small);
resLenasmall=zeros(a,b,c);
resLenasmall(:, :, 1) = get_res_image(imageLena_small(:, :, 1), coeffs_y);
resLenasmall(:, :, 2) = get_res_image(imageLena_small(:, :, 2), coeffs_c);
resLenasmall(:, :, 3) = get_res_image(imageLena_small(:, :, 3), coeffs_c);
pmfresLena_small=stats_marg(resLenasmall,-255:255);
[BinaryTree,HuffCode,BinCode,Codelengths]= buildHuffman(pmfresLena_small);
% Encoding
bytestreamY = enc_huffman_new( resLenaY(:) +256, BinCode, Codelengths);
bytestreamCb = enc_huffman_new( resLenaCb(:) +256, BinCode, Codelengths);
bytestreamCr = enc_huffman_new( resLenaCr(:) +256, BinCode, Codelengths);
bytestream = [bytestreamY;bytestreamCb;bytestreamCr];
% Decoding
ReconstructedY = double(reshape( dec_huffman_new ( bytestreamY, BinaryTree, max(size(resLenaY(:))) ), size(resLenaY))) - 256;
ReconstructedCb = double(reshape( dec_huffman_new ( bytestreamCb, BinaryTree, max(size(resLenaCb(:))) ), size(resLenaCb))) - 256;
ReconstructedCr = double(reshape( dec_huffman_new ( bytestreamCr, BinaryTree, max(size(resLenaCr(:))) ), size(resLenaCr))) - 256;
%% Reconstruction
RecY = get_rec_image(ReconstructedY, coeffs_y);
RecCb = get_rec_image(ReconstructedCb, coeffs_c);
RecCr = get_rec_image(ReconstructedCr, coeffs_c);
%Upsample CbCr component
rec_image = upsample_yuv(RecY, RecCb, RecCr);
rec_image = ictYCbCr2RGB(rec_image);
%% evaluation and show results
figure
subplot(121)
imshow(uint8(imageLena)), title('Original Image')
subplot(122)

PSNR = calcPSNR(imageLena, rec_image);
imshow(uint8(rec_image)), title(sprintf('Reconstructed Image, PSNR = %.2f dB', PSNR))
BPP = numel(bytestream) * 8 / (numel(imageLena)/3);
CompressionRatio = 24/BPP;

fprintf('Bit Rate         = %.2f bit/pixel\n', BPP);
fprintf('CompressionRatio = %.2f\n', CompressionRatio);
fprintf('PSNR             = %.2f dB\n', PSNR);

% Put all sub-functions which are called in your script here.
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
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
% Output        : MSE      (Mean Squared Error)
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

function image_yuv_subsampled = subsample_yuv(I_yuv)
%Input: yuv image
%Output: yuv(Chrominance subsampled by factor of 2)
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
    for i = 1:2
        I_chroma_subsampled(:, :, i) = I_chroma_subsampled(:, :, i)';
    end
    image_yuv_subsampled = struct('Y', I_y,...,
                                          'Cb', I_chroma_subsampled(:, :, 1),...,
                                          'Cr', I_chroma_subsampled(:, :, 2));
end

function image_yuv_upsampled = upsample_yuv(Y, U, V)
%Input: yuv image
%Output: yuv(Chrominance upsampled by factor of 2)
    N = 10;
    % Wrap Round
    U_pad = padarray(U, [2 2], 'replicate', 'both');
    V_pad = padarray(V, [2 2], 'replicate', 'both');
    % Resample(upsample)
    temp = resample(U_pad, 2, 1, N);
    U_upsampled = resample(temp', 2, 1, N);
    U_upsampled = U_upsampled';
    
    temp = resample(V_pad, 2, 1, N);
    V_upsampled = resample(temp', 2, 1, N);
    V_upsampled = V_upsampled';
    % Crop back
    size_U = size(U_upsampled);
    size_V = size(V_upsampled);
    U_rec=U_upsampled(5:size_U(1)-4,5:size_U(2)-4,:);
    V_rec=V_upsampled(5:size_V(1)-4,5:size_V(2)-4,:);
    image_yuv_upsampled(:, :, 1)=Y;
    image_yuv_upsampled(:, :, 2) = U_rec;
    image_yuv_upsampled(:, :, 3) = V_rec;
end

function resImage = get_res_image(Image, coeffs)
%Input: Original Image, coefficients of predictor(third-order)
%Output: Residual Image
    [M, N, C] = size(Image);
    preImage = zeros(M, N, C);
    resImage = zeros(M, N, C);
    preImage(1, :, :) = round(Image(1, :, :));
    preImage(:, 1, :) = round(Image(:, 1, :));
    resImage(1, :, :) = round(Image(1, :, :));
    resImage(:, 1, :) = round(Image(:, 1, :));
    for m = 2:M
        for n = 2:N
            preImage(m, n, 1) = coeffs(1) * preImage(m, n - 1, 1) + ...,
                                           coeffs(2) * preImage(m - 1, n - 1, 1) + ...,
                                           coeffs(3) * preImage(m - 1, n, 1);
            resImage(m, n, 1) = round(Image(m, n, 1) - preImage(m, n, 1));
            preImage(m, n, 1) = preImage(m, n, 1) + resImage(m, n, 1);
        end
    end
end

function pmf = stats_marg(image, range)
    pmf = hist(image(:), range);
    pmf = pmf ./ sum(pmf);
end

function recImage = get_rec_image(resImage, coeffs)
    [M, N, C] = size(resImage);
    recImage = zeros(M, N, C);
    recImage(1, :, :) = round(resImage(1, :, :));
    recImage(:, 1, :) = round(resImage(:, 1, :));
    for m = 2:M
        for n = 2:N
            recImage(m, n, 1) = coeffs(1) * recImage(m, n - 1, 1) + ...,
                                           coeffs(2) * recImage(m - 1, n - 1, 1) + ...,
                                           coeffs(3) * recImage(m - 1, n, 1);
            recImage(m, n, 1) = recImage(m, n, 1) + resImage(m, n, 1);
        end
    end
end