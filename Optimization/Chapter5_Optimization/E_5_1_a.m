clc,clear;
lena_small = double(imread('lena_small.tif'));

%% For lena.tif image
Lena       = double(imread('lena.tif'));
scales = 1; % quantization scale factor, for E(4-1), we just evaluate scale factor of 1
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    k_small  = IntraEncode(lena_small, qScale);
    k = IntraEncode(Lena, qScale);
    
    %% use pmf of k_small to build and train huffman table
    % insert your code here
    pmf_k_small = stats_marg(k_small, min(k):max(k));
    [BinaryTree, HuffCode, BinCode, Codelengths]=buildHuffman(pmf_k_small);
    %% use trained table to encode k to get the bytestream
    % insert your code here
    bytestream=enc_huffman_new(k - min(k) + 1, BinCode, Codelengths );
    bitPerPixel_lena(scaleIdx) = (numel(bytestream)*8) / (numel(Lena)/3);
    %% image reconstruction
    k_rec=double(dec_huffman_new(bytestream,BinaryTree,size(k(:),1)))+min(k)-1;
    I_rec = IntraDecode(k_rec, size(Lena),qScale);
    PSNR_lena(scaleIdx) = calcPSNR(Lena, I_rec);
    fprintf('QP Lena: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', qScale, bitPerPixel_lena(scaleIdx), PSNR_lena(scaleIdx))
end

%% For foreman0020.bmp image
foreman0020 = double(imread('foreman0020.bmp'));
scales = 1; % quantization scale factor, for E(4-1), we just evaluate scale factor of 1
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    k_small  = IntraEncode(lena_small, qScale);
    k        = IntraEncode(foreman0020, qScale);
    
    %% use pmf of k_small to build and train huffman table
    % insert your code here
    pmf_k_small = stats_marg(k_small, min(k):max(k));
    [BinaryTree, HuffCode, BinCode, Codelengths]=buildHuffman(pmf_k_small);
    %% use trained table to encode k to get the bytestream
    % insert your code here
    bytestream=enc_huffman_new(k - min(k) + 1, BinCode, Codelengths );
    bitPerPixel_foreman(scaleIdx) = (numel(bytestream)*8) / (numel(foreman0020)/3);
    %% image reconstruction
    k_rec=double(dec_huffman_new(bytestream, BinaryTree, size(k(:),1)))+min(k)-1;
    I_rec = IntraDecode(k_rec, size(foreman0020),qScale);
    PSNR_foreman(scaleIdx) = calcPSNR(foreman0020, I_rec);
    fprintf('QP Foreman: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', qScale, bitPerPixel_foreman(scaleIdx), PSNR_foreman(scaleIdx))
end

%% R-D plot
figure;
plot(bitPerPixel_lena, PSNR_lena, 'b-x');
hold on;
plot(bitPerPixel_foreman, PSNR_foreman, 'r-*');
xlabel('bpp in bits/pixel');
ylabel('PSNR in dB');
title('R-D plot for different quantization scale');
legend('Lena', 'Foreman');
%% put all used sub-functions here.
function dst = IntraDecode(image, img_size , qScale)
%  Function Name : IntraDecode.m
%  Input         : image (zero-run encoded image, 1xN)
%                  img_size (original image size)
%                  qScale(quantization scale)
%  Output        : dst   (decoded image)
    EOB = 1000;
    image_zzd = ZeroRunDec_EoB(image, EOB);
    num_rows = img_size(1) / 8 * 64;
    num_columns = img_size(2) / 8;
    image_zzd = reshape(image_zzd(:), [num_rows, num_columns * img_size(3)]);   %correct
    [M, N] = size(image_zzd);
    image_dezig = zeros(img_size);  %Correct
    for i = 1:N
        temp = blockproc(image_zzd(:, i), [64, 1], @(block_struct) DeZigZag8x8(block_struct.data));
        current_dim = mod(i, 3);
        if current_dim == 0
            current_dim = 3;
        end
        current_index = floor((i - 1)/3);
        image_dezig(:, current_index*8 + 1: (current_index + 1)*8, current_dim) = temp;
    end
   
    
    image_dequant = blockproc(image_dezig, [8, 8], @(block_struct) DeQuant8x8(block_struct.data, qScale));
    image_IDCT = blockproc(image_dequant, [8, 8], @(block_struct) IDCT8x8(block_struct.data));
    dst = ictYCbCr2RGB(image_IDCT);
end

function dst = IntraEncode(image, qScale)
%  Function Name : IntraEncode.m
%  Input         : image (Original RGB Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN)
    imageYUV = ictRGB2YCbCr(image);
    EOB = 1000;
    %DCT Transform
    imageYUV_DCT = blockproc(imageYUV, [8, 8], @(block_struct) DCT8x8(block_struct.data));
    imageYUV_quant = blockproc(imageYUV_DCT, [8, 8], @(block_struct) Quant8x8(block_struct.data, qScale));
    imageYUV_zz = blockproc(imageYUV_quant, [8, 8], @(block_struct) ZigZag8x8(block_struct.data));
    dst = ZeroRunEnc_EoB(imageYUV_zz(:), EOB);
end
%% and many more functions
function coeff = DCT8x8(block)
%  Input         : block    (Original Image block, 8x8x3)
%
%  Output        : coeff    (DCT coefficients after transformation, 8x8x3)
    coeff = zeros(size(block));
    [M, N, C] = size(block);
    % Y = AXA'
    for c = 1:C
        coeff(:,:,c)=dct2(block(:,:,c));
    end
end

function block = IDCT8x8(coeff)
%  Function Name : IDCT8x8.m
%  Input         : coeff (DCT Coefficients) 8*8*N
%  Output        : block (original image block) 8*8*N
    block = zeros(size(coeff));
    [~, ~, C] = size(coeff);
    for c = 1:C
        block(:,:,c)=idct2(coeff(:,:,c));
    end
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

function pmf = stats_marg(image, range)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[counts,~]=hist(image(:),range);
pmf=counts/sum(counts);
end

function quant = Quant8x8(dct_block, qScale)
%  Input         : dct_block (Original Coefficients, 8x8x3)
%                  qScale (Quantization Parameter, scalar)
%
%  Output        : quant (Quantized Coefficients, 8x8x3)
    L = qScale * [16, 11, 10, 16, 24, 40, 51, 61;
                        12, 12, 14, 19, 26, 58, 60, 55;
                        14, 13, 16, 24, 40, 57, 69, 56;
                        14, 17, 22, 29, 51, 87, 80, 62;
                        18, 55, 37, 56, 68, 109, 103, 77;
                        24, 35, 55, 64, 81, 104, 113, 92;
                        49, 64, 78, 87, 103, 121, 120, 101;
                        72, 92, 95, 98, 112, 100, 103, 99];
    
    C =  qScale * [17, 18, 24, 47, 99, 99, 99, 99;
                         18, 21, 26, 66, 99, 99, 99, 99;
                         24, 13, 56, 99, 99, 99, 99, 99;
                         47, 66, 99, 99, 99, 99, 99, 99;
                         99, 99, 99, 99, 99, 99, 99, 99;
                         99, 99, 99, 99, 99, 99, 99, 99;
                         99, 99, 99, 99, 99, 99, 99, 99;
                         99, 99, 99, 99, 99, 99, 99, 99;];
     quant = zeros(size(dct_block));
     quant(:, :, 1) = round(dct_block(:, :, 1) ./ L);
     quant(:, :, 2) = round(dct_block(:, :, 2) ./ C);
     quant(:, :, 3) = round(dct_block(:, :, 3) ./ C);
end

function dct_block = DeQuant8x8(quant_block, qScale)
%  Function Name : DeQuant8x8.m
%  Input         : quant_block  (Quantized Block, 8x8x3)
%                  qScale       (Quantization Parameter, scalar)
%
%  Output        : dct_block    (Dequantized DCT coefficients, 8x8x3)
    L = qScale * [16, 11, 10, 16, 24, 40, 51, 61;
                        12, 12, 14, 19, 26, 58, 60, 55;
                        14, 13, 16, 24, 40, 57, 69, 56;
                        14, 17, 22, 29, 51, 87, 80, 62;
                        18, 55, 37, 56, 68, 109, 103, 77;
                        24, 35, 55, 64, 81, 104, 113, 92;
                        49, 64, 78, 87, 103, 121, 120, 101;
                        72, 92, 95, 98, 112, 100, 103, 99];
    
    C =  qScale * [17, 18, 24, 47, 99, 99, 99, 99;
                         18, 21, 26, 66, 99, 99, 99, 99;
                         24, 13, 56, 99, 99, 99, 99, 99;
                         47, 66, 99, 99, 99, 99, 99, 99;
                         99, 99, 99, 99, 99, 99, 99, 99;
                         99, 99, 99, 99, 99, 99, 99, 99;
                         99, 99, 99, 99, 99, 99, 99, 99;
                         99, 99, 99, 99, 99, 99, 99, 99;];
                     
     dct_block = zeros(size(quant_block));
     dct_block(:, :, 1) = quant_block(:, :, 1) .* L;
     dct_block(:, :, 2) = quant_block(:, :, 2) .* C;
     dct_block(:, :, 3) = quant_block(:, :, 3) .* C;
end

function zz = ZigZag8x8(quant)
%  Input         : quant (Quantized Coefficients, 8x8xN)
%
%  Output        : zz (zig-zag scaned Coefficients, 64xN)
    ZigZag =    [1     2    6    7    15   16   28   29;
                     3     5    8    14   17   27   30   43;
                     4     9    13   18   26   31   42   44;
                     10    12   19   25   32   41   45   54;
                     11    20   24   33   40   46   53   55;
                     21    23   34   39   47   52   56   61;
                     22    35   38   48   51   57   60   62;
                     36    37   49   50   58   59   63   64];
    [M, N, C] = size(quant);
    zz = zeros(M * N, C);
    for c = 1:C
        temp = quant(:, :, c);
        zz(ZigZag(:), c) = temp(:);
    end
end

function coeffs = DeZigZag8x8(zz)
%  Function Name : DeZigZag8x8.m
%  Input         : zz    (Coefficients in zig-zag order)
%
%  Output        : coeffs(DCT coefficients in original order)
    [~, N] = size(zz);
    coeffs = zeros([8, 8, N]);
    ZigZag =    [1     2    6    7    15   16   28   29;
                 3     5    8    14   17   27   30   43;
                 4     9    13   18   26   31   42   44;
                 10    12   19   25   32   41   45   54;
                 11    20   24   33   40   46   53   55;
                 21    23   34   39   47   52   56   61;
                 22    35   38   48   51   57   60   62;
                 36    37   49   50   58   59   63   64];
    for i = 1:N
        ith_zz = zz(:, i);
        temp = ith_zz(ZigZag(:));
        temp = reshape(temp, 8, 8);
        coeffs(:, :, i) = temp;
    end
end

function zze = ZeroRunEnc_EoB(zz, EOB)
%  Input         : zz (Zig-zag scanned sequence, 1xN)
%                  EOB (End Of Block symbol, scalar)
%
%  Output        : zze (zero-run-level encoded sequence, 1xM)
    zze = zeros(size(zz));  %pre-allocate memory
    pointer_zze = 1;    %Using indexing, which is much faster
    len = length(zz);
    zeros_rep = 0;  %Number of repetitions of zeros
    for i = 1:len
        temp = zz(i);
        % When current symbol is 0
        if temp == 0
            % When this 0 is the first 0 in a string
            if zeros_rep == 0
                zze(pointer_zze:pointer_zze + 1) = [0, 0];
                zeros_rep = 1;
                pointer_zze = pointer_zze + 2;
            % When this zeros is the following 0 in a string, rep += 1
            else
                zze(pointer_zze - 1) = zeros_rep;
                zeros_rep = zeros_rep + 1;
            end
        % When the current symbol is not 0
        else 
            zeros_rep = 0;
            zze(pointer_zze) = temp;
            pointer_zze = pointer_zze + 1;
        end
        % Check if the last symbol of 8x8 block is 0
        if (mod(i, 64) == 0) && (zz(i) == 0)
            zze(pointer_zze - 2) = EOB;
            pointer_zze = pointer_zze - 1;
            zeros_rep = 0;
        end
    end
    
    zze(pointer_zze:end) = [];
    % Process end of the sequence
    if zze(end) == 0 | zze(end - 1) == 0
        zze(end - 1:end) = [];
        zze(end + 1) = EOB;
    end
end

function dst = ZeroRunDec_EoB(src, EoB)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB signs)
%                  EoB (end of block sign)
%
%  Output        : dst (reconstructed zig-zag scanned sequence 1xN)
    [M, N] = size(src);
    dst = zeros(1, 100 * N);
    last_el_is_zero = 0;
    pointer = 1;
    for i = 1:length(src)
        temp = src(i);
        if src(i) == EoB
            if mod(pointer, 64) == 0    %at the last position of one block
                dst(pointer) = 0;
                pointer = pointer + 1;
            else
                num_zeros = 64 - mod(pointer, 64) + 1;
                dst(pointer : pointer + num_zeros - 1) = zeros(1, num_zeros);
                pointer = pointer + num_zeros;
            end
        %Last symbol is 0 and the current symbol is not 0
        elseif (src(i) == 0) && (~last_el_is_zero)
            last_el_is_zero = 1;
            dst(pointer) = 0;
            pointer = pointer + 1;
        elseif last_el_is_zero
            dst(pointer:pointer + src(i) - 1) = zeros(1, src(i));
            last_el_is_zero = 0;
            pointer = pointer + src(i);
        else
            dst(pointer) = src(i);
            pointer = pointer + 1;
        end
        
    end
    %Process the end of dst
    dst(pointer : end) = [];
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