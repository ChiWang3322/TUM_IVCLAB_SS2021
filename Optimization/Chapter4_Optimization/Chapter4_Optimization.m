clc,clear
%% Define the parameters of codec
scales_video = [0.07,0.2,0.4,0.8,1.0,1.5,2,3,4,4.5];
% scales_still = [0.15,0.3,0.7,1.0,1.5,3,5];
scales_still = [0.15, 0.3, 0.7, 1.0, 3, 4, 5, 10];
EOB = 4000;
MIN_THR = -2000;
MAX_THR = 4000;
range = MIN_THR : MAX_THR;
PyramidLevel = 3;
%% Load lena_small image
lena_small = double(imread('lena_small.tif'));

%% Specify the path of frames
directory = './ForemanSequence';    %Current folder path
frames_dir = dir(fullfile(directory,'*.bmp'));  
num_frames = length(frames_dir);
frames = cell(1, num_frames);
frames_rec = cell(1, num_frames);
%% Read frames to be encoded
for i = 1:num_frames
    frames{i} = double(imread(fullfile(directory, frames_dir(i).name)));
end
[H, W, C] = size(frames{1});
%% Pre-define psnr and bit rate variable
PSNR_each_frame = zeros(1, num_frames);
BPP_each_frame = zeros(1, num_frames);
PSNR_mean_still = zeros(1, length(scales_still));
BPP_mean_still = zeros(1, length(scales_still));
PSNR_mean_video = zeros(1, length(scales_video));
BPP_mean_video = zeros(1, length(scales_video));
% %% Still image codec(Image pyramid)
% fprintf('Start Still Image Codec......\n');
% for i = 1:length(scales_still)
%     qScale = scales_still(i);
%     % Train Huffmann code using lena_small
%     pmf_k_training = zeros(1, MAX_THR - MIN_THR + 1);
%     for j = 1:num_frames
%         image_rgb = frames{j};
%         image_yuv = ictRGB2YCbCr(image_rgb);
%         ImagePyramid = GetImagePyramid(image_yuv, PyramidLevel);
%         % Update Huffmann table
%         k = IntraEncode(ImagePyramid{PyramidLevel}, qScale, EOB, false);
%         pmf_k = stats_marg(k, range);
%         pmf_k_training = (pmf_k + pmf_k_training) / sum((pmf_k + pmf_k_training));
%         [BinaryTree, HuffCode, BinCode, Codelengths]=buildHuffman(pmf_k_training);
%         % Train Huffmann table for error image
%         k_err = IntraEncode(ImagePyramid{1}, qScale, EOB, false);
%         if j == 1
%             pmf_k_err = stats_marg(k_err, range);
%             [BinaryTreeErr, HuffCodeErr, BinCodeErr, CodelengthsErr]=buildHuffman(pmf_k_err);
%         end
%         offset = -MIN_THR + 1;
%         Bytestream = cell(1, PyramidLevel);
%         % Encode last level
%         Bytestream{PyramidLevel} = enc_huffman_new(k + offset, BinCode, Codelengths );
%         % Encode error image
%         k_err = cell(1, PyramidLevel);
%         for p = 1:PyramidLevel - 1
%             k_err{p} = IntraEncode(ImagePyramid{p}, qScale, EOB, false);
%             Bytestream{p} = enc_huffman_new(k_err{p} + offset, BinCodeErr, CodelengthsErr );
%         end
%         % Compute BPP
%         BPP_all = zeros(1, PyramidLevel);
%         for p = 1:PyramidLevel
%             BPP_all(p) = (numel(Bytestream{p}) * 8);
%         end
%         BPP = (sum(BPP_all) - BPP_all(2)) / (numel(image_rgb) / 3);
%         % Reconstruct reference image and calculate PSNR
%         Pyramid_rec_yuv = cell(1, PyramidLevel);
%         % Reconstruct error image
%         for p = 1:PyramidLevel - 1
%             k_rec_err = double(dec_huffman_new(Bytestream{p}, BinaryTreeErr, size(k_err{p}(:), 1))) - offset;
%             pyramid_size = [H / 2^(p - 1), W / 2^(p - 1), C];
%             Pyramid_rec_yuv{p} = IntraDecode(k_rec_err, pyramid_size, qScale, EOB, false);
%         end
%         % Reconstruct last level
%         k_rec = double(dec_huffman_new(Bytestream{PyramidLevel}, BinaryTree, size(k(:), 1))) - offset;
%         last_level_size = [H / 2^(PyramidLevel - 1), W / 2^(PyramidLevel - 1), C];
%         Pyramid_rec_yuv{PyramidLevel} = IntraDecode(k_rec, last_level_size, qScale, EOB, false);
%         rec_image_yuv = ImagePyramid_rec(Pyramid_rec_yuv);
%         rec_image_rgb = ictYCbCr2RGB(rec_image_yuv);
%         PSNR = calcPSNR(image_rgb, rec_image_rgb);
%         BPP_each_frame(j) = BPP;
%         PSNR_each_frame(j) = PSNR;
%         fprintf('frame: %.0f QP: %f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', j, qScale, BPP, PSNR);
%     end
%     PSNR_mean_still(i) = mean(PSNR_each_frame);
%     BPP_mean_still(i) = mean(BPP_each_frame);
%     fprintf('All frames average: QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', qScale, BPP_mean_still(i), PSNR_mean_still(i));
%     fprintf('--------------------------------------------------------------------------------\n');
% end
%% Still image codec (Adaptive huffmann table)
fprintf('Start Still Image Codec......\n');
for i = 1:length(scales_still)
    qScale = scales_still(i);
    % Train Huffmann code using lena_small
    pmf_k_training = zeros(1, MAX_THR - MIN_THR + 1);
    for j = 1:num_frames
        image_rgb = double(imread(fullfile(directory, frames_dir(j).name)));
        k = IntraEncode(image_rgb, qScale, EOB, true);
        pmf_k = stats_marg(k, range);
        pmf_k_training = (pmf_k + pmf_k_training) / sum((pmf_k + pmf_k_training));
        [BinaryTree, HuffCode, BinCode, Codelengths]=buildHuffman(pmf_k_training);
        % Encode byte stream of reference frame
        offset = -MIN_THR + 1;
        bytestream = enc_huffman_new(k + offset, BinCode, Codelengths );
        BPP = (numel(bytestream) * 8) / (numel(image_rgb) / 3);
        % Reconstruct reference image and calculate PSNR
        k_rec = double(dec_huffman_new(bytestream,BinaryTree,size(k(:),1))) - offset;
        rec_image_rgb = IntraDecode(k_rec, size(image_rgb), qScale, EOB, true); 
        PSNR = calcPSNR(image_rgb, rec_image_rgb);
        BPP_each_frame(j) = BPP;
        PSNR_each_frame(j) = PSNR;
        fprintf('frame: %.0f QP: %f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', j, qScale, BPP, PSNR);
    end
    PSNR_mean_still(i) = mean(PSNR_each_frame);
    BPP_mean_still(i) = mean(BPP_each_frame);
    fprintf('All frames average: QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', qScale, BPP_mean_still(i), PSNR_mean_still(i));
    fprintf('--------------------------------------------------------------------------------\n');
end
%% Save Data
DataPath = '../PlotComparison/';
DataName = 'still_codec_values_optimized.mat';
fullpath = strcat(DataPath, DataName);
save(fullpath, 'BPP_mean_still', 'PSNR_mean_still');
%% visualization
figure;

plot(BPP_mean_still, PSNR_mean_still, '--o', 'LineWidth' , 2, 'MarkerSize', 8);
for i = 1 : length(scales_still)
    text(BPP_mean_still(i) + 0.1, PSNR_mean_still(i), num2str(scales_still(i), '%.2f'));
end

grid on
hold on

plot(BPP_mean_video, PSNR_mean_video, '--x', 'LineWidth' , 2, 'MarkerSize', 8);
for i = 1 : length(scales_video)
    text(BPP_mean_video(i) + 0.1, PSNR_mean_video(i), num2str(scales_video(i), '%.2f'));
end

legend('Still-Image Codec with Corresponding qScale', 'Video Codec with Corresponding qScale', 'Location', 'best');
title('Rate-Distortion Plot');
xlabel('Bit Rate [bits/pixel]');
ylabel('PSNR [dB]');
%% put all used sub-functions here.
function dst = IntraDecode(image, img_size , qScale, EOB, ict)
%  Function Name : IntraDecode.m
%  Input         : image (zero-run encoded image, 1xN)
%                  img_size (original image size)
%                  qScale(quantization scale)
%  Output        : dst   (decoded image)
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
    if ict == true
        dst = ictYCbCr2RGB(image_IDCT);
    else
        dst = image_IDCT;
    end
end

function dst = IntraEncode(image, qScale, EOB, ict)
%  Function Name : IntraEncode.m
%  Input         : image (Original RGB Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN)
    if ict == true
        imageYUV = ictRGB2YCbCr(image);
    else
        imageYUV = image;
    end
    
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

%%
function [ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( p )
global y
p=p(:)/sum(p)+eps;              % normalize histogram
p1=p;                           % working copy
c=cell(length(p1),1);			% generate cell structure 
for i=1:length(p1)				% initialize structure
   c{i}=i;						
end
while size(c)-2					% build Huffman tree
	[p1,i]=sort(p1);			% Sort probabilities
	c=c(i);						% Reorder tree.
	c{2}={c{1},c{2}};           % merge branch 1 to 2
    c(1)=[];	                % omit 1
	p1(2)=p1(1)+p1(2);          % merge Probabilities 1 and 2 
    p1(1)=[];	                % remove 1
end
getcodes(c,[]);                  % recurse to find codes
code=char(y);
[numCodes maxlength] = size(code); % get maximum codeword length
length_b=0;
HuffCode=zeros(1,numCodes);
for symbol=1:numCodes
    for bit=1:maxlength
        length_b=bit;
        if(code(symbol,bit)==char(49)) HuffCode(symbol) = HuffCode(symbol)+2^(bit-1)*(double(code(symbol,bit))-48);
        elseif(code(symbol,bit)==char(48))
        else 
            length_b=bit-1;
            break;
        end;
    end;
    Codelengths(symbol)=length_b;
end;
BinaryTree = c;
BinCode = code;
clear global y;
return
end
%----------------------------------------------------------------
function getcodes(a,dum)       
global y                            % in every level: use the same y
if isa(a,'cell')                    % if there are more branches...go on
         getcodes(a{1},[dum 0]);    % 
         getcodes(a{2},[dum 1]);
else   
   y{a}=char(48+dum);   
end
end
%% 
%--------------------------------------------------------------
%
%
%
%           %%%    %%%       %%%      %%%%%%%%
%           %%%    %%%      %%%     %%%%%%%%%
%           %%%    %%%     %%%    %%%%
%           %%%    %%%    %%%    %%%
%           %%%    %%%   %%%    %%%
%           %%%    %%%  %%%    %%%
%           %%%    %%% %%%    %%%
%           %%%    %%%%%%    %%%
%           %%%    %%%%%     %%%
%           %%%    %%%%       %%%%%%%%%%%%
%           %%%    %%%          %%%%%%%%%   BUILDHUFFMAN.M
%
%
% description:  creatre a huffman table from a given distribution
%
% input:        data              - Data to be encoded (indices to codewords!!!!
%               BinCode           - Binary version of the Code created by buildHuffman
%               Codelengths       - Array of Codelengthes created by buildHuffman
%
% returnvalue:  bytestream        - the encoded bytestream
%
% Course:       Image and Video Compression
%               Prof. Eckehard Steinbach
%
%-----------------------------------------------------------------------------------

function [bytestream] = enc_huffman_new( data, BinCode, Codelengths)

a = BinCode(data(:),:)';
b = a(:);
mat = zeros(ceil(length(b)/8)*8,1);
p  = 1;
for i = 1:length(b)
    if b(i)~=' '
        mat(p,1) = b(i)-48;
        p = p+1;
    end
end
p = p-1;
mat = mat(1:ceil(p/8)*8);
d = reshape(mat,8,ceil(p/8))';
multi = [1 2 4 8 16 32 64 128];
bytestream = sum(d.*repmat(multi,size(d,1),1),2);

end



%% 
%--------------------------------------------------------------
%
%
%
%           %%%    %%%       %%%      %%%%%%%%
%           %%%    %%%      %%%     %%%%%%%%%            
%           %%%    %%%     %%%    %%%%
%           %%%    %%%    %%%    %%%
%           %%%    %%%   %%%    %%%
%           %%%    %%%  %%%    %%%
%           %%%    %%% %%%    %%%
%           %%%    %%%%%%    %%%
%           %%%    %%%%%     %%% 
%           %%%    %%%%       %%%%%%%%%%%%
%           %%%    %%%          %%%%%%%%%   BUILDHUFFMAN.M
%
%
% description:  creatre a huffman table from a given distribution
%
% input:        bytestream        - Encoded bitstream
%               BinaryTree        - Binary Tree of the Code created by buildHuffma
%               nr_symbols        - Number of symbols to decode
%
% returnvalue:  output            - decoded data
%
% Course:       Image and Video Compression
%               Prof. Eckehard Steinbach
%
%
%-----------------------------------------------------------------------------------

function [output] = dec_huffman_new (bytestream, BinaryTree, nr_symbols)

output = zeros(1,nr_symbols);
ctemp = BinaryTree;

dec = zeros(size(bytestream,1),8);
for i = 8:-1:1
    dec(:,i) = rem(bytestream,2);
    bytestream = floor(bytestream/2);
end

dec = dec(:,end:-1:1)';
a = dec(:);

i = 1;
p = 1;
while(i <= nr_symbols)&&p<=max(size(a))
    while(isa(ctemp,'cell'))
        next = a(p)+1;
        p = p+1;
        ctemp = ctemp{next};
    end;
    output(i) = ctemp;
    ctemp = BinaryTree;
    i=i+1;
end;
end






% ctemp = BinaryTree;
% i = 1;
% p = 1;
% while(i <= nr_symbols)
%     while(isa(ctemp,'cell'))
%         next = a(p)+1;
%         next
% p = p+1;
%         ctemp = ctemp{next};
%     end;
%     output2(i) = ctemp;
%     ctemp = BinaryTree;
%     i=i+1;
% end



%return
