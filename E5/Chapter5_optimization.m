clc,clear
%% Define the parameters of codec
scales_video = [0.07,0.2,0.4,0.8,1.0,1.5,2,3,4,4.5];
scales_still = [0.15,0.3,0.7,1.0,1.5,3,5,7,10];
MIN_THR = -1000;
MAX_THR = 4000;
EOB = 4000;
range = MIN_THR : MAX_THR;
SearchRange = 4;
NumReferenceFrame = 3;
%% Load lena_small image
lena_small = double(imread('lena_small.tif'));

%% Specify the path of the frames to be encoded
directory = './ForemanSequence';    %Current folder path
frames_dir = dir(fullfile(directory,'*.bmp'));  
num_frames = length(frames_dir);
frames = cell(1, num_frames);   % Include all frames to be encoded, rgb format
frames_rec = cell(1, num_frames);
frames_rec_yuv = cell(1, num_frames);
%% Read frames to be encoded
for j = 1 : num_frames
    frames{j}=double(imread(fullfile(directory, frames_dir(j).name)));
end
%% Pre-define psnr and bit rate variable
PSNR_each_frame = zeros(1, num_frames);
BPP_each_frame = zeros(1, num_frames);
PSNR_mean_still = zeros(1, length(scales_still));
BPP_mean_still = zeros(1, length(scales_still));
PSNR_mean_video = zeros(1, length(scales_video));
BPP_mean_video = zeros(1, length(scales_video));
%% Video codec
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Start Video Codec......\n');
for i = 1:length(scales_video)
    PSNR_each_frame = zeros(1, num_frames);
    BPP_each_frame = zeros(1, num_frames);
    qScale = scales_video(i);
    % Build huffmann table using training image
    [BinaryTree, HuffCode, BinCode, Codelengths] = TrainHuffTable(qScale, EOB, 'lena_small.tif');
    frames_rec = cell(1, num_frames);
    frames_rec_yuv = cell(1, num_frames);
    for j = 1:num_frames
        % Intra-Encode the first frame
        if j == 1
            frame1_yuv = ictRGB2YCbCr(frames{j});
            frame1_intra_encode = IntraEncode(frame1_yuv, qScale, EOB, false);
            % Huffmann encoding
            offset = -MIN_THR + 1;
            bytestream=enc_huffman_new(frame1_intra_encode + offset, BinCode, Codelengths );
            % Huffmann decoding
            frame1_intra_encode_rec = double(dec_huffman_new(bytestream, BinaryTree, size(frame1_intra_encode(:), 1))) - offset;
            % Intra-Decoding
            frame1_rec_rgb = IntraDecode(frame1_intra_encode_rec, size(frame1_yuv), qScale, EOB, true); 
            BPP_each_frame(1) = (numel(bytestream) * 8) / (numel(frame1_yuv) / 3);
            PSNR_each_frame(1) = calcPSNR(frame1_rec_rgb, frames{1});
            frames_rec{1} = frame1_rec_rgb;
            frames_rec_yuv{1} = ictRGB2YCbCr(frame1_rec_rgb);
        % Motion compensation
        else
            % Get reference frame buffer
            if j <= NumReferenceFrame
                ref_image = cell(1, j - 1);
                for k = 1: j - 1
                    ref_image{k} = frames_rec_yuv{k};
                end
            elseif j > NumReferenceFrame
                ref_image = cell(1, NumReferenceFrame);
                for k = 1:NumReferenceFrame
                     ref_image{k} = frames_rec_yuv{j - k};
                end
            end
        
            frame_yuv = ictRGB2YCbCr(frames{j});
            % Get motion vectors and error image
            [motion_vector, mv_index] = SSD_frac_mul_ref(ref_image, frame_yuv(:, :, 1));
            rec_frame_yuv = SSD_rec_frac_mul_ref(ref_image, motion_vector, mv_index);
            error_image = frame_yuv - rec_frame_yuv;
            % Intra-Encode error_image
            error_image_intra_encode = IntraEncode(error_image, qScale, EOB, false);
            % Build Huffmann code table for motion vector and error image
            if j == 2
                pmf_mv = stats_marg(motion_vector, 1:(SearchRange * 4 + 1)^2);
                [BinaryTreeMV, HuffCodeMV, BinCodeMV, CodelengthsMV]=buildHuffman(pmf_mv);
                pmf_err = stats_marg(error_image_intra_encode, -2000:4000);
                [BinaryTreeErr, HuffCodeErr, BinCodeErr, CodelengthsErr]=buildHuffman(pmf_err);
            end
            % Huffmann encoding motion vector and error image
            off_setMV = 0;
            bytestreamMV = enc_huffman_new(motion_vector + off_setMV, BinCodeMV, CodelengthsMV);

            off_setZR = 2001;
            bytestreamZR = enc_huffman_new(error_image_intra_encode + off_setZR, BinCodeErr, CodelengthsErr);
            % Huffmann decoding motion vector and error image
            dec_error_intra_encode = double(reshape(dec_huffman_new(bytestreamZR, BinaryTreeErr, max(size(error_image_intra_encode(:)))), size(error_image_intra_encode))) - off_setZR;
            dec_MV = double(reshape(dec_huffman_new(bytestreamMV, BinaryTreeMV, max(size(motion_vector(:)))), size(motion_vector))) - off_setMV;
            dec_error_image = IntraDecode(dec_error_intra_encode, size(frame_yuv), qScale, EOB, false);
            % Reconstruct frame
            rec_frame = dec_error_image + SSD_rec_frac_mul_ref(ref_image, motion_vector, mv_index);
            frames_rec_yuv{j} = rec_frame;
            rec_frame_rgb = ictYCbCr2RGB(rec_frame);
            frames_rec{j} = rec_frame_rgb;
            bppMV = (numel(bytestreamMV)*8) / (numel(frames{j})/3);
            bppZR = (numel(bytestreamZR)*8) / (numel(frames{j})/3);
            BPP_each_frame(j) = bppMV + bppZR;
            PSNR_each_frame(j) = calcPSNR(rec_frame_rgb, frames{j});
        end
        fprintf('frame: %.0f QP: %.2f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', j, qScale, BPP_each_frame(j), PSNR_each_frame(j));
    end
    BPP_mean_video(i) = mean(BPP_each_frame);
    PSNR_mean_video(i) = mean(PSNR_each_frame);
    fprintf('Average for QP: %.2f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', qScale, BPP_mean_video(i), PSNR_mean_video(i));
    fprintf('--------------------------------------------------------------------------------\n');
end
%% Save Data
DataPath = './PlotComparison/';
DataName = 'video_codec_values_SSD_mul_optim_huff.mat';
fullpath = strcat(DataPath, DataName);
save(fullpath, 'BPP_mean_video', 'PSNR_mean_video');
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
function [BinaryTree, HuffCode, BinCode, Codelengths] = TrainHuffTable(qScale, EOB, path, pmf_train)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
     if ~exist('path','var')
        path = struct;
        path(1).name = 'images\sequences\akiyo20_40_RGB';
        path(2).name = 'images\sequences\coastguard20_40_RGB';
        path(3).name = 'images\sequences\foreman20_40_RGB';
        path(4).name = 'images\sequences\news20_40_RGB';
        path(5).name = 'images\sequences\silent20_40_RGB';
     end
     if ~exist('pmf_train','var')
        pmf_train = [];
     end
     if ~exist('EOB','var')
        EOB = 4000;
     end
     MIN_VAL = -1000;
     MAX_VAL = 4000;
     range = MIN_VAL : MAX_VAL;
     train_image = double(imread(path));
     k_training  = IntraEncode(train_image, qScale, EOB, true);
     pmf_k_training = stats_marg(k_training, range);
     [BinaryTree, HuffCode, BinCode, Codelengths]=buildHuffman(pmf_k_training);
%      for i = 1:length(path)
%          directory = path(i).name;
%          images_dir = dir(fullfile(directory,'*.bmp'));
%          num_images = length(images_dir);
%          for j = 1:num_images
%              train_image=double(imread(fullfile(directory, images_dir(j).name)));
%              k_training  = IntraEncode(train_image, qScale, EOB, true);
%              pmf_k_training = stats_marg(k_training, range);
%              temp = sum((pmf_train + pmf_k_training));
%              pmf_train = (pmf_train + pmf_k_training) / temp;
%          end
%      end
%      [BinaryTree, HuffCode, BinCode, Codelengths]=buildHuffman(pmf_k_training);
end

function rec_image = SSD_rec(ref_image, motion_vectors)
%  Input         : ref_image(Reference Image, YCbCr image)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)
    block_size = [8, 8];
    rec_image = zeros(size(ref_image));
    [M, N, ~] = size(rec_image);
    SearchRange = 4;

    ref_image_interp = zeros(M * 2 - 1 + 4 * SearchRange, N * 2 - 1 + 4 * SearchRange);
    for i = 1:3
        temp = interp2(ref_image(:, :, i), 1);
        temp = padarray(temp, [2 * SearchRange, 2 * SearchRange], 0, 'both');
        ref_image_interp(:, :, i) = temp;
    end

    for m = 1:block_size(1):M
        for n = 1:block_size(2):N
            mv = motion_vectors(floor(m / 8) + 1, floor(n / 8) + 1);
            offset = SearchRange * 2;
            sz = [2 * offset + 1, 2 * offset + 1];
            [col, row] = ind2sub(sz, mv);
            col = col - offset - 1;
            row = row - offset - 1;
            loc = [m, n];
            ref_loc = (loc - 1) * 2  + offset + 1 + [row, col];
            rec_image(m : m + block_size(1) - 1, n : n + block_size(2) - 1, :) = ...,
                ref_image_interp(ref_loc(1) : 2 : ref_loc(1) + block_size(1) * 2 - 2, ref_loc(2) : 2 : ref_loc(2) + block_size(2) * 2 - 2, :);
        end
    end
end

function motion_vectors_indices = SSD(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
    function [x, y, step] = LogarithmicSearch(center_loc, step, ref_image, block, cur_x, cur_y, offset)
        [M, N, ~] = size(block);
        current_block = ref_image(center_loc(1)  : 2 : center_loc(1)  + 2 * M - 2,...,
                                                center_loc(2)  : 2 : center_loc(2)  + 2 * N - 2);
        up_block = ref_image(center_loc(1) - step  : 2 :center_loc(1) - step  + 2 * M - 2,...,
                                         center_loc(2)  : 2 : center_loc(2)  + 2 * N - 2);
        bottom_block = ref_image(center_loc(1) + step  : 2 : center_loc(1) + step  + 2 * M - 2,...,
                                                center_loc(2)  : 2 :center_loc(2)  + 2 * N - 2);
        left_block = ref_image(center_loc(1)  : 2 :center_loc(1)  + 2 * M - 2,...,
                                          center_loc(2) - step  : 2 :center_loc(2) - step  + 2 * N - 2);
        right_block = ref_image(center_loc(1)  : 2 :center_loc(1)  + 2 * M - 2,...,
                                            center_loc(2) + step  : 2 : center_loc(2) + step  + 2 * N - 2);
        SAD_current = sum((current_block - block).^2,'all');
        SAD_up = sum((up_block - block).^2,'all');
        SAD_bottom = sum((bottom_block - block).^2,'all');
        SAD_left = sum((left_block - block).^2,'all');
        SAD_right = sum((right_block - block).^2,'all');
        if cur_x == -offset
            SAD_left = Inf;
        elseif cur_x == offset
            SAD_right = Inf;
        elseif cur_y == offset
            SAD_bottom = Inf;
        elseif cur_y == -offset
            SAD_up = Inf;
        end
        SAD_array = [SAD_bottom, SAD_current, SAD_left, SAD_right, SAD_up];
        min_SAD = min(SAD_array);
        % Move center point
        if min_SAD == SAD_current
            step = step / 2;
            y = cur_y;
            x = cur_x;
        elseif min_SAD == SAD_bottom
            y = cur_y + step;
            x = cur_x;
        elseif min_SAD == SAD_up
            y = cur_y - step;
            x = cur_x;
        elseif min_SAD == SAD_left
            x = cur_x - step;
            y = cur_y;
        elseif min_SAD == SAD_right
            x = cur_x + step;     
            y = cur_y;
        end
    end
    function motion_vector_indice = BlockSSD(block, loc, ref_image, SearchRange)
        %Searching range +-4 pixels
        offset = SearchRange * 2;
        step = offset/2;
        ref_loc = (loc - 1) * 2  + offset + 1;  %location on reference image
        [M, N] = size(block);
        best_loc = ref_loc;
        x = 0;  %vector
        y = 0;
        while true
            center_loc = ref_loc + [y, x];
            
            % if center point touchs border
            if x == -offset | x == offset | y == -offset | y == offset
                step = 1;
                [x, y, step] = LogarithmicSearch(center_loc, step, ref_image, block, x, y, offset);
                best_loc = [x + offset + 1, y + offset + 1];
                break
            end
            [x, y, step] = LogarithmicSearch(center_loc, step, ref_image, block, x, y, offset);
            if step == 1
                [x, y, step] = LogarithmicSearch(center_loc, step, ref_image, block, x, y, offset);
                best_loc = [x + offset + 1, y + offset + 1];
                break
            end
        end
        
        %Change Matrix indice to linear indice
        motion_vector_indice = sub2ind([17, 17], best_loc(1), best_loc(2));
    end
    %Bilinear interpolation
    SearchRange = 4;
    ref_image_interp = interp2(ref_image, 1);
    ref_image_interp = padarray(ref_image_interp, [2 * SearchRange, 2 * SearchRange], 0, 'both');
    [W, H, ~] = size(image);
    block_size = [8, 8];
    motion_vectors_indices = zeros(W / block_size(1), H / block_size(2));
    for w = 1 : block_size(1) : W
        for h = 1 : block_size(2) : H
            loc = [w, h];
            block = image( w:w + block_size(1) - 1, h:h + block_size(2) - 1 );
            motion_vectors_indices(floor(w / 8) + 1, floor(h / 8) + 1) = BlockSSD(block, loc, ref_image_interp, SearchRange);
            
        end
    end
end

function [mv_indices, mv_index] = SSD_frac_mul_ref(ref_image, image)
%  Input         : ref_image(Reference Image, size: 1 x # ref_frame cell)
%                  image (Current Image, size: height x width, YUV format)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
    [M, N, ~] = size(image);
    num_ref_frame = length(ref_image);
    block_size = 8;
    mv_indices_mul = zeros(M / block_size, N / block_size, num_ref_frame);
    SAD = zeros(M / block_size, N / block_size, num_ref_frame);

    for i = 1:num_ref_frame
        reference_image = ref_image{i};
        [mv_indices_mul(:, :, i), SAD(:, :, i)] = SSD_frac(reference_image(:, :, 1), image(:,:,1));
    end
    
    [W, H, ~] = size(SAD);
    mv_indices = zeros(W, H);
    mv_index = zeros(W, H);
    for w = 1:W
        for h = 1:H
            [~, min_SAD_index] = min(SAD(w, h, :));
            mv_index(w, h) = min_SAD_index;
            mv_indices(w, h) = mv_indices_mul(w, h, min_SAD_index);
        end
    end
end

function rec_image = SSD_rec_frac_mul_ref(ref_image, motion_vectors, motion_index)
%  Input         : ref_image(Reference Image, YCbCr image, struct)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)
    block_size = [8, 8];
    rec_image = zeros(size(ref_image{1}));
    [M, N, ~] = size(rec_image);
    SearchRange = 4;
    
    ref_image_interp = cell(1, length(ref_image));
    for i = 1:length(ref_image)
        reference_image = ref_image{i};
        temp_interp_image = zeros(2 * M - 1 + 4 * SearchRange, ...,
                                                 2 * N - 1 + + 4 * SearchRange, 3);
        for j = 1:3
            temp = interp2(reference_image(:, :, j), 1);
            temp = padarray(temp, [2 * SearchRange, 2 * SearchRange], 0, 'both');
            temp_interp_image(:, :, j) = temp;
        end
        ref_image_interp{i} = temp_interp_image;
    end


    for m = 1:block_size(1):M
        for n = 1:block_size(2):N
            mv = motion_vectors(floor(m / 8) + 1, floor(n / 8) + 1);
            index = motion_index(floor(m / 8) + 1, floor(n / 8) + 1);
            reference_image = ref_image_interp{index};

            offset = SearchRange * 2;
            sz = [2 * offset + 1, 2 * offset + 1];
            [col, row] = ind2sub(sz, mv);
            col = col - offset - 1;
            row = row - offset - 1;
            loc = [m, n];
            ref_loc = (loc - 1) * 2  + offset + 1 + [row, col];
            rec_image(m : m + block_size(1) - 1, n : n + block_size(2) - 1, :) = ...,
                reference_image(ref_loc(1) : 2 : ref_loc(1) + block_size(1) * 2 - 2, ref_loc(2) : 2 : ref_loc(2) + block_size(2) * 2 - 2, :);
        end
    end
end

function [motion_vectors_indices, SADArray] = SSD_frac(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
    function [x, y, step, SAD] = LogarithmicSearch(center_loc, step, ref_image, block, cur_x, cur_y, offset)
        [M, N, ~] = size(block);
        current_block = ref_image(center_loc(1)  : 2 : center_loc(1)  + 2 * M - 2,...,
                                                center_loc(2)  : 2 : center_loc(2)  + 2 * N - 2);
        up_block = ref_image(center_loc(1) - step  : 2 :center_loc(1) - step  + 2 * M - 2,...,
                                         center_loc(2)  : 2 : center_loc(2)  + 2 * N - 2);
        bottom_block = ref_image(center_loc(1) + step  : 2 : center_loc(1) + step  + 2 * M - 2,...,
                                                center_loc(2)  : 2 :center_loc(2)  + 2 * N - 2);
        left_block = ref_image(center_loc(1)  : 2 :center_loc(1)  + 2 * M - 2,...,
                                          center_loc(2) - step  : 2 :center_loc(2) - step  + 2 * N - 2);
        right_block = ref_image(center_loc(1)  : 2 :center_loc(1)  + 2 * M - 2,...,
                                            center_loc(2) + step  : 2 : center_loc(2) + step  + 2 * N - 2);
        SAD_current = sum((current_block - block).^2,'all');
        SAD_up = sum((up_block - block).^2,'all');
        SAD_bottom = sum((bottom_block - block).^2,'all');
        SAD_left = sum((left_block - block).^2,'all');
        SAD_right = sum((right_block - block).^2,'all');
        if cur_x == -offset
            SAD_left = Inf;
        elseif cur_x == offset
            SAD_right = Inf;
        elseif cur_y == offset
            SAD_bottom = Inf;
        elseif cur_y == -offset
            SAD_up = Inf;
        end
        SAD_array = [SAD_bottom, SAD_current, SAD_left, SAD_right, SAD_up];
        SAD = min(SAD_array);
        % Move center point
        if SAD == SAD_current
            step = step / 2;
            y = cur_y;
            x = cur_x;
        elseif SAD == SAD_bottom
            y = cur_y + step;
            x = cur_x;
        elseif SAD == SAD_up
            y = cur_y - step;
            x = cur_x;
        elseif SAD == SAD_left
            x = cur_x - step;
            y = cur_y;
        elseif SAD == SAD_right
            x = cur_x + step;     
            y = cur_y;
        end
    end

    function [motion_vector_indice, MIN_SAD] = BlockSSD(block, loc, ref_image, SearchRange)
        %Searching range +-4 pixels
        offset = SearchRange * 2;
        step = offset/2;
        ref_loc = (loc - 1) * 2  + offset + 1;  %location on reference image
%         [M, N] = size(block);
        best_loc = ref_loc;
        x = 0;  %vector
        y = 0;
        while true
            center_loc = ref_loc + [y, x];
            
            % if center point touchs border
            if x == -offset || x == offset || y == -offset || y == offset
                step = 1;
                [x, y, ~, MIN_SAD] = LogarithmicSearch(center_loc, step, ref_image, block, x, y, offset);
                best_loc = [x + offset + 1, y + offset + 1];
                break
            end
            [x, y, step, ~] = LogarithmicSearch(center_loc, step, ref_image, block, x, y, offset);
            if step == 1
                [x, y, ~, MIN_SAD] = LogarithmicSearch(center_loc, step, ref_image, block, x, y, offset);
                best_loc = [x + offset + 1, y + offset + 1];
                break
            end
        end
        
        %Change Matrix indice to linear indice
        motion_vector_indice = sub2ind([17, 17], best_loc(1), best_loc(2));
    end



    %Bilinear interpolation
    SearchRange = 4;
    ref_image_interp = interp2(ref_image, 1);
    ref_image_interp = padarray(ref_image_interp, [2 * SearchRange, 2 * SearchRange], 0, 'both');
    [W, H, ~] = size(image);
    block_size = [8, 8];
    motion_vectors_indices = zeros(W / block_size(1), H / block_size(2));
    SADArray = zeros(W / block_size(1), H / block_size(2));
    for w = 1 : block_size(1) : W
        for h = 1 : block_size(2) : H
            loc = [w, h];
            block = image( w:w + block_size(1) - 1, h:h + block_size(2) - 1 );
            rowIndex = floor(w / 8) + 1;
            columnIndex = floor(h / 8) + 1;
            [motion_vectors_indices(rowIndex, columnIndex), SADArray(rowIndex, columnIndex)] ...,
                                    = BlockSSD(block, loc, ref_image_interp, SearchRange);
            
        end
    end
end
  
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
