%% Add Path
path(path,'data/images');     
path(path,'encoder');
path(path,'decoder'); 
path(path,'analysis'); 
%% Main
bits         = 8;
epsilon      = 0.1;
block_size   = 2;
%% lena small for VQ training
image_small  = double(imread('lena_small.tif'));
% Note: (optional) Temp_clusters contain Intermediate Quantization Representatives for each iteration, e.g. for visualization of the representatives after each iteration
% The variable Temp_clusters can be empty after returning from the VectorQuantizer function.
[clusters, Temp_clusters] = VectorQuantizer(image_small, bits, epsilon, block_size);
qImage_small              = ApplyVectorQuantizer(image_small, clusters, block_size);
%% Huffman table training
pmfqLenaSmall = stats_marg(qImage_small, 1:2^bits);
[BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmfqLenaSmall);
%% 
image  = double(imread('lena.tif'));
qImage = ApplyVectorQuantizer(image, clusters, block_size);
%% Huffman encoding
bytestream = enc_huffman_new(qImage(:), BinCode, Codelengths);
%%
bpp  = (numel(bytestream) * 8) / (numel(image)/3);
%% Huffman decoding
qReconst_image = double(reshape(dec_huffman_new(bytestream, BinaryTree, max(size(qImage(:))) ), size(qImage)));
%%
reconst_image  = InvVectorQuantizer(qReconst_image, clusters, block_size);
PSNR = calcPSNR(image, reconst_image);

% %%
% image = double(imread('lena_small.tif'));
% bits         = 8;
% epsilon      = 0.1;
% bsize   = 2;

%%

function [clusters, Temp_clusters] = VectorQuantizer(image, bits, epsilon, bsize)
    %% Lloyd Max
    M = 2^(bits);
    uniform_t_q = linspace(0, 1, M+1) * 256;
    uniform_x_q = uniform_t_q(1:end-1) + (uniform_t_q(2:end)-uniform_t_q(1:end-1))/2;

    clusters = [repmat(uniform_x_q', [1, 4]), zeros(length(uniform_x_q), 5)];
    dist = [1; zeros(length(uniform_x_q)-1, 1)];
    
    % Create Blocks of size bsizexbsize -> vec those blocks
    image1D = [image(:, :, 1), image(:, :, 2), image(:, :, 3)]; 
    N1 = bsize*ones(1,size(image1D, 1)/bsize);
    N2 = bsize*ones(1,size(image1D, 2)/bsize);

    cellImage1D = mat2cell(image1D, N1, N2);
    vecFCN = @(blk) blk(:);
    matImage1DVec = cell2mat(cellfun(vecFCN, cellImage1D, 'UniformOutput', false));
    vecImg = reshape(matImage1DVec(:), [4, numel(image1D)/4])';
    %%
    Temp_clusters = {zeros(M, 4), zeros(M, 4), zeros(M, 4), zeros(M, 4), zeros(M, 4)};
    count = 1;
    while true
        old_dist = dist;
        [I, D]   = knnsearch(clusters(:, 1:4), vecImg, 'Distance', 'euclidean');
        
        for rep = 1:size(clusters, 1)
            clusters(rep, 9) = length(find(I==rep));
            if clusters(rep, 9) > 0
                clusters(rep, 5:8) = sum(vecImg(find(I==rep), :));
                clusters(rep, 1:4) = 1./clusters(rep, 9) .* clusters(rep, 5:8);
            end
            dist(rep) = sum(D(find(I==rep)).^2)/length(D);
        end
        Temp_clusters(count) = mat2cell(clusters(:, 1:4), M, 4);
        count = count + 1;
        %% Cell Spliting
        zeroClusters = find(clusters(:, 9) == 0);
        for idx = 1:length(zeroClusters)
            idxBiggestCluster = find(clusters(:, 9) == max(clusters(:, 9)));
            clusters(zeroClusters(idx), :) = clusters(idxBiggestCluster(1), :);
            clusters(zeroClusters(idx), 4) = clusters(zeroClusters(idx), 4) +1;
            clusters(zeroClusters(idx), 9) = floor(max(clusters(:, 9))/2);
            clusters(idxBiggestCluster(1), 9) = ceil(clusters(idxBiggestCluster(1), 9)/2);
        end
        
        %%
        if abs(sum(dist) - sum(old_dist))/sum(old_dist) < epsilon 
            break;
        else
            clusters(:, 5:8) = zeros(length(uniform_x_q), 4);
            clusters(:, 9) = zeros(length(uniform_x_q), 1);
        end

    end
    clusters(:, 5:end) = [];
    save('Temp_clusters', 'Temp_clusters');
end


function pmf = stats_marg(image, range)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[counts,~]=hist(image(:),range);
pmf=counts/sum(counts);
end

function qImage = ApplyVectorQuantizer(image, clusters, bsize)
%  Function Name : ApplyVectorQuantizer.m
%  Input         : image    (Original Image)
%                  clusters (Quantization Representatives)
%                  bsize    (Block Size)
%  Output        : qImage   (Quantized Image)
% Create Blocks of size bsizexbsize -> vec those blocks
    [M, N, C] = size(image);
    vecImage = zeros(M/bsize * N/bsize * C, bsize * bsize);
    i = 1;
    for c = 1:C
        for n = 1:bsize:N
            for m = 1:bsize:M
                cur_block = image(m:m + bsize - 1, n:n + bsize - 1, c);
                vecImage(i, :) = reshape(cur_block(:), 1, []);
                i = i + 1;
            end
        end
    end
    [I, ~] = knnsearch(clusters, vecImage, 'Distance', 'euclidean');
    qImage = reshape(I, [size(image, 1)/bsize, size(image, 2)/bsize, size(image, 3)]);
end

function image = InvVectorQuantizer(qImage, clusters, block_size)
%  Function Name : VectorQuantizer.m
%  Input         : qImage     (Quantized Image)
%                  clusters   (Quantization clusters)
%                  block_size (Block Size)
%  Output        : image      (Dequantized Images)
    [Mq, Nq, Cq] = size(qImage);
    image = zeros(Mq * block_size, Nq * block_size, Cq);
    for c = 1 : Cq
        for m = 1 : Mq
            for n = 1 : Nq
                index = qImage(m, n, c);
                image( (m - 1) * block_size + 1 : m * block_size,...,
                         (n - 1) * block_size + 1 : n * block_size, c) = reshape(clusters(index, :), [block_size, block_size]);
            end
        end
    end
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