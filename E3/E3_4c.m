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

%% sub-functions
function [clusters, Temp_clusters] = VectorQuantizer(image, bits, epsilon, bsize)
    load 'Temp_clusters'
    %Get vectorized image
    [M, N, C] = size(image);
    num_blocks = M/bsize * N/bsize * C;
    vecImage = zeros(num_blocks, bsize * bsize);
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
    
    %Initialize clusters(Uniform Initialization)
    clusters = 0 : 2^bits - 1;
    clusters = reshape( (clusters + 0.5) * floor(256 / 2^bits), [], 1);
    clusters = repmat(clusters, 1, bsize*bsize);
    clusters(:, 5) = 0;
    
%     clusters = zeros(2^bits, bsize*bsize);
%     random_index = randperm(num_blocks, 2^bits);
%     clusters = vecImage(random_index, :);
%     clusters(:, 5) = 0;
    %Cluster Optimization
    J = inf;
    count = 0;
    while true
        [I, D] = knnsearch(clusters(:, 1:bsize*bsize), vecImage, 'Distance', 'euclidean');
        for i = 1:2^bits
            index = (I == i);
            clusters(i, 1:bsize*bsize) = mean(vecImage(index, :), 1);
            clusters(i, end) = sum(index);
        end
        %Cell spliting
        offset = 2;
        emptyClusters = find(clusters(:, end) == 0);
        if ~isempty(emptyClusters)     
           for index = 1:length(emptyClusters)
                index_larggest_cluster = find((clusters(:, end) == max(clusters(:, end))));
                clusters(emptyClusters(index), :) = clusters(index_larggest_cluster(1), :);
                clusters(emptyClusters(index), 1:bsize*bsize) = clusters(emptyClusters(index), 1:bsize*bsize) - offset;
                clusters(emptyClusters(index), end) = floor(max(clusters(:, end))/2);
                clusters(index_larggest_cluster(1), end) = max(clusters(:, end)) - floor(max(clusters(:, end))/2);
            end 
        end
        %Cost evaluation
        temp = J;
        J = mean(D.^2);
        if (abs(temp - J) / J) < epsilon
            break
        end

        
        count = count + 1;
    end
    clusters(:, end) = [];
    Temp_clusters = 0;
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