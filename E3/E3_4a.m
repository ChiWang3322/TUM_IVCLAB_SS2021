%% DO NOT MODIFY THIS CODE
path(path,'data/images');     
path(path,'encoder');
path(path,'decoder'); 
%% This code will call your function and should work without any changes
imageLena_small = double(imread('lena_small.tif'));
imageLena = double(imread('lena.tif'));
% image1D = [imageLena(:, :, 1), imageLena(:, :, 2), imageLena(:, :, 3)]; 
bits         = 8;
epsilon      = 0.1;
block_size   = 2;
[clusters, Temp_clusters] = VectorQuantizer(imageLena_small, bits, epsilon, block_size);
save('clusters', 'clusters');
%% Test your function
qImage_small = ApplyVectorQuantizer(imageLena_small, clusters, block_size);
qImage = ApplyVectorQuantizer(imageLena, clusters, block_size);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");


% function qImage = ApplyVectorQuantizer(image, clusters, bsize)
% %  Function Name : ApplyVectorQuantizer.m
% %  Input         : image    (Original Image)
% %                  clusters (Quantization Representatives)
% %                  bsize    (Block Size)
% %  Output        : qImage   (Quantized Image)
%     % Create Blocks of size bsizexbsize -> vec those blocks
%     image1D = [image(:, :, 1), image(:, :, 2), image(:, :, 3)]; 
%     N1 = bsize*ones(1,size(image1D, 1)/bsize);
%     N2 = bsize*ones(1,size(image1D, 2)/bsize);
% 
%     cellImage1D = mat2cell(image1D, N1, N2);
%     vecFCN = @(blk) blk(:);
%     matImage1DVec = cell2mat(cellfun(vecFCN, cellImage1D, 'UniformOutput', false));
%     vecImg = reshape(matImage1DVec(:), [4, numel(image1D)/4])';
% 
%     [qImage, ~] = knnsearch(clusters, vecImg, 'Distance', 'euclidean');
%     qImage = reshape(qImage, [size(image, 1)/bsize, size(image, 2)/bsize, size(image, 3)]);
% 
% end



%%