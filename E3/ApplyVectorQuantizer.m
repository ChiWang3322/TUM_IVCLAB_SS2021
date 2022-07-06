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