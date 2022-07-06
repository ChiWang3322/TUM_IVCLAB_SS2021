function [clusters, Temp_clusters] = VectorQuantizer(image, bits, epsilon, bsize)

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
    
    %Initialize clusters(random Initialization)
    clusters = 0 : 2^bits - 1;
    clusters = reshape( (clusters + 0.5) * floor(256 / 2^bits), [], 1);
    clusters = repmat(clusters, 1, bsize*bsize);
%     clusters = zeros(2^bits, bsize*bsize);
%     random_index = randperm(num_blocks, 2^bits);
%     clusters = vecImage(random_index, :);
    %Cluster Optimization
    J = inf;
    count = 0;
    while true
        [I, D] = knnsearch(clusters, vecImage, 'Distance', 'euclidean');
        temp = J;
        J = mean(D.^2);
        if (abs(temp - J) / J) < epsilon
            break
        end
        count = count + 1;
        for i = 1:2^bits
            index = (I == i);
            clusters(i, :) = mean(vecImage(index, :), 1);
        end
    end
    Temp_clusters = 0;
end