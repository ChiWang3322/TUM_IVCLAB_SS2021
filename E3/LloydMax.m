function [qImage, clusters] = LloydMax(image, bits, epsilon)
%  Input         : image (Original RGB Image)
%                  bits (bits for quantization)
%                  epsilon (Stop Condition)
%  Output        : qImage (Quantized Image)
%                  clusters (Quantization Table)
    %Get initial clusters
    range = 0:255;
    [counts,~]=hist(image(:),range);
    unique_value = unique(image(:));
    max_val = max(unique_value);
    min_val = min(unique_value);
    step = (max_val - min_val)/2^bits;
    clusters = zeros(2^bits, 1);
    j = 1;
    for i = min_val:step:max_val
        if(i == 255)
            break
        end
        index = (range >= i & range < i + step);
        weights = zeros(1, 256);
        weights(index) = (counts(index) / sum(counts(index)));
        clusters(j) = sum(weights.*range);
        j = j + 1;
    end
   %Optimization
    J = inf;
    temp = 0;
    qImage = zeros(size(image));
    count = 0;
    while true
        [D, I] = pdist2(clusters(:, 1), image(:), 'euclidean', 'Smallest', 1);
        temp = J;
        J = mean(D.^2);
        for i = 1:2^bits
            index = (I == i);
            clusters(i, 2) = sum(image(index));
            clusters(i, 3) = sum(index);
            clusters(i, 1) = clusters(i, 2) / clusters(i, 3);
        end
        count = count + 1;
        if (abs(temp - J) / J) < epsilon
            break
        end

    end
%     [idx, clusters_kmeans] = kmeans(image(:), 2^bits);
%     clusters_kmeans = sort(clusters_kmeans);
%     clusters_kmeans = round(clusters_kmeans);
    clusters = round(clusters);
    qImage = reshape(I, size(image));
    % end