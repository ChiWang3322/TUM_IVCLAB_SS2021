function [err_image, modeArray] = GetIntraErrorImage(image)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    ResImage8x8 = zeros(size(image));
    [M, N, C] = size(image);
    modeArray8x8 = ones(M / 8, N / 8, 3) * 10;
    TempImage = zeros(M + 1, N + 1, 3);
    for c = 1:C
        TempImage(:, :, c) = padarray(image(:, :, c), [1 1], 'pre', 'replicate');
    end
    for c = 1:C
        for i = 1:8:M
            for j = 1:8:N
                block = TempImage(i : i + 8, j : j + 8, c);
                [ResImage8x8(i : i + 7, j : j + 7, c), modeArray8x8(floor(i / 8) + 1, floor(j / 8) + 1, c)] = IntraPrediction8x8(block);
            end
        end
    end
end

