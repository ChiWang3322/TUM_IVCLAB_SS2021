function [resData] = DPCM(data)
%DPCM Summary of this function goes here
%   Detailed explanation goes here
    coeffs = [1/2, 1/4, 1/4];
    [M, N, C] = size(data);

    preData = zeros(M, N, C);
    preData(1:M, 1, :) = data(1:M, 1, :);
    preData(1, 1:N, :) = data(1, 1:N, :);

    resData = zeros(M, N, C);
    for c = 1:C
        for i = 2:M
            for j = 2: N
                preData(i, j, c) = preData(i, j - 1, c) * coeffs(1) + preData(i - 1, j, c) * coeffs(2)...,
                                            + preData(i - 1, j - 1, c) * coeffs(3);
            end
        end
    end
    resData = round(data - preData);
    
end

