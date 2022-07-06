function [DPCM_code] = DPCM(data)
%DPCM Summary of this function goes here
%Input : data N * 3
%Output: DPCM_code: Huffmann encoded

    DPCM_data = zeros(size(data));
    
    for i = 1:3
        temp = diff(data(:, i));
        DPCM_data(1, i) = data(1, i);
        DPCM_data(2 : end, i) = temp;
    end
    
end

