function [] = ShowPyramid(Pyramid)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    Level = length(Pyramid);
    for j = 1:Level
        figure
        imshow(uint8(Pyramid{j}))
    end
end

