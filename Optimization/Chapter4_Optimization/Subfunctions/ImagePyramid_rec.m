function [rec_image] = ImagePyramid_rec(ImagePyramid)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    level = length(ImagePyramid);
    rec_image = ImagePyramid{level};
    for j = level - 1:-1:1
        up_image = Upsample(rec_image, 2);
        rec_image = up_image + ImagePyramid{j};
    end

    
    function [image_upsampled] = Upsample(image, factor)
        [W, H, C] = size(image);
        image_upsampled = zeros(W * factor, H * factor, C);
        padding_length = 4;
        image_pad=padarray(image, [padding_length padding_length], 'replicate', 'both');
        N = 10;
        for i = 1:3
            temp1 = resample(image_pad(:, :, i), factor, 1, N);
            temp2 = resample(temp1', factor, 1, N);
            temp2 = temp2';
            temp2 = temp2(padding_length * 2 + 1 : end - padding_length * 2,...
                                    padding_length * 2 + 1 : end - padding_length * 2);
            image_upsampled(:, :, i) = temp2;
        end
    end



end

