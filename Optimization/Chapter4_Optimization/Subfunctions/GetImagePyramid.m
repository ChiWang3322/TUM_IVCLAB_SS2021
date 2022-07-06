function [ImagePyramid] = GetImagePyramid(ori_Image, Level)
% Input : ori_Image (original image, yuv420 format)
%            Level (Depth of the pyramid)
% Output : ImagePyramid (1 x Level cell)
    ImagePyramid = cell(1, Level);
    CurrentImage = ori_Image;
    for j = 1:Level - 1
        SubImage = Subsample(CurrentImage, 2);
        UpImage = Upsample(SubImage, 2);
        ImagePyramid{j} = CurrentImage - UpImage;
        CurrentImage = Subsample(CurrentImage, 2);
    end
    ImagePyramid{Level} = CurrentImage;


    function [image_subsampled] = Subsample(image, factor)
        N = 10;
        [W, H, C] = size(image);
        padding_length = 4;
        image_subsampled = zeros(W / factor, H / factor, C);
        image_pad=padarray(image, [padding_length padding_length], 'replicate', 'both');
        % subsample
        for i = 1:C
            temp1 = resample(image_pad(:,:,i),1, factor, N);
            temp2 = resample(temp1', 1, 2, N);
            temp2 = temp2';
            temp2 = temp2(padding_length / factor + 1:end - padding_length / factor, ...
                                    padding_length / factor + 1:end - padding_length / factor);
            image_subsampled(:, :, i) = temp2;
        end
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

