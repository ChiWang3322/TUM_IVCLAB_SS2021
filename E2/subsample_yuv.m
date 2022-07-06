function image_yuv_subsampled = subsample_yuv(I_yuv)
%Input: yuv image
%Output: yuv(Chrominance subsampled by factor of 2)
    %Wrap around
    I_y = I_yuv(:, :, 1);
    I_yuv=padarray(I_yuv, [4 4], 'replicate', 'both');
    %Resample(subsample)
    %Define parameter N
    N = 10;
    %Subsample
    for i = 2:3
        temp = resample(I_yuv(:,:,i),1, 2, N);
        I_chroma(:, :, i - 1) = resample(temp', 1, 2, N);
    end
    %Crop back
    size_chroma=size(I_chroma);
    I_chroma_subsampled=I_chroma(3:size_chroma(1)-2,3:size_chroma(2)-2,:);
    for i = 1:2
        I_chroma_subsampled(:, :, i) = I_chroma_subsampled(:, :, i);
    end
    image_yuv_subsampled = struct('Y', I_y,...,
                                          'Cb', I_chroma_subsampled(:, :, 1),...,
                                          'Cr', I_chroma_subsampled(:, :, 2));
end