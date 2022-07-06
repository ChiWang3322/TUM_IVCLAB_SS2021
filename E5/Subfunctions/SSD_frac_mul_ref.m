function [mv_indices, mv_index] = SSD_frac_mul_ref(ref_image, image)
%  Input         : ref_image(Reference Image, size: 1 x # ref_frame cell)
%                  image (Current Image, size: height x width, YUV format)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
    [M, N, ~] = size(image);
    num_ref_frame = length(ref_image);
    block_size = 8;
    mv_indices_mul = zeros(M / block_size, N / block_size, num_ref_frame);
    SAD = zeros(M / block_size, N / block_size, num_ref_frame);

    for i = 1:num_ref_frame
        reference_image = ref_image{i};
        [mv_indices_mul(:, :, i), SAD(:, :, i)] = SSD_frac(reference_image(:, :, 1), image(:,:,1));
    end
    
    [W, H, ~] = size(SAD);
    mv_indices = zeros(W, H);
    mv_index = zeros(W, H);
    for w = 1:W
        for h = 1:H
            [~, min_SAD_index] = min(SAD(w, h, :));
            mv_index(w, h) = min_SAD_index;
            mv_indices(w, h) = mv_indices_mul(w, h, min_SAD_index);
        end
    end


  