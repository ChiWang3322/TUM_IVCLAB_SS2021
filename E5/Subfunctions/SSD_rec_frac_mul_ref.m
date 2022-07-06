function rec_image = SSD_rec_frac_mul_ref(ref_image, motion_vectors, motion_index)
%  Input         : ref_image(Reference Image, YCbCr image, struct)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)
    block_size = [8, 8];
    rec_image = zeros(size(ref_image{1}));
    [M, N, ~] = size(rec_image);
    SearchRange = 4;
    
    ref_image_interp = cell(1, length(ref_image));
    for i = 1:length(ref_image)
        reference_image = ref_image{i};
        temp_interp_image = zeros(2 * M - 1 + 4 * SearchRange, ...,
                                                 2 * N - 1 + + 4 * SearchRange, 3);
        for j = 1:3
            temp = interp2(reference_image(:, :, j), 1);
            temp = padarray(temp, [2 * SearchRange, 2 * SearchRange], 0, 'both');
            temp_interp_image(:, :, j) = temp;
        end
        ref_image_interp{i} = temp_interp_image;
    end


    for m = 1:block_size(1):M
        for n = 1:block_size(2):N
            mv = motion_vectors(floor(m / 8) + 1, floor(n / 8) + 1);
            index = motion_index(floor(m / 8) + 1, floor(n / 8) + 1);
            reference_image = ref_image_interp{index};

            offset = SearchRange * 2;
            sz = [2 * offset + 1, 2 * offset + 1];
            [col, row] = ind2sub(sz, mv);
            col = col - offset - 1;
            row = row - offset - 1;
            loc = [m, n];
            ref_loc = (loc - 1) * 2  + offset + 1 + [row, col];
            rec_image(m : m + block_size(1) - 1, n : n + block_size(2) - 1, :) = ...,
                reference_image(ref_loc(1) : 2 : ref_loc(1) + block_size(1) * 2 - 2, ref_loc(2) : 2 : ref_loc(2) + block_size(2) * 2 - 2, :);
        end
    end
end
