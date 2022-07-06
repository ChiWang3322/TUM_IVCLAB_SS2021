function rec_image = SSD_rec_frac(ref_image, motion_vectors)
%  Input         : ref_image(Reference Image, YCbCr image)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)
    block_size = [8, 8];
    rec_image = zeros(size(ref_image));
    [M, N, ~] = size(rec_image);
    SearchRange = 4;

    ref_image_interp = zeros(M * 2 - 1 + 4 * SearchRange, N * 2 - 1 + 4 * SearchRange);
    for i = 1:3
        temp = interp2(ref_image(:, :, i), 1);
        temp = padarray(temp, [2 * SearchRange, 2 * SearchRange], 0, 'both');
        ref_image_interp(:, :, i) = temp;
    end

    for m = 1:block_size(1):M
        for n = 1:block_size(2):N
            mv = motion_vectors(floor(m / 8) + 1, floor(n / 8) + 1);
            offset = SearchRange * 2;
            sz = [2 * offset + 1, 2 * offset + 1];
            [col, row] = ind2sub(sz, mv);
            col = col - offset - 1;
            row = row - offset - 1;
            loc = [m, n];
            ref_loc = (loc - 1) * 2  + offset + 1 + [row, col];
            rec_image(m : m + block_size(1) - 1, n : n + block_size(2) - 1, :) = ...,
                ref_image_interp(ref_loc(1) : 2 : ref_loc(1) + block_size(1) * 2 - 2, ref_loc(2) : 2 : ref_loc(2) + block_size(2) * 2 - 2, :);
        end
    end
end
