function rec_image = SSD_rec(ref_image, motion_vectors)
%  Input         : ref_image(Reference Image, YCbCr image)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)
    rec_image = zeros(size(ref_image));
    ref_image = padarray(ref_image, [4, 4], 0, 'both');
    block_size = [8, 8];
    [M, N, ~] = size(rec_image);
    for m = 1:block_size(1):M
        for n = 1:block_size(2):N
            mv = motion_vectors(floor(m / 8) + 1, floor(n / 8) + 1);
            sz = [9 9];
            [col, row] = ind2sub(sz, mv);
            col = col - 5;
            row = row - 5;
            ref_loc = [m, n] + [row, col] + [4 4];
            rec_image(m : m + block_size(1) - 1, n : n + block_size(2) - 1, :) = ...,
                ref_image(ref_loc(1) : ref_loc(1) + block_size(1) - 1, ref_loc(2) : ref_loc(2) + block_size(2) - 1, :);
        end
    end
end
