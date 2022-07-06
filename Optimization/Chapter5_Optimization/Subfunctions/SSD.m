function motion_vectors_indices = SSD(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )

    function motion_vector_indice = BlockSSD(block, loc, ref_image)
        %Searching range +-4 pixels
        search_range = 4;
        ref_loc = loc + search_range;  %location on reference image
        [M, N] = size(block);
        MIN_SSE = inf;
        best_loc = ref_loc;
        for x = -4:4
            for y = -4:4
                current_block = ref_image(ref_loc(1) + x : ref_loc(1) + x + M - 1,...,
                                                        ref_loc(2) + y : ref_loc(2) + y + N - 1);
                SSE = sum((current_block - block).^2, 'all');
                if SSE < MIN_SSE
                    MIN_SSE = SSE;
                    best_loc = [x + 5, y + 5];
                end
            end
        end
        %Change Matrix indice to linear indice
        motion_vector_indice = sub2ind([9, 9], best_loc(2), best_loc(1));
    end


    ref_image = padarray(ref_image, [4, 4], 0, 'both');
    [W, H, ~] = size(image);
    block_size = [8, 8];
    motion_vectors_indices = zeros(W / block_size(1), H / block_size(2)) + 10;
    for w = 1 : block_size(1) : W
        for h = 1 : block_size(2) : H
            loc = [w, h];
            block = image( w:w + block_size(1) - 1, h:h + block_size(2) - 1 );
            motion_vectors_indices(floor(w / 8) + 1, floor(h / 8) + 1) = BlockSSD(block, loc, ref_image);
            
        end
    end
end

