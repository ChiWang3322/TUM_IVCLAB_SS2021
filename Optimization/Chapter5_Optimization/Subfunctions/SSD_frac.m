function [motion_vectors_indices, SADArray] = SSD_frac(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
    function [x, y, step, SAD] = LogarithmicSearch(center_loc, step, ref_image, block, cur_x, cur_y, offset)
        [M, N, ~] = size(block);
        current_block = ref_image(center_loc(1)  : 2 : center_loc(1)  + 2 * M - 2,...,
                                                center_loc(2)  : 2 : center_loc(2)  + 2 * N - 2);
        up_block = ref_image(center_loc(1) - step  : 2 :center_loc(1) - step  + 2 * M - 2,...,
                                         center_loc(2)  : 2 : center_loc(2)  + 2 * N - 2);
        bottom_block = ref_image(center_loc(1) + step  : 2 : center_loc(1) + step  + 2 * M - 2,...,
                                                center_loc(2)  : 2 :center_loc(2)  + 2 * N - 2);
        left_block = ref_image(center_loc(1)  : 2 :center_loc(1)  + 2 * M - 2,...,
                                          center_loc(2) - step  : 2 :center_loc(2) - step  + 2 * N - 2);
        right_block = ref_image(center_loc(1)  : 2 :center_loc(1)  + 2 * M - 2,...,
                                            center_loc(2) + step  : 2 : center_loc(2) + step  + 2 * N - 2);
        SAD_current = sum((current_block - block).^2,'all');
        SAD_up = sum((up_block - block).^2,'all');
        SAD_bottom = sum((bottom_block - block).^2,'all');
        SAD_left = sum((left_block - block).^2,'all');
        SAD_right = sum((right_block - block).^2,'all');
        if cur_x == -offset
            SAD_left = Inf;
        end
        if cur_x == offset
            SAD_right = Inf;
        end
        if cur_y == offset
            SAD_bottom = Inf;
        end
        if cur_y == -offset
            SAD_up = Inf;
        end
        SAD_array = [SAD_bottom, SAD_current, SAD_left, SAD_right, SAD_up];
        SAD = min(SAD_array);
        % Move center point
        if SAD == SAD_current
            step = step / 2;
            y = cur_y;
            x = cur_x;
        elseif SAD == SAD_bottom
            y = cur_y + step;
            x = cur_x;
        elseif SAD == SAD_up
            y = cur_y - step;
            x = cur_x;
        elseif SAD == SAD_left
            x = cur_x - step;
            y = cur_y;
        elseif SAD == SAD_right
            x = cur_x + step;     
            y = cur_y;
        end
    end

    function [motion_vector_indice, MIN_SAD] = BlockSSD(block, loc, ref_image, SearchRange)
        %Searching range +-4 pixels
        offset = SearchRange * 2;
        step = offset / 2;
        ref_loc = (loc - 1) * 2  + offset + 1;  %location on reference image
%         [M, N] = size(block);
        best_loc = ref_loc;
        x = 0;  %vector
        y = 0;
        while true
            center_loc = ref_loc + [y, x];
            
            % if center point touchs border
            if x == -offset || x == offset || y == -offset || y == offset
                step = 1;
                [x, y, ~, MIN_SAD] = LogarithmicSearch(center_loc, step, ref_image, block, x, y, offset);
                best_loc = [x + offset + 1, y + offset + 1];
                break
            end
            [x, y, step, ~] = LogarithmicSearch(center_loc, step, ref_image, block, x, y, offset);
            if step == 1
                [x, y, ~, MIN_SAD] = LogarithmicSearch(center_loc, step, ref_image, block, x, y, offset);
                best_loc = [x + offset + 1, y + offset + 1];
                break
            end
        end
        
        %Change Matrix indice to linear indice
        motion_vector_indice = sub2ind([17, 17], best_loc(1), best_loc(2));
    end



    %Bilinear interpolation
    SearchRange = 4;
    ref_image_interp = interp2(ref_image, 1);
    ref_image_interp = padarray(ref_image_interp, [2 * SearchRange, 2 * SearchRange], 0, 'both');
    [W, H, ~] = size(image);
    block_size = [8, 8];
    motion_vectors_indices = zeros(W / block_size(1), H / block_size(2));
    SADArray = zeros(W / block_size(1), H / block_size(2));
    for w = 1 : block_size(1) : W
        for h = 1 : block_size(2) : H
            loc = [w, h];
            block = image( w:w + block_size(1) - 1, h:h + block_size(2) - 1 );
            rowIndex = floor(w / 8) + 1;
            columnIndex = floor(h / 8) + 1;
            [motion_vectors_indices(rowIndex, columnIndex), SADArray(rowIndex, columnIndex)] ...,
                                    = BlockSSD(block, loc, ref_image_interp, SearchRange);
            
        end
    end
end