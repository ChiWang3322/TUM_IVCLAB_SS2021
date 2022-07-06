function motion_vectors_indices = SSD_8x8_Optimized(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
    function [x, y, step] = LogarithmicSearch(center_loc, step, ref_image, block, cur_x, cur_y)
        [M, N, ~] = size(block);
        current_block = ref_image(center_loc(1)  : center_loc(1)  + M - 1,...,
                                                center_loc(2)  : center_loc(2)  + N - 1);
        up_block = ref_image(center_loc(1) - step  : center_loc(1) - step  + M - 1,...,
                                         center_loc(2)  : center_loc(2)  + N - 1);
        bottom_block = ref_image(center_loc(1) + step  : center_loc(1) + step  + M - 1,...,
                                                center_loc(2)  : center_loc(2)  + N - 1);
        left_block = ref_image(center_loc(1)  : center_loc(1)  + M - 1,...,
                                          center_loc(2) - step  : center_loc(2) - step  + N - 1);
        right_block = ref_image(center_loc(1)  : center_loc(1)  + M - 1,...,
                                            center_loc(2) + step  : center_loc(2) + step  + N - 1);
        SAD_current = sum((current_block - block).^2,'all');
        SAD_up = sum((up_block - block).^2,'all');
        SAD_bottom = sum((bottom_block - block).^2,'all');
        SAD_left = sum((left_block - block).^2,'all');
        SAD_right = sum((right_block - block).^2,'all');
        if cur_x == -4
            SAD_left = Inf;
        elseif cur_x == 4
            SAD_right = Inf;
        elseif cur_y == 4
            SAD_bottom = Inf;
        elseif cur_y == -4
            SAD_up = Inf;
        end
        SAD_array = [SAD_bottom, SAD_current, SAD_left, SAD_right, SAD_up];
        min_SAD = min(SAD_array);
        % Move center point
        if min_SAD == SAD_current
            step = 1;
            y = cur_y;
            x = cur_x;
        elseif min_SAD == SAD_bottom
            y = cur_y + step;
            x = cur_x;
        elseif min_SAD == SAD_up
            y = cur_y - step;
            x = cur_x;
        elseif min_SAD == SAD_left
            x = cur_x - step;
            y = cur_y;
        elseif min_SAD == SAD_right
            x = cur_x + step;     
            y = cur_y;
        end
    end
    function motion_vector_indice = BlockSSD(block, loc, ref_image)
        %Searching range +-4 pixels
        search_range = 4;
        step = 2;
        ref_loc = loc + search_range;  %location on reference image
        [M, N] = size(block);
        MIN_SSE = inf;
        best_loc = ref_loc;
        center_loc = ref_loc;
        x = 0;  %vector
        y = 0;
        while true
            center_loc = ref_loc + [y, x];
            
            % if center point touchs border
            if x == -4 | x == 4 | y == -4 | y == 4
                step = 1;
                [x, y, step] = LogarithmicSearch(center_loc, step, ref_image, block, x, y);
                best_loc = [x + 5, y + 5];
                break
            end
            [x, y, step] = LogarithmicSearch(center_loc, step, ref_image, block, x, y);
            if step == 1
                [x, y, step] = LogarithmicSearch(center_loc, step, ref_image, block, x, y);
                best_loc = [x + 5, y + 5];
                break
            end
        end

%         for x = -4:4
%             for y = -4:4
%                
%                 SSE = sum((current_block - block).^2, 'all');
%                 if SSE < MIN_SSE
%                     MIN_SSE = SSE;
%                     best_loc = [x + 5, y + 5];
%                 end
%             end
%         end
        %Change Matrix indice to linear indice
        motion_vector_indice = sub2ind([9, 9], best_loc(1), best_loc(2));
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