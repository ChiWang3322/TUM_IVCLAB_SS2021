function [Image] = IntraPredictionDecode8x8(ResImage, BorderImage, modeArray8x8)
%Input : ResImage (Resisual image, height x width x 3)
%           BorderArray (1x2 cell, first cell represents upper border, second is left border)
%           modeArray (height/4 x width/4 x 3, prediction mode of each block)
%Output : Image (Reconstructed image)
% 
    [M, N, C] = size(ResImage);
    block_size = 2;
    Image = zeros(M + 1, N + 1, 3);
    Image = BorderImage;
    for c = 1:C
        for i = 1:block_size:M
            for j = 1:block_size:N
                block = Image(i : i + block_size, j : j + block_size, c);
                mode = modeArray8x8(floor(i / block_size) + 1, floor(j / block_size) + 1, c);
                if mode == 0
                    PredBlock = mode0(block, block_size);
                elseif mode == 1
                    PredBlock = mode1(block, block_size);
                elseif mode == 2
                    PredBlock = mode2(block, block_size);
                else
                    PredBlock = mode3(block, block_size);
                end
                ResBlock = ResImage(i : i + block_size - 1, j : j + block_size - 1, c);
                temp = PredBlock + ResBlock;
                Image(i + 1 : i + block_size, j + 1 : j + block_size, c) = temp;
                                                                                                                
            end
        end
    end
    Image(1, :, :) = [];
    Image(:, 1, :) = [];
end


%% Mode function
%Vertical
function [PredBlock] = mode0(block, bsize)
    PredBlock = zeros(bsize, bsize);
    for i = 1:bsize
        PredBlock(:, i) = block(1, i + 1);
    end
end

%Horizontal
function [PredBlock] = mode1(block, bsize)
    PredBlock = zeros(bsize, bsize);
    for i = 1 : bsize
        PredBlock(i, :) = block(i + 1, 1);
    end
end

%DC
function [PredBlock] = mode2(block, bsize)
    DC = sum(block(1, :)) + sum(block(:, 1)) - block(1, 1);
    DC = DC / (bsize + 1 + bsize);
    PredBlock = ones(bsize, bsize) * DC;

end

%Diagnal (down-right)
function [PredBlock] = mode3(block, bsize)
    PredBlock = zeros(bsize, bsize);
    for i = 1:bsize
        Value = block(1, i);
        temp = PredBlock(1 : end - i + 1, i : end);
        for j = 1:bsize - i + 1
            temp(j, j) = Value;
        end
        PredBlock(1 : end - i + 1, i : end) = temp;
    end

    for i = 1:bsize
        Value = block(i, 1);
        temp = PredBlock(i : end, 1 : end - i + 1);
        for j = 1:bsize - i + 1
            temp(j, j) = Value;
        end
        PredBlock(i : end, 1 : end - i + 1) = temp;
    end
end


