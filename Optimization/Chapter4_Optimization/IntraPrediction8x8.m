function [ResImage, modeArray] = IntraPrediction8x8(image)
%Input : image
%         
%Output : ResImage (4x4 resblcok)
%             modeArray (0-3)
    [M, N, C] = size(image);
    block_size = 2;
    ResImage = zeros(M, N, C);
    modeArray = ones(M / block_size, N / block_size, 3) * 10;
    TempImage = zeros(M + 1, N + 1, 3);
    for c = 1:C
        TempImage(:, :, c) = padarray(image(:, :, c), [1 1], 'pre', 'replicate');
    end
    for c = 1:C
        for i = 1:block_size:M
            for j = 1:block_size:N
                block = TempImage(i : i + block_size, j : j + block_size, c);
                ResBlockCell = {};
                Num_modes = 4;
                SSDArray = zeros(1, Num_modes);
                [ResBlockCell{1}, SSDArray(1)] = mode0(block, block_size);
                [ResBlockCell{2}, SSDArray(2)] = mode1(block, block_size);
                [ResBlockCell{3}, SSDArray(3)] = mode2(block, block_size);
                [ResBlockCell{4}, SSDArray(4)] = mode3(block, block_size);
                [~, mode] = min(SSDArray);
                ResBlock = ResBlockCell{mode};
                modeArray(floor(i / block_size) + 1, floor(j / block_size) + 1, c) = mode - 1;
                ResImage(i : i + block_size - 1, j : j + block_size - 1, c) = ResBlock;
            end
        end
    end
end


%% Mode function
%Vertical
function [ResBlock, SSD] = mode0(block, bsize)
    ResBlock = zeros(bsize, bsize);
    for i = 1:bsize
        ResBlock(:, i) = block(1, i + 1);
    end
    ResBlock = block(2:end, 2:end) - ResBlock;
    SSD = sum(abs(ResBlock), 'all');
end

%Horizontal
function [ResBlock, SSD] = mode1(block, bsize)
    ResBlock = zeros(bsize, bsize);
    for i = 1 : bsize
        ResBlock(i, :) = block(i + 1, 1);
    end
    ResBlock = block(2:end, 2:end) - ResBlock;
    SSD = sum(abs(ResBlock), 'all');
end
%DC
function [ResBlock, SSD] = mode2(block, bsize)
    DC = sum(block(1, :)) + sum(block(:, 1)) - block(1, 1);
    DC = DC / (bsize + 1 + bsize);
    ResBlock = block(2:end, 2:end) - ones(bsize, bsize) * DC;
    SSD = sum(abs(ResBlock), 'all');
end

%Diagnal (down-right)
function [ResBlock, SSD] = mode3(block, bsize)
    ResBlock = zeros(bsize, bsize);
    for i = 1:bsize
        Value = block(1, i);
        temp = ResBlock(1 : end - i + 1, i : end);
        for j = 1:bsize - i + 1
            temp(j, j) = Value;
        end
        ResBlock(1 : end - i + 1, i : end) = temp;
    end

    for i = 1:bsize
        Value = block(i, 1);
        temp = ResBlock(i : end, 1 : end - i + 1);
        for j = 1:bsize - i + 1
            temp(j, j) = Value;
        end
        ResBlock(i : end, 1 : end - i + 1) = temp;
    end
    ResBlock = block(2:end, 2:end) - ResBlock;
    SSD = sum(abs(ResBlock), 'all');
end


