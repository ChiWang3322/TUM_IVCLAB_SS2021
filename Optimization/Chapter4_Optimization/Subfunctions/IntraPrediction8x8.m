function [ResBlock, mode] = IntraPrediction8x8(block)
%Input : block (9x9 block)
%Output : ResBlock (8x8 resblcok)
%              mode (0-3)
    ResBlockCell = {};
    Num_modes = 4;
    SSDArray = zeros(1, Num_modes);
    [ResBlockCell{1}, SSDArray(1)] = mode0(block);
    [ResBlockCell{2}, SSDArray(2)] = mode1(block);
    [ResBlockCell{3}, SSDArray(3)] = mode2(block);
    [ResBlockCell{4}, SSDArray(4)] = mode3(block);
    [~, mode] = min(SSDArray);
    ResBlock = ResBlockCell{mode};
    mode = mode - 1;

    %Vertical
    function [ResBlock, SSD] = mode0(block)
        ResBlock = zeros(8, 8);
        for i = 1:8
            ResBlock(:, i) = block(1, i + 1);
        end
        ResBlock = block(2:end, 2:end) - ResBlock;
        SSD = sum(abs(ResBlock), 'all');
    end
        
    %Horizontal
    function [ResBlock, SSD] = mode1(block)
        ResBlock = zeros(8, 8);
        for i = 1:8
            ResBlock(i, :) = block(i + 1, 1);
        end
        ResBlock = block(2:end, 2:end) - ResBlock;
        SSD = sum(abs(ResBlock), 'all');
    end
    %DC
    function [ResBlock, SSD] = mode2(block)
        DC = sum(block(1, :)) + sum(block(:, 1)) - block(1, 1);
        DC = DC / (9 + 8);
        ResBlock = zeros(8, 8);
        ResBlock = block(2:end, 2:end) - ones(8, 8) * DC;
        SSD = sum(abs(ResBlock), 'all');
    end
    
    %Diagnal
    function [ResBlock, SSD] = mode3(block)
        ResBlock = zeros(8, 8);
        for i = 1:8
            Value = block(1, i);
            temp = ResBlock(1 : end - i + 1, i : end);
            for j = 1:8 - i + 1
                temp(j, j) = Value;
            end
            ResBlock(1 : end - i + 1, i : end) = temp;
        end
        
        for i = 1:8
            Value = block(i, 1);
            temp = ResBlock(i : end, 1 : end - i + 1);
            for j = 1:8 - i + 1
                temp(j, j) = Value;
            end
            ResBlock(i : end, 1 : end - i + 1) = temp;
        end
        ResBlock = block(2:end, 2:end) - ResBlock;
        SSD = sum(abs(ResBlock), 'all');
        
        
    end
end

