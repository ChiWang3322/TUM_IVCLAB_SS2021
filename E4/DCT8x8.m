function coeff = DCT8x8(block)
%  Input         : block    (Original Image block, 8x8x3)
%
%  Output        : coeff    (DCT coefficients after transformation, 8x8x3)
    coeff = zeros(size(block));
    [M, N, C] = size(block);
    % Y = AXA'
    for c = 1:C
        coeff(:, :, c) = dct(block(:, :, c));   %AX
        coeff(:, :, c) = dct(coeff(:, :, c)');  %A*(AX)' = Y'
        coeff(:, :, c) = coeff(:, :, c)';   
    end
end