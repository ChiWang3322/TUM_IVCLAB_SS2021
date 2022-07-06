function block = IDCT8x8(coeff)
%  Function Name : IDCT8x8.m
%  Input         : coeff (DCT Coefficients) 8*8*3
%  Output        : block (original image block) 8*8*3
    block = zeros(size(coeff));
    [~, ~, C] = size(coeff);
    for c = 1:C
        block(:, :, c) = idct(coeff(:, :, c));
        block(:, :, c) = idct(block(:, :, c)');
        block(:, :, c) = block(:, :, c)';
    end
end