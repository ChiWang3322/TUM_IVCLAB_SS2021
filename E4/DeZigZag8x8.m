function coeffs = DeZigZag8x8(zz)
%  Function Name : DeZigZag8x8.m
%  Input         : zz    (Coefficients in zig-zag order)
%
%  Output        : coeffs(DCT coefficients in original order)
    coeffs = zeros([8, 8, 3]);
    ZigZag =    [1     2    6    7    15   16   28   29;
                 3     5    8    14   17   27   30   43;
                 4     9    13   18   26   31   42   44;
                 10    12   19   25   32   41   45   54;
                 11    20   24   33   40   46   53   55;
                 21    23   34   39   47   52   56   61;
                 22    35   38   48   51   57   60   62;
                 36    37   49   50   58   59   63   64];
    for i = 1:3
        ith_zz = zz(:, i);
        temp = ith_zz(ZigZag(:));
        temp = reshape(temp, 8, 8);
        coeffs(:, :, i) = temp;
    end
end