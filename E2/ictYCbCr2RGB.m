function rgb = ictYCbCr2RGB(yuv)
% Input         : yuv (Original YCbCr image)
% Output        : rgb (RGB Image after transformation)
% YOUR CODE HERE
    y = yuv(:, :, 1);
    Cb = yuv(:, :, 2);
    Cr = yuv(:, :, 3);
    rgb(:, :, 1) = y + 1.402 * Cr;
    rgb(:, :, 2) = y - 0.344 * Cb - 0.714 * Cr;
    rgb(:, :, 3) = y + 1.772 * Cb;
end