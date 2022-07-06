% Read Image
imageLena = double(imread('data/images/lena_small.tif'));
%Convert image to YUV domain
imageLena_yuv = ictRGB2YCbCr(imageLena);
%Define third-order predictor
a1_y = 7/8; a2_y = -1/2; a3_y = 5/8;
a1_c = 3/8; a2_c = -1/4; a3_c = 7/8;
% create the predictor and obtain the residual image
[M, N, C] = size(imageLena);
preImage = zeros(M, N, C);
resImage  = zeros(M, N, C);
%Copy the first column and the first row of original image to res and pre
preImage(1, :, :) = round(imageLena_yuv(1, :, :));
preImage(:, 1, :) = round(imageLena_yuv(:, 1, :));
resImage(1, :, :) = round(imageLena_yuv(1, :, :));
resImage(:, 1, :) = round(imageLena_yuv(:, 1, :));
%Predict Y channel
for m = 2:M
    for n = 2:N
        preImage(m, n, 1) = 7/8 * preImage(m, n - 1, 1) + ...,
                                       (-1/2) * preImage(m - 1, n - 1, 1) + ...,
                                       5/8 * preImage(m - 1, n, 1);
        resImage(m, n, 1) = round(imageLena_yuv(m, n, 1) - preImage(m, n, 1));
        preImage(m, n, 1) = preImage(m, n, 1) + resImage(m, n, 1);
    end
end
%Predict CbCr channel
for c = 2:3
    for m = 2:M
        for n = 2:N
            preImage(m, n, c) = 3/8 * preImage(m, n - 1, c) + ...,
                                           (-1/4) * preImage(m - 1, n - 1, c) + ...,
                                           7/8 * preImage(m - 1, n, c);
            resImage(m, n, c) = round(imageLena_yuv(m, n, c) - preImage(m, n, c));
            preImage(m, n, c) = preImage(m, n, c) + resImage(m, n, c);
        end;
    end
end
% get the PMF of the residual image
pmfRes    = stats_marg(resImage, -255:255);
% calculate the entropy of the residual image
H_res     = calc_entropy(pmfRes);

fprintf('H_err_ThreePixel   = %.2f bit/pixel\n',H_res);
%Huffmann encoding
[ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( pmfRes );
fprintf('Maximum code length   = %d\n',max(Codelengths));
fprintf('Minimum code length   = %d\n',min(Codelengths));
plot(Codelengths);

% Put all sub-functions which are called in your script here.
function pmf = stats_marg(image, range)
    pmf = hist(image(:), range);
    pmf = pmf ./ sum(pmf);
end

function H = calc_entropy(pmf)
    pmf = pmf(:);
    H = -sum(nonzeros(pmf).*log2(nonzeros(pmf)));
end

function yuv = ictRGB2YCbCr(rgb)
% Input         : rgb (Original RGB Image)
% Output        : yuv (YCbCr image after transformation)
% YOUR CODE HERE
    r = rgb(:, :, 1);
    g = rgb(:, :, 2);
    b = rgb(:, :, 3);
    yuv(:, :, 1) = 0.299 * r + 0.587 * g + 0.114 * b;
    yuv(:, :, 2) = -0.169 * r - 0.331 * g + 0.5 * b;
    yuv(:, :, 3) = 0.5 * r - 0.419 * g - 0.081 * b;
end