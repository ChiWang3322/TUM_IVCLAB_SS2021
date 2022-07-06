% Read Image
imageLena = double(imread('data/images/lena.tif'));
%Define the first-order predictor
a1 = 1;
% create the predictor and obtain the residual image
[M, N, C] = size(imageLena);
preImage = zeros(size(imageLena));
preImage(:, 1, :) = imageLena(:, 1, :);
for n = 2:N
    preImage(:, n, :) = a1 * imageLena(:, n-1, :);
end
resImage  = zeros(size(imageLena));
resImage = imageLena - preImage;
% get the PMF of the residual image
pmfRes    = stats_marg(resImage, -255:255);
% calculate the entropy of the residual image
H_res     = calc_entropy(pmfRes);

fprintf('H_err_OnePixel   = %.2f bit/pixel\n',H_res);

% Put all sub-functions which are called in your script here.
function pmf = stats_marg(image, range)
    pmf = hist(image(:), range);
    pmf = pmf ./ sum(pmf);
end

function H = calc_entropy(pmf)
    pmf = pmf(:);
    H = -sum(nonzeros(pmf).*log2(nonzeros(pmf)));
end