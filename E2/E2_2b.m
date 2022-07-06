% Read Image
imageLena = double(imread('data/images/lena.tif'));
% Calculate Joint PMF
jpmfLena  = stats_joint(imageLena);
% Calculate Joint Entropy
Hjoint    = calc_entropy(jpmfLena);
fprintf('H_joint = %.2f bit/pixel pair\n', Hjoint);

% Put all sub-functions which are called in your script here.
function pmf = stats_joint(image)
%  Input         : image (Original Image)
%
%  Output        : pmf   (Probability Mass Function)
    pmf = zeros(256, 256);
    [M, N, C] = size(image);
    for c = 1:C
        for m = 1:M
            for n = 1:N/2
                pmf(image(m, 2 * n - 1,c), image(m, 2 * n, c)) = ...,
                pmf(image(m, 2 * n - 1,c), image(m, 2 * n, c)) + 1;
            end
        end
    end
    mesh(pmf);
    pmf = pmf / sum(pmf(:));
end

function H = calc_entropy(pmf)
%  Input         : pmf   (Probability Mass Function)
%
%  Output        : H     (Entropy in bits)
    pmf = pmf(:);
    H = -sum(nonzeros(pmf).*log2(nonzeros(pmf)));
end