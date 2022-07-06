imageLena     = double(imread('data/images/lena.tif'));
imageSail     = double(imread('data/images/sail.tif'));
imageSmandril = double(imread('data/images/smandril.tif'));

range = 0:255

pmfLena       = stats_marg(imageLena, range);
HLena         = calc_entropy(pmfLena);

pmfSail       = stats_marg(imageSail, range);
HSail         = calc_entropy(pmfSail);

pmfSmandril   = stats_marg(imageSmandril, range);
HSmandril     = calc_entropy(pmfSmandril);

fprintf('--------------Using individual code table--------------\n');
fprintf('lena.tif      H = %.2f bit/pixel\n', HLena);
fprintf('sail.tif      H = %.2f bit/pixel\n', HSail);
fprintf('smandril.tif  H = %.2f bit/pixel\n', HSmandril);

% Put all sub-functions which are called in your script here.
function pmf = stats_marg(image, range)
    pmf = hist(image(:), range);
    pmf = pmf ./ sum(pmf);
end

function H = calc_entropy(pmf)
    pmf = pmf(:);
    H = -sum(nonzeros(pmf).*log2(nonzeros(pmf)));
end