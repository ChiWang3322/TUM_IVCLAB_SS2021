imageLena     = double(imread('data/images/lena.tif'));
imageSail     = double(imread('data/images/sail.tif'));
imageSmandril = double(imread('data/images/smandril.tif'));

range = 0:255;

pmfLena       = stats_marg(imageLena, range);
pmfSail       = stats_marg(imageSail, range);
pmfSmandril   = stats_marg(imageSmandril, range);

mergedPMF     = pmfLena + pmfSail + pmfSmandril;
mergedPMF     = mergedPMF / sum(mergedPMF(:));

minCodeLengthLena     = min_code_length(mergedPMF, pmfLena);
minCodeLengthSail     = min_code_length(mergedPMF, pmfSail);
minCodeLengthSmandril = min_code_length(mergedPMF, pmfSmandril);

fprintf('--------------Using merged code table--------------\n');
fprintf('lena.tif      H = %.2f bit/pixel\n', minCodeLengthLena);
fprintf('sail.tif      H = %.2f bit/pixel\n', minCodeLengthSail);
fprintf('smandril.tif  H = %.2f bit/pixel\n', minCodeLengthSmandril);

% Put all sub-functions which are called in your script here.
function pmf = stats_marg(image, range)
    pmf = hist(image(:), range);
    pmf = pmf ./ sum(pmf);
end

function H = min_code_length(pmf_table, pmf_image)
    H = -sum(pmf_image'.*log2(nonzeros(pmf_table)));
end