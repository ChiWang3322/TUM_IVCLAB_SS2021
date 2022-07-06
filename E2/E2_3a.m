%% DO NOT MODIFY THIS CODE
%% This code will call your function and should work without any changes
imageLena = double(imread('data/images/lena.tif'));
H   = stats_cond(imageLena);
fprintf('H_cond = %.2f bit/pixel\n',H);

fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");