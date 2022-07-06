%% DO NOT MODIFY THIS CODE
path(path,'data/images');     
path(path,'encoder');
path(path,'decoder'); 
%% This code will call your function and should work without any changes
imageLena_small = double(imread('lena_small.tif'));
imageLena = double(imread('lena.tif'));

qImage = {};
qImage_small = {};
bits = 1 : 1 : 7;
for bit = bits
    qImage{end+1} = UniQuant(imageLena, bit);
    qImage_small{end+1} =  UniQuant(imageLena_small, bit);
end
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");
save('qImage.mat', 'qImage');
save('qImage_small.mat', 'qImage_small');