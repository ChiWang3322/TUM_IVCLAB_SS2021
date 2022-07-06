%% DO NOT MODIFY THIS CODE
path(path,'data/images');     
path(path,'encoder');
path(path,'decoder'); 
%% This code will call your function and should work without any changes
load('qImage.mat')
load('qImage_small.mat')

recImage = {};
recImage_small = {};
bits = 1 : 1 : 7;
for bit = bits
    recImage{end+1} = InvUniQuant(qImage{bit}, bit);
    recImage_small{end+1} = InvUniQuant(qImage_small{bit}, bit);
end
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");