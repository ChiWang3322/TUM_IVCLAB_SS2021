%% DO NOT MODIFY THIS CODE
%% This code will call your function and should work without any changes
block = randi(255,[8,8]);
coeff = DCT8x8(block);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% DO NOT MODIFY THIS CODE
%% This code will call your function and should work without any changes
block1 = randi(255,8,8,3);
coeff1 = DCT8x8(block1);
rec1   = IDCT8x8(coeff1);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");