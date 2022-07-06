%% DO NOT MODIFY THIS CODE
%% This code will call your function and should work without any changes
quant = randi(10, [8, 8]);
ZigZag8x8(quant);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% DO NOT MODIFY THIS CODE
%% This code will call your function and should work without any changes
quant = randi(255,8,8,3);
zz    = ZigZag8x8(quant);
coeff = DeZigZag8x8(zz);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");