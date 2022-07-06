%% DO NOT MODIFY THIS CODE
%% This code will call your function and should work without any changes
EOB = 1000;
A = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2 3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 0 0];

B = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2  3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 0 1];

C = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2 3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 1 0];

D = [2 -2 3 1 -1 2 0 0 -1 2 -1 -1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0, ...
    0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0, ...
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0, ...
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

A_solution = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 EOB];

B_solution = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 0 1 1];

C_solution = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 0 0 1 EOB];

D_solution = [2 -2 3 1 -1 2 0 1 -1 2 -1 -1 1 0 24 -1 0 23 -1 EOB];

A_mysol = ZeroRunEnc_EoB(A, EOB);
B_mysol = ZeroRunEnc_EoB(B, EOB);
C_mysol = ZeroRunEnc_EoB(C, EOB);
D_mysol = ZeroRunEnc_EoB(D, EOB);
fprintf('Solution of A correct? %s\n', string(sum(A_mysol == A_solution) == length(A_solution)));
fprintf('Solution of B correct? %s\n', string(sum(B_mysol == B_solution) == length(B_solution)));
fprintf('Solution of C correct? %s\n', string(sum(C_mysol == C_solution) == length(C_solution)));
fprintf('Solution of D correct? %s\n', string(sum(D_mysol == D_solution) == length(D_solution)));

%% Test a sequence
% test on a sequence
load('foreman10_residual_zero_run');
load('foreman10_residual_zig_zag');
EOB = 1000;
zero_run_enc = ZeroRunEnc_EoB(foreman10_residual_zig_zag, EOB);
fprintf("Your solution:\n");
zero_run_enc(1,1:100)
fprintf("Our solution:\n");
foreman10_residual_zero_run(1,1:100)
fprintf("Correct? %d\n", sum(zero_run_enc(1,1:100) == foreman10_residual_zero_run(1,1:100)))
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");