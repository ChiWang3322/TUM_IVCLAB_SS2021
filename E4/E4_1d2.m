%% DO NOT MODIFY THIS CODE
%% This code will call your function and should work without any changes
EOB = 1000;
A = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 EOB];

B = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 0 1 1];

C = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 0 0 1 EOB];

A_solution = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2 3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 0 0];

B_solution = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2  3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 0 1];

C_solution = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2 3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 1 0];

% Run learner solution.
A_mysol = ZeroRunDec_EoB(A, EOB);
B_mysol = ZeroRunDec_EoB(B, EOB);
C_mysol = ZeroRunDec_EoB(C, EOB);

fprintf('Solution of A correct? %s\n', string(sum(A_mysol == A_solution) == length(A_solution)));
fprintf('Solution of B correct? %s\n', string(sum(B_mysol == B_solution) == length(B_solution)));
fprintf('Solution of C correct? %s\n', string(sum(C_mysol == C_solution) == length(C_solution)));

% assert(isequal(length(A_solution),length(A_mysol)), 'The length of reconstructed A is %d, but the length of original A is %d', length(zzA), length(A));
% assert(isequal(length(B_solution),length(B_mysol)), 'The length of reconstructed B is %d, but the length of original B is %d', length(zzB), length(B));
% assert(isequal(length(C_solution),length(C_mysol)), 'The length of reconstructed C is %d, but the length of original C is %d', length(zzC), length(C));

fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");