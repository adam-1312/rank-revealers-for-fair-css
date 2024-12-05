% Test script for CPQR
%rng(37);
m = 500; n = 50; k = 20;

iters = 1000;

% Initialize
error = Inf(1,iters); mu_B = error;
tic
for i = 1:iters
    % fprintf('Iteration #%d\n',i);

    A = rand(m,n);
    
    % Compute
    [Q,R,Pi,gamma,R12,A2] = CPQR(A,k);
    % [Q,R,Pi,R12,R22] = lowQR(A,k); Q = Q(:, 1:k);
    
    % Calculate error
    A_k = Q*[R R12];
    error(i) = norm(A*Pi-A_k);
    % fprintf('CPQR approximation error is %.3e\n',error);
    
    % Calculate success of pivoting strategy
    mu_B(i) = max(gammaQR(A*Pi,k),1);
    % fprintf('CPQR success value %.3e\n',mu_B);
end
toc

% plot(1:iters, error)
% figure
histogram(mu_B')