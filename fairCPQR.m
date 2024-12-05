% Fair CPQR for CSS with 2 row groups
function [P] = fairCPQR(A, B,kf)
    [~,n] = size(A);
    P = eye(n);

    % Start matrices
    R_hat_A = A; R_hat_B = B;

    % First iteration
    % Pivoting strategy: select column with biggest spectral norm
    norms_A = vecnorm(R_hat_A); norms_B = vecnorm(R_hat_B);
    [~,j_A] = max(norms_A); [~,j_B] = max(norms_A);
    if norms_A(j_A) >= norms_B(j_B) % Select A or B to be compensated
        j = j_A;
    else
        j = j_B;
    end
    
    % Permute columns 
    P(:,[1,j]) = P(:,[j,1]); 
    R_hat_A(:,[1,j]) = R_hat_A(:,[j,1]); norms_A([1,j]) = norms_A([j,1]);
    R_hat_B(:,[1,j]) = R_hat_B(:,[j,1]); norms_B([1,j]) = norms_B([j,1]);

    % Householder transformations and update
    R_A = norm(R_hat_A(:,1)); Q_A = R_hat_A(:,1) / R_A;
    V_A = Q_A' * R_hat_A(:,2:n); R_hat_A(:,2:n) = R_hat_A(:,2:n) - Q_A * V_A;
    norms_A(1) = 0; norms_A(2:n) = sqrt(norms_A(2:n).^2 - V_A.^2);

    R_B = norm(R_hat_B(:,1)); Q_B = R_hat_B(:,1) / R_B;
    V_B = Q_B' * R_hat_B(:,2:n); R_hat_B(:,2:n) = R_hat_B(:,2:n) - Q_B * V_B;
    norms_B(1) = 0; norms_B(2:n) = sqrt(norms_B(2:n).^2 - V_B.^2);
    
    for k = 2:kf
        % Pivoting strategy
        [~,j_A] = max(norms_A); [~,j_B] = max(norms_A);
        if norms_A(j_A) >= norms_B(j_B) % Select A or B to be compensated
            j = j_A;
        else
            j = j_B;
        end

        % Permute columns
        P(:,[k,j]) = P(:,[j,k]); 
        R_hat_A(:,[k,j]) = R_hat_A(:,[j,k]); norms_A([k,j]) = norms_A([j,k]);
        R_hat_B(:,[k,j]) = R_hat_B(:,[j,k]); norms_B([k,j]) = norms_B([j,k]);
        
        % Householder transformations and update
        V_A(:,[1,j-k+1]) = V_A(:,[j-k+1,1]);
        r = norm(R_hat_A(:,k)); Q_A = [Q_A,R_hat_A(:,k)/r]; R_A = [R_A,V_A(:,1);zeros(1,k-1),r];
        b = Q_A(:,k)' * R_hat_A(:,(k+1):n); V_A(:,1) = []; V_A = [V_A;b];
        R_hat_A(:,(k+1):n) = R_hat_A(:,(k+1):n) - Q_A(:,k) * b;
        norms_A(k) = 0; norms_A((k+1):n) = sqrt(norms_A((k+1):n).^2 - b.^2);

        V_B(:,[1,j-k+1]) = V_B(:,[j-k+1,1]);
        r = norm(R_hat_B(:,k)); Q_B = [Q_B,R_hat_B(:,k)/r]; R_B = [R_B,V_B(:,1);zeros(1,k-1),r];
        b = Q_B(:,k)' * R_hat_B(:,(k+1):n); V_B(:,1) = []; V_B = [V_B;b];
        R_hat_B(:,(k+1):n) = R_hat_B(:,(k+1):n) - Q_B(:,k) * b;
        norms_B(k) = 0; norms_B((k+1):n) = sqrt(norms_B((k+1):n).^2 - b.^2);
    end
end