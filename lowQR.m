function [Q,R_11,P,R_12,R_22] = lowQR(A,k)
    [m, n] = size(A);
    perm = 1:n;

    R = zeros(m,n);
    [Q,hat_R] = qr(A);

    for i = 1:k
        [~,~,v] = svds(hat_R,1,'largest');
        [~,c] = max(abs(v));
        
        P = eye(n-i+1);
        P(:, [1 c]) = P(:, [c 1]);
        perm([i i+c-1]) = perm([i+c-1 i]);

        hat_R = hat_R * P;
        [Q_i,hat_R] = qr(hat_R);
        R(i:end,i) = hat_R(:,1);
        R(i,i:end) = hat_R(1,:);
        hat_R = hat_R(2:end,2:end);

        Q_i_tmp = eye(m); Q_i_tmp(i:end,i:end)=Q_i;
        Q = Q*Q_i_tmp;
    end

    P = eye(n); P = P(:, perm);
    % P=perm;
    R_11 = R(1:k,1:k);
    R_12 = R(1:k,k+1:end);
    R_22 = hat_R;
end