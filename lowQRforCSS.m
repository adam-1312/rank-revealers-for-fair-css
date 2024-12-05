function [Q,R_11,P,R_12,R_22] = lowQRforCSS(A,k)
    [m, n] = size(A);
    perm = 1:n;
    
    Q = 0;
    R = zeros(m,n);
    hat_R = qr(A);

    for i = 1:k
        [~,~,v] = svds(hat_R,1,'largest');
        [~,c] = max(abs(v));
        
        P = eye(n-i+1);
        P(:, [1 c]) = P(:, [c 1]);
        perm([i i+c-1]) = perm([i+c-1 i]);

        hat_R = hat_R * P;
        hat_R = qr(hat_R);
        R(i:end,i) = hat_R(:,1);
        R(i,i:end) = hat_R(1,:);
        hat_R = hat_R(2:end,2:end);
    end

    P = eye(n); P = P(:, perm);
    % P=perm;
    R_11 = R(1:k,1:k);
    R_12 = R(1:k,k+1:end);
    R_22 = hat_R;
end