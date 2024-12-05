% Fair CPQR for CSS with 2 row groups
function [P] = fairLowQRforCSS(A,B,k)
    [m_1, n] = size(A); [m_2, ~] = size(B);
    perm = 1:n;
    
    Q = 0;
    R_A = zeros(m_1,n); R_B = zeros(m_2,n);
    hat_R_A = qr(A); hat_R_B = qr(B);

    for i = 1:k
        [~,~,v_A] = svds(hat_R_A,1,'largest');
        [~,~,v_B] = svds(hat_R_B,1,'largest');
        [key_A,c_A] = max(abs(v_A)); [key_B,c_B] = max(abs(v_B));
        if key_A >= key_B % Select A or B to be compensated
            c = c_A;
        else
            c = c_B;
        end
        P = eye(n-i+1);
        P(:, [1 c]) = P(:, [c 1]);
        perm([i i+c-1]) = perm([i+c-1 i]);

        hat_R_A = hat_R_A * P; hat_R_B = hat_R_B * P;
        hat_R_A = qr(hat_R_A); hat_R_B = qr(hat_R_B);
        R_A(i:end,i) = hat_R_A(:,1); R_B(i:end,i) = hat_R_B(:,1);
        R_A(i,i:end) = hat_R_A(1,:); R_B(i,i:end) = hat_R_B(1,:);
        hat_R_A = hat_R_A(2:end,2:end); hat_R_B = hat_R_B(2:end,2:end);
    end

    P = eye(n); P = P(:, perm);
    % P=perm;
end