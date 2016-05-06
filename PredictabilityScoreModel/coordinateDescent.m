function [validationError, w] = coordinateDescent(trX, trY, valX, valY, lambda, w, b)
    d = size(trX, 1);
    N = size(trX, 2);
    valN = size(valX, 2);
    delta = 1;
    Xt = transpose(trX);
    valXt = transpose(valX);
    a = 2 * sum(trX.^2, 2);
    lossValue = norm(Xt*w + b - trY).^2 + lambda * sum(abs(w));
    while(delta >= 0.01)
    %            disp(['loss function value : ', num2str(lossValue)]);
        r = trY - (Xt*w + b); % compute residual vector r        
        b_old = b;
        b = (1/N)*sum(r + b_old); % update b
        r = r + b_old - b; % update r
        for k=1:d
            ck = 2* trX(k, :)*(r + w(k)*Xt(:, k));
            wk_old = w(k);
            if (ck < -lambda)
                w(k) = (ck + lambda)/a(k);
            elseif (ck > lambda)
                    w(k) = (ck - lambda)/a(k);
            else w(k) = 0;
            end
            r = r + (wk_old-w(k))*Xt(:, k);
        end
        lossValue_old = lossValue;
        lossValue = norm(Xt*w + b - trY).^2 + lambda * sum(abs(w));
        delta = sum(lossValue_old - lossValue);
    end
    validationError = norm(valY-(valXt*w+b))*sqrt(1/valN);