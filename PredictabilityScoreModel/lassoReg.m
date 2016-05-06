function [w, b, trainMAE, lossValue] = lassoReg(trX, trY, k)
[lambdas,~] = kfold_lasso_lambda_tuner(trX, trY, k);
lambda = lambdas(size(lambdas, 1)-1);

d = size(trX, 1);
N = size(trX, 2);
b = 0;
w = ones(d,1);
trXt = transpose(trX);
a = 2 * sum(trX.^2, 2);
lossValue = norm(trXt*w + b - trY).^2 + lambda * sum(abs(w));
delta = 1;
while(delta >= 0.1)
    %disp(num2str(lossValue));
    r = trY - (trXt*w + b); % compute residual vector r        
    b_old = b;
    b = (1/N)*sum(r + b_old); % update b
    r = r + b_old - b; % update r
    for k=1:d
        ck = 2* trX(k, :)*(r + w(k)*trXt(:, k));
        wk_old = w(k);
        if (ck < -lambda)
            w(k) = (ck + lambda)/a(k);
        elseif (ck > lambda)
                w(k) = (ck - lambda)/a(k);
        else w(k) = 0;
        end
        r = r + (wk_old-w(k))*trXt(:, k);
    end
    lossValue_old = lossValue;
    lossValue = norm(trXt*w + b - trY).^2 + lambda * sum(abs(w));
    delta = sum(lossValue_old - lossValue);
end

trainMAE = sum(abs(trY-(trXt*w+b)))*(1/N);
resultCorr = corr(trY, trXt*w+b);
disp(['correlation: ' num2str(resultCorr)])
lossValue = norm(trXt*w + b - trY).^2 + lambda * sum(abs(w));

end