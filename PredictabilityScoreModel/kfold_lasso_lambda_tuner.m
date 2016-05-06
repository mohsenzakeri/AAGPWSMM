function [lambdas, validationErrors] = kfold_lasso_lambda_tuner(trX, trY, k)
    N = size(trY, 1);
    d = size(trX, 1);
    fold = floor(N/k);
    validationErrors = zeros(1);
    lambdas = zeros(1);

    lambda = 2*norm(trX*(trY-(1/N)*sum(trY)), Inf);
    w = ones(d,1);
    b = 0;
    
    validationError = norm(trY-(trX'*w+b))*sqrt(1/N);    
    validationDiff = 1;
    % change lamda until validation error starts to increase or 
    % doesn't have a specific improvement.
    minLoops = 0;
    while (validationDiff > 0 || minLoops < 3)        
        tic
        % Regularization Path
%        disp(['lamda: ', num2str(lamda)]);
        old_validationError = validationError;
        k_valErr = 0;
        for i=0:k-1
            tr_startidx = fold*i+1;
            tr_endidx = mod(tr_startidx + (fold * (k-1)), N);
            if tr_endidx == 0
                tr_endidx = N;
            end
            if tr_startidx <= tr_endidx
                tr_idx = tr_startidx:tr_endidx;
            else
                tr_idx = [1:tr_endidx, tr_startidx:N];
            end
            val_idx = setdiff(1:N, tr_idx);
            [valErr, w] = coordinateDescent(trX(:, tr_idx),trY(tr_idx),...
            trX(:, val_idx), trY(val_idx), lambda, w, b);
            k_valErr = k_valErr + valErr;
        end
        validationError = k_valErr / k;
        validationDiff = sum(old_validationError - validationError);
        nonzero = length(w(w~=0));
        validationErrors = [validationErrors;validationError];
        lambdas = [lambdas;lambda];
        lambda = lambda * 0.5; %% change lamda by a constant ratio of 2
        minLoops = minLoops + 1;
        disp([num2str(minLoops) ' -> time: ' num2str(toc) ', val_diff: ' num2str(validationDiff)]);
    end    
end