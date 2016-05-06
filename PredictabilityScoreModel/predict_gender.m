function [traccuracy,tstccuracy] = predict_gender(trLbs,trD,tstLbs,tstD,model)
    [C,accuracy] = TuneC(trLbs, trD,model,1,100,10000);
    %C for scored values with logistic - libsvm 0
    %C = 7901;
    %C for scored values with svm - libsvm 5
    %C = 4501;
    %C for score with predictability
    %C = 7301
    %C for not scored values svm - libsvm 5
    %C = 101;
    %C for not scored values logistic - libsvm 0
    %C = 9701;
    user_model = train(trLbs, trD ,[sprintf('-s %f -c %f',model,C)]); %4101
    [predicted_label, traccuracy, prob_estimates] = predict(trLbs, trD, user_model);
    [predicted_label, tstccuracy, prob_estimates] = predict(tstLbs, tstD, user_model);
end