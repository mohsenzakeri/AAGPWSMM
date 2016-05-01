load('./data/tr_usr_norm');
load('./data/tst_usr_norm');

usr = sort(unique(tr_usr_norm(:,1)));

[~,usrIdx] = ismember(tr_usr_norm(:,1),usr);

m = max(usrIdx);
n = max(tr_usr_norm(:,2));
trainFeat = sparse(usrIdx,tr_usr_norm(:,2),tr_usr_norm(:,3),m,n);

%Find train genders
genders = findUserGenders(usr);

%[C,accuracy] = TuneC(genders, trainFeat,1,1,100,10000);
%SVM 5
%C = 6801;
%SVM 1
C = 1001;
%logistic 0
%C = 9501;
user_svm_model = train(genders, trainFeat ,[sprintf('-s 1 -c %f',C)]); %9501
[predicted_label, accuracy, prob_estimates] = predict(genders, trainFeat, user_svm_model, ['-b 1']);

usr = sort(unique(tst_usr_norm(:,1)));
[~,usrIdx] = ismember(tst_usr_norm(:,1),usr);

m = max(usrIdx);
n = max(tr_usr_norm(:,2));
usr_testFeat = sparse(usrIdx,tst_usr_norm(:,2),tst_usr_norm(:,3),m,n);
test_genders = findUserGenders(usr);

[predicted_label, accuracy, prob_estimates] = predict(test_genders, usr_testFeat, user_log_model,['-b 1']);

