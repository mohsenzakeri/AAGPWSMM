load('./data/tr_usr_norm');
usr = sort(unique(tr_usr_norm(:,1)));
[~,usrIdx] = ismember(tr_usr_norm(:,1),usr);

m = max(usrIdx);
n = max(tr_usr_norm(:,2));
trainFeat = sparse(usrIdx,tr_usr_norm(:,2),tr_usr_norm(:,3),m,n);

%Find train genders
fid = fopen('./data/ml.usr.age.gender.csv');
format = '%f %f %f';
C = textscan(fid,format, 1000000,'delimiter', '\t','CommentStyle','user'); 
user_gend = [C{1},C{3}];
[~, gendIdx] = ismember(usr,user_gend(:,1));
genders = user_gend(gendIdx,2);

%Tunning user model
%[C,accuracy] = TuneC(genders, trainFeat,0,1,100,10000);
C = 9501;

user_log_model = train(genders, trainFeat ,[sprintf('-s 0 -c %f',C)]); %9501
[predicted_label, accuracy, prob_estimates] = predict(genders, trainFeat, user_log_model, ['-b 1']);

load('./data/tr_ss_norm');
load('./data/tr_usr_ss');
tr_scores = scoreSubSequences(tr_ss_norm,tr_usr_ss, user_log_model);
sorted_scores = buildScoreSubModel(tr_scores,tr_usr_ss);


load('./data/tr_usr_ss');
usrids = tr_usr_ss(:,1);
usr = sort(unique(usrids));
[~,usrIdx] = ismember(sorted_scores(:,1),usr);
trainFeat = sparse(usrIdx,sorted_scores(:,2),sorted_scores(:,3));
fid = fopen('./data/ml.usr.age.gender.csv');
format = '%f %f %f';
C = textscan(fid,format,'delimiter', '\t','CommentStyle','user'); 
fclose(fid);
user_gend = [C{1},C{3}];
[~, gndIdx] = ismember(usr,user_gend(:,1));
genders = user_gend(gndIdx,2);

[C,accuracy] = TuneC(genders, trainFeat,0,1,100,10000);
% logistic 0
% C = 101;
% logistic 6
% C = 701
% logistic 7
% C = 501
% svm 5, 1, 3
%C = 1;
% svm 2
% C = 101;
% svm 2
% c = 401;
mih_model = train(genders, trainFeat ,[sprintf('-s 0 -c %f',C)]); 
[predicted_label, accuracy, prob_estimates] = predict(genders, trainFeat, mih_model,['-b 1']);

load('./data/tst_ss_norm');
load('./data/tst_usr_ss');
load('./data/tst_usr_ss');
tst_scores = scoreSubSequences(tst_ss_norm,tst_usr_ss, user_log_model);
sorted_scores = buildScoreSubModel(tst_scores,tst_usr_ss);

load('./data/tst_usr_ss');
usrids = tst_usr_ss(:,1);
usr_subids = tst_usr_ss(:,2);
usr = sort(unique(usrids));
[~,usrIdx] = ismember(sorted_scores(:,1),usr);
testFeat = sparse(usrIdx,sorted_scores(:,2),sorted_scores(:,3));

[~, gndIdx] = ismember(usr,user_gend(:,1));
test_genders = user_gend(gndIdx,2);

[predicted_label, accuracy, prob_estimates] = predict(test_genders, testFeat, mih_model,['-b 1']);


