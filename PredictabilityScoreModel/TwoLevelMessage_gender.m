fid = fopen('./data/1k1gram_feat.csv');
format = '%s %f';
C = textscan(fid,format,'delimiter', ',','CommentStyle','feat,freq'); 
feat = C{1};
freq = C{2};
fclose(fid);

fid = fopen('./data/msgs_tr_10_1k1gram.csv');
format = '%s %f %s %s %f';
C = textscan(fid,format,'delimiter', ',','CommentStyle','id,'); 
msgs_tr_10_1k1gram_id = C{2};
msgs_tr_10_1k1gram_feat = C{3};
msgs_tr_10_1k1gram_norm = C{5}; 
fclose(fid);

load('./data/usrMsgMap');
uniqueMessages = unique(msgs_tr_10_1k1gram_id);
[~, MsgIdxs]=ismember(msgs_tr_10_1k1gram_id, uniqueMessages);
[~, MsgIdxsTotal]=ismember(uniqueMessages, usrMsgMap(:, 2));
[~, featIdxs] = ismember(msgs_tr_10_1k1gram_feat, feat(:, 1));

trainFeat = sparse(MsgIdxs,featIdxs,msgs_tr_10_1k1gram_norm);

%Find train genders
fid = fopen('./data/ml.usr.age.gender.csv');
format = '%f %f %f';
C = textscan(fid,format, 1000000,'delimiter', '\t','CommentStyle','user'); 
user_gend = [C{1},C{3}];

user_of_message = usrMsgMap(MsgIdxsTotal,1);
[~, gendIdx] = ismember(user_of_message,user_gend(:,1));
genders = user_gend(gendIdx,2);

% %logistic model
disp('logistic model')
% tunning
[C,accuracy] = TuneC(genders, trainFeat,0,1,100,10000);
%C = 1
message_log_model = train(genders, trainFeat ,[sprintf('-s 0 -c %f',C)]); %3101
[predicted_label, accuracy, prob_estimates] = predict(genders, trainFeat, message_log_model, ['-b 1']);
evalModel = [genders,predicted_label,prob_estimates];

error = abs(genders-prob_estimates(:,2));
%predictability = abs(prob_estimates(:,2)-0.5);

% %regression model
disp('regression model')
%[w, b, trainMAE, lossValue] = lassoReg(trainFeat', predictability, 3);
[w, b, trainMAE, lossValue] = lassoReg(trainFeat', error, 3);
% save('w','w')
% save('b','b')
%assign score to all train messages
% load('w')
% load('b')

%files are already generated. Original message train and test files are
%needed for the following commands which are commented.
%usr_newFeat = generateNewUserFeat('./data/msgs_tr_1k1gram.csv', usrMsgMap, w, b, feat);
%save('usr_newFeat', 'usr_newFeat');
%usr_tst_newFeat = generateNewUserFeat('./data/msgs_tst_1k1gram.csv', usrMsgMap, w, b, feat);
%save('usr_tst_newFeat', 'usr_tst_newFeat');

%So we load the precomputed features.
load('./data/usr_newFeat');
load('./data/usr_tst_newFeat');
%find labels
fid = fopen('./data/usrs_tr_1k1gram.csv');
format = '%s %f %s %s %f';
C = textscan(fid,format,'delimiter', ',','CommentStyle','id,'); 
usrs_tr_10_1k1gram_id = C{2};
usrs_10_1k1gram_feat = C{3};
usrs_tr_10_1k1gram_norm = C{5}; 
fclose(fid);
fid = fopen('./data/usrs_tst_1k1gram.csv');
format = '%s %f %s %s %f';
C = textscan(fid,format, 400000,'delimiter', ',','CommentStyle','id,'); 
usrs_tsts_10_1k1gram_id = C{2};
usrs_tst_10_1k1gram_feat = C{3};
usrs_tst_10_1k1gram_norm = C{5}; 
fclose(fid);
trLbs = findUserGenders(usrs_tr_10_1k1gram_id);
tstLbs = findUserGenders(usrs_tsts_10_1k1gram_id);

[traccuracy,tstccuracy] = predict_gender(trLbs,sparse(usr_newFeat),tstLbs,sparse(usr_tst_newFeat),0);
