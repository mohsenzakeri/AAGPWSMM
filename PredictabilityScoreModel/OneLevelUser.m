fid = fopen('./data/1k1gram_feat.csv');
format = '%s %f';
C = textscan(fid,format,'delimiter', ',','CommentStyle','feat,freq'); 
feat = C{1};
freq = C{2};
fclose(fid);

fid = fopen('./data/usrs_tr_1k1gram.csv');
format = '%s %f %s %s %f';
C = textscan(fid,format,'delimiter', ',','CommentStyle','id,'); 
usrs_tr_10_1k1gram_id = C{2};
usrs_10_1k1gram_feat = C{3};
usrs_tr_10_1k1gram_norm = C{5}; 
fclose(fid);

[~, featIdxs] = ismember(usrs_10_1k1gram_feat, feat(:, 1));

load('usrMsgMap');
usr = unique(usrMsgMap(:, 1));
usr = sort(usr);

uniqueUsers_tr = sort(unique(usrs_tr_10_1k1gram_id));
[~, usrIdx] = ismember(usrs_tr_10_1k1gram_id,uniqueUsers_tr);
trainFeat = sparse(usrIdx, featIdxs, usrs_tr_10_1k1gram_norm);
tr_genders = findUserGenders(usrs_tr_10_1k1gram_id);

%%%Now reading tst user
fid = fopen('./data/usrs_tst_1k1gram.csv');
format = '%s %f %s %s %f';
C = textscan(fid,format,'delimiter', ',','CommentStyle','id,'); 
usrs_tsts_10_1k1gram_id = C{2};
usrs_tst_10_1k1gram_feat = C{3};
usrs_tst_10_1k1gram_norm = C{5}; 
fclose(fid);

[~, tstfeatIdxs] = ismember(usrs_tst_10_1k1gram_feat, feat(:, 1));
uniquetUsers_tst = sort(unique(usrs_tsts_10_1k1gram_id));
[~, usrtstIdx] = ismember(usrs_tsts_10_1k1gram_id,uniquetUsers_tst);
testFeat = sparse(usrtstIdx, tstfeatIdxs, usrs_tst_10_1k1gram_norm);
tst_genders = findUserGenders(usrs_tsts_10_1k1gram_id);

[traccuracy,tstccuracy] = predict_gender(tr_genders,trainFeat,tst_genders,testFeat,0);
