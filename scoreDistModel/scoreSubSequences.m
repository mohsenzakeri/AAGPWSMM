function scores = scoreSubSequences(ss_norm, usr_ss, user_log_model)

    fid = fopen('./data/1k1gram_feat.csv');
    format = '%s %f';
    C = textscan(fid,format,'delimiter', ',','CommentStyle','feat,freq'); 
    feat = C{1};
    fclose(fid);

    
    usrids = usr_ss(:,1);
    subids_ofuser = usr_ss(:,2);
    
    subid = ss_norm(:,1);
    oneGFeat = ss_norm(:,2);
    norm = ss_norm(:,3);


    m = max(subid);
    n = length(feat);
    tstX = sparse(subid, oneGFeat, norm, m, n); 

    [predicted_label, accuracy, prob_estimates] = predict(ones(size(tstX,1),1),tstX,user_log_model, ('-b 1') );

    [~, currUsrSubIdxs]=ismember(sort(unique(subid)), subids_ofuser);

    result_users = usrids(currUsrSubIdxs);
    result_subIds = sort(unique(subid));
    result_preds = prob_estimates(:,1);


    scores = [result_users,result_subIds,result_preds];
end