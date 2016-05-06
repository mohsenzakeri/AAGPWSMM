function usr_newFeat = generateNewUserFeat(fileName, usrMsgMap, weights, b, feat) 

epsilon = 0.000001;

usr = unique(usrMsgMap(:, 1));
usr = sort(usr);
msg = sort(usrMsgMap(:, 2));

fid = fopen(fileName);
blocksize = 1000000;
format = '%s %f %s %s %f';
remainingMsgids = [];
remainingOneGFeats = [];
remainingNorms = [];

usrGender = zeros(length(usr), length(weights));

loopCounter = 1;
disp(['iter' ',duration(s)'])
while ~feof(fid)
    tic
    C = textscan(fid,format, blocksize,'CommentStyle','id,', 'delimiter', ','); 
    msgid = C{2};
    oneGFeat = C{3};
    norm = C{5};
    
    msgid = [remainingMsgids; msgid];
    oneGFeat = [remainingOneGFeats; oneGFeat];
    norm = [remainingNorms; norm];
    lastMsgIdIdxs = find(msgid == msgid(length(msgid)));
    remainingMsgids = msgid(lastMsgIdIdxs, :);
    remainingOneGFeats = oneGFeat(lastMsgIdIdxs, :);
    remainingNorms = norm(lastMsgIdIdxs, :);
    
    notLastMsgId = setdiff(1:length(msgid), lastMsgIdIdxs);
    msgid = msgid(notLastMsgId, :);
    oneGFeat = oneGFeat(notLastMsgId, :);
    norm = norm(notLastMsgId, :);        
    
    [~, msgIdx] = ismember(msgid, msg);
    [~, oneGFeatIdx] = ismember(oneGFeat, feat);
    %msgid = msgid(msgIdx>0);
    %norm = norm(msgIdx>0);
    %oneGFeatIdx = oneGFeatIdx(msgIdx>0);
    %msgIdx = msgIdx(msgIdx>0);
    % declaring size of the sparse matrix
    
    [~, ia, ~] = unique([msgIdx, oneGFeatIdx], 'rows');
    uniqueRows = sort(ia);
    msgid = msgid(uniqueRows);
    msgIdx = msgIdx(uniqueRows);
    oneGFeatIdx = oneGFeatIdx(uniqueRows);
    norm = norm(uniqueRows);
    
    
    m = max(msgIdx);
    n = length(feat);
    tstX = sparse(msgIdx, oneGFeatIdx, norm, m, n);
    
    %%Start of Model Call
    %%Here You Generate the predicted Y values for the list of sparse messages
    tstY = tstX * weights + b;
    tstY(tstY == 0) = epsilon;
    score = 1./tstY;
%    score = abs(tstY);
    sparse_score = sparse(msgIdx, oneGFeatIdx, score(msgIdx), m, n);
    val = tstX .* sparse_score;
    %%End of Model Call
    
    [~, currUsrMsgIdxs]=ismember(msgid, usrMsgMap(:, 2));
    [~, usrIdx] = ismember(usrMsgMap(currUsrMsgIdxs, 1), usr);
    [groups,~,grpMems] = unique(usrIdx);
    genderSum = zeros(length(groups), size(val, 2));
    for c=1:size(val, 2)
        [~, ~, x] = find(val(:, c));
        genderSum(1:max(grpMems(oneGFeatIdx == c)), c) = accumarray(grpMems(oneGFeatIdx == c),x);
    end
    usrGender(groups, :) = usrGender(groups, :) + genderSum;

    disp([num2str(loopCounter) ',' num2str(ceil(toc))]);
    loopCounter = loopCounter + 1;
end
fclose(fid);

usr_newFeat = usrGender./repmat(sum(usrGender, 2), 1, size(usrGender, 2));
usr_newFeat = usr_newFeat(~isnan(usr_newFeat(:, 1)), :);

end
