function labels = findUserGenders(userids)
    uniqueUsers = sort(unique(userids));
    fid = fopen('./data/ml.usr.age.gender.csv');
    format = '%f %f %f';
    C = textscan(fid,format, 1000000,'delimiter', '\t','CommentStyle','user'); 
    fclose(fid);
    user_gend = [C{1},C{3}];
    [~, gndIdx] = ismember(uniqueUsers,user_gend(:,1));
    labels = user_gend(gndIdx,2);
end