function genders = findUserGenders(userids)
    fid = fopen('./data/ml.usr.age.gender.csv');
    format = '%f %f %f';
    C = textscan(fid,format, 1000000,'delimiter', '\t','CommentStyle','user'); 
    user_gend = [C{1},C{3}];
    [~, gendIdx] = ismember(userids,user_gend(:,1));
    genders = user_gend(gendIdx,2);
end