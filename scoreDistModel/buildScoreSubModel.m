function scoredUsers = buildScoreSubModel(scores,usr_ss)
    usrids = usr_ss(:,1);
    usr = sort(unique(usrids));
    sorted_scores = sortrows(scores,-3);

    for i=1:length(usr)
        ss_count = length(sorted_scores(sorted_scores(:,1)==usr(i),2));
        sorted_scores(sorted_scores(:,1)==usr(i),2) = [1:ss_count];
    end
    scoredUsers = sorted_scores;     
end