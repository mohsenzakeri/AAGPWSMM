function [C,accuracy] = TuneC(trLbs,trD,model,startIter,step,endIter)
    cValues = [startIter:step:endIter];
    models = zeros(numel(cValues),1);
    for i=1:numel(cValues)
        i
        models(i) = train(trLbs, trD,[sprintf('-s %f -c %f -v 5',model,cValues(i))]);
    end
    [accuracy,arg] = max(models);
    C = cValues(arg);                 
end        
