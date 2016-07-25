function [correct, error, meanScore] = computeStatistics(labels, predictedLabels, predictionScores)
%COMPUTESTATISTICS Summary of this function goes here
%   Detailed explanation goes here

start_idx = 60000;
success = 0;
failure = 0;

for i = 1:size(predictedLabels, 2)
    
    if predictedLabels(i) == labels(start_idx + i);
        
        success = success + 1;
        
    else
        
        failure = failure + 1;
        
    end
    
end

correct = success / size(predictedLabels, 2);
error = failure / size(predictedLabels, 2);
meanScore = mean(predictionScores);

end

