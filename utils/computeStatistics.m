function [correct, error, meanScore, meanCorrectScore, meanNonCorrectScore] = computeStatistics(labels, predictedLabels, ...
            predictionScores, correctlyPredictionScores, nonCorrectlyPredictionScores)
%COMPUTESTATISTICS Summary of this function goes here
%   Detailed explanation goes here

diff = predictedLabels == labels;

correct = sum(diff == 1) / size(predictedLabels, 2);
error = sum(diff == 0) / size(predictedLabels, 2);
meanScore = mean(predictionScores);
meanCorrectScore = mean(correctlyPredictionScores);
meanNonCorrectScore = mean(nonCorrectlyPredictionScores);

end