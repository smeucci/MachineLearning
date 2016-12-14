function results = test(varargin)
%MAIN Summary of this function goes here
%   Detailed explanation goes here


%% Setup the environment
setup;


%% Parsing the optional parameters
p = inputParser;
p.KeepUnmatched = true;
defaultNumTesting = 10000;
defaultType = 'normal';
defaultEpsilon = 0.1;
defaultVerbose = true;
addOptional(p, 'NumTesting', defaultNumTesting, @(x) isnumeric(x));
addOptional(p, 'Type', defaultType, @(x) ischar(x));
addOptional(p, 'Epsilon', defaultEpsilon, @(x) isnumeric(x));
addOptional(p, 'Verbose', defaultVerbose, @(x) islogical(x));
parse(p, varargin{:});
num_testing = p.Results.NumTesting;
type = p.Results.Type;
epsilon = p.Results.Epsilon;
verbose = p.Results.Verbose;

assert(verbose == true || verbose == false, 'Error: Verbose must be a boolean');
assert(num_testing <= defaultNumTesting, 'Error: NumTesting must NOT be greater than 10000.');
assert(num_testing > 0, 'Error: NumTesting must be greater than 0.');
assert(strcmp(type, 'normal') == 1 || strcmp(type, 'adversarial') == 1, ...
    'Error: Type must be either "normal" or "adversarial."');


%% Parsing the training parameters
opts.expDir = fullfile('data', 'mnist-baseline') ;
opts.dataDir = fullfile('data', 'mnist') ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
opts.train = struct() ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;


%% Load the net or train a new one
if exist('data/mnist-baseline/mnist-cnn.mat', 'file')
    net = load('data/mnist-baseline/mnist-cnn.mat');
else
    [net, info] = cnn_mnist_train();
end


%% Load the mnist dataset
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts.dataDir) ;
  if ~exist(opts.expDir, 'dir')
      mkdir(opts.expDir) ;
  end
  save(opts.imdbPath, '-struct', 'imdb') ;
end


%% Testing

predictedLabels = zeros(1, num_testing);
predictionScores = zeros(1, num_testing);
correctlyPredictionScores = zeros(1, num_testing);
nonCorrectlyPredictionScores = zeros(1, num_testing);

images = imdb.images.data(:,:,:,imdb.images.set == 3);
labels = imdb.images.labels(imdb.images.set == 3);

for i = 1:num_testing
   
    im = images(:,:,:,i);
    label = labels(i);

    % Compute the adversarial example if type equals 'adversarial'
    if strcmp(type, 'adversarial')
        im = getAdversarial(net, im, label, epsilon);
    end
    
    res = simplenn(net, im);

    scores = squeeze(gather(res(end).x));
    [bestScore, best] = max(scores);

    predictedLabels(i) = best;
    predictionScores(i) = bestScore;
    
    if label == best
        correctlyPredictionScores(i) = predictionScores(i);
    else
        nonCorrectlyPredictionScores(i) = predictionScores(i);
    end
    
    if verbose == true
        fprintf('test image: %d # label: %d - predicted: %d - score: %.4f\n', i, label, best, bestScore);
    end
    
end

correctlyPredictionScores = correctlyPredictionScores(correctlyPredictionScores ~= 0);
nonCorrectlyPredictionScores = nonCorrectlyPredictionScores(nonCorrectlyPredictionScores ~= 0);

%% Compute statistics about the prediction
[correct, error, meanScore, meanCorrectScore, meanNonCorrectScore] = computeStatistics(labels(1:num_testing), ...
    predictedLabels, predictionScores, correctlyPredictionScores, nonCorrectlyPredictionScores);


%% Output

results.numberOfImages = num_testing;
results.type = type;
if strcmp(type, 'adversarial') == 1
    results.epsilon = epsilon;
end
results.correctlyPredicted = correct;
results.errorRate = error;
results.averageConfidence = meanScore;
results.averageCorrectConfidence = meanCorrectScore;
results.averageNonCorrectConfidence = meanNonCorrectScore;
results.labels = labels(1:num_testing);
results.predictedLabels = predictedLabels;
results.scores = predictionScores;

fprintf('\n## Mnist Dataset Test##\n');
fprintf('---------------------------\n');
fprintf('Number of images: %d\n', num_testing);
fprintf('Type: %s\n', type);
if strcmp(type, 'adversarial') == 1
    fprintf('Epsilon: %.2f\n', epsilon);
end
fprintf('Correctly predicted: %.4f\n', correct);
fprintf('Error rate: %.4f\n', error);
fprintf('Average confidence: %.4f\n', meanScore);
fprintf('Average correct confidence: %.4f\n', meanCorrectScore);
fprintf('Average non correct confidence: %.4f\n', meanNonCorrectScore);
fprintf('---------------------------\n\n');

end
