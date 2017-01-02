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
defaultModel = 'standard';
defaultEpsilon = 0.1;
defaultVerbose = true;
addOptional(p, 'NumTesting', defaultNumTesting, @(x) isnumeric(x));
addOptional(p, 'Type', defaultType, @(x) ischar(x));
addOptional(p, 'Model', defaultModel, @(x) ischar(x));
addOptional(p, 'Epsilon', defaultEpsilon, @(x) isnumeric(x));
addOptional(p, 'Verbose', defaultVerbose, @(x) islogical(x));
parse(p, varargin{:});
num_testing = p.Results.NumTesting;
type = p.Results.Type;
model = p.Results.Model;
epsilon = p.Results.Epsilon;
verbose = p.Results.Verbose;

assert(verbose == true || verbose == false, 'Error: Verbose must be a boolean');
assert(num_testing <= defaultNumTesting, 'Error: NumTesting must NOT be greater than 10000.');
assert(num_testing > 0, 'Error: NumTesting must be greater than 0.');
assert(strcmp(type, 'normal') == 1 || strcmp(type, 'adversarial') == 1, ...
    'Error: Type must be either "normal" or "adversarial."');


%% Parsing the training parameters
suffix = '';
if strcmp(model, 'standard')
    suffix = '-base';
elseif strcmp(model, 'mixed')
    suffix = '-mix';
elseif strcmp(model, 'adversarial')
    suffix = '-adv';
end
opts.expDir = fullfile('data', ['mnist-baseline', suffix]) ;
opts.dataDir = fullfile('data', 'mnist') ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
opts.train = struct() ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;


%% Load the net or train a new one
fprintf(' - Loading the net ...\n');
if exist([opts.expDir, '/mnist-cnn.mat'], 'file')
    net = load([opts.expDir, '/mnist-cnn.mat']);
else
    [net, info] = train('type', model);
end


%% Load the mnist dataset
fprintf(' - Loading the dataset ...\n');
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

images = imdb.images.data(:,:,:,imdb.images.set == 3);
labels = imdb.images.labels(imdb.images.set == 3);

images = images(:,:,:,1:num_testing);
labels = labels(1:num_testing);

if strcmp(type, 'adversarial')
    fprintf(' - Computing adversarial examples ...\n');
    images = adversarial(net, images, labels, epsilon);
end
fprintf(' - Classifying ...\n');
res = simplenn(net, images, [], []);

scores = squeeze(gather(res(end).x));
[predictionScores, predictedLabels] = max(scores);
diff = labels - predictedLabels;
correctlyPredictionScores = predictionScores(diff == 0);
nonCorrectlyPredictionScores = predictionScores(diff ~= 0);

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

fprintf('\n\n## Mnist Dataset Test##\n');
fprintf('---------------------------\n');
fprintf('Number of images: %d\n', num_testing);
fprintf('Model: %s\n', model);
fprintf('Type: %s\n', type);
fprintf('Correctly predicted: %.4f\n', correct);
fprintf('Error rate: %.4f\n', error);
fprintf('Average confidence: %.4f\n', meanScore);
fprintf('Average correct confidence: %.4f\n', meanCorrectScore);
fprintf('Average non correct confidence: %.4f\n', meanNonCorrectScore);
fprintf('---------------------------\n\n');

end
