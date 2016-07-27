function main(varargin)
%MAIN Summary of this function goes here
%   Detailed explanation goes here

setup;

opts.expDir = fullfile('data', ['mnist-baseline']) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data', 'mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;


if exist('data/mnist-baseline/mnist-cnn.mat', 'file')
    net = load('data/mnist-baseline/mnist-cnn.mat');
else
    [net, info] = cnn_mnist_custom();
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts.dataDir) ;
  if ~exist(opts.expDir, 'dir')
      mkdir(opts.expDir) ;
  end
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Testing

idx = 1;
predictedLabels = zeros(1, 10000);
predictionScores = zeros(1, 10000);
for i = 1:size(imdb.images.data, 4)
    
    if imdb.images.set(i) == 3
        
        fprintf('test image: %d', idx);
        
        im = imdb.images.data(:,:,:, i);
        labels = imdb.images.labels(i);
        
        % run the CNN
        im = getAdversarial(net, im, labels, 0.1);
        res = vl_simplenn(net, im);
        
        scores = squeeze(gather(res(end).x));
        [bestScore, best] = max(scores);
        
        predictedLabels(idx) = best;
        predictionScores(idx) = bestScore;
        
        fprintf(' # label: %d - predicted: %d - score: %.4f\n', labels, best, bestScore);
        
        idx = idx + 1;
        
    end
    
end

[correct, error, meanScore] = computeStatistics(imdb.images.labels, predictedLabels, predictionScores);

fprintf('Correctly predicted %.4f\n', correct);
fprintf('Error rate %.4f\n', error);
fprintf('Average confidence %.4f\n', meanScore);

end

