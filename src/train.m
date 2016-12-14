function [net, info] = train(varargin)
%CNN_MNIST_TRAIN Summary of this function goes here
%   Detailed explanation goes here


%% Setup the environment
setup;

%% Custom parameters
opts.type = 'standard'; % other option are 'adversarial' and 'mixed'
opts.eps = 0.1;
opts.alfa = 0.5;
opts.epochs = 30;


%% Parsing the training parameters
opts.expDir = fullfile('data', 'mnist-baseline') ;
opts.dataDir = fullfile('data', 'mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts.dataDir = fullfile('data','mnist') ;
opts.expDir = fullfile('data','mnist-baseline') ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
opts.train.batchSize = 100 ;
opts.train.continue = true ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;
opts.train.numEpochs = opts.epochs ;


%%  Prepare the data

net = initializeCNN() ;

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts.dataDir) ;
  if ~exist(opts.expDir, 'dir'),  mkdir(opts.expDir) ; end
  if ~exist(opts.dataDir, 'dir'),  mkdir(opts.dataDir) ; end
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),0:9,'UniformOutput',false) ;


%% Training phase

[net, info] = cnn_train(net, imdb, @getBatch, ...
      'expDir', opts.expDir, ...
      net.meta.trainOpts, ...
      opts.train, ...
      'val', find(imdb.images.set == 2), ...
      'type', opts.type, ...
      'eps', opts.eps, ...
      'alfa', opts.alfa) ;


%% Save the result for later use

net.layers{end} = struct('type', 'softmax') ;
net = vl_simplenn_tidy(net);
save([opts.expDir, '/mnist-cnn.mat'], '-struct', 'net') ;


end