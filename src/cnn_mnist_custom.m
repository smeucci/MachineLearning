function [net, info] = cnn_mnist_custom(varargin)
%MAIN Summary of this function goes here
%   Detailed explanation goes here

setup;

opts.expDir = fullfile('data', ['mnist-baseline']) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data', 'mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts.dataDir = fullfile('data','mnist') ;
opts.expDir = fullfile('data','mnist-baseline') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 100 ;
opts.train.numEpochs = 30 ;
opts.train.continue = true ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;


% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

net = initializeCNN() ;

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts.dataDir) ;
  if ~exist(opts.expDir, 'dir')
      mkdir(opts.expDir) ;
  end
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),0:9,'UniformOutput',false) ;


% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train(net, imdb, @getBatch, ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 2)) ;

% Save the result for later use
net.layers{end} = struct('type', 'softmax') ;
net = vl_simplenn_tidy(net);
save([opts.expDir, '/mnist-cnn.mat'], '-struct', 'net') ;

end