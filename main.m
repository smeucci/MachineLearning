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


% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

net = initializeCNN() ;

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;


% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train(net, imdb, @getBatch, ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imdb.images.data_mean ;
save([opts.expDir, '/mnist-cnn.mat'], '-struct', 'net') ;

end