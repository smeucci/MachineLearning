function [imdb] = getMnistImdb(dataDir)
%GETMNISTIMDB Summary of this function goes here
%   Detailed explanation goes here

files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(dataDir, 'dir')
  mkdir(dataDir) ;
end

% Load train set
train_images = loadMNISTImages([dataDir '/' files{1,1}]);
train_labels = loadMNISTLabels([dataDir '/' files{1,2}]);

train_images = reshape(train_images(:,:), 28, 28, 60e3);
train_labels = double(train_labels(:)') + 1;

% Load test set
test_images = loadMNISTImages([dataDir '/' files{1,3}]);
test_labels = loadMNISTLabels([dataDir '/' files{1,4}]);

test_images = reshape(test_images(:,:), 28, 28, 10e3);
test_labels = double(test_labels(:)') + 1;

set = [ones(1,numel(train_labels) - 5e3) 2*ones(1, 5e3) 3*ones(1,numel(test_labels))];

data = single(reshape(cat(3, train_images, test_images),28,28,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;
data = cellfun(@(x) remap(x, [min(min(x)), max(max(x))], [0, 1]), {data(:,:,1,:)}, 'UniformOutput', false);
data = data{1, 1};

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, train_labels, test_labels) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;

end