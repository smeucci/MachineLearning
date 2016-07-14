function [images, labels] = getBatches(imdb, batch)
%GETBATCHES Summary of this function goes here
%   Detailed explanation goes here

images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

end