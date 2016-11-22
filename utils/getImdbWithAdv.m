function [imdb] = getImdbWithAdv(imdb, net)
%GETIMDBWITHADV Summary of this function goes here
%   Detailed explanation goes here

images = cat(4, imdb.images.data(:,:,:,imdb.images.set == 1), ...
                imdb.images.data(:,:,:,imdb.images.set == 2));
            
labels = cat(2, imdb.images.labels(imdb.images.set == 1), ...
                imdb.images.labels(imdb.images.set == 2));

for i = 1:size(images, 4)
    if mod(i, 2)
        im = images(:,:,:,i);
        label = labels(i);
        im = getAdversarial(net, im, label, 0.1);
        images(:,:,:,i) = im;
        fprintf('%d\n', i);
    end
end

imdb.images.data = cat(4, images, imdb.images.data(:,:,:,imdb.images.set == 3));

end