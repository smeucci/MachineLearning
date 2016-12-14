function res = backward(net, res, dzdy, type, opts)
%BACKWARD Summary of this function goes here
%   Detailed explanation goes here


if strcmp(type, 'mixed')
    net.layers{end}.class = [ net.layers{end}.class,  net.layers{end}.class];
end


if opts.doder
  res(opts.n+1).dzdx = dzdy ;
  for i=opts.n:-1:opts.backPropLim
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type

      case 'conv'
        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
          vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
                      'pad', l.pad, ...
                      'stride', l.stride, ...
                      l.opts{:}, ...
                      opts.cudnnValue{:}) ;

      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                    'pad', l.pad, 'stride', l.stride, ...
                    'method', l.method, ...
                    l.opts{:}, ...
                    opts.cudnnValue{:}) ;

      case 'softmax'
        res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;

      case 'loss'
        res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;

      case 'softmaxloss'
         res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;

      case 'relu'
        if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
        if ~isempty(res(i).x)
          res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
        end

      case 'dropout'
        if testMode
          res(i).dzdx = res(i+1).dzdx ;
        else
          res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, ...
                                     'mask', res(i+1).aux) ;
        end

    end % layers

    switch l.type
      case {'conv', 'convt', 'bnorm'}
        if ~opts.accumulate
          res(i).dzdw = dzdw ;
        else
          for j=1:numel(dzdw)
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
        end
        dzdw = [] ;
    end
    if opts.conserveMemory && ~net.layers{i}.precious && i ~= opts.n
      res(i+1).dzdx = [] ;
      res(i+1).x = [] ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
  if i > 1 && i == opts.backPropLim && opts.conserveMemory && ~net.layers{i}.precious
    res(i).dzdx = [] ;
    res(i).x = [] ;
  end
end

end
