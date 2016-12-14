function res = forward(net, res, type, opts)
%FORWARD Summary of this function goes here
%   Detailed explanation goes here


if strcmp(type, 'mixed')
    net.layers{end}.class = [ net.layers{end}.class,  net.layers{end}.class];
end


for i=1:opts.n
  if opts.skipForward, break; end;
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        l.opts{:}, ...
        opts.cudnnValue{:}) ;
    
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
        'pad', l.pad, 'stride', l.stride, ...
        'method', l.method, ...
        l.opts{:}, ...
        opts.cudnnValue{:}) ;
    
    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;

    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;

    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
      
    case 'relu'
      if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
      res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;

    case 'dropout'
      if testMode
        res(i+1).x = res(i).x ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end

    otherwise
      error('Unknown layer type ''%s''.', l.type) ;
  end

 % optionally forget intermediate results
  needsBProp = opts.doder && i >= opts.backPropLim;
  forget = opts.conserveMemory && ~needsBProp ;
  if i > 1
    lp = net.layers{i-1} ;
    % forget RELU input, even for BPROP
    forget = forget && (~needsBProp || (strcmp(l.type, 'relu') && ~lp.precious)) ;
    forget = forget && ~(strcmp(lp.type, 'loss') || strcmp(lp.type, 'softmaxloss')) ;
    forget = forget && ~lp.precious ;
  end
  if forget
    res(i).x = [] ;
  end

  res(i).time = toc(res(i).time) ;
end

end
