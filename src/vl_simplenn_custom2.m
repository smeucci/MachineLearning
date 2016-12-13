function res = vl_simplenn_custom2(net, x, dzdy, res, varargin)
%VL_SIMPLENN_CUSTOM Summary of this function goes here
%   Detailed explanation goes here


%% Setup %%
%%-------%%

opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;
opts.eps = 0.1;
opts.alfa = 0.5;
opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;
assert(opts.backPropDepth > 0, 'Invalid `backPropDepth` value (!>0)');
backPropLim = max(n - opts.backPropDepth + 1, 1);

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
  if opts.skipForward
    error('simplenn:skipForwardNoBackwPass', ...
      '`skipForward` valid only when backward pass is computed.');
  end
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

switch lower(opts.mode)
  case 'normal'
    testMode = false ;
  case 'test'
    testMode = true ;
  otherwise
    error('Unknown mode ''%s''.', opts. mode) ;
end

if nargin <= 3 || isempty(res)
  if opts.skipForward
    error('simplenn:skipForwardEmptyRes', ...
    'RES structure must be provided for `skipForward`.');
  end
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'stats', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;

  cln_res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'stats', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;

  adv_res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'stats', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;

end

if ~opts.skipForward
  res(1).x = x ;
  cln_res(1).x = x;
  
  adv_x = x;
  for h = 1:size(x, 4)
      adv_x(:,:,1,h) = getAdversarial(net, adv_x(:,:,1,h), net.layers{end}.class(h), opts.eps);
  end
  
  adv_res(1).x = adv_x;
  
end


%% Forward pass %%
%%--------------%%

for i=1:n
  if opts.skipForward, break; end;
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        l.opts{:}, ...
        cudnn{:}) ;
    
      cln_res(i+1).x = res(i+1).x;
    
      adv_res(i+1).x = vl_nnconv(adv_res(i).x, l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        l.opts{:}, ...
        cudnn{:}) ;
    
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
        'pad', l.pad, 'stride', l.stride, ...
        'method', l.method, ...
        l.opts{:}, ...
        cudnn{:}) ;
    
      cln_res(i+1).x = res(i+1).x;
    
      adv_res(i+1).x = vl_nnpool(adv_res(i).x, l.pool, ...
        'pad', l.pad, 'stride', l.stride, ...
        'method', l.method, ...
        l.opts{:}, ...
        cudnn{:}) ;
    
    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;
      cln_res(i+1).x = res(i+1).x;
      adv_res(i+1).x = vl_nnsoftmax(adv_res(i).x) ;

    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;
      cln_res(i+1).x = res(i+1).x;
      adv_res(i+1).x = vl_nnloss(adv_res(i).x, l.class) ;

    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
      cln_res(i+1).x = res(i+1).x;
      adv_res(i+1).x = vl_nnsoftmaxloss(adv_res(i).x, l.class) ;
      
    case 'relu'
      if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
      res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;
      cln_res(i+1).x = res(i+1).x;
      adv_res(i+1).x = vl_nnrelu(adv_res(i).x,[],leak{:}) ;

    case 'dropout'
      if testMode
        res(i+1).x = res(i).x ;
        cln_res(i+1).x = res(i+1).x;
        adv_res(i+1).x = adv_res(i).x ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
        cln_res(i+1).x = res(i+1).x;
        cln_res(i+1).aux = res(i+1).aux;
        [adv_res(i+1).x, adv_res(i+1).aux] = vl_nndropout(adv_res(i).x, 'rate', l.rate) ;
      end

    otherwise
      error('Unknown layer type ''%s''.', l.type) ;
  end

 % optionally forget intermediate results
  needsBProp = doder && i >= backPropLim;
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


%% Backward pass %%
%%---------------%%

if doder
  res(n+1).dzdx = dzdy ;
  cln_res(n+1).dzdx = dzdy;
  adv_res(n+1).dzdx = dzdy ;
  for i=n:-1:backPropLim
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type

      case 'conv'
        [cln_res(i).dzdx, cln_dzdw{1}, cln_dzdw{2}] = ...
          vl_nnconv(cln_res(i).x, l.weights{1}, l.weights{2}, cln_res(i+1).dzdx, ...
          'pad', l.pad, ...
          'stride', l.stride, ...
          l.opts{:}, ...
          cudnn{:}) ;
      
        [adv_res(i).dzdx, adv_dzdw{1}, adv_dzdw{2}] = ...
          vl_nnconv(adv_res(i).x, l.weights{1}, l.weights{2}, adv_res(i+1).dzdx, ...
          'pad', l.pad, ...
          'stride', l.stride, ...
          l.opts{:}, ...
          cudnn{:}) ;
      
        res(i).dzdx = opts.alfa*cln_res(i).dzdx + (1 - opts.alfa)*adv_res(i).dzdx;
        dzdw{1} = opts.alfa*cln_dzdw{1} + (1 - opts.alfa)*adv_dzdw{1};
        dzdw{2} = opts.alfa*cln_dzdw{2} + (1 - opts.alfa)*adv_dzdw{2};

      case 'pool'
        cln_res(i).dzdx = vl_nnpool(cln_res(i).x, l.pool, cln_res(i+1).dzdx, ...
                    'pad', l.pad, 'stride', l.stride, ...
                    'method', l.method, ...
                    l.opts{:}, ...
                    cudnn{:}) ;
         
        adv_res(i).dzdx = vl_nnpool(adv_res(i).x, l.pool, adv_res(i+1).dzdx, ...
                    'pad', l.pad, 'stride', l.stride, ...
                    'method', l.method, ...
                    l.opts{:}, ...
                    cudnn{:}) ;
                
        res(i).dzdx = opts.alfa*cln_res(i).dzdx + (1 - opts.alfa)*adv_res(i).dzdx; 

      case 'softmax'
        cln_res(i).dzdx = vl_nnsoftmax(cln_res(i).x, cln_res(i+1).dzdx) ;
        adv_res(i).dzdx = vl_nnsoftmax(adv_res(i).x, adv_res(i+1).dzdx) ;
        res(i).dzdx = opts.alfa*cln_res(i).dzdx + (1 - opts.alfa)*adv_res(i).dzdx;

      case 'loss'
        cln_res(i).dzdx = vl_nnloss(cln_res(i).x, l.class, cln_res(i+1).dzdx) ;
        adv_res(i).dzdx = vl_nnloss(adv_res(i).x, l.class, adv_res(i+1).dzdx) ;
        res(i).dzdx = opts.alfa*cln_res(i).dzdx + (1 - opts.alfa)*adv_res(i).dzdx;

      case 'softmaxloss'
        cln_res(i).dzdx = vl_nnsoftmaxloss(cln_res(i).x, l.class, cln_res(i+1).dzdx) ;
        adv_res(i).dzdx = vl_nnsoftmaxloss(adv_res(i).x, l.class, adv_res(i+1).dzdx) ;
        res(i).dzdx = opts.alfa*cln_res(i).dzdx + (1 - opts.alfa)*adv_res(i).dzdx;

      case 'relu'
        if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
        if ~isempty(res(i).x)
          cln_res(i).dzdx = vl_nnrelu(cln_res(i).x, cln_res(i+1).dzdx, leak{:}) ;
          adv_res(i).dzdx = vl_nnrelu(adv_res(i).x, adv_res(i+1).dzdx, leak{:}) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          cln_res(i).dzdx = vl_nnrelu(cln_res(i+1).x, cln_res(i+1).dzdx, leak{:}) ;
          adv_res(i).dzdx = vl_nnrelu(adv_res(i+1).x, adv_res(i+1).dzdx, leak{:}) ;
        end
        res(i).dzdx = opts.alfa*cln_res(i).dzdx + (1 - opts.alfa)*adv_res(i).dzdx;

      case 'dropout'
        if testMode
          cln_res(i).dzdx = cln_res(i+1).dzdx ;
          adv_res(i).dzdx = adv_res(i+1).dzdx ;
        else
          cln_res(i).dzdx = vl_nndropout(cln_res(i).x, cln_res(i+1).dzdx, ...
                                     'mask', cln_res(i+1).aux) ;
          adv_res(i).dzdx = vl_nndropout(adv_res(i).x, adv_res(i+1).dzdx, ...
                                     'mask', adv_res(i+1).aux) ;
        end
        res(i).dzdx = opts.alfa*cln_res(i).dzdx + (1 - opts.alfa)*adv_res(i).dzdx;

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
    if opts.conserveMemory && ~net.layers{i}.precious && i ~= n
      res(i+1).dzdx = [] ;
      res(i+1).x = [] ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
  if i > 1 && i == backPropLim && opts.conserveMemory && ~net.layers{i}.precious
    res(i).dzdx = [] ;
    res(i).x = [] ;
  end
end


end