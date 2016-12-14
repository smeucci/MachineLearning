function [res, opts] = simplenn_opts(net, x, dzdy, res, varargin)
%SIMPLENN_OPTS Summary of this function goes here
%   Detailed explanation goes here


% Custom parameters
opts.type = 'standard'; % other option are 'adversarial' and 'mixed'
opts.eps = 0.1;
opts.alfa = 0.5;

%Training parameters
opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;
opts = vl_argparse(opts, varargin);

opts.n = numel(net.layers) ;
assert(opts.backPropDepth > 0, 'Invalid `backPropDepth` value (!>0)');
opts.backPropLim = max(opts.n - opts.backPropDepth + 1, 1);

if (nargin <= 2) || isempty(dzdy)
  opts.doder = false ;
  if opts.skipForward
    error('simplenn:skipForwardNoBackwPass', ...
      '`skipForward` valid only when backward pass is computed.');
  end
else
  opts.doder = true ;
end

if opts.cudnn
  opts.cudnnValue = {'CuDNN'} ;
else
  opts.cudnnValue = {'NoCuDNN'} ;
end

switch lower(opts.mode)
  case 'normal'
    opts.testMode = false ;
  case 'test'
    opts.testMode = true ;
  otherwise
    error('Unknown mode ''%s''.', opts. mode) ;
end

if nargin <= 3 || isempty(res)
  if opts.skipForward
    error('simplenn:skipForwardEmptyRes', ...
    'RES structure must be provided for `skipForward`.');
  end
  res = struct(...
    'x', cell(1,opts.n+1), ...
    'dzdx', cell(1,opts.n+1), ...
    'dzdw', cell(1,opts.n+1), ...
    'aux', cell(1,opts.n+1), ...
    'stats', cell(1,opts.n+1), ...
    'time', num2cell(zeros(1,opts.n+1)), ...
    'backwardTime', num2cell(zeros(1,opts.n+1))) ;
end

if ~opts.skipForward
  res(1).x = x ;
end

end
