function res = simplenn(net, x, dzdy, res, varargin)
%SIMPLENN Summary of this function goes here
%   Detailed explanation goes here


%% Setup %%

[res, opts] = simplenn_opts(net, x, dzdy, res, varargin{:});


%% Forward %%

if strcmp(opts.type, 'mixed') || strcmp(opts.type, 'adversarial')
    adv_res = res;
end

res = forward(net, res, 'standard', opts);
res = backward(net, res, dzdy, 'standard', opts);

if strcmp(opts.type, 'mixed') || strcmp(opts.type, 'adversarial')
    cln_res = res;
end

% Get adversarial examples
if strcmp(opts.type, 'mixed') || strcmp(opts.type, 'adversarial')
    
    adv_res(1).x = res(1).x + opts.eps*sign(res(1).dzdx);
    
    if strcmp(opts.type, 'mixed')
        adv_res(1).x = cat(4, res(1).x, adv_res(1).x);
    end
    
    adv_res = forward(net, adv_res, opts.type, opts);
    adv_res = backward(net, adv_res, dzdy, opts.type, opts);
    
    if strcmp(opts.type, 'adversarial')
        for i=1:opts.n
            if ~isempty(cln_res(i).dzdw) && ~isempty(adv_res(i).dzdw)
                res(i).dzdw{1} = opts.alfa*cln_res(i).dzdw{1} + (1 - opts.alfa)*adv_res(i).dzdw{1};
                res(i).dzdw{2} = opts.alfa*cln_res(i).dzdw{2} + (1 - opts.alfa)*adv_res(i).dzdw{2};
            end
        end  
    end
    
    if strcmp(opts.type, 'mixed')
       res = adv_res; 
    end
    
end


end

