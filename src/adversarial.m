function adv = adversarial(net, im, label, eps)
%ADVERSARIAL Summary of this function goes here
%   Detailed explanation goes here

dzdy = 1;
res = simplenn(net, im, [], []);
dzdx = vl_nnsoftmaxloss(res(end).x, label, dzdy);
res = simplenn(net, im, dzdx, res, 'skipForward', true);

adv = res(1).x + eps*sign(res(1).dzdx);

end

