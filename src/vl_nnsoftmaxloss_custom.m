function  Y = vl_nnsoftmaxloss_custom(X,c,eps,alfa)
%VL_NNSOFTMAXLOSS_CUSTOM Summary of this function goes here
%   Detailed explanation goes here

dzdy = 1;

Ya = vl_nnsoftmaxloss(X, c);

dYa = vl_nnsoftmaxloss(X, c, dzdy);

Yb = vl_nnsoftmaxloss(X + eps*sign(dYa), c);

Y = alfa*Ya + (1 - alfa)*Yb;

end

