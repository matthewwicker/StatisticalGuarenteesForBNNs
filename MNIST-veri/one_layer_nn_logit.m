function logit = one_layer_nn_logit(W1,b1,W2,b2, X)
X = X(:);

logit = W2*max(W1*X + b1,0) + b2;

end