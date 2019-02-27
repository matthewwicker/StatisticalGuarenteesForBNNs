function confidence = one_layer_nn_confidence(W1,b1,W2,b2, X)
X = X(:);

logit = W2*max(W1*X + b1,0) + b2;
confidence = 1/(1+ exp(-logit));

end