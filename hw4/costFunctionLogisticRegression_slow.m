function [J, grad] = costFunctionLogisticRegression_slow(theta, X, y, lambda)
% costFunctionLogisticRegression Compute cost and gradient for logistic regression with regularization
%    [J, grad] = costFunctionLogisticRegression(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% number of training examples
n = length(y); 

% pre-allocate space for gradient
grad = zeros(size(theta));

% Logistic Regression Cost Function
J = (1/n)*sum(-y.*(log(sigmoid(X*theta))) -(1-y).*log(1-(sigmoid(X*theta)))) + (lambda/(2*n))*sum(theta(2:end).^2);

for i=1:length(theta)
    if i == 1 %no regularization on the constant term
        grad(i) = (1/n)*sum((sigmoid(X*theta)-y).*X(:,i));
    else
        grad(i) = (1/n)*sum((sigmoid(X*theta)-y).*X(:,i)) + lambda*theta(i)/n;
    end
end

end
