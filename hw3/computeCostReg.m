function J = computeCostReg(xData, y, theta, lambda)
%computeCostReg Compute cost for regularized linear regression
%   J = computeCostReg(xData, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in xData and y
% Input:
%   Xdata, size nxD
%   Theta, size Dx1
%   Y, size nx1 
%   lambda is the regularization coefficient
%       Where n is the number of samples, and D is the dimension 
%       of the sample plus 1 (the plus 1 accounts for the constant column)
% Output- J, the least squares cost
 
% Cost w/out regularization
%J = sum((xData*theta -y).^2) /(2*length(y));
%Cost w/regularization
J = sum((xData*theta -y).^2) /(2*length(y)) + (lambda/(2*length(y)))*sum(theta(2:end).^2);

end
