function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta


% New matrix is matrix product of theta and each row of matrix X, i.e. X * theta
ThetaTransposeX = X * theta;

% Calculate hypothesis for each entry (i.e. plug each sample into sigmoid to check what result is vs. what it should be (y))
h = sigmoid(ThetaTransposeX);

% Calculate logistic regression cost, y is vector of all correct results, h is a vector of all hypothesized results
J = 1/m * (sum(-y.*log(h) - (1 - y).*log(1 - h)));

% Calculate gradient for all values of theta (i.e. partial derivatives of cost function)
grad = 1/m .* (X.' *  (h - y)); 



% =============================================================

end
