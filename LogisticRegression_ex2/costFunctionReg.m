function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Note our cost function and gradient are the same as non-regularized, only we have to factor in a lambda for all elements EXCEPT theta(1) to regularize (avoid overfitting). theta(1) is avoided since we don't need to worry about overfitting height
[J, grad] = costFunction(theta, X, y);

% first element of temp is made to 0 to avoid regularizing first theta
temp = theta;
temp(1) = 0;
J = J + ((lambda / (2*m) * sum(temp.^2)));
% repeat process for gradient calculation
temp = ((lambda / m) .* theta);
temp(1) = 0;
grad = grad + temp;


% =============================================================

end
