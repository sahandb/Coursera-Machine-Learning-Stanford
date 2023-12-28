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

h = sigmoid(X*theta);

% Calculate unregularized cost
% using vector multiplication. The size of each argument is (m x 1),
% and we want the vector product to be a scalar,
% so use a transposition so that (1 x m) times (m x 1) gives a result of (1 x 1), a scalar
unreg_cost = ((-y)'*log(h) - (1-y)'*log(1-h))/m;

theta(1) = 0;

reg_cost = (lambda / (2 * m)) * (theta'*theta);

J = unreg_cost + reg_cost;

% calculate grads
grad = (X'*(h - y) + lambda*theta)/m;

% =============================================================

end
