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

%   theta:(n+1) x 1
%   X:        m x (n+1)
%   h,y:      m x 1
%   grad: (n+1) x 1

h = sigmoid(X*theta);
%Skip bias term for calculation at position 0 (index 1)
J = (1/m)*sum((-y.*log(h))-((1-y).*log(1-h))) + lambda/(2*m)*sum(theta(2:size(theta)).^2);
%Separate case for theta0 bias term so not to regularize theta1 (replace), others follow pattern
%so calculate in parallel:
grad = X'*(h - y)/m + lambda*theta/m;
%do not regularize theta1 (theta0 bias term is replaced here)
grad(1,:) = grad(1,:) - lambda*theta(1,:)/m;
% =============================================================

end
