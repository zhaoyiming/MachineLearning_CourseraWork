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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% 
% Note: grad should have the same dimensions as theta
%

z=sigmoid(X*theta);
J=(1*sum((-y).*log(z)-(1-y).*log(1-z))./m);
##sum=0;
##
##gr=zeros(m);
##
##for iter = 1:m
##  x=[X(iter,1),X(iter,2),X(iter,3)];
##  z=transpose(theta)*transpose(x);
##  
##  gr(iter)=sigmoid(z);
##  sum=sum+(-y(iter)*log(sigmoid(z))-(1-y(iter))*log(1-sigmoid(z)));
##    
##endfor
##J=1/m*sum;
##
##n=length(theta);
##
##for iters = 1:n
##  sums=0;
##  for iter = 1:m
##     sums=sums+(gr(iter)-y(iter))*X(iter,iters);
##  endfor
##  grad(iters)=1/m*sums;
##endfor
##




% =============================================================

end
