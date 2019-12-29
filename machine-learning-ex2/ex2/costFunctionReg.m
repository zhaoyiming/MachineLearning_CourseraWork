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



[J, grad] = costFunction(theta, X, y);
temp=0;
for iter=2:size(theta)
  temp=temp+theta(iter)^2;
endfor
J=J+lambda/(2*m)*temp;

%¼ÇÂ¼x
gr=zeros(m);
for iter = 1:m
  x=[X(iter,1),X(iter,2),X(iter,3)];
  z=transpose(theta)*transpose(x); 
  gr(iter)=sigmoid(z);
endfor
%theta1
sums=0;
for iter = 1:m
   sums=sums+(gr(iter)-y(iter))*X(iter,1);
endfor
grad(1)=1/m*sums;

%theta others
for iters=2:size(theta)
  sums=0;
  for iter = 1:m
     sums=sums+(gr(iter)-y(iter))*X(iter,iters)+lambda/m*theta(iters);
  endfor
  grad(iters)=1/m*sums;
endfor



% =============================================================

end
