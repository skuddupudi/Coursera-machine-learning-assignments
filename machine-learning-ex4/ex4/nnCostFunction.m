function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% You need to return the following variables correctly 
J = 0;
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));


for i = 1:m,
% ====================== YOUR CODE HERE ======================
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% Instructions: You should complete the code by working through the
  x = X(i,1:input_layer_size);
  x = x';
  x = [1; x];
  z2 = (theta1 * x);
  a2 = sigmoid(z2);

  a2 = [1; a2];
  z3 = (theta2 *a2);
  a3 = sigmoid(z3);

  Y = zeros(num_labels,1);
  for j = 1:num_labels
    if(j==y(i, 1))
      Y(j,1) = 1;
    end
  end
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
  
  
  D3 = a3 - Y;
  D2 = theta2'*D3 .* (a2.*(1-a2));
  
  J = J + (-1/m)*(Y'*log(a3) + (1-Y)'*log(1-a3));
  
  theta2_grad = (theta2_grad + D3*a2');
  theta1_grad = (theta1_grad + D2(2:end, :)*x');
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
end

theta2_grad = theta2_grad/m;
theta1_grad = theta1_grad/m;

if lambda != 0
thetar1 = [zeros(size(theta1,1), 1) theta1(:, 2:end)];
thetar2 = [zeros(size(theta2,1), 1) theta2(:, 2:end)];
J = J + (lambda/(2*m))*(sum(sum(thetar1.*thetar1)) + sum(sum(thetar2.*thetar2)));

theta1_grad = (theta1_grad + (lambda/m)*thetar1);
theta2_grad = (theta2_grad + (lambda/m)*thetar2);
end;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [theta1_grad(:) ; theta2_grad(:)];


end
