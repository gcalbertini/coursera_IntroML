function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); %m x 1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

%theta dimensions := S_(j+1) x ((S_j)+1)
%Theta1 := 25 x 401
%Theta2 := 10 x 26

% Theta1:
%     Row 1:= theta for all nodes from input layer->1st node of hidden layer
%     Row 2:= theta for all nodes from input layer->2nd node of hidden layer
%     ...
%     Column 1:= theta for input layer 1st node->all nodes from hidden layer
%     Column 2:= theta for input layer 2nd node->all nodes from hidden layer
%     ...
%
% Theta2:
%     1st row indicates: theta for all nodes from hidden layer->1st node of output layer
%     2nd row indicates: theta for all nodes from hidden layer->2nd node of output layer
%     ...
%     1st Column indicates: theta for hidden layer 1st node->all nodes from output layer
%     2nd Column indicates: theta for hidden layer 1st node->all nodes from output layer
%     ...
%Rows (m) are image sample count, columns (n) are features in each
%sample...

%X is a1 with bias node
a1 = [ones(m,1) X]; %m x num_features+1, 5000 x 401

z2 = a1*Theta1'; %5000 x 25
a2 = sigmoid(z2);%5000 x 25
a2 =  [ones(size(a2,1),1) a2]; %Add hidden layer bias node, 5000 x 26

z3 = a2*Theta2'; %5000 x 10
a3 = sigmoid(z3);%5000 x 10

[prob, p] = max(a3,[],2); %max element in each row corr. greatest probability of chosen index for given sample

% =========================================================================


end
