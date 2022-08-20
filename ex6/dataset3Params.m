function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

eg = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
results = zeros(size(eg,1),3);
iteration = 0;

C1=C;
sigma1=sigma;
model1 = svmTrain(X, y, C1, @(x1, x2)gaussianKernel(x1, x2, sigma1));
predictions = svmPredict(model1, Xval);
error = mean(double(predictions ~= yval));
temp = error;

for i=1:size(eg,1)
    C1 = eg(i);
    for j=1:size(eg,1)
        iteration = iteration + 1;
        if C1==1 && sigma1==0.3
            error1 = temp;
        else
            sigma1 = eg(j);
            model1 = svmTrain(X, y, C1, @(x1, x2) gaussianKernel(x1, x2, sigma1));
            predictions = svmPredict(model1, Xval);
            error1 = mean(double(predictions ~= yval));
        end
        results(iteration,:) = [C1 sigma1 error1];
        if error1<=error
            error = error1;
            C = C1;
            sigma = sigma1;
        end
    end
end
results;
iteration;
% =========================================================================

end
