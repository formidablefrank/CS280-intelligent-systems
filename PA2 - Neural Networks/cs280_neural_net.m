% Simple Neural Network Demo (CS280) by Pros Naval
clear;
% Define Architecture of NN
n_in = 3; %number of input features
n_h1 = 7; %number of neurons in hidden1
n_h2 = 5; %number of neurons in hidden2
n_out = 3; %number of output features

eta = 0.1; % Learning Rate

% Pre-allocate storage and initialize weights + biases
x_in = zeros(n_in,1); %zeros,rand(nrows,ncols), input vector
w_h1 = -0.1+(0.1+0.1)*rand(n_h1,n_in); % each row denotes weights coming to the node from previous layer
b_h1 = -0.1+(0.1+0.1)*rand(n_h1,1); % bias on that layer: each row denotes bias to that node
w_h2 = -0.1+(0.1+0.1)*rand(n_h2,n_h1);
b_h2 = -0.1+(0.1+0.1)*rand(n_h2,1);
w_out = -0.1+(0.1+0.1)*rand(n_out,n_h2);
b_out = -0.1+(0.1+0.1)*rand(n_out,1);
d_out = zeros(n_out,1); % output vector
% Training Data
N = 8;
X = [
0 0 0;
0 0 1;
0 1 0;
0 1 1;
1 0 0;
1 0 1;
1 1 0;
1 1 1]; %input vectors
Y = [
0 0 0;
1 1 0;
1 0 1;
0 1 1;
0 1 1;
1 0 0;
1 1 0;
0 0 0]; %desired outputs for each object above


max_epoch = 30000
% TRAINING PHASE
totalerr = zeros(max_epoch,1); % each row denotes error from each epoch
for q = 1:max_epoch
    p = randperm(N); % shuffle patterns e.g. {7,3,5,1,2,4,6,0}
    for n = 1:N
        nn = p(n); % get random number from p
        % read data
        x_in = X(nn,:)'; % input vector, X(row_no, all cols)
        d_out = Y(nn,:)'; % desired output vector
    % forward pass
        % hidden layer 1
        v_h1 = w_h1*x_in + b_h1; % dot product, matrix addition
        y_h1 = 1./(1+exp(-v_h1)); % vectorized function eval
        % hidden layer 2
        v_h2 = w_h2*y_h1 + b_h2;
        y_h2 = 1./(1+exp(-v_h2));
        % output layer
        v_out = w_out*y_h2 + b_out;
        out = 1./(1+exp(-v_out));
   % error backpropagation %
        % compute error
        err = d_out - out;
        % compute gradient in output layer
        delta_out = err.*out.*(1 - out);
        % compute gradient in hidden layer 2
        delta_h2 = y_h2.*(1-y_h2).*(w_out'*delta_out);
        % compute gradient in hidden layer 1
        delta_h1 = y_h1.*(1-y_h1).*(w_h2'*delta_h2);
        % update weights and biases in output layer
        w_out = w_out + eta.*delta_out*y_h2';
        b_out = b_out + eta.*delta_out;
        % update weights and biases in hidden layer 2
        w_h2 = w_h2 + eta.*delta_h2*y_h1';
        b_h2 = b_h2 + eta.*delta_h2;
        % update weights and biases in hidden layer 1
        w_h1 = w_h1 + eta.*delta_h1*x_in';
        b_h1 = b_h1 + eta.*delta_h1;
    end
    totalerr(q) = totalerr(q) + sum(err.*err);
    if mod(q,500) == 0
        fprintf('iteration: %d Error: %f\n',q,totalerr(q));
    end
    % if termination condition is satisfied save weights and exit
    if totalerr(q) < 0.001
        break
    end
end
% TEST PHASE
nn_output = zeros(size(Y));
for n = 1:N
    % read data
    x_in = X(n,:)';
    d_out = Y(n,:)';
    % hidden layer 1
    v_h1 = w_h1*x_in + b_h1;
    y_h1 = 1./(1+exp(-v_h1));
    % hidden layer 2
    v_h2 = w_h2*y_h1 + b_h2;
    y_h2 = 1./(1+exp(-v_h2));
    % output layer
    v_out = w_out*y_h2 + b_out;
    out = 1./(1+exp(-v_out));
    nn_output(n,:) = ge(out',0.5);
    [x_in' nn_output(n,:)]
end

fprintf('Total Bits with error: %d\n', sum(sum(abs(Y-nn_output))));
fprintf('Total epochs: %d\n',q);
fprintf('Network Error at termination: %f\n',totalerr(q));
figure;
plot(totalerr(1:q));

% Things to try
% 1) Do not shuffle the patterns. Observe the error curve. Did the network learn ?
% 2) Change learning rate to 0.5 or 0.01. Observe the error curve. Did the network learn ?
% 3) Change the architecture (e.g. 3-2-2-3; 3-20-35-3)
