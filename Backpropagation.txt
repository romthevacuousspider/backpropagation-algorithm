%number of nodes
NHIDDENS = 3;
NINPUTS = 2;
NOUTPUTS = 1;

% weights input to hidden
in_to_hidden = zeros(NINPUTS, NHIDDENS);
in_to_hidden(1,1) = 0.1;in_to_hidden(1,2) = 0.2;in_to_hidden(1,3) = 0;
in_to_hidden(2,1) = 0;in_to_hidden(2,2) = -0.1; in_to_hidden(2,3) = 0.1;

%weights hidden to out
hidden_to_out =zeros(NHIDDENS, NOUTPUTS);
hidden_to_out(1,1) = -0.1;hidden_to_out(2,1) = 0.15; hidden_to_out(3,1) = -0.2;

%weights input to output
in_to_output = zeros(NINPUTS, NOUTPUTS);
in_to_output(1,1) = 0.25; in_to_output(2,1) = 0.2;

% Hyperparameters
eta = 0.2;

% X1, X2, Target_Out
for cycle = 1:2
    % Input - Output
    if cycle == 1
        Xes = [0 1];
        target_output = 1;
    elseif cycle == 2
        Xes = [1 0];
        target_output = 1;
    end
    
    % Forward pass
    NetIn_hidden = zeros(1, NHIDDENS);
    
    %Input to hidden 
    % y1, y2, y3
    for i = 1:NINPUTS
        for j = 1:NHIDDENS
            NetIn_hidden(j) = NetIn_hidden(j) + Xes(i) * in_to_hidden(i, j);
        end
    end
    
    HiddenLayerOut = 1./(1 + exp(-NetIn_hidden));
    
    %Hidden to Out
    % y1*w9 + y2*w8 + y3*w7
    NetIn_output_hidden = zeros(1, NHIDDENS);
    for k = 1:NHIDDENS
        NetIn_output_hidden(k) = HiddenLayerOut(k) * hidden_to_out(k);
    end

    %input to Out
    % x1*w2 + x2*w5
    NetIn_output_direct = zeros(1, NINPUTS);
    for i = 1:NINPUTS
        NetIn_output_direct(i) = Xes(i) * in_to_output(i);
    end
    
    %y_out
    NetIn_output = sum(NetIn_output_hidden) + sum(NetIn_output_direct);
    NetOut = NetIn_output;
    
    % Backward pass
    error = target_output - NetOut;
    
    % beta 1 , beta2 , beta 3
    beta_hidden = zeros(1, NHIDDENS);
    beta_output = error;
    
    for j = 1:NHIDDENS
        beta_hidden(j) = beta_output * hidden_to_out(j) * HiddenLayerOut(j) * (1 - HiddenLayerOut(j));
    end
    
    % Update weights for hidden to output connections
    for k = 1:NHIDDENS
        hidden_to_out(k) = hidden_to_out(k) + eta * beta_output * HiddenLayerOut(k);
    end
    
    % Update weights for input to hidden connections
    for i = 1:NINPUTS
        for j = 1:NHIDDENS
            % Only update the weight if the weight is not 0
            if in_to_hidden(i, j) ~= 0
                in_to_hidden(i, j) = in_to_hidden(i, j) + eta * beta_hidden(j) * Xes(i);
            end
        end
    end
    
    % Update weights for direct connections from input to output
    for i = 1:NINPUTS
        in_to_output(i) = in_to_output(i) + eta * beta_output * Xes(i);
    end
    
    % Display updated weights
    fprintf('Updated Weights cycle %d:\n', cycle);
    fprintf('Weights from input to hidden:\n');
    disp(in_to_hidden);
    fprintf('Weights from hidden to output:\n');
    disp(hidden_to_out);
    fprintf('Weights from input to output:\n');
    disp(in_to_output);
end
