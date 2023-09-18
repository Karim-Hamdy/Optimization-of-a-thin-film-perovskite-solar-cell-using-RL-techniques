
clear all
clc
%% Actions vector
A1= 0:1:2;
A2= 0:1:2;
A3= 0:1:2;
A4= 0:1:2;
A5= 0:1:2;
ActionVector= transpose(combvec(A1,A2, A3,A4, A5)); % Combination Function

%% Reset and Step functions creation
ResetHandle = @() myResetFunction;
StepHandle = @(Action,LoggedSignals) myStepFunction(Action,LoggedSignals,ActionVector);
%% Observation
ObservationInfo = rlNumericSpec([6 1],'LowerLimit',[-50;-50; -50; -50;-50;0],'UpperLimit', [ 50;50;50; 50; 50; 5000]); % Define lower and upper limits
%% Action definition
ActionInfo = rlFiniteSetSpec(1:243);
%% Environment 
env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle); % Check inputs
%% Agent

% Neural network
criticNet = [
    imageInputLayer([1 6 1],"Name","state","Normalization","none")
    fullyConnectedLayer(256,"Name","Fully_256_1")
    tanhLayer("Name","tanh_activation1")
    fullyConnectedLayer(256,"Name","Fully_256_2")
    tanhLayer("Name","tanh_activation2")
    fullyConnectedLayer(128,"Name","Fully_128_1")
    tanhLayer("Name","tanh_activation3")
    fullyConnectedLayer(128,"Name","Fully_128_2")
    tanhLayer("Name","tanh_activation4")
    fullyConnectedLayer(64,"Name","Fully_64_1")
    tanhLayer("Name","tanh_activation5")
    fullyConnectedLayer(64,"Name","Fully_64_2")
    reluLayer("Name","relu_activation1")
    fullyConnectedLayer(1,"Name","output")];

actorNet = [
    imageInputLayer([1 6 1],"Name","state","Normalization","none")
    fullyConnectedLayer(2056,"Name","Fully_256_1")
    tanhLayer("Name","tanh_activation1")
    fullyConnectedLayer(2056,"Name","Fully_256_2")
    tanhLayer("Name","tanh_activation2")
    fullyConnectedLayer(1528,"Name","Fully_128_1")
    tanhLayer("Name","tanh_activation3")
    fullyConnectedLayer(1528,"Name","Fully_128_2")
    tanhLayer("Name","tanh_activation4")
    fullyConnectedLayer(1024,"Name","Fully_64_1")
    tanhLayer("Name","tanh_activation5")
    fullyConnectedLayer(1024,"Name","Fully_64_2")
    reluLayer("Name","relu_activation1")
    fullyConnectedLayer(243,"Name","output")];

% Agent Creation

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

criticOpts = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1,'UseDevice','cpu');
critic = rlValueRepresentation(criticNet,obsInfo,'Observation',{'state'},criticOpts);

actorOpts = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1,'UseDevice','cpu');
actor = rlDiscreteCategoricalActor(actorNet,obsInfo,actInfo);

agentOpts = rlACAgentOptions(...
    'NumStepsToLookAhead',64, ...
    'EntropyLossWeight',0.3, ...
    'DiscountFactor',0.9);
agent = rlACAgent(actor,critic,agentOpts);

%% Training 

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 100, ...
    'MaxStepsPerEpisode', 1, ...critic
    'Verbose', true, ...
    'Plots','training-progress',...
    'ScoreAveragingWindowLength',10,...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',1000,...
    'SaveAgentCriteria',"EpisodeReward" ,...
    'SaveAgentValue', 66,...
    'SaveAgentDirectory', pwd + "\agents\");  % Check options

doTraining = 1;
if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
end 


