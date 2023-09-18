function [Observation,Reward,IsDone,LoggedSignals] = myStepFunction(Action,LoggedSignals,ActionVector)

%% Parameters unpacking 
% State=LoggedSignals.State;
D1= LoggedSignals.State(1);
D2= LoggedSignals.State(2);
D3= LoggedSignals.State(3);
D4= LoggedSignals.State(4);
D5= LoggedSignals.State(5);
Absorption= LoggedSignals.State(6);
%% Action 
Actionvec= ActionVector(Action,:);
%% Parameters Updating 
if Actionvec(1,1) == 0
    D1=D1;
elseif Actionvec(1,1) == 1
        D1=D1+0.0025;     %SiO2 height 2.5 nm increase 
else 
D1=D1-0.0025;
end 

if Actionvec(1,2) == 0
    D2=D2;
elseif Actionvec(1,2) == 1
        D2=D2+0.0025;      %Teeth height 2.5 nm increase 
else 
D2=D2-0.0025;
end

if Actionvec(1,3) == 0
    D3=D3;
elseif Actionvec(1,3) == 1
        D3=D3+0.25;    %Number of teeth 0.25 increase
else 
D3=D3-0.25;
end 

if Actionvec(1,4) == 0
    D4=D4;
elseif Actionvec(1,4) == 1
        D4=D4+0.1;              %Ratio 0.1 increase
else 
D4=D4-0.1;
end 

if Actionvec(1,5) == 0
    D5=D5;
elseif Actionvec(1,5) == 1
        D5=D5+0.1;           %Nonuniformity 0.1 increase
else 
D5=D5-0.1;
end 
%% Applying physics 
    % Lumerical function
Absorption=Lumlink(D1*(10^-6),D2*(10^-6),D3,D4,D5);
%% Updating logged signal
LoggedSignal.State= [ D1; D2; D3; D4; D5; Absorption];
Observation = LoggedSignal.State;
fileID= fopen('Initialization.txt','w');
fprintf(fileID,'%f ',Observation);
fclose(fileID);

%% Reward 
Reward = 0;
x= (Absorption)*100;
if Absorption >= 0.613888  
    Reward = Reward+abs(x) ;
else
    Reward= Reward-x;
end 

if Reward == 0
    IsDone = 1;
else
    IsDone =0;
end 

%% Displaying state of step
 disp(' Action: '+string(Actionvec));
 disp(' Absorption: '+string(Absorption));
 disp(' Reward: '+string(Reward));
 disp(' Observation: '+string(Observation));
 
end