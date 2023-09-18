function [InitialObservation,LoggedSignals] = myResetFunction()
fileID= fopen('Initialization.txt','r');
formatSpec= '%f';
A= fscanf(fileID,formatSpec);
fclose(fileID);
D1= A(1);
D2= A(2);
D3= A(3);
D4= A(4);
D5= A(5);
Absorption= A(6);
LoggedSignals.State= [ D1; D2; D3; D4; D5; Absorption];
InitialObservation = LoggedSignals.State;
end