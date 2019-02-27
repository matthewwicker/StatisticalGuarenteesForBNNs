function script_verification(delta,epsilon,pth)
%script_verification('model','../../','test_0.csv','../../',0.01,0.2)
%modelName = 'model';
%modelDir = '../../';
%fileName = 'test_0.csv';
%fileDir = '../../';
%delta = 0.01;
%epsilon = 0.3;
addpath('DeepGO')
plotFlag = false; 
verbose = false;
GO_verifier(delta,plotFlag,verbose,epsilon,pth)
if ~verbose
    exit
end
end
