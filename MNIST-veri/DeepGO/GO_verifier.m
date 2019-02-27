function GO_verifier(delta,plotFlag,verbose,epsilon,pth)
%% Code for paper "Confidence Reachability Analysis on DNNs"
% Compare Robustness of Different DNNs for MNIST Classification
% clear data and window
%clearvars -except modelNames modelIdx modelDir fileDir imageFileName delta plotFlag verbose epsilon

%fprintf('%s\n',class(pth))

fid = fopen(strcat(pth,'/out.txt'),'w');
fprintf(fid,'?');

%clc
%clear
%close all;

% Define Global varibale to save internal memory
global outputNN_ind
global LB
global UB
global PIXEL_IND
global EPSILON
global LIP_CONST
global R
global INPUT_IMAGE
global MAX_ITERATION
global imageTempK
global tStart
global tStart_M
global BOUNDERROR
global AllImage
global Kall
global LogitAll
global AllImage_M
global Kall_M
global LogitAll_M
global balaceNum
global nn_logit
global nn_confidence
balaceNum = 0.001;
%load DNN_1

%net = importKerasNetwork('tmp/model.json','WeightFile','tmp/modelweights.h5', ...
%    'OutputLayerType','regression');
% Here is where we need to edit the tmp value
W1 = csvread(strcat(pth,'/W0.txt'));
W2 = csvread(strcat(pth,'/W1.txt'));
b1 = csvread(strcat(pth,'/b0.txt'));
b2 = csvread(strcat(pth,'/b1.txt'));
nn_logit = @(X)one_layer_nn_logit(W1',b1,W2',b2, X);
nn_confidence = @(X)one_layer_nn_confidence(W1',b1,W2',b2, X);

%load MNIST_Dataset


%featureCell = {[5,5;5,6;5,7;5,8;5,9;5,10;6,10;7,10];
%    [7,6;7,7;8,6;8,7;9,6;9,7;10,6;10,7];
%    [11,4;11,5;11,6;11,7;11,8;11,9;10,5;10,6];
%    [8,4;8,5;8,6;8,7;8,8;8,9;9,9;10,9]
%    };

featureCell = {[5,5;5,6;5,7;5,8;5,9];
    };



%%
saveImageNum = 10; %WHAT IS THIS?
imageIndex = 133;
featureIndexZ = 1;

%%
Kall = 1;
LogitAll = [];
Kall_M = 1;
LogitAll_M = [];
%convnet = convnet_1;
%convnet = net;
%logitlayer = 'fc';
%numOfLayers = length(convnet.Layers);
%logitlayer = convnet.Layers(numOfLayers-2).Name;
INPUT_IMAGE = csvread(strcat(pth,'/inp.txt'));
%========================================
%INPUT_IMAGE = XTest(:,:,:,imageIndex);
[imgRowNum,imgColNum] = size(INPUT_IMAGE);
AllImage = zeros(imgRowNum,imgRowNum);
AllImage_M = zeros(imgRowNum,imgRowNum);


%[outputNN_start,outputNN_ind] = max(activations(convnet,INPUT_IMAGE,logitlayer));
[outputNN_start,outputNN_ind] = max(nn_logit(INPUT_IMAGE));



%[outputNN_start_Conf,~] = max(activations(convnet,INPUT_IMAGE,'softmax'));
[outputNN_start_Conf,~] = max(nn_confidence(INPUT_IMAGE));

upperBound = outputNN_start;
PIXEL_IND = featureCell{featureIndexZ,1};

%PIXEL_IND = [PIXEL_IND];
%sum_temp = epsilon * ones(size(PIXEL_IND,1),1);
LB = zeros(size(PIXEL_IND,1),1);
UB = zeros(size(PIXEL_IND,1),1);

for ii = 1:size(PIXEL_IND,1)
    LB(ii) = max(INPUT_IMAGE(PIXEL_IND(ii,1), PIXEL_IND(ii,2)) - epsilon,0.0) ;
    UB(ii) = min(INPUT_IMAGE(PIXEL_IND(ii,1), PIXEL_IND(ii,2)) + epsilon,1.0) ;
end
%LB = zeros(size(PIXEL_IND,1),1);
%UB = ones(size(PIXEL_IND,1),1);
%EPSILON = 0.003*ones(size(PIXEL_IND,1),1); %???????????????
EPSILON = 0.0003*ones(size(PIXEL_IND,1),1);
R = linspace(1.1,2,size(PIXEL_IND,1));
R = R';

LIP_CONST = 8*ones(size(PIXEL_IND,1),1);
MAX_ITERATION = 100*ones(size(PIXEL_IND,1));
BOUNDERROR = linspace(0.05,0.5,size(PIXEL_IND,1));
BOUNDERROR =BOUNDERROR';

pixelOptIndex = size(PIXEL_IND,1);
x = INPUT_IMAGE(PIXEL_IND(pixelOptIndex,1),PIXEL_IND(pixelOptIndex,2));
imageTempK = INPUT_IMAGE';
imageTempK = imageTempK(:);
if verbose
    disp('Now Calculate the Lower Boundary of Range')
end
if verbose
    tStart = tic;
    tic
end
[zLGO,imageReult] = evalFunDNNmin(pixelOptIndex,x,INPUT_IMAGE,verbose);
if verbose
    toc
    min_time = toc(tStart);
end
%
[B_sort,I_sort] = sort(LogitAll);
LValue = B_sort(1);
Lindex = I_sort;
AllImage = AllImage(:,:,I_sort);
AllImage = AllImage(:,:,1:saveImageNum);
Min_Fig = AllImage(:,:,1);
%ConfidenceAll = activations(convnet,reshape(AllImage,[14,14,1,...
%    size(AllImage,3)]),'softmax');
ConfidenceAll = zeros(size(AllImage,3),1);
for ii = 1:size(AllImage,3)
    ConfidenceAll(ii) = nn_confidence(AllImage(:,:,ii));
end
%ConfidenceAll = activations(convnet,reshape(AllImage,[14,14,1,...
%    size(AllImage,3)]),softmax_layer_name);

ConfTargetAll = ConfidenceAll(:,outputNN_ind);
[CValue,Cindex]= min(ConfTargetAll);

%%
if verbose
    disp('Now Calculate the Ubber Boundary of Range')
end
if verbose
    tStart_M = tic;
    tic
end
[zLGO_M,imageReult_M] = evalFunDNNmax(pixelOptIndex,x,INPUT_IMAGE,verbose);
if verbose
    toc
    max_time = toc(tStart_M);
end
[B_sort_max,I_sort_max] = sort(LogitAll_M);
Lindex_M = I_sort_max(end);
LValue_M = B_sort_max(end);

AllImage_M = AllImage_M(:,:,I_sort_max);
AllImage_M = AllImage_M(:,:,end-saveImageNum+1:end);
Max_Fig = AllImage_M(:,:,end);
%ConfidenceAll_M = activations(convnet,reshape(AllImage_M,[14,14,1,...
%    size(AllImage_M,3)]),'softmax');
%ConfidenceAll_M = activations(convnet,reshape(AllImage_M,[14,14,1,...
%    size(AllImage_M,3)]),softmax_layer_name);
ConfidenceAll_M = zeros(size(AllImage_M,3),1);
for ii = 1:size(AllImage_M,3)
    ConfidenceAll_M(ii) = nn_confidence(AllImage_M(:,:,ii));
end
ConfTargetAll_M = ConfidenceAll_M(:,outputNN_ind);
[CValue_M,Cindex_M]= max(ConfTargetAll_M);

%%
if plotFlag
    figure;
    subplot(1,3,1)
    imshow(INPUT_IMAGE);
    title({'Input Image';num2str(outputNN_start);num2str(outputNN_start_Conf)});
    
    subplot(1,3,2)
    imshow(Min_Fig);
    title({'Lower Boundary Image';num2str(LValue);num2str(CValue)});
    
    subplot(1,3,3)
    imshow(Max_Fig);
    title({'Ubber Boundary Image';num2str(LValue_M);num2str(CValue_M)});
    
    fprintf('\nGiven Test Image: Logit = %4.4f; Confidence = %4.4f\n',...
        outputNN_start,min(outputNN_start_Conf))
    fprintf('Lower Boundary:  Logit = %4.4f; Confidence = %4.4f\n',...
        LValue,min(ConfTargetAll))
    fprintf('Ubber Boundary:  Logit = %4.4f; Confidence = %4.4f\n',...
        LValue_M,max(ConfTargetAll_M))
    fprintf('Calculation Time: Lower Bound = %4.3fs; Upper Bound = %4.3fs\n',...
        min_time,max_time)
    drawnow
end
%disp(outputNN_start_Conf - CValue)
%disp(CValue_M - outputNN_start_Conf)
%if ((outputNN_start_Conf - CValue) > delta) || ((CValue_M - outputNN_start_Conf) > delta)
    %disp('0')
%    fprintf(fid,'0!');
%else
    %disp('1')
%    fprintf(fid,'1!');
%end
if delta ~= -1
    if ((outputNN_start_Conf - CValue) > delta) || ((CValue_M - outputNN_start_Conf) > delta)
        %disp('0')
        fprintf(fid,'0!');
    else
        %disp('1')
        fprintf(fid,'1!');
    end
else    
    target = outputNN_start_Conf;
    if target <= 0.5
        %value = CValue;
        value = CValue_M;
    else
        %value = CValue_M;
        value = CValue;
    end
    draw = rand(1);
    t = 0;
    v = 0;
    if(value < draw)
    	v = 1;
    end
    if(target < draw)
        t = 1;
    end
    if(t ~= v)
        fprintf(fid,'0!');
    else
        fprintf(fid,'1!');
    end
end
fclose(fid);
end


