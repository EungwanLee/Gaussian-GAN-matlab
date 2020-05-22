clc
clear

%% Initialization
XTrain = randn(10000,50);

%% Data 
XNew = reshape(XTrain', [1,1,size(XTrain',1),size(XTrain',2)]);

%% Create Generator Network Layers
Generator_units = 50;
numLatentInputs = 100;

filterSize = [1 1];
numFilters = 128;

layersGenerator = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    transposedConv2dLayer(filterSize,2*numFilters,'Name','tconv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    transposedConv2dLayer(filterSize,1*numFilters,'Name','tconv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    transposedConv2dLayer(filterSize,50,'Name','tconv3')];
    
lgraphGenerator = layerGraph(layersGenerator);

dlnetGenerator = dlnetwork(lgraphGenerator)

%% Create Discriminator Network Layers
Discriminator_units = 100;
layersDiscriminator = [
    imageInputLayer([1 1 size(XNew,3)],'Normalization','none','Name','in')
    fullyConnectedLayer(Discriminator_units, 'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(Discriminator_units, 'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(Discriminator_units, 'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1, 'Name','fc4')];

lgraphDiscriminator = layerGraph(layersDiscriminator);

dlnetDiscriminator = dlnetwork(lgraphDiscriminator)

figure
subplot(1,2,1)
plot(lgraphGenerator)
title("Generator")

subplot(1,2,2)
plot(lgraphDiscriminator)
title("Discriminator")

%% Specify Training Options
numEpochs = 16;
miniBatchSize = 32;

learnRateGenerator = 0.001;
learnRateDiscriminator = 0.01;

numObservations = size(XTrain,1);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);

trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];

gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;

executionEnvironment = "auto";

%% Train Model
ZValidation = rand(1,1,numLatentInputs,32,'single');
dlZValidation = dlarray(ZValidation,'SSCB');

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZValidation = gpuArray(dlZValidation);
end

% figure
iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    % Shuffle data
    idx = randperm(size(XTrain,1),size(XTrain,1)); 
    XNew = XNew(:,:,:,idx);
    
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        iteration
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X = XNew(:,:,:,idx);
        
        % Concatenate mini-batch of data and generate latent inputs for the
        % generator network.        
        Z = rand(1,1,numLatentInputs,size(X,4),'single');
        
        % Convert mini-batch of data to dlarray specify the dimension labels
        % 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlarray(X, 'SSCB');
        dlZ = dlarray(Z, 'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
            dlZ = gpuArray(dlZ);
        end
        
        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [gradientsGenerator, gradientsDiscriminator, stateGenerator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ);
        dlnetGenerator.State = stateGenerator;
        
        % Update the discriminator network parameters.
        [dlnetDiscriminator.Learnables,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator.Learnables, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            learnRateDiscriminator, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [dlnetGenerator.Learnables,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator.Learnables, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            learnRateGenerator, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Every 100 iterations, display batch of generated images using the
        % held-out generator input.
%         if mod(iteration,100) == 0 || iteration == 1
%             ep=100;
% 
%             for data_count=1:1:ep
%                 ZValidation = rand(1,1,numLatentInputs,32,'single');
%                 dlZValidation = dlarray(ZValidation,'SSCB');
%                 if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
%                     dlZValidation = gpuArray(dlZValidation);
%                 end
%                     dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);
%                     for ii=1:1:32
%                         for jj=1:1:50
%                             AA(1,jj+(50*(ii-1))+((32*50)*(data_count-1)))=dlXGeneratedValidation(1,1,jj,ii);
%                         end
%                     end
%             end
%             [p,x] = hist(AA,300);
%             D = duration(0,0,toc(start),'Format','hh:mm:ss');
%             title(...
%                 "Epoch: " + i + ", " + ...
%                 "Iteration: " + iteration + ", " + ...
%                 "Elapsed: " + string(D))
%             
%             drawnow
%         end
    end
end
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
