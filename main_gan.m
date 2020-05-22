clc
clear

%% Initialization
XTrain = randn(50000,50);

%% Data 
XNew = reshape(XTrain', [1,1,size(XTrain',1),size(XTrain',2)]);

%% Create Generator Network Layers
Generator_units = 50;
numLatentInputs = 100;
layersGenerator = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    fullyConnectedLayer(Generator_units, 'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(Generator_units, 'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer([1 1 size(XNew,3),50], 'Name','fc3')];
    
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
numEpochs = 5000;
miniBatchSize = 32;

learnRateGenerator = 0.001;
learnRateDiscriminator = 0.01;

numObservations = numel(XNew);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);

trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];

gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;

executionEnvironment = "auto";

%% Train Model
ZValidation = randn(1,1,numLatentInputs,32,'single');
dlZValidation = dlarray(ZValidation,'SSCB');

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZValidation = gpuArray(dlZValidation);
end

figure
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
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X = XNew(:,:,:,idx);
        
        % Concatenate mini-batch of data and generate latent inputs for the
        % generator network.        
        Z = randn(1,1,numLatentInputs,size(X,4),'single');
        
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
        
%         % Every 100 iterations, display batch of generated images using the
%         % held-out generator input.
%         if mod(iteration,100) == 0 || iteration == 1
%             
%             % Generate images using the held-out generator input.
%             dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);
%             
%             % Rescale the images in the range [0 1] and display the images.
%             I = imtile(extractdata(dlXGeneratedValidation));
%             I = rescale(I);
%             image(I)
%             
%             % Update the title with training progress information.
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
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
