%% Voltage Imaging Slice Test Analysis Code
% For next version: Layer a loop on top of this to iterate across all images in the folder,
% and sorting based on Condition sequence from ephys. Also use ephys data
% to determine what images to NOT use.
% Remember: dF/F =( F(t) - F0)/F0 

%% Load files and setup
% Step 1: Read the ephys file
ephysFilePath = char(uigetdir('/Volumes/Elements/Data/Voltage Imaging/VoltImg_slice_test/Ephys Data')); % Select and set root folder where all experiments with cells you want to analyze are located
ephysFileDir = dir(ephysFilePath);
load([ephysFileDir(3).folder, '/', ephysFileDir(3).name]);

% Get dv parameters and condition sequence
dvCondSequence = ExpStruct.dvStepParams.dvCondSequence; % sequence of randomized dv trials
dvToTest = ExpStruct.dvStepParams.dvToTest; % dv steps to be simulated by current injection
nConds = length(unique(dvCondSequence)); % number of conditions (dv steps)

pulseStart = ExpStruct.dvStepParams.pulseStart;
sweepDur = ExpStruct.dvStepParams.sweepDur;
nPulses = ExpStruct.dvStepParams.nPulses;
pulseFreq = ExpStruct.dvStepParams.pulseFreq;
imagingFreq = 330.30;
Fs = ExpStruct.Fs;

% Step 2: Identify the folder containing the imaging files correspdonding to the electrophysiology recording
ImgsFilePath = char(uigetdir('/Volumes/Elements/Data/Voltage Imaging/VoltImg_slice_test/Imaging Data')); % Select and set root folder where all experiments with cells you want to analyze are located
ImgfolderContents = dir(ImgsFilePath);

% Step 2b: revise cond sequence to match actual trials recorded
dvCondSequence = dvCondSequence(1:(size(ImgfolderContents, 1)-3));

% Step 3: Identify bad trials, to skip in analyses below
vsTest_inputs = ExpStruct.dvStepParams.vsTest_inputs*2;
baselineAllTrials = [];
excludeTrials = [];
for tt = 1:size(vsTest_inputs, 2)
    baseline = mean(vsTest_inputs(1:0.001*pulseStart*Fs, tt));
    baselineAllTrials = [baselineAllTrials, baseline];
    
    if baselineAllTrials(tt) > -50
        excludeTrials = [excludeTrials, tt];
    end
end

% Step 3: Identify GEVI
UpOrDown = input('1 for upward GEVI, 2 for downward GEVI ', 's');

voltImgTest_Analysis.ephysData.Fs = Fs;
voltImgTest_Analysis.pulseParams.pulseStart = pulseStart;
voltImgTest_Analysis.pulseParams.sweepDur = sweepDur;
voltImgTest_Analysis.pulseParams.nPulses = nPulses;
voltImgTest_Analysis.pulseParams.pulseFreq = pulseFreq;
voltImgTest_Analysis.imagingFreq = imagingFreq;
voltImgTest_Analysis.dvCondSequence = dvCondSequence;
voltImgTest_Analysis.dvToTest = dvToTest;
voltImgTest_Analysis.Rinput = ExpStruct.dvStepParams.Rinput;
voltImgTest_Analysis.fRinput = ExpStruct.dvStepParams.fRinput;
voltImgTest_Analysis.ephysData.vsTest_inputs = vsTest_inputs;
voltImgTest_Analysis.ephysData.baselineAllTrials = baselineAllTrials;
voltImgTest_Analysis.ephysData.excludeTrials = excludeTrials;

%% Calculate ROI mask
% Step 1: Calculate mask for ROI using spiking resp
if UpOrDown == '1'
    maxDvTrials = find(dvCondSequence == max(unique(dvCondSequence)));
elseif UpOrDown == '2'
    maxDvTrials = find(dvCondSequence == min(unique(dvCondSequence)));
end

maxDvStack = [];
for tt = 1:length(maxDvTrials)
    if ismember(maxDvTrials(tt), excludeTrials)
        continue
    end
    currImgPath = [ImgfolderContents(maxDvTrials(tt)+2).folder, '/', ImgfolderContents(maxDvTrials(tt)+2).name];
    info = imfinfo(currImgPath);
    numFrames = numel(info);    

    % Preallocate the image stack
    imageStack = zeros(info(1).Height, info(1).Width, numFrames);

    % Read each frame and store in the stack
    for frameIndex = 1:numFrames
        imageStack(:,:,frameIndex) = imread(currImgPath, 'Index', frameIndex, 'Info', info);
    end
    
%     figure(10)
%     imagesc(mean(imageStack, 3));
%     axis equal
%     axis image
    
    maxDvStack = cat(3, maxDvStack, imageStack);
end

meanMaxDvStack = mean(maxDvStack(:, :, :), 3);
meanFluorMaxDvStack = meanMaxDvStack;
figure(10); clf; imagesc(meanFluorMaxDvStack); axis equal; axis image; colorbar; % caxis([-7 -4]);

% Old Method: Automatically set cutoff from whole mean
% meanMaxDvStack(meanMaxDvStack <= cutOffFluor) = 0;
% meanMaxDvStack(meanMaxDvStack > 0) = 1;
% roiStack = meanMaxDvStack;
% figure(20); imagesc(roiStack); axis equal; axis image;
% [roiX, roiY] = find(roiStack);
% imagesc(meanMaxDvStack);
% axis equal
% axis image

%%%%%
% New Method: Hand select cell or area of interest
roiX = []; roiY = [];
roiHandSelect = drawfreehand;
roiHandSelectMask = createMask(roiHandSelect);
[roiX, roiY] = find(roiHandSelectMask);

% Calculate mean fluorescence of the area of interest and then form into final ROI mask

roiMeanMaxDvStack = zeros(size(maxDvStack, 1), size(maxDvStack, 2));
for rr = 1:length(roiX)
    roiMeanMaxDvStack(roiX(rr), roiY(rr)) = mean(maxDvStack(roiX(rr), roiY(rr),:), 3);
end

figure(11)
imagesc(roiMeanMaxDvStack); axis equal; axis image; colorbar;

% Old version: Calculate mean fluorescence of the area of interest and then form into final ROI mask
% roiMeanMaxDvStack = mean(maxDvStack(min(roiX):max(roiX), roiY, :), 3);
% imagesc(roiMeanMaxDvStack)

stdFluor = std(nonzeros(roiMeanMaxDvStack));
meanFluor = mean(nonzeros(roiMeanMaxDvStack));

% Designate cutoff fluorescence for pixels to be selected for ROI
cutOffFluor = meanFluor; %stdFluor*1+ meanFluor; % currently cutoff is 1 standard devs from mean fluorescence

roiMeanMaxDvStack(roiMeanMaxDvStack <= cutOffFluor) = 0;
roiMeanMaxDvStack(roiMeanMaxDvStack > 0) = 1;
roiStack = roiMeanMaxDvStack;
figure(20); imagesc(roiStack); axis equal; axis image;
[roiX, roiY] = find(roiStack);

%% Calculate df and mean df for all conditions
% Step 2: Calculate mean df across all trials across all conditions
dfAllConds = cell(nConds, 1);
traceAllConds = cell(nConds, 1);
f0Start = 0.01;
f0End = 0.03;

for tt = 1:size(vsTest_inputs, 2)
    if ismember(tt, excludeTrials)
        continue
    end
    
    % Read the multi-frame image
    currImgPath = [ImgfolderContents(tt+2).folder, '/', ImgfolderContents(tt+2).name];
    info = imfinfo(currImgPath);
    numFrames = numel(info);

    % Preallocate the image stack
    imageStack = zeros(info(1).Height, info(1).Width, numFrames);
    
    % Read each frame and store in the stack
    for frameIndex = 1:numFrames
        imageStack(:,:,frameIndex) = imread(currImgPath, 'Index', frameIndex, 'Info', info);
    end
    
    % Designate ROI (2 methods)
%   % Strategy 1: Hand-select the region of interest (ROI)
%     figure();
%     imshow(imageStack(:, :, 1), []);
%     title('Select ROI');
%     roi2 = round(getPosition(imrect));
    
    % Strategy 2: Pre-select (designate) ROI area, assume point-per-pixel
%     width = 26;
%     height = 31;
%     x = 55;
%     y = 1; 
%     roi = [x, y, width, height]; % Replace with the coordinates and dimensions of your ROI
%     
%     figure(10)
%     for ff = 2:size(imageStack, 3)
%         imagesc(imageStack(:,:,ff))
%         hold on;
%         rectangle('Position',[x, y, width, height]);
%         axis square;
%         axis equal;
%         axis image;
%         pause
%     end
     
      % Strategy 3: Customize ROI based on shape of fluorescence off averaged image
%     meanImg = mean(imageStack, 3);
%     stdFluor = std2(meanImg);
%     meanFluor = mean2(meanImg);
%     cutOffFluor = stdFluor*2 + meanFluor;
%     
%     meanImg(meanImg <= cutOffFluor) = 0;
%     meanImg(meanImg > 0) = 1;
%     [roiX, roiY] = find(meanImg);
%     
%     figure(10)
%     imagesc(meanImg);
%     axis equal;
%     axis image;    
   
% %     Extract the pixel intensity values
%     df = [];
%     for ff = 2:size(imageStack, 3)
%         % Read current frame
%         currentFrame = imageStack(:, :, ff);
%         
%         % Extract ROI pixel intensities
% %         roiPixels = currentFrame(roi(2):roi(2)+roi(4)-1, roi(1):roi(1)+roi(3)-1);
%         roiPixels = currentFrame(roiX, roiY);
%         % Step 4: Calculate intensity changes
%         previousFrame = imageStack(:, :, ff-1);
% %         previousRoiPixels = previousFrame(roi(2):roi(2)+roi(4)-1, roi(1):roi(1)+roi(3)-1);
%         previousRoiPixels = previousFrame(roiX, roiY);
%         intensityChange = mean(abs(roiPixels(:) - previousRoiPixels(:))); % Example: using mean intensity difference
%         
%         df = [df; intensityChange];
%     end
    
%    Extract the pixel intensity values (new option 2)    
    df = [];
    roiPixels = [];
    previousRoiPixels = [];
    for ff = 2:size(imageStack, 3)
        % Read current frame
        currentFrame = imageStack(:, :, ff); 
        % Read previous frame
        previousFrame = imageStack(:, :, ff-1);

            for rr = 1:size(roiX, 1)
                roiPixels(rr) = currentFrame(roiX(rr), roiY(rr));
                previousRoiPixels(rr) = previousFrame(roiX(rr), roiY(rr));
            end

        intensityChange = mean(abs(roiPixels(:) - previousRoiPixels(:)));
        df = [df; intensityChange];
    end
    
    df = (df - mean(df(ceil(imagingFreq*f0Start):floor(imagingFreq*f0End))))/mean(df(ceil(imagingFreq*f0Start):floor(imagingFreq*f0End)));

    dfAllConds{dvCondSequence(tt), 1} = [dfAllConds{dvCondSequence(tt), 1}, df];
    traceAllConds{dvCondSequence(tt), 1} = [traceAllConds{dvCondSequence(tt), 1}, vsTest_inputs(:, tt)];
end

meanDfAllConds = [];
meanTraceAllConds = [];
for cc = 1:nConds
    meanDfAllConds = [meanDfAllConds, mean(dfAllConds{cc}, 2)];
    meanTraceAllConds = [meanTraceAllConds, mean(traceAllConds{cc}, 2)];
end

for cc = 1:nConds
    figure(50+cc)
    clf
    hold on
    for pp = 1:nPulses
        xline((pulseStart/1000)+(pp-1)/pulseFreq, 'LineWidth', 2, 'color', [.8 .8 .8]);
    end
    yyaxis left
    plot(linspace(0, size(meanDfAllConds, 1)/imagingFreq, size(meanDfAllConds, 1)), meanDfAllConds(:, cc), 'linewidth', 2, 'color', 'k')
    ylabel('mean dF')
    yyaxis right
    plot(linspace(0, size(meanTraceAllConds, 1)/Fs, size(meanTraceAllConds, 1)), meanTraceAllConds(:, cc), 'linewidth', 1, 'color', [0.3010 0.7450 0.9330])
    hold off
    title(['dV = ', num2str(abs(mean(meanTraceAllConds(1:(ExpStruct.dvStepParams.pulseStart/1000)*Fs, cc)))+(max(meanTraceAllConds(((ExpStruct.dvStepParams.pulseStart/1000)*Fs):((ExpStruct.dvStepParams.pulseStart/1000)*Fs+((1000/ExpStruct.dvStepParams.pulseFreq)/1000*Fs)), cc)))), ' (1st pulse) ',...
        num2str(abs(mean(meanTraceAllConds(1:(ExpStruct.dvStepParams.pulseStart/1000)*Fs, cc)))+(max(meanTraceAllConds(((ExpStruct.dvStepParams.pulseStart/1000)*Fs+((1000/ExpStruct.dvStepParams.pulseFreq)/1000*Fs)):end, cc)))), ' (2nd pulse)'])
    xlabel('Time (s)')
    ylabel('mV')
%     xlim([0.1, size(meanDfAllConds, 1)/imagingFreq])
%     ylim([9 15]);
end

voltImgTest_Analysis.maxDvStack = maxDvStack;
voltImgTest_Analysis.meanFluorMaxDvStack = meanFluorMaxDvStack;
voltImgTest_Analysis.stdFluor = stdFluor;
voltImgTest_Analysis.meanFluor = meanFluor;
voltImgTest_Analysis.cutOffFluor = cutOffFluor; 
voltImgTest_Analysis.roiStack = roiStack;
voltImgTest_Analysis.dfAllConds = dfAllConds;
voltImgTest_Analysis.traceAllConds = traceAllConds;
voltImgTest_Analysis.meanDfAllConds = meanDfAllConds;
voltImgTest_Analysis.meanTraceAllConds = meanTraceAllConds;

%% dF over voltage
figure(40);
clf
hold on
for cc = 1:nConds
    scatter(abs(mean(meanTraceAllConds(1:(ExpStruct.dvStepParams.pulseStart/1000)*Fs, cc)))+(max(meanTraceAllConds(((ExpStruct.dvStepParams.pulseStart/1000)*Fs):((ExpStruct.dvStepParams.pulseStart/1000)*Fs+((1000/ExpStruct.dvStepParams.pulseFreq)/1000*Fs)), cc))),...
        max(-meanDfAllConds(ceil((ExpStruct.dvStepParams.pulseStart/1000)*imagingFreq):floor(((ExpStruct.dvStepParams.pulseStart/1000)*imagingFreq+((1000/ExpStruct.dvStepParams.pulseFreq)/1000*imagingFreq))), cc)), 50, 'k', 'filled');
    scatter(abs(mean(meanTraceAllConds(1:(ExpStruct.dvStepParams.pulseStart/1000)*Fs, cc)))+(max(meanTraceAllConds(((ExpStruct.dvStepParams.pulseStart/1000)*Fs+((1000/ExpStruct.dvStepParams.pulseFreq)/1000*Fs)):end, cc))),...
        max(-meanDfAllConds(ceil(((ExpStruct.dvStepParams.pulseStart/1000)*imagingFreq+((1000/ExpStruct.dvStepParams.pulseFreq)/1000*imagingFreq))):end, cc)), 50, 'k', 'filled');
end
xlabel('dV (mV)')
ylabel('dF')
axis square;
set(gca,'linewidth',1.5, 'fontsize', 12)
hold off
%% Save Analysis Results
directory = 'E:\Data\Voltage Imaging\VoltImg_slice_test\Analysis Results';
voltImgTest_Analysis.mouseID = ['voltImgTest_Analysis_', num2str(ExpStruct.mouseID)];
fileName = [num2str(voltImgTest_Analysis.mouseID), '.mat'];
save(fullfile(directory, fileName), 'voltImgTest_Analysis', '-v7.3');

