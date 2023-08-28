% Assuming you have 'reference_data' and 'original_data' as your datasets

dfWholeSweep = [];
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
    dfWholeSweep = [dfWholeSweep; intensityChange];
end

% Resample the original_data to match the reference_data sampling rate
time_reference = (0:length(vsTest_inputs)-1)/Fs;
time_original = (0:length(dfWholeSweep)-1)/imagingFreq;

% Interpolate the original_data to match the time values of reference_data
resampled_original_data = interp1(time_original, dfWholeSweep, time_reference, 'linear');

% Calculate cross-correlation and find time shift
cross_corr = xcorr(vsTest_inputs, resampled_original_data);
[~, max_index] = max(cross_corr);
time_shift = max_index - length(vsTest_inputs) + 1;

% Apply the time shift to the resampled_original_data
aligned_original_data = circshift(resampled_original_data, time_shift);

% Plot the aligned data for visualization
figure;
plot(time_reference, vsTest_inputs, 'b', time_reference, aligned_original_data, 'r');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Reference Data', 'Aligned Original Data');

%%
% Assuming you have 'high_sampled_data' as your dataset sampled at 20 kHz

% Define the downsampling factor
downsampling_factor = round(20000 / 330.30);

% Downsample the high_sampled_data
downsampled_data = downsample(vsTest_inputs, downsampling_factor);

% Create time array for the downsampled data
time_downsampled = (0:length(downsampled_data)-1) / 330.30;

% Plot the downsampled data
plot(time_downsampled, downsampled_data);
xlabel('Time (s)');
ylabel('Amplitude');
title('Downsampled Data at 330.30 Hz');