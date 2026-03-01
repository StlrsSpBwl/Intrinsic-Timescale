edf_path = "/Users/jiangruitong/Library/CloudStorage/GoogleDrive-ruitongj@andrew.cmu.edu/Shared drives/NML_shared/PapersInPrep/Journals/Larry/EEG spectral properties during adaptation of new light-dark cycle/Data/Data for baseline setting/Virgil/Virgil.0001.edf";
bandHz = [1 40];
bandAlpha = [8 12];

test_signal_TC  = tc_signal_from_edf_file(edf_path,bandHz);
%%
bandBeta = [13 30];
test_power_TC = tc_power_from_edf_file(edf_path,bandBeta);
%%
test_power_TC_alpha = tc_power_from_edf_file(edf_path,bandAlpha);

test_power_TC_gamma = tc_power_from_edf_file(edf_path,[30 80]); % don't use gamma due to EMG artifact
%%
test_power_TC_delta = tc_power_from_edf_file(edf_path,[1 4]);
%%
test_power_TC_theta = tc_power_from_edf_file(edf_path,[4 8]);
%% now we do the whole batch analysis
input_folder='/Users/jiangruitong/Library/CloudStorage/GoogleDrive-ruitongj@andrew.cmu.edu/Shared drives/NML_shared/PapersInPrep/Journals/Larry/EEG spectral properties during adaptation of new light-dark cycle/Data/Data for baseline setting';
output_folder = '/Users/jiangruitong/Library/CloudStorage/GoogleDrive-ruitongj@andrew.cmu.edu/Shared drives/NML_shared/PapersInPrep/Journals/Larry/Temporal Correlations and Performance Fluctuation/Intrinsic Timescale Time Series';
EEG_TC_batch(input_folder,output_folder,'Virgil');
EEG_TC_batch(input_folder,output_folder,'Goku');
EEG_TC_batch(input_folder,output_folder,'Baldface');

%% concatenating the data and visualize the time series
TC_data_folder = '/Users/jiangruitong/Library/CloudStorage/GoogleDrive-ruitongj@andrew.cmu.edu/Shared drives/NML_shared/PapersInPrep/Journals/Larry/Temporal Correlations and Performance Fluctuation/Intrinsic Timescale Time Series';
subject_name = {"Virgil","Goku","Baldface"};
parameters_folder = "TC";