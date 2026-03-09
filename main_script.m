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
bandNames = {'delta','theta','alpha','beta'};
% concatenate the data into a continuous time table
for nS =1:numel(subject_name)
    for nB = 1:numel(bandNames)
        full_folder = fullfile(TC_data_folder,subject_name{nS},parameters_folder);
        files = dir(fullfile(full_folder,sprintf('*_%s_TC.mat',bandNames{nB})));
        data = cell(1,numel(files));
        for nF = 1:numel(files)
            tmp = load(fullfile(files(nF).folder,files(nF).name));%load the data first
            % valid=tmp.keep;
            data{nF} = timetable(tmp.time',tmp.TC, ...
                'VariableNames',{'TC'});
        end
        TC_data.(subject_name{nS}).(bandNames{nB}) = vertcat(data{:});
    end
end
save(fullfile(TC_data_folder,'full_tc_timeseries.mat'),'TC_data') % save the data
%% visualize the time series with masking (save the figures)
figure_folder = '/Users/jiangruitong/Library/CloudStorage/GoogleDrive-ruitongj@andrew.cmu.edu/Shared drives/NML_shared/PapersInPrep/Journals/Larry/Temporal Correlations and Performance Fluctuation/Time Traces Figures';
subject_name = fieldnames(TC_data);
band_name = fieldnames(TC_data.Baldface);
light_on = 6;
light_off = 19;
for nS = 1:numel(subject_name)
    for nB = 1:numel(band_name)
        fig = figure;
        save_figure_name = strcat(subject_name{nS},'_',band_name{nB},'.png');
        TCTimeTrace(TC_data.(subject_name{nS}),light_on,light_off,band_name{nB});
        set(fig,'Units','normalized','OuterPosition',[0 0 1 1])
        saveas(fig,fullfile(figure_folder,save_figure_name))
        close(fig)
    end
end
%% include the smooth data (using moving average)
figure_folder_smooth='/Users/jiangruitong/Library/CloudStorage/GoogleDrive-ruitongj@andrew.cmu.edu/Shared drives/NML_shared/PapersInPrep/Journals/Larry/Temporal Correlations and Performance Fluctuation/Time Traces Figures/Smooth Trace';
for nS=1:numel(subject_name)
    for nB=1:numel(band_name)
        fig = figure;
        save_figure_name_smooth = strcat(subject_name{nS},'_',band_name{nB},'_smooth.png');
        TCTimeTrace(TC_data.(subject_name{nS}),light_on,light_off,band_name{nB},true,120);
        set(fig,'Units','normalized','OuterPosition',[0 0 1 1])
        saveas(fig,fullfile(figure_folder_smooth,save_figure_name_smooth))
        close(fig)
    end
end
%% visualization polar plot style
%circadian_rose_shaded(time_points,in_data,time_res,color,pct_lo,pct_hi)
figure_folder_rose = '/Users/jiangruitong/Library/CloudStorage/GoogleDrive-ruitongj@andrew.cmu.edu/Shared drives/NML_shared/PapersInPrep/Journals/Larry/Temporal Correlations and Performance Fluctuation/Rose Plot';
for nS=1:numel(subject_name)
    for nB=1:numel(band_name)
        fig = figure;
        save_figure_name_rose = strcat(subject_name{nS},'_',band_name{nB},'_rose.png');
        data_strcut = TC_data.(subject_name{nS}).(band_name{nB});
        circadian_rose_shaded(data_strcut.Time,data_strcut.TC,1,[0.2 0.4 0.8],25,75,1,strcat(subject_name{nS},{' '},band_name{nB}))
        set(fig,'Units','normalized','OuterPosition',[0 0 1 1])
        saveas(fig,fullfile(figure_folder_rose,save_figure_name_rose))
        close(fig)
    end
end
%% plot out the peridogram (Virgil)
figure;
colors_mapping = struct('delta',[0.1 0.1 0.7],'theta',[0.2 0.6 0.2], ...
    'alpha',[0.8 0.2 0.2],'beta',[0.9 0.5 0.1]);
batch_perio

