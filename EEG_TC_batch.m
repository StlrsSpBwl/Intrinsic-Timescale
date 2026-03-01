function EEG_TC_batch(input_folder,output_folder,subject_name)
bands = struct( ...
    'delta',[1 4], ...
    'theta',[4 8], ...
    'alpha',[8 12], ...
    'beta',[12 30]);
bandNames = fieldnames(bands);
file_path = fullfile(input_folder,subject_name);
file_dir = dir(file_path);
file_dir = file_dir(~[file_dir.isdir]&[file_dir.bytes]>10000);
outdir = fullfile(output_folder,subject_name,"TC");
if ~exist(outdir,"dir")
    mkdir(outdir);
end

for nF =1:length(file_dir)
    edf_path=fullfile(file_dir(nF).folder,file_dir(nF).name);
    name_segment=split(file_dir(nF).name,".");
    disp(edf_path)
    for nB = 1:numel(bandNames)
        bandHz = bands.(bandNames{nB});
        out=tc_power_from_edf_file(edf_path,bandHz);
        save_name = sprintf('%s%s_%s_TC.mat', ...
            subject_name,name_segment{2},bandNames{nB});
        save(fullfile(outdir,save_name),"-struct",'out');
    end
end
end