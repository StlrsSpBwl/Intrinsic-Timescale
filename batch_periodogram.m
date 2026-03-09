function batch_periodogram(TC_struct,subject,colors_mapping)
    % bandNames will be a structure of the bandname and the color
    % associating to the band
    bandNames = fieldnames(TC_struct.(subject));
    figure;
    for nB = 1:numel(bandNames)
        subplot(numel(bandNames),1,nB)
        [~,~]=TC_periodogram(TC_struct.(subject),bandNames{nB},colors_mapping.(bandNames{nB}),0);
    end
end