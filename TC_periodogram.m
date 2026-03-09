function [period, P1_norm]=TC_periodogram(TC_struct,band_name,color,down_sample_flag,title_string)
    if nargin<3, color = [0.2 0.4 0.8]; end
    if nargin<4, down_sample_flag=0; end
    if nargin<5, title_string = band_name; end
    TC_timetable = TC_struct.(band_name);
    % Resample to hourly
    if down_sample_flag
        tc_hourly = retime(TC_timetable,'hourly',@(x) median(x,'omitnan'));
        tc_vals = fillmissing(tc_hourly.TC,'nearest');
        Fs = 1/3600;
    else
        tc_vals = fillmissing(TC_timetable.TC,'nearest');
        dt = seconds(median(diff(TC_timetable.Time)));
        Fs = 1/dt;
    end
    
    % FFT
    L = numel(tc_vals);
    freq=fft(tc_vals);
    f=Fs*(0:(L/2))/L;
    P2=abs(freq/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1)=2*P1(2:end-1);
    period = (1./f)/3600; % convert into hour
    P1_norm = P1/sum(P1);
    % Identify the dominant period
    [~,iPeak]=max(P1_norm(2:end));
    % Plot
    plot(period,P1_norm,'Color',color,'LineWidth',4);
    hold on
    xline(period(iPeak+1),'k--','LineWidth',2,'HandleVisibility','off');
    xlim([0 48])
    xticks(0:12:48)
    xlabel('Period (Hr)','FontSize',18);
    ylabel('Power Spectral Density (norm.)','FontSize',18);
    if nargin>=2 && ~isempty(band_name)
        title(title_string, 'FontSize', 18);
    end
    set(gca,'FontSize',18,'LineWidth',2)
    box off
end