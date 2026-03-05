function TCTimeTrace(TC_struct,light_on_time,light_off_time,band_name,smooth_flag,smooth_win)
    if nargin<5,smooth_flag = false;end
    if nargin<6, smooth_win = 120;end
    TimeTable = TC_struct.(band_name);
    t = TimeTable.Time;
    value = TimeTable.TC;
    if smooth_flag
        raw_color = [171/255,217/255,233/255];
        smooth_color = [44/255,123/255,182/255];
        plot(t,value,"Color",raw_color)
        hold on
        plot(t,movmean(value,smooth_win,'omitnan'),'Color',smooth_color,'LineWidth',2)
    else
        smooth_color = [44/255,123/255,182/255];
        plot(t,value,"Color",smooth_color,'LineWidth',1)
    end
    ylim([0 max(value)*1.1]) %TC is non-negative
    xlim([min(t),max(t)])
    % shading based on light-on and off time
    start_time = t(1);
    end_time = t(end);
    total_days = ceil(days(end_time-start_time));
    tick_time = [];
    for day = 0:total_days
        if light_on_time>light_off_time
            light_off_start = dateshift(start_time,'start','day','next')+days(day)-hours(24-light_off_time);
            light_off_end = dateshift(start_time,'start','day','next')+days(day)-hours(24-light_on_time);
        else
            light_off_start = dateshift(start_time,'start','day','next')+days(day)-hours(24-light_off_time);
            light_off_end = dateshift(start_time,'start','day','next')+days(day+1)-hours(24-light_on_time);
        end
        tick_time = [tick_time,light_off_start];
        patch([light_off_start,light_off_end,light_off_end,light_off_start], ...
            [0,0,max(value)*1.1,max(value)*1.1], ...
            [0.5 0.5 0.5],'FaceAlpha',0.3,'EdgeColor','none');
    end
    set(gca,'FontSize',20)
    xticks(tick_time)
    day_number = 1:total_days;
    new_tick_labels = arrayfun(@(x) sprintf("Day %d",x),day_number,'UniformOutput',false);
    xticklabels(new_tick_labels);
    ylabel("TC (s)",'FontSize',24)
    title(band_name,'FontSize',24)
    box off
end