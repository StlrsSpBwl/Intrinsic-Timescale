function out = tc_signal_from_edf_file(edf_path, bandHz)

    if nargin<2 || isempty(bandHz)
        bandHz = [1 40];
    end

    % --- Parameters (Zilio/Honey style) ---
    P.win_sec     = 120;
    P.step_sec    = 30;
    P.maxLag_sec  = 0.5;      % Zilio et al 2012
    P.accBadZ     = 6;
    P.eegAmp      = 0.3;
    P.maxBadFrac  = 0.30;

    P.filtOrder   = 4;
    P.pad_sec     = 5;

    P.Nshuff      = 200;

    % --- Load EDF ---
    data = edfread(edf_path,'TimeOutputType','datetime');
    cn = data.Properties.VariableNames;

    EEG = cell2mat(data{:,contains(cn,'EEG')}(:,1));
    Acc = data{:,contains(cn,'Accelerometer')};

    Acc_x = cell2mat(Acc(:,1));
    Acc_y = cell2mat(Acc(:,2));
    Acc_z = cell2mat(Acc(:,3));

    fs_EEG = numel(data{1,contains(cn,'EEG')}{1});
    fs_Acc = numel(Acc{1,1});

    % --- Accel processing ---
    analog2accel = @(x) (68.6/2047)*x-68.6;
    Acc_x = analog2accel(fillmissing(Acc_x,'nearest'));
    Acc_y = analog2accel(fillmissing(Acc_y,'nearest'));
    Acc_z = analog2accel(fillmissing(Acc_z,'nearest'));
    acc_mag = sqrt(Acc_x.^2 + Acc_y.^2 + Acc_z.^2);
    % acc_zs  = zscore(resample(acc_mag, fs_EEG, fs_Acc));
    acc_mag_rs = resample(acc_mag,fs_EEG,fs_Acc);
    % more robust z-score calculation
    medA = median(acc_mag_rs,'omitnan');
    madA = mad(acc_mag_rs,1);
    acc_zs = (acc_mag_rs-medA)/(1.4826*madA+eps);

    % --- Bandpass filter design ---
    [b,a] = butter(P.filtOrder, bandHz/(fs_EEG/2), 'bandpass');

    % --- Windowing ---
    winS    = round(P.win_sec*fs_EEG);
    stepS   = round(P.step_sec*fs_EEG);
    maxLagS = round(P.maxLag_sec*fs_EEG);
    padS    = round(P.pad_sec*fs_EEG);

    nW = floor((length(EEG)-winS)/stepS)+1;

    ACW = nan(nW,1);                % will store full width at half maximum (seconds)
    keep = false(nW,1);

    Nshuff = P.Nshuff;
    ACW_surr_all = nan(nW, Nshuff);

    % timestamps (center of each window)
    t0 = data.("Record Time")(1);
    time_dt = t0 + seconds((0:nW-1)*P.step_sec + P.win_sec/2);

    % --- Progress bar ---
    h = waitbar(0,'Signal ACW: processing windows...');
    cleanup = onCleanup(@() (ishandle(h) && close(h)));

    thr = 0.5;   % half of lag0 peak (=1 under 'coeff'): threshold of FWHM of ACF function is essentially 0.5

    for w = 1:nW
        if mod(w,20)==0 || w==1 || w==nW
            if ~ishandle(h), error('Progress bar closed by user.'); end
            waitbar(w/nW, h, sprintf('Signal ACW window %d/%d', w, nW));
        end

        i0 = (w-1)*stepS + 1;
        i1 = i0 + winS - 1;

        seg_raw = EEG(i0:i1);
        acc_raw = acc_zs(i0:i1);

        badFrac = mean((abs(acc_raw)>P.accBadZ) | (abs(seg_raw)>P.eegAmp) | isnan(seg_raw));
        if badFrac > P.maxBadFrac
            continue
        end

        % filter with padding
        p0 = max(1, i0-padS);
        p1 = min(length(EEG), i1+padS);
        segp = double(EEG(p0:p1));
        segp = fillmissing(segp,'nearest');

        segp_f = filtfilt(b,a,segp);
        seg = segp_f((i0-p0+1):(i1-p0+1));
        seg = seg - mean(seg);

        % --- ACF normalized ---
        r = xcorr(seg, maxLagS, 'coeff');       % lags -maxLag..+maxLag
        acf0 = r(maxLagS+1:end);                % lag 0..maxLag (includes lag0)

        % --- Full width at half maximum (positive-lag half-width ×2) ---
        k = find(acf0(2:end) <= thr, 1, 'first');   % search from lag1
        if isempty(k)
            % no crossing within maxLag -> censor at full width 2*maxLag
            ACW(w) = 2*P.maxLag_sec;
        else
            t_half = k / fs_EEG;                    
            ACW(w) = 2*t_half;                      % FULL width
        end

        keep(w) = true;

        % --- Surrogates: Müller-style temporal scrambling (permute time samples) ---
        % doing phase randomization won't change anything due to
        % Wiener-Khinchin theorem
        for s = 1:Nshuff
            segS = seg(randperm(numel(seg)));
            rS = xcorr(segS, maxLagS, 'coeff');
            acfS0 = rS(maxLagS+1:end);

            kS = find(acfS0(2:end) <= thr, 1, 'first');
            if isempty(kS)
                ACW_surr_all(w,s) = 2*P.maxLag_sec;
            else
                ACW_surr_all(w,s) = 2*(kS/fs_EEG);
            end
        end
    end

    % --- Summaries + p-values (mid-p tie handling) ---
    ACW_surr_mean = mean(ACW_surr_all,2,'omitnan');
    ACW_surr_p05  = prctile(ACW_surr_all,5,2);
    ACW_surr_p95  = prctile(ACW_surr_all,95,2);

    p_hi = nan(nW,1);
    p_lo = nan(nW,1);
    for w = 1:nW
        if ~keep(w) || isnan(ACW(w)), continue; end
        svals = ACW_surr_all(w,:);
        svals = svals(~isnan(svals));
        if isempty(svals), continue; end
        gt = sum(svals >  ACW(w));
        eq = sum(svals == ACW(w));
        lt = sum(svals <  ACW(w));
        N  = numel(svals);
        p_hi(w) = (1 + gt + 0.5*eq) / (1 + N);
        p_lo(w) = (1 + lt + 0.5*eq) / (1 + N);
    end

    out.ACW = ACW;
    out.time = time_dt;
    out.keep = keep;
    out.bandHz = bandHz;

    out.ACW_surr_all  = ACW_surr_all;
    out.ACW_surr_mean = ACW_surr_mean;
    out.ACW_surr_p05  = ACW_surr_p05;
    out.ACW_surr_p95  = ACW_surr_p95;

    out.p_hi = p_hi;
    out.p_lo = p_lo;
    out.params = P;
end