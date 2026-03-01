function out = tc_power_from_edf_file(edf_path, bandHz)

    if nargin<2 || isempty(bandHz)
        bandHz = [1 40];
    end

    % --- Parameters (Müller-style) ---
    P.bin_sec     = 0.5;        % change to 0.125 if you want finer resolution
    P.win_sec     = 120;
    P.step_sec    = 30;
    P.maxLag_sec  = 60;

    P.baseline_sec = [40 60]; % BUG REPORT FIX: baseline window

    P.accBadZ     = 4;
    P.eegAmp      = 0.3;
    P.maxBadFrac  = 0.30;

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
    acc_zs  = zscore(resample(acc_mag, fs_EEG, fs_Acc));

    % --- Bin EEG to power time series ---
    L = round(P.bin_sec * fs_EEG);
    nBins = floor(length(EEG)/L);

    EEG = EEG(1:nBins*L);
    acc_zs = acc_zs(1:nBins*L);

    EEGb = reshape(EEG, L, nBins)';
    ACCb = reshape(acc_zs, L, nBins)';
    % use rms for rejection
    acc_mag_rs = resample(acc_mag,fs_EEG,fs_Acc);
    acc_mag_rs = acc_mag_rs(1:nBins*L);
    Acc_mag = reshape(acc_mag_rs,L,nBins)';
    acc_rms = nan(nBins,1);
    for i=1:nBins
        acc_rms(i)=sqrt(mean(Acc_mag(i,:).^2,'omitnan'));
    end
    medA = median(acc_rms,'omitnan');
    madA = mad(acc_rms,1);
    zA = (acc_rms-medA)/(1.4826*madA+eps);
    bad_acc = zA>10;


    powTS = nan(nBins,1);
    bad   = false(nBins,1);

    h = waitbar(0,'Power TC: computing binned power...');
    cleanup = onCleanup(@() (ishandle(h) && close(h)));
    n_nan =0; 
    n_eeg=0;
    n_acc=0;
    for i = 1:nBins
        if mod(i,2000)==0 || i==1 || i==nBins
            if ~ishandle(h), error('Progress bar closed by user.'); end
            waitbar(i/nBins, h, sprintf('Power bins %d/%d', i, nBins));
        end

        seg = EEGb(i,:);
        % acc = ACCb(i,:);

        % if any(isnan(seg)) || max(abs(seg))>P.eegAmp || bad_acc(i)
        %     bad(i) = true;
        %     continue
        % end
        if any(isnan(seg))
            bad(i)=true;n_nan=n_nan+1;continue
        end
        if max(abs(seg))>P.eegAmp
            bad(i)=true;n_eeg=n_eeg+1;continue
        end
        if bad_acc(i)
            bad(i)=true;n_acc = n_acc+1;continue
        end

        [Pxx,f] = pwelch(seg, hann(L), 0, 2^nextpow2(L), fs_EEG);
        idx = (f>=bandHz(1)) & (f<=bandHz(2));
        if ~any(idx)
            bad(i) = true;
            continue
        end
        powTS(i) = log10(median(Pxx(idx),'omitnan'));
        if ~isfinite(powTS(i)), bad(i)=true; end
    end
    fprintf('Rjection: NaN=%.1f%%, EEG amp=%.1f%%, Accel=%.1f%%,Total=%.1f%%', ...
        100*n_nan/nBins,100*n_eeg/nBins,100*n_acc/nBins,100*mean(bad));
    powTS_interp = powTS;
    nanMask = isnan(powTS)|~isfinite(powTS);
    goodIdx = find(~nanMask);
    if numel(goodIdx)>=2
        powTS_interp = interp1(goodIdx,powTS(goodIdx),(1:nBins)',"linear",'extrap');
    end
    fprintf('Power bins: %d total, %d bad (%.1f%%), interpolated.\n', ...
        nBins, sum(nanMask),100*mean(nanMask));

    % --- Windowed ACF of power TS ---
    acfWin  = round(P.win_sec / P.bin_sec);
    acfStep = round(P.step_sec / P.bin_sec);
    maxLag  = round(P.maxLag_sec / P.bin_sec);

    nW = floor((nBins-acfWin)/acfStep)+1;

    TC = nan(nW,1);
    keep = false(nW,1);   % CRITICAL BUG FIX: initialize keep

    lags_sec = (1:maxLag) * P.bin_sec;
    baseIdx = (lags_sec>=P.baseline_sec(1)) & (lags_sec<=P.baseline_sec(2));

    waitbar(0,h,'Power TC: computing TC windows...');

    for w = 1:nW
        if mod(w,50)==0 || w==1 || w==nW
            if ~ishandle(h), error('Progress bar closed by user.'); end
            waitbar(w/nW, h, sprintf('Power windows %d/%d', w, nW));
        end

        i0 = (w-1)*acfStep + 1;
        i1 = i0 + acfWin - 1;

        if mean(bad(i0:i1)) > P.maxBadFrac
            continue
        end

        seg = powTS_interp(i0:i1);
        if any(isnan(seg)) || any(~isfinite(seg))
            continue
        end
        seg = seg - mean(seg);

        r = xcorr(seg, maxLag, 'coeff');
        acf = r(maxLag+2:end);    % lag 1..maxLag (Müller excludes lag0)

        baseline = median(acf(baseIdx),'omitnan');
        thr = 0.5*(acf(1) - baseline) + baseline;  % BUG REPORT FIX

        k = find(acf <= thr, 1, 'first');
        if isempty(k)
            TC(w) = P.maxLag_sec;
        else
            TC(w) = lags_sec(k);
        end

        keep(w) = true;
    end

    % timestamps (center of each window)
    t0 = data.("Record Time")(1);
    time_dt = t0 + seconds((0:nW-1)*P.step_sec + P.win_sec/2);

    % --- Surrogates: Müller-style temporal scrambling of power TS (permute good bins) ---
    Nshuff = P.Nshuff;
    TC_surr_all = nan(nW, Nshuff);

    goodBins = ~bad & ~isnan(powTS) & isfinite(powTS);
    v = powTS(goodBins);

    waitbar(0,h,'Power TC: running surrogates...');

    for s = 1:Nshuff
        if mod(s,5)==0 || s==1 || s==Nshuff
            if ~ishandle(h), error('Progress bar closed by user.'); end
            waitbar(s/Nshuff, h, sprintf('Surrogate %d/%d', s, Nshuff));
        end

        powShuff = powTS;
        powShuff(goodBins) = v(randperm(numel(v))); % permutation to scramble the data

        for w = 1:nW
            if ~keep(w) || isnan(TC(w)), continue; end

            i0 = (w-1)*acfStep + 1;
            i1 = i0 + acfWin - 1;

            segS = powShuff(i0:i1);
            if any(isnan(segS)) || any(~isfinite(segS)), continue; end
            segS = segS - mean(segS);

            rS = xcorr(segS, maxLag, 'coeff');
            acfS = rS(maxLag+2:end);

            baselineS = median(acfS(baseIdx),'omitnan');
            thrS = 0.5*(acfS(1) - baselineS) + baselineS;

            kS = find(acfS <= thrS, 1, 'first');
            if isempty(kS)
                TC_surr_all(w,s) = P.maxLag_sec;
            else
                TC_surr_all(w,s) = lags_sec(kS);
            end
        end
    end

    TC_surr_mean = mean(TC_surr_all,2,'omitnan');
    TC_surr_p05  = prctile(TC_surr_all,5,2);
    TC_surr_p95  = prctile(TC_surr_all,95,2);

    % mid-p p-values
    p_hi = nan(nW,1);
    p_lo = nan(nW,1);
    for w = 1:nW
        if ~keep(w) || isnan(TC(w)), continue; end
        svals = TC_surr_all(w,:);
        svals = svals(~isnan(svals));
        if isempty(svals), continue; end
        gt = sum(svals >  TC(w));
        eq = sum(svals == TC(w));
        lt = sum(svals <  TC(w));
        N  = numel(svals);
        p_hi(w) = (1 + gt + 0.5*eq) / (1 + N);
        p_lo(w) = (1 + lt + 0.5*eq) / (1 + N);
    end

    out.TC = TC;
    out.time = time_dt;
    out.keep = keep;
    out.bandHz = bandHz;

    out.powTS = powTS;
    out.badBins = bad;

    out.TC_surr_all  = TC_surr_all;
    out.TC_surr_mean = TC_surr_mean;
    out.TC_surr_p05  = TC_surr_p05;
    out.TC_surr_p95  = TC_surr_p95;

    out.p_hi = p_hi;
    out.p_lo = p_lo;
    out.params = P;
end