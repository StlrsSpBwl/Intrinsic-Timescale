function out = tc_from_edf_file(edf_path)
% compute TC from one edf using the method from Muller et. al 2025
% step 1: high-gamma (56~96 Hz) log10 power in 125 ms bins
% step 2: acf in 120s windows with 90 s overlap, max lag 60s (excluding lag
% 0)
% step 3: median ACF across windows
% step 4: TC = first crossing below half-max threshold relative to 40s~60s baseline
% step 5: surrogate TC from time-shuffled high-gamma power

% Parameters
%P.fs_EEG = 500;
P.powWin_sec= 0.125;
P.hgBand = [56 96];
P.nfft = 64;
P.acfWin_sec = 120;
P.acfStep_sec = 30; %step of 30s = 90s overlap
P.maxLag_sec = 60;
P.baseline_sec = [40 60];
P.accBadZ = 4;
P.powBadZ=4;
P.maxBadFrac = 0.15;
P.accMinRaw = 1800;
P.analog2accel =@(x) (68.6/2047)*x-68.6;

% Load EDF data 
data = edfread(edf_path,"TimeOutputType","datetime");
cn = data.Properties.VariableNames;
Acc_idx = contains(cn,'Accelerometer');
EEG_idx = contains(cn,'EEG');
EEG_col = data{:,EEG_idx};
Acc_col = data{:,Acc_idx};

EEG = cell2mat(EEG_col(:,1));
Acc_x = cell2mat(Acc_col(:,1));
Acc_y=cell2mat(Acc_col(:,2));
Acc_z = cell2mat(Acc_col(:,3));
fs_Acc = numel(Acc_col{1,1});
fs_EEG = numel(EEG_col{1,1});
Acc_x(Acc_x<P.accMinRaw) = NaN;
Acc_y(Acc_y<P.accMinRaw)=NaN;
Acc_z(Acc_z<P.accMinRaw)=NaN;
Acc_x = P.analog2accel(fillmissing(Acc_x,'nearest'));
Acc_y = P.analog2accel(fillmissing(Acc_y,'nearest'));
Acc_z = P.analog2accel(fillmissing(Acc_z,'nearest'));
acc_mag = sqrt(Acc_x.^2+Acc_y.^2+Acc_z.^2);
% Resampling accel magnitude to EEG fs
acc_resamp = resample(acc_mag,fs_EEG,fs_Acc);
acc_z = zscore(acc_resamp);
% Build 125 ms bins
L = round(P.powWin_sec*fs_EEG);
dt=L/fs_EEG;
nBins = floor(length(EEG)/L);
EEG = EEG(1:nBins*L);
need = nBins*L;
if length(acc_z)<need
    acc_z(end+1:need)=NaN;
else
    acc_z = acc_z(1:need);
end

EEGBins = reshape(EEG,L,nBins)';
AccBins = reshape(acc_z,L,nBins)';
% High-gamma powr per 125ms
hgPow = nan(nBins,1);
accMax = nan(nBins,1);
hgLoRatio = nan(nBins,1);
bbPow = nan(nBins,1);
for i =1:nBins
    x = EEGBins(i,:);
    a = AccBins(i,:);
    accMax(i)=max(abs(a));
    if any(isnan(x))||max(abs(x))>0.3
        hgPow(i)=NaN;
        continue
    end
    [Pxx,f] = pwelch(x,hann(L),0,P.nfft,fs_EEG);
    % Band indices
    hgIdx = (f>=56&f<=96);
    loIdx = (f>=8&f<=30);
    bbIdx = (f>=1 & f<=120);
    if ~(any(hgIdx)&&any(loIdx)&&any(bbIdx))
        hgPow(i) = NaN;
        continue
    end
    % hgIdx = (f>=P.hgBand(1)) & (f<=P.hgBand(2));
    % if any(hgIdx)
    %     hgPow(i) = median(Pxx(hgIdx),'omitnan');
    % end
    % Powers
    hg = median(Pxx(hgIdx),'omitnan');
    lo = median(Pxx(loIdx),'omitnan');
    bb = median(Pxx(bbIdx),'omitnan');
    hgPow(i) = log10(hg);
    % Diagnostics for later rejection
    hgLoRatio(i)=log10(hg)-log10(lo);
    bbPow(i)=log10(bb);
end
hgPow = log10(hgPow);
% Bad-bin mask (accel+power)
badBins = false(nBins,1);
badBins = badBins | (accMax>P.accBadZ);
% hg power spike
zHg = zscore(hgPow);
badBins = badBins | (zHg>P.powBadZ);
% EMG/spectral tilt
zTilt = zscore(hgLoRatio);
badBins = badBins | (zTilt>4);
% broadband bursts
zBB = zscore(bbPow);
badBins = badBins | (zBB>4);
% NaN always bad
badBins = badBins|isnan(hgPow);
% Compute ACFs in 120 s windows
acfWin_n = round(P.acfWin_sec/dt);
acfStep_n = round(P.acfStep_sec/dt);
maxLag_n = round(P.maxLag_sec/dt);
nW = floor((nBins-acfWin_n)/acfStep_n)+1;
acfMat = nan(nW,maxLag_n);
keepWin = false(nW,1);
for w=1:nW
    i0 = (w-1)*acfStep_n+1;
    i1 = i0+acfWin_n-1;
    if i1>nBins
        break
    end
    if mean(badBins(i0:i1))>P.maxBadFrac
        continue
    end
    x=hgPow(i0:i1);
    if any(isnan(x))
        continue
    end
    x=x-mean(x);
    r=xcorr(x,maxLag_n,'coeff');
    acfMat(w,:)=r(maxLag_n+2:end);%ignore the first item coz it's lag 0
    keepWin(w)=true;
end
lags_sec = (1:maxLag_n)*dt;
% TC per 120s window (one value every ~30s step)
baseIdx = lags_sec>=P.baseline_sec(1)&lags_sec<=P.baseline_sec(2);
TC_series = nan(nW,1);
for w=1:nW
    if ~keepWin(w),continue;end
    acf = acfMat(w,:);
    baseline = median(acf(baseIdx),'omitnan');
    thr=baseline+0.5*(acf(1)-baseline);%half max is the threshold
    k = find(acf<thr,1,'first');
    if isempty(k)
        TC_series(w)=P.maxLag_sec;
    else
        TC_series(w)=lags_sec(k);
    end
end
%global summary
TC_real = median(TC_series,'omitnan');
%optional daily median
acfMed = median(acfMat(keepWin,:),1,'omitnan');

% Create timestamps (datetime)
TC_time_sec = ((0:nW-1)*P.acfStep_sec)+P.acfWin_sec/2;
t0=data.("Record Time")(1);
TC_time_dt = t0+seconds(TC_time_sec);

% surrogate TC time series
Nshuff = 200;
TC_surr_all = nan(nW,Nshuff);
hgPow_fill = hgPow;
hgPow_fill(isnan(hgPow_fill))=datasample(hgPow(~isnan(hgPow)),sum(isnan(hgPow_fill)),'Replace',true);
for s = 1:Nshuff
    hgShuff = hgPow_fill(randperm(length(hgPow_fill)));
    for w=1:nW
        if ~keepWin(w)
            continue;
        end
        i0=(w-1)*acfStep_n+1;
        i1 = i0+acfWin_n-1;
        seg = hgShuff(i0:i1);
        seg = seg-mean(seg);
        r = xcorr(seg,maxLag_n,'coeff');
        acfS = r(maxLag_n+2:end);
        baselineS = median(acfS(baseIdx),'omitnan');
        thrS = baselineS +0.5*(acfS(1)-baselineS);
        kS = find(acfS<thrS,1,'first');
        if isempty(kS)
            TC_surr_all(w,s)=P.maxLag_sec;
        else
            TC_surr_all(w,s)=lags_sec(kS);
        end
    end
end
% get the summary
TC_surr_mean = mean(TC_surr_all,2,'omitnan');
TC_surr_p05=prctile(TC_surr_all,5,2);
TC_surr_p95=prctile(TC_surr_all,95,2);
out.TC_series = TC_series;
out.TC_time_sec = TC_time_sec;
out.TC_time_dt = TC_time_dt;
out.TC_surr_mean = TC_surr_mean;
out.TC_surr_all = TC_surr_all;
out.TC_surr_p05 = TC_surr_p05;
out.TC_surr_p95 = TC_surr_p95;
% out.TC_real = TC_real;
% out.TC_surr = TC_surr;
end
