function TC = tc_from_hgpow(hgPow,badBins,P)
dt=P.powWin_sec;
acfWin_n = round(P.acfWin_sec/dt);
acfStep_n = round(P.acfStep_sec/dt);
maxLag_n = round(P.maxLag_sec/dt);
nBins = length(hgPow);
nW = floor((nBins-acfWin_n)/acfStep_n)+1;
acfMat = nan(nW, maxLag_n);
keepWin = false(nW,1);
for w=1:nW
    i0=(w-1)*acfStep_n+1;
    i1 = i0+acfWin_n-1;
    if i1>nBins
        break
    end
    if mean(badBins(i0:i1))>P.maxBadFrac||any(isnan(hgPow(i0:i1)))
        continue
    end
    x=hgPow(i0:i1);
    x=x-mean(x);
    r = xcorr(x,maxLag_n,'coeff');
    acfMat(w,:)=r(maxLag_n+2:end);
    keepWin(w)=true;
end
acfMed = median(acfMat(keepWin,:),1,'omitnan');
lags_sec = (1:maxLag_n)*dt;
baseIdx = lags_sec>=P.baseline_sec(1)&lags_sec<=P.baseline_sec(2);
baseline = median(acfMed(baseIdx),'omitnan');
thr = baseline +0.5*(acfMed(1)-baseline);
k = find(acfMed<thr,1,"first");
if isempty(k)
    TC=P.maxLag_sec;
else
    TC=lags_sec(k);
end
end