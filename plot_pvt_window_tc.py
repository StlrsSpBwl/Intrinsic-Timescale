def plot_pvt_window_tc(df, tc_floor=0.5, min_n=10, figsize=(14, 8)):
    """
    Bar plots of TC metrics at the 3 PVT session windows.
    Per-day then averaged (mean ± SEM).
    """
    subjects = df.columns.tolist()
    windows = {'Morning\n(9-10)': 9, 'Noon\n(12-13)': 12, 'Afternoon\n(15-16)': 15}
    metrics = ['mean', 'pct_floor', 'cv', 'p95']
    labels = {'mean': 'Mean TC (s)', 'pct_floor': f'% at floor (≤{tc_floor}s)',
              'cv': 'CV', 'p95': '95th percentile (s)'}
    colors = {'Morning\n(9-10)': '#E24B4A', 'Noon\n(12-13)': '#378ADD',
              'Afternoon\n(15-16)': '#1D9E75'}

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Compute per-day metrics at each window
    data = {}
    for subj in subjects:
        data[subj] = {}
        vals = df[subj].dropna()
        for wname, hr in windows.items():
            day_groups = vals[vals.index.hour == hr].groupby(vals[vals.index.hour == hr].index.date)
            day_vals = []
            for date, grp in day_groups:
                x = grp.values
                if len(x) < min_n:
                    continue
                day_vals.append({
                    'mean': np.mean(x),
                    'pct_floor': (x <= tc_floor).mean() * 100,
                    'cv': np.std(x, ddof=1) / np.mean(x) if np.mean(x) > 0 else np.nan,
                    'p95': np.percentile(x, 95),
                })
            dm = pd.DataFrame(day_vals)
            data[subj][wname] = {
                m: (dm[m].mean(), dm[m].sem()) for m in metrics
            }

    n_subj = len(subjects)
    fig, axes = plt.subplots(len(metrics), n_subj, figsize=figsize, squeeze=False)

    x_pos = np.arange(len(windows))
    width = 0.6
    wnames = list(windows.keys())

    for col, subj in enumerate(subjects):
        for row, m in enumerate(metrics):
            ax = axes[row, col]
            means = [data[subj][w][m][0] for w in wnames]
            sems = [data[subj][w][m][1] for w in wnames]
            bars = ax.bar(x_pos, means, width, yerr=sems, capsize=5,
                          color=[colors[w] for w in wnames], alpha=0.8,
                          edgecolor='white', linewidth=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(wnames if row == len(metrics) - 1 else [''] * 3)
            ax.set_ylabel(labels[m])
            if row == 0:
                ax.set_title(subj, fontsize=14, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig