#pip install scikit-posthocs
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
from scikit_posthocs import posthoc_dunn

def pvt_eda(df_trials):
    """
    df_trials: one subject's trials DataFrame from compile_pvt_data()
        (columns: ['subject', 'session', 'trial', 'rt'])
    """
    # Summary stats per session
    summary = df_trials.groupby('session')['rt'].agg(
        median = 'median',
        min='min',
        max='max',
        q1 = lambda x:x.quantile(0.25)
        q3 = lambda x:x.quantile(0.75)
    )
    summary['iqr'] = summary['q3'] - summary['q1']
    print('===Descriptive stats===')
    print(summary[['min','q1','median','q3','max','iqr']])
    # Normality test (Shapiro-Wilk per session)
    print('\n===Shapiro-Wilk normality test===')
    groups = {}
    all_normal=True
    for session, grp in df_trials.groupby('session'):
        rt = grp['rt'].values
        groups[session] = rt
        stat, p = sp_stats.shapiro(rt)
        normal = p > 0.05
        all_normal &= normal
        print(f' {session}:W={stat:.4f},p={p:.4g} {"(normal)" if normal else "(non-normal)"}')
    group_list = list(groups.values())
    group_names = list(groups.keys())
    if len(group_list)<2:
        print('\nOnly one session - skipping comparisons.')
    elif all_normal:
        print('\n===ANOVA(all groups normal)===')
        f_stat,p_val = sp_stats.f_oneway(*group_list)
        print(f' F={f_stat:.4f}, p={p_val:.4g}')
        if p_val<0.05:
            print('\n===Post-hoc: Tukey HSD===')
            tukey = sp_stats.tukey_hsd(*group_list)
            for i in range(len(group_names)):
                for j in range(i+1,len(group_names)):
                    p=tukey.pvalue[i][j]
                    print(f' {group_names[i]} vs {group_names[j]}: p={p:.4g} {"(significant)" if p<0.05 else ""}')
    else:
        print('\n===Kruskal-Wallis (non-normal groups)===')
        h_stat,p_val = sp_stats.kruskal(*group_list)
        print(f' H={h_stat:.4f}, p={p_val:.4g}')
        if p_val<0.05:
            print('\n===Post-hoc: Dunn\'s test===')
            dunn = posthoc_dunn(df_trials, val_col='rt',group_col='session', p_adjust='bonferroni')
            print(dunn.to_string())


    # KDE overlay plot
    fig,ax = plt.subplots(figsize=(8,4))
    for session, grp in df_trials.groupby('session'):
        grp['rt'].plot.kde(ax=ax, label=session, legend=True,linewidth=1.5)
    ax.set_xlabel('Reaction Time (s)')
    ax.set_ylabel('Density')
    ax.set_xlim(0,df_trials['rt'].quantile(0.99)*1.2)
    ax.legend()
    ax.set_title(f"{df_trials['subject'].iloc[0]} - PVT RT Distributions by session")
    plt.tight_layout()
    plt.show()
    return summary