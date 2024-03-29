import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ProcessingHelpers
import seaborn as sns
import colorsys

def plot_grouped_by_RUL(df_sub, leg=True, cols_data=None):

    if cols_data is None: 
        cols = [col for col in df_sub.columns if len(df_sub[col].unique()) > 2]
        cols_data = [col for col in cols if col.startswith('sen') or col.startswith('os')]

    g = sns.PairGrid(data=df_sub, x_vars="RUL", y_vars=cols_data,hue="unit_nr", height=2, aspect=6,)
    g = g.map(plt.plot, alpha=0.5)
    g = g.set(xlim=(df_sub['RUL'].max(),df_sub['RUL'].min()))
    if leg:
        g = g.add_legend()


def plot_rolling_by_unit(df_sub):
    nrs = df_sub['unit_nr'].unique()[:5]
    N = len(nrs)    
    
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    colors = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    fig, axes = plt.subplots(len(df_sub.columns),1, figsize = (15,22))
    axes = axes.flatten()
    
    for unit_nr, color in zip(nrs, colors):
        idx = df_sub['unit_nr']==unit_nr
        df_sub2 = df_sub.loc[idx].copy()
        plot_rolling_stats(df_sub2, 0.1, color, axes)
    
    for ax in axes:
        ax.set_xlim((df_sub.index.max(), df_sub.index.min()))
        
def plot_rolling_stats(df_sub2, wind_p, color, axes):
    wind_n = int(len(df_sub2) * wind_p)    
    
    for col, ax in zip(df_sub2.columns, axes):
        ser_std = df_sub2[col].rolling(wind_n).std()[::wind_n]
        ser_mean = df_sub2[col].rolling(wind_n).mean()[::wind_n]
        plot_line(ser_mean+ser_std, ser_mean, ser_mean-ser_std, color, ax=ax)
    
def plot_line(ser_high, ser_mid, ser_low, col, ax=None, fig=None):
    if ax is None:
        ax = plt.gca
    ax.plot(ser_high.index, ser_high.values, marker='o', linestyle='dashed', color=col)
    ax.plot(ser_mid.index, ser_mid.values, marker='o', color=col)
    ax.plot(ser_low.index, ser_low.values, marker='o', linestyle='dashed', color=col)
    
def plot_imshow(df, resample=True):
    cols_data = [col for col in df.columns if col.startswith('s')] 
    fig, axes = plt.subplots(int(np.ceil(len(cols_data)/3)), 3, figsize=(17, 10))
    axes = axes.flatten()

    m = len(df['unit_nr'].unique())

    if resample:
        n = 3 * m
    else:
        n = int(df['time'].values.max())

    t = np.linspace(0, 1, m)
    dc_tmp = dict([(col, np.zeros((m, n))) for col in cols_data])

    for i, (nr, d) in enumerate(df.groupby('unit_nr')):
        tmp = ProcessingHelpers.resample_fixed(d, n) if resample else d

        for ax, col in zip(axes, cols_data):
            v = np.zeros((1, n))
            v[:] = np.nan
            v[0, (n - len(tmp)):] = tmp[col].values
            dc_tmp[col][i, :] = v

    for ax, col in zip(axes, cols_data):

        mat = dc_tmp[col]
        ax.imshow(mat)
        ax.set_ylabel(col)
        ax.grid(False)

    for i in range(len(axes) - len(cols_data)):
        fig.delaxes(axes[-(i+1)])