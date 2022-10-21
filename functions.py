from __future__ import division
from audioop import minmax
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy.stats.mstats as stat
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as stats
from sklearn.preprocessing import StandardScaler
from scipy.signal import periodogram

def plot_plt(title, xlabel, ylabel, data, filename, scalexy=[True, True], x_scale_log=False, y_scale_log=False):
    if x_scale_log:
        plt.xscale('log')
    if y_scale_log:
        plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(data, scalex=scalexy[0], scaley=scalexy[1])
    plt.savefig('./charts/%d' % filename)
    plt.close()


def plot_hist(title, xlabel, ylabel, data, filename, bins, range, x_scale_log=False, y_scale_log=False, close=True,
              alpha=1, legend=0, bar=False, height=0):
    if x_scale_log:
        plt.xscale('log')
    if y_scale_log:
        plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if bar:
        hist = plt.bar(height, data, width=(range[1]-range[0])/bins, alpha=alpha)
    else:
        hist = plt.hist(data, bins=bins, range=range, alpha=alpha)
    if legend:
        plt.legend(legend)
    if close:
        plt.savefig(filename)
        plt.close()

    return hist


def zwroc_stopy_zwrotu(df):
    stopy_arr = []
    for i in range(len(df) - 1):
        stopy_arr.append((df[0][i + 1] - df[0][i]) / df[0][i])
    return stopy_arr

