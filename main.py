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
from functions import zwroc_stopy_zwrotu, plot_plt, plot_hist
matplotlib.rcParams['agg.path.chunksize'] = 10000


dane = pd.read_csv(r'Guz_dane.csv', header=None)
print(dane)
dane.head()



# Stworzenie wykresu pobranych danych w skali liniowej i pół-logarytmicznej.

plot_plt("wykres pobranych danych w skali liniowej", 'czas', 'wartość', dane,
         '01_wykresPobranychDanychWSkaliLiniowej.png')
plot_plt("wykres pobranych danych w skali pół-logarytmicznej", 'czas', 'wartość', dane,
         '02_wykresPobranychDanychWSkaliPolLogarytmicznej.png', y_scale_log=True)

# Obliczenie logarytmicznych stóp zwrotu (logarithmic returns) pobranych danych oraz stworzenie ich wykresu.

stopyZwrotu = zwroc_stopy_zwrotu(dane)
plot_plt("wykres logarytmicznych stóp zwrotu (dane)", 'czas', 'logarytmiczna stopa zwrotu', stopyZwrotu,
         '03_LogarytmiczneStopyZwrotu(dane).png')

# Znormalizowanie szeregu stóp zwrotu (średnia równa 0 i odchylenie standardowe 1).

stopyZwrotu_ = np.array(stopyZwrotu).reshape(-1, 1)
scaler = StandardScaler()
scaler.fit(stopyZwrotu_)
stopyZwrotu_norm = scaler.transform(stopyZwrotu_)

print("Srednia (stopy zwrotu) = ", round(stopyZwrotu_norm.mean()))
print("Odchylenie Standardowe (stopy zwrotu) = ", round(stopyZwrotu_norm.std()))

# Wyrysowanie danych.
plot_plt("wykres znormalizowanych logarytmicznych stóp zwrotu (dane)", 'czas', 'logarytmiczna stopa zwrotu',
         stopyZwrotu_norm, '04_ZnormalizowaneLogarytmiczneStopyZwrotu(dane).png')

# ########################################################## TEMAT 2 ################################################

# Wygenerowanie danych o rozkładzie normalnym (white noise).
whiteNoise = np.random.normal(loc=0, scale=1, size=len(stopyZwrotu))

print("Srednia (white noise) = ", round(whiteNoise.mean()))
print("Odchylenie Standardowe (white noise) = ", round(whiteNoise.std()))

# Stworzenie wykresu wygenerowanych danych.
plot_plt("white noise", 'czas', 'wartość', whiteNoise, '05_WhiteNoise.png')

# Policzenie sumy skumulowanej (ruch Browna).
brown = np.cumsum(whiteNoise)

# Stworzenie wykresu dla sumy skumulowanej.
plot_plt("ruchy Browna", 'czas', 'wartość', brown, '06_RuchyBrowna.png')

# ########################################################## TEMAT 3 ################################################

# # Dla danych giełdowych (stopy zwrotu) i wygenerowanych policzenie histogramów.
# osobno
histStopyZwrotu = plot_hist("Histogram dla stóp zwrotu", 'wartość', 'częstość', stopyZwrotu_norm,
                            "07_HistogramDlaStopZwrotuNorm", bins=100, range=(-20, 20))
histWhiteNoise = plot_hist("Histogram dla white noise", 'wartość', 'częstość', whiteNoise, "08_HistogramDlaWhiteNoise",
                           bins=100,range=(-20, 20))
# skala log
plot_hist("Histogram dla stóp zwrotu(skala pół-logarytmiczna)", 'wartość', 'częstość', stopyZwrotu_norm,
          "07a_HistogramDlaStopZwrotuNorm(pollogarytmiczna)", bins=100, range=(-20, 20), y_scale_log=True)
plot_hist("Histogram dla white noise(skala pół-logarytmiczna)", 'wartość', 'częstość', whiteNoise,
          "08a_HistogramDlaWhiteNoise(pollogarytmiczna)", bins=100, range=(-20, 20), y_scale_log=True)

# wspolny
plot_hist("Histogramy dla stop zwrotu i white noise", 'wartość', 'częstośc', whiteNoise,
          "09_histogramy(pologarytmiczna).png", bins=100, range=(-20, 20), y_scale_log=True, close=False)
plot_hist("Histogramy dla stop zwrotu i white noise", 'wartość', 'częstośc', stopyZwrotu_norm,
          "09_histogramy(pologarytmiczna).png", bins=100, range=(-20, 20), y_scale_log=True, alpha=.5,
          legend=["white noise", "stopy zwrotu"])

# policzenie sumy binow
StopySuma = sum(histStopyZwrotu[0][:])
print("Suma binów (stopy zwrotu):", round(StopySuma))
WhiteSuma = sum(histWhiteNoise[0][:])
print("Suma binów (white noise):", round(WhiteSuma))

# Znormalizowanie histogramów (suma wartości słupków powinna być równa 1).

plot_hist("Znormalizowany histogram dla stóp zwrotu", 'wartość', 'częstość', (histStopyZwrotu[0] / StopySuma),
          "10_ZnormalizowanyHistogramDlaStopZwrotu.png", bins=100, range=(-20, 20), y_scale_log=True, bar=True,
          height=histStopyZwrotu[1][:-1])
plot_hist("Znormalizowany histogram dla white noise", 'wartość', 'częstość', (histWhiteNoise[0] / WhiteSuma),
          "11_ZnormalizowanyHistogramDlaWhiteNoise.png", bins=100, range=(-20, 20), y_scale_log=True, bar=True,
          height=histWhiteNoise[1][:-1])

# Policzenie parametrów rozkładów (skośności oraz kurtozy).
print("Dane giełdowe")
print("   Kurtoza: ", stat.kurtosis(stopyZwrotu_norm))
print("   Skośność: ", stat.skew(stopyZwrotu_norm))

print("White Noise")
print("   Kurtoza: ", stat.kurtosis(whiteNoise))
print("   Skośność: ", stat.skew(whiteNoise))

# Przedstawienie obu znormalizowanych rozkładów na jednym wykresie.
plot_hist("Histogramy dla stop zwrotu i white noise", 'wartość', 'częstośc', (histWhiteNoise[0] / WhiteSuma),
          "12_znormalizowane histogramy(pologarytmiczna).png", bins=100, range=(-20, 20), y_scale_log=True, bar=True,
          height=histWhiteNoise[1][:-1], close=False)
plot_hist("Histogramy dla stop zwrotu i white noise", 'wartość', 'częstośc', (histStopyZwrotu[0] / StopySuma),
          "12_znormalizowane histogramy(pologarytmiczna).png", bins=100, range=(-20, 20), y_scale_log=True, bar=True,
          height=histStopyZwrotu[1][:-1], alpha=.5, legend=["white noise", "stopy zwrotu"])

StopySuma_norm = sum(histStopyZwrotu[0] / StopySuma)
WhiteSuma_norm = sum(histWhiteNoise[0] / WhiteSuma)

print("Histogram znormalizowany - Suma binów (stopy zwrotu):", round(StopySuma_norm))
print("Histogram znormalizowany - Suma binów (white noise):", round(WhiteSuma_norm))

######################################################### TEMAT 4 ################################################

# Dla danych giełdowych (stopy zwrotu) i wygenerowanych (biały szum) policzenie funkcji autokorelacji.
#
abs_stopy = np.abs(stopyZwrotu_norm)  # modul stop zwrotu

acf_stopy = stats.tsa.acf(stopyZwrotu_norm, nlags=len(stopyZwrotu_norm) - 1)  # autokorelacja stop zwrotu
acf_stopy_abs = stats.tsa.acf(abs_stopy, nlags=len(stopyZwrotu_norm) - 1)  # autokorelacja modulu stop zwrotu

abs_white_noise = np.abs(whiteNoise)
np.reshape(abs_white_noise, (len(abs_white_noise), 1))  # modul bialego szumu

acf_white_noise = stats.tsa.acf(whiteNoise, nlags=len(whiteNoise) - 1)  # autokorelacja bialego szumu
acf_white_noise_abs = stats.tsa.acf(abs_white_noise, nlags=len(abs_white_noise) - 1)  # autokorelacja modulu bialego szumu


plt.ylim(-0.006, 0.006)
plot_plt("Autokorelacja stóp zwrotu", 'przesunięcie', 'korelacja', acf_stopy, "13_AutokorelacjaStopZwrotu.png")

plt.ylim(-0.006, 0.006)
plot_plt("Autokorelacja white noise", 'przesunięcie', 'korelacja', acf_white_noise, "14_AutokorelacjaWhiteNoise.png")

# Policzenie funkcji autokorelacji dla MODUŁU stóp zwrotu oraz modułu danych wygenerowanych. Sporządzenie odpowiedniego wykresu.

# plt.ylim(-0.025, 0.075)
# plot_plt("Autokorelacja modułu stóp zwrotu", 'przesunięcie', 'korelacja', acf_stopy_abs,
#          "15_AutokorelacjaModuluStopZwrotu.png")
#
#
# plot_plt("Autokorelacja modułu stóp zwrotu (skala logarytmiczna)", 'przesunięcie', 'korelacja', acf_stopy_abs,
#          "15a_AutokorelacjaModuluStopZwrotuogLog.png", x_scale_log=True, y_scale_log=True)
#
# plt.ylim(-0.025, 0.075)
# plot_plt("Autokorelacja modułu white noise", 'przesunięcie', 'korelacja', acf_white_noise_abs, "16_AutokorelacjaModuluWhiteNoise.png")
#
#
# plot_plt("Autokorelacja modułu white noise (skala logarytmiczna)", 'przesunięcie', 'korelacja', acf_white_noise_abs,
#          "16a_AutokorelacjaModuluWhiteNoiseLogLog.png", x_scale_log=True, y_scale_log=True)

# Policzenie widma mocy dla danych giełdowych (stopy zwrotu) i wygenerowanych (biały szum).

# Widmo mocy sygnału jest kwadratem modułu widma
# power spectral density - Dla sygnałów rzeczywistych, periodogram jest równy kwadratowi modułu widma. Przedstawia on rozklad mocy widmowej sygnału na jednostkę częstotliwości.


stopyZwrotu_norm = np.reshape(stopyZwrotu_norm, (-1, (len(stopyZwrotu_norm))))[0]

power_spectrum_stopy_zwrotu = np.abs(np.fft.fft(stopyZwrotu_norm)) ** 2
power_spectrum_white_noise = np.abs(np.fft.fft(whiteNoise)) ** 2
power_spectrum_stopy_zwrotu_abs = np.abs(np.fft.fft(abs(stopyZwrotu_norm))) ** 2
power_spectrum_white_noise_abs = np.abs(np.fft.fft(abs_white_noise)) ** 2

plt.ylim(1e-1, 1e+8)
plot_plt("Widmo mocy dla danych giełdowych", 'częstotliwość', 'widmo mocy', power_spectrum_stopy_zwrotu,
         "17_WidmoMocyDlaDanychGieldowych", x_scale_log=True, y_scale_log=True)

plot_plt("Widmo mocy dla white noise", 'częstotliwość', 'widmo mocy', power_spectrum_white_noise,
         "18_WidmoMocyDlaWhiteNoise", x_scale_log=True, y_scale_log=True)

# # Policzenie widma mocy dla modułu stóp zwrotu oraz modułu danych wygenerowanych. Sporządzenie odpowiedniego wykresu.
plot_plt("Widmo mocy dla modułu danych giełdowych", 'częstotliwość', 'widmo mocy', power_spectrum_stopy_zwrotu_abs,
         "19_WidmoMocyDlaModuluDanychGieldowych", x_scale_log=True, y_scale_log=True)

plot_plt("Widmo mocy dla modułu white noise", 'częstotliwość', 'widmo mocy', power_spectrum_white_noise_abs,
         "20_WidmoMocyDlaModuluWhiteNoise", x_scale_log=True, y_scale_log=True)
