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



#Pobranie danych giełdowych np. ze strony https://finance.yahoo.com

dane = pd.read_csv(r'../MSFT.csv')
print(dane)
dane.head()

# Stworzenie wykresu pobranych danych w skali liniowej i pół-logarytmicznej.
plt.title("wykres pobranych danych w skali liniowej")
plt.xlabel('czas')
plt.ylabel('wartość')
plt.plot(dane['Open'], scaley=True, scalex=True)
plt.savefig('01wykres pobranych danych w skali liniowej.png')
plt.close()


plt.yscale('log')
plt.title("wykres pobranych danych w skali pół-logarytmicznej")
plt.xlabel('czas')
plt.ylabel('wartość')
plt.plot(dane['Open'], scaley=True, scalex=True)
plt.savefig('02wykres pobranych danych w skali pół-logarytmicznej.png')
plt.close()

# Obliczenie logarytmicznych stóp zwrotu (logarithmic returns) pobranych danych oraz stworzenie ich wykresu.

def zwroc_stopy_zwrotu(df):
    stopy_arr = []
    for i in range(len(df)-1):
        stopy_arr.append((df['Open'][i+1] - df['Open'][i])/df['Open'][i])
    return stopy_arr



stopyZwrotu = zwroc_stopy_zwrotu(dane)
plt.title("logarytmiczne stopy zwrotu (dane)")
plt.xlabel('czas')
plt.ylabel('logarytmiczna stopa zwrotu')
plt.plot(stopyZwrotu,scaley=True, scalex=True)
plt.savefig('03logarytmiczne stopy zwrotu (dane).png')
plt.close()

# Znormalizowanie szeregu stóp zwrotu (średnia równa 0 i odchylenie standardowe 1).

stopyZwrotu_ = np.array(stopyZwrotu).reshape(-1,1)
scaler = StandardScaler()
scaler.fit(stopyZwrotu_)
stopyZwrotu_norm = scaler.transform(stopyZwrotu_)

print("Srednia (stopy zwrotu) = ", round(stopyZwrotu_norm.mean()))
print("Odchylenie Standardowe (stopy zwrotu) = ", round(stopyZwrotu_norm.std()))


# Wyrysowanie danych.
plt.title("znormalizowane logarytmiczne stopy zwrotu (dane)")
plt.xlabel('czas')
plt.ylabel('logarytmiczna stopa zwrotu')
plt.plot(stopyZwrotu_norm,scaley=True, scalex=True)
plt.savefig('04znormalizowane logarytmiczne stopy zwrotu (dane).png')
plt.close()

# ########################################################## TEMAT 2 ################################################

# Wygenerowanie danych o rozkładzie normalnym (white noise).
whiteNoise = np.random.normal(loc=0, scale=1, size=9074)
print("Srednia (white noise) = ", round(whiteNoise.mean()))
print("Odchylenie Standardowe (white noise) = ", round(whiteNoise.std()))

# Stworzenie wykresu pobranych danych.
plt.title("white noise")
plt.xlabel('czas')
plt.ylabel('wartość')
plt.plot(whiteNoise,scaley=True, scalex=True)
plt.savefig('05white noise.png')
plt.close()


# Policzenie sumy skumulowanej (ruch Browna).
brown = np.cumsum(whiteNoise)

# Stworzenie wykresu dla sumy skumulowanej.
plt.title("Ruchy Browna")
plt.xlabel('czas')
plt.ylabel('wartość')
plt.plot(brown,scaley=True, scalex=True)
plt.savefig('06Ruchy Browna.png')
plt.close()


# ########################################################## TEMAT 3 ################################################

# Dla danych giełdowych (stopy zwrotu) i wygenerowanych policzenie histogramów.
fig = plt.figure(figsize=(15,15))
histStopyZwrotu = plt.hist(stopyZwrotu_norm, bins=500, range=(-10, 10))
plt.title("histogram dla stóp zwortu")
plt.xlabel('wartość')
plt.ylabel('częstość')
plt.savefig('07histogram dla stóp zwortu.png')
plt.close()



fig = plt.figure(figsize=(15,15))
histWhiteNoise = plt.hist(whiteNoise, bins=500, range=(-10, 10))
plt.title("histogram dla white noise")
plt.xlabel('wartość')
plt.ylabel('częstość')
plt.savefig('08histogram dla white noise.png')
plt.close()



fig = plt.figure(figsize=(15,15))
plt.hist(whiteNoise, bins=500, range=(-10, 10))
plt.hist(stopyZwrotu_norm, bins=500, range=(-10, 10), alpha=.5, color='r')
plt.legend(["white noise", "stopy zwrotu"])
plt.title("histogramy")
plt.xlabel('wartość')
plt.ylabel('częstość')
plt.savefig('09histogramy.png')
plt.close()

StopySuma = sum(histStopyZwrotu[0][:])
WhiteSuma = sum(histWhiteNoise[0][:])

print("Suma binów (stopy zwrotu):", round(StopySuma))
print("Suma binów (white noise):", round(WhiteSuma))


# Znormalizowanie histogramów (suma wartości słupków powinna być równa 1).

fig = plt.figure(figsize=(15,15))
plt.plot(histStopyZwrotu[1][:-1], (histStopyZwrotu[0]/StopySuma))
plt.title("znormalizowany histogram dla stóp zwrotu")
plt.xlabel('wartość')
plt.ylabel('częstość')
plt.savefig('10znormalizowany histogram dla stóp zwrotu.png')
plt.close()


fig = plt.figure(figsize=(15,15))
plt.plot(histWhiteNoise[1][:-1], (histWhiteNoise[0]/WhiteSuma))
plt.title("znormalizowany histogram dla stóp zwrotu")
plt.xlabel('wartość')
plt.ylabel('częstość')
plt.savefig('11znormalizowany histogram dla stóp zwrotu.png')
plt.close()



# Policzenie parametrów rozkładów (skośności oraz kurtozy).
print("Dane giełdowe")
print("   Kurtoza: ", stat.kurtosis(stopyZwrotu_norm))
print("   Skośność: ", stat.skew(stopyZwrotu_norm))

print("White Noise")
print("   Kurtoza: ", stat.kurtosis(whiteNoise))
print("   Skośność: ", stat.skew(whiteNoise))


# Przedstawienie obu znormalizowanych rozkładów na jednym wykresie.

fig = plt.figure(figsize=(15,15))
plt.bar(histWhiteNoise[1][:-1], (histWhiteNoise[0]/WhiteSuma), width=0.1)
plt.bar(histStopyZwrotu[1][:-1], (histStopyZwrotu[0]/StopySuma), width=0.1, alpha=.2, color='r')
plt.legend(["white noise", "stopy zwrotu"])
plt.title("znormalizowane histogramy")
plt.xlabel('wartość')
plt.ylabel('częstość')
plt.savefig('12znormalizowane_histogramy.png')
plt.close()


StopySuma_norm = sum(histStopyZwrotu[0]/StopySuma)
WhiteSuma_norm = sum(histWhiteNoise[0]/WhiteSuma)

print("Histogram znormalizowany - Suma binów (stopy zwrotu):", round(StopySuma_norm))
print("Histogram znormalizowany - Suma binów (white noise):", round(WhiteSuma_norm))


# ########################################################## TEMAT 4 ################################################

# Dla danych giełdowych (stopy zwrotu) i wygenerowanych (biały szum) policzenie funkcji autokorelacji.

fig = plot_acf((stopyZwrotu_norm), lags=len(stopyZwrotu_norm) - 1, zero=False, auto_ylims=True)
plt.title('Autokorelacja stóp zwrotu')
plt.xlabel("przesunięcie")
plt.ylabel("korelacja")
plt.savefig('13Autokorelacja stóp zwrotu.png')
plt.close()

fig = plot_acf((whiteNoise), lags=len(whiteNoise) - 1, zero=False, auto_ylims=True)
plt.title('Autokorelacja białego szumu')
plt.xlabel("przesunięcie")
plt.ylabel("korelacja")
plt.savefig('14Autokorelacja białego szumu.png')
plt.close()


# Policzenie funkcji autokorelacji dla modułu stóp zwrotu oraz modułu danych wygenerowanych. Sporządzenie odpowiedniego wykresu.

abs_stopy = np.abs(stopyZwrotu_norm)
acf_stopy = stats.tsa.acf(abs_stopy, nlags=len(stopyZwrotu_norm) - 1)

plt.plot(acf_stopy[1:], scaley=True, scalex=True)
plt.xlabel("przesunięcie")
plt.ylabel("korelacja")
plt.title('Autokorelacja modułu stóp zwrotu')
plt.savefig('15Autokorelacja modułu stóp zwrotu.png')
plt.close()

abs_white_noise = np.abs(whiteNoise)
np.reshape(abs_white_noise, (len(abs_white_noise), 1))
acf_noise = stats.tsa.acf(abs_white_noise, nlags=len(abs_white_noise) - 1)

plt.plot(acf_noise[1:], scaley=True, scalex=True)
plt.xlabel("przesunięcie")
plt.ylabel("korelacja")
plt.title('Autokorelacja modułu białego szumu')
plt.savefig('16Autokorelacja modułu białego szumu.png')
plt.close()


# skala log
abs_stopy = np.abs(stopyZwrotu_norm)
acf_stopy = stats.tsa.acf(abs_stopy, nlags=len(stopyZwrotu_norm) - 1)

plt.yscale('log')
plt.plot(acf_stopy[1:], scaley=True, scalex=True)
plt.xlabel("przesunięcie")
plt.ylabel("korelacja dodatnia")
plt.title('Autokorelacja modułu stóp zwrotu - skala logarytmiczna')
plt.savefig('17Autokorelacja modułu stóp zwrotu - skala logarytmiczna.png')
plt.close()

abs_white_noise = np.abs(whiteNoise)
np.reshape(abs_white_noise, (len(abs_white_noise), 1))
acf_noise = stats.tsa.acf(abs_white_noise, nlags=len(abs_white_noise) - 1)

plt.plot(acf_noise[1:], scaley=True, scalex=True)
plt.xlabel("przesunięcie")
plt.ylabel("korelacja dodatnia")
plt.title('Autokorelacja modułu białego szumu - skala logarytmiczna')
plt.yscale('log')
plt.savefig('18Autokorelacja modułu białego szumu - skala logarytmiczna.png')
plt.close()



# Policzenie widma mocy dla danych giełdowych (stopy zwrotu) i wygenerowanych (biały szum).

#Widmo mocy sygnału jest kwadratem modułu widma
#power spectral density - Dla sygnałów rzeczywistych, periodogram jest równy kwadratowi modułu widma. Przedstawia on rozklad mocy widmowej sygnału na jednostkę częstotliwości.


power_spectrum_stopy_zwrotu = np.abs(np.fft.fft(stopyZwrotu_norm))**2
freqs = np.fft.fftfreq(stopyZwrotu_norm.size)
idx = np.argsort(freqs)

fig = plt.figure(figsize=(15,15))
plt.plot(freqs[idx], power_spectrum_stopy_zwrotu[idx], scaley=True, scalex=True)
plt.xlabel('częstotliwość')
plt.ylabel('widmo mocy')
plt.title('Widmo mocy dla danych giełdowych')
plt.savefig("19Widmo mocy dla danych giełdowych")
plt.close()

# ------

power_spectrum_white_noise = np.abs(np.fft.fft(whiteNoise))**2
freqs = np.fft.fftfreq(whiteNoise.size)
idx = np.argsort(freqs)

fig = plt.figure(figsize=(15,15))
plt.plot(freqs[idx], power_spectrum_white_noise[idx], scaley=True, scalex=True)
plt.xlabel('częstotliwość')
plt.ylabel('widmo mocy')
plt.title('Widmo mocy dla white noise')
plt.savefig("20Widmo mocy dla white noise")
plt.close()

# Policzenie widma mocy dla modułu stóp zwrotu oraz modułu danych wygenerowanych. Sporządzenie odpowiedniego wykresu.

power_spectrum_stopy_zwrotu_abs = np.abs(np.fft.fft(abs_stopy))**2
freqs = np.fft.fftfreq(abs_stopy.size)
idx = np.argsort(freqs)

fig = plt.figure(figsize=(15,15))
plt.plot(freqs[idx], power_spectrum_stopy_zwrotu_abs[idx], scaley=True, scalex=True)
plt.xlabel('częstotliwość')
plt.ylabel('widmo mocy')
plt.title('Widmo mocy dla modułu danych giełdowych')
plt.savefig("21Widmo mocy dla modułu danych giełdowych")
plt.close()

# ------

power_spectrum_white_noise_abs = np.abs(np.fft.fft(abs_white_noise))**2
freqs = np.fft.fftfreq(abs_white_noise.size)
idx = np.argsort(freqs)

fig = plt.figure(figsize=(15,15))
plt.plot(freqs[idx], power_spectrum_white_noise_abs[idx], scaley=True, scalex=True)
plt.xlabel('częstotliwość')
plt.ylabel('widmo mocy')
plt.title('Widmo mocy dla modułu white noise')
plt.savefig("22Widmo mocy dla modułu white noise")
plt.close()


# plt.plot(abs_stopy)
# plt.show()
# plt.plot(abs_white_noise)
# plt.show()