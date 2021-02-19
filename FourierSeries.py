import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
#function defining
def func(x):
    return 1 - x
#defining types of functions even/odd
def even_func(x):
    return func(x) if x > 0 else func(-x)
def odd_func(x):
    return func(x) if x > 0 else -func(-x)
#setting freq of func
def freq():
    return 2 * np.pi / 2
#formula for An coeff
def coeff_a(n, func)-> float:
    return integrate.quad(lambda x: 2 * func(x) * np.cos(n * x * freq()), 0, 1)[0]
#formula for Bn coeff
def coeff_b(n, func)-> float:
    return integrate.quad(lambda x: 2 * func(x) * np.sin(n * x * freq()), 0, 1)[0]
#calculating of n coeffs for func with func type
def calc_coeffs(N, coeff_func, func_type):
    res = np.zeros(N)
    for i in range(0, len(res)):
        res[i] = coeff_func(i, func_type)
    return res
def calc_func(func, s, f, coeff, N):
    res = np.zeros(N)
    for i in range(0, len(res)):
        if s <= f: 
            s = s + coeff
            res[i] = func(s)
    return res
#Fourier series for even and odd func(put aCoeffs np.zeros(1) for b_coeffs and similar for a_coeffs)
def furier_s(x, a_coeffs, b_coeffs):
    results = np.zeros(len(x))
    a_len = len(a_coeffs)
    b_len = len(b_coeffs)
    for i in range(0, len(x)):
        y = a_coeffs[0] / 2 + b_coeffs[0] / 2
        for j in range(1, a_len): y += a_coeffs[j] * np.cos(j * x[i] * freq())
        for j in range(1, b_len): y += b_coeffs[j] * np.sin(j * x[i] * freq())
        results[i] = y
    return results
#creating subplots
fig, axs = plt.subplots(2, 2)
fig.suptitle("Signal processing methods task 1")
#An coeffs for even
axs[0][0].set_title("An coeffs for even")
a_coeffs = calc_coeffs(100,  coeff_a, even_func)
axs[0][0].plot(np.arange(0, 100), a_coeffs, 'tab:green')
axs[0][0].grid(True)
#Bn coeffs for odd
axs[1][0].set_title("Bn coeffs for odd")
b_coeffs = calc_coeffs(100, coeff_b, odd_func)#odd_func_offset | odd_func
axs[1][0].plot(np.arange(0, 100), b_coeffs , 'tab:purple')
axs[1][0].grid(True)
#Result of even func approximation
axs[0][1].set_title("Result of even func approximation")
axs[0][1].plot(np.arange(-1,1,0.01), furier_s(np.arange(-1,1,0.01), a_coeffs[:len(a_coeffs) // 2], np.zeros(1)), 'tab:red')
axs[0][1].plot(np.arange(-1,1,0.01), calc_func(even_func, -1, 1, 0.01, 200))
#Result of odd func approximation
axs[1][1].set_title("Result of odd func approximation")
axs[1][1].plot(np.arange(-1,1,0.01), furier_s(np.arange(-1,1,0.01), np.zeros(1), b_coeffs[:len(b_coeffs) // 2]), 'tab:red')
axs[1][1].plot(np.arange(-1,1,0.01), calc_func(odd_func, -1, 1, 0.01, 200))
fig.tight_layout()
plt.show();