import numpy as np
import matplotlib.pyplot as plt

def compute_dtft(x, n, w):
    X = np.zeros(len(w), dtype=complex)
    for i in range(len(w)):
        X[i] = np.sum(x * np.exp(-1j * w[i] * n))
    return X



# (a) x[n] = 3 * (5) ^ (- |n - 2|) for all n
def x_a(n):
    return 3 * (5 ** (-np.abs(n - 2)))

# (b) x[n] = alpha ^ n * cos(omega_{0}*n + phi) * u[n], |alpha| < 1
def x_b(n, alpha=0.9, omega_0=1, phi=0):
    return (alpha ** n) * np.cos(omega_0 * n + phi) * (n >= 0)

# (c) x[n] = 7 for all n
def x_c(n):
    return 7

# (d) x[n] = A * cos(omega_{0}*n + phi) for all n
def x_d(n, A=1, omega_0=1, phi=0):
    return A * np.cos(omega_0 * n + phi)

# (e) x[n] = A * sin(omega_{0}*n + phi) * (u[n] - u[n - 9])
def x_e(n, A=1, omega_0=1, phi=0):
    return A * np.sin(omega_0 * n + phi) * ((n >= 0) & (n < 9))





def plot_dtft(w, X, title):


    plt.figure(figsize=(12, 6))
    plt.suptitle(title,)
    plt.subplot(1, 2, 1)
    plt.stem(w / (2 * np.pi), 20 * np.log10(np.abs(X)), use_line_collection=True)
    plt.title('Magnitude (dB)')
    plt.xlabel('Normalized Frequency (cycles/sample)')
    plt.ylabel('|X(e^(jω))| (dB)')



    plt.subplot(1, 2, 2)
    plt.stem(w / (2 * np.pi), np.angle(X), use_line_collection=True)
    plt.title('Phase')
    plt.xlabel('Normalized Frequency (cycles/sample)')
    plt.ylabel('∠X(e^(jω))')
    plt.show()




# ///// q1
    

# Example usage for sequence (a)
# N = 20
# n = np.arange(-N, N, dtype=float)
# w = np.linspace(-np.pi, np.pi, 1000)

# x = x_a(n)
# X = compute_dtft(x, n, w)
# plot_dtft(w, X, "Sequence (a): x[n] = 3 * (5) ^ (- |n - 2|)")

# # Example usage for sequence (b)
# N = 20
# n = np.arange(-N, N, dtype=float)
# w = np.linspace(-np.pi, np.pi, 1000)

# x = x_b(n)
# X = compute_dtft(x, n, w)
# plot_dtft(w, X, "Sequence (b): x[n] = alpha ^ n * cos(omega_{0}*n + phi) * u[n], |alpha| < 1")

# # Example usage for sequence (c)
# N = 20
# n = np.arange(-N, N, dtype=float)
# w = np.linspace(-np.pi, np.pi, 1000)

# x = x_c(n)
# X = compute_dtft(x, n, w)
# plot_dtft(w, X, "Sequence (c): x[n] = 7")

# # Example usage for sequence (d)
# N = 20
# n = np.arange(-N, N, dtype=float)
# w = np.linspace(-np.pi, np.pi, 1000)

# x = x_d(n)
# X = compute_dtft(x, n, w)
# plot_dtft(w, X, "Sequence (d): x[n] = A * cos(omega_{0}*n + phi)")

# # Example usage for sequence (e)
# N = 20
# n = np.arange(-N, N, dtype=float)
# w = np.linspace(-np.pi, np.pi, 1000)

# x = x_e(n)
# X = compute_dtft(x, n, w)
# plot_dtft(w, X, "Sequence (e): x[n] = A * sin(omega_{0}*n + phi) * (u[n] - u[n - 9])")

# ---------------------------------------
    

    # //// q2
from scipy.signal import freqz
def plot_dtft_and_freqz(w, X, title, n, x):
    plt.figure(figsize=(12, 6))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.stem(w / (2 * np.pi), 20 * np.log10(np.abs(X)), use_line_collection=True)
    plt.title('Magnitude (dB)')
    plt.xlabel('Normalized Frequency (cycles/sample)')
    plt.ylabel('|X(e^(jω))| (dB)')

    plt.subplot(1, 2, 2)
    plt.stem(w / (2 * np.pi), np.angle(X), use_line_collection=True)
    plt.title('Phase')
    plt.xlabel('Normalized Frequency (cycles/sample)')
    plt.ylabel('∠X(e^(jω))')

    # Compute and plot freqz
    w_freqz, H_freqz = freqz(x, worN=w / (2 * np.pi))
    plt.subplot(1, 2, 1)
    plt.plot(w / (2 * np.pi), 20 * np.log10(np.abs(H_freqz)), 'r--')
    plt.subplot(1, 2, 2)
    plt.plot(w / (2 * np.pi), np.angle(H_freqz), 'r--')

    plt.show()

# Example usage for sequence (a)
N = 20
n = np.arange(-N, N, dtype=float)
w = np.linspace(-np.pi, np.pi, 1000)

x = x_a(n)
X = compute_dtft(x, n, w)
plot_dtft_and_freqz(w, X, "Sequence (a): x[n] = 3 * (5) ^ (- |n - 2|)", n, x)

# Example usage for sequence (b)
N = 20
n = np.arange(-N, N, dtype=float)
w = np.linspace(-np.pi, np.pi, 1000)

x = x_b(n)
X = compute_dtft(x, n, w)
plot_dtft_and_freqz(w, X, "Sequence (b): x[n] = alpha ^ n * cos(omega_{0}*n + phi) * u[n], |alpha| < 1", n, x)

# Example usage for sequence (c)
N = 20
n = np.arange(-N, N, dtype=float)
w = np.linspace(-np.pi, np.pi, 1000)

x = x_c(n)
X = compute_dtft(x, n, w)
plot_dtft_and_freqz(w, X, "Sequence (c): x[n] = 7", n, x)

# Example usage for sequence (d)
N = 20
n = np.arange(-N, N, dtype=float)
w = np.linspace(-np.pi, np.pi, 1000)

x = x_d(n)
X = compute_dtft(x, n, w)
plot_dtft_and_freqz(w, X, "Sequence (d): x[n] = A * cos(omega_{0}*n + phi)", n, x)

# Example usage for sequence (e)
N = 20
n = np.arange(-N, N, dtype=float)
w = np.linspace(-np.pi, np.pi, 1000)

x = x_e(n)
X = compute_dtft(x, n, w)
plot_dtft_and_freqz(w, X, "Sequence (e): x[n] = A * sin(omega_{0}*n + phi) * (u[n] - u[n - 9])", n, x)
