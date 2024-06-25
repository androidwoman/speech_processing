import numpy as np
def compute_dtft(x, n, w):
    # Computes Discrete-time Fourier Transform
    # [X] = compute_dtft(x,n,w)
    #
    # X = DTFT values computed at frequencies w
    # x is a finite-duration sequence over n
    # n is the vector of "time" values over which the computation is performed
    # w is a vector of frequencies used in the output
    X = np.zeros(len(w), dtype=complex)
    for k in range(len(w)):
        X[k] = np.sum(x * np.exp(-1j * w[k] * n))
    return X