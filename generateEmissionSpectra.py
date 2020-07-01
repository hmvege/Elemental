import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():
    pass

    # colors = np.array((np.arange(255), np.arange(255), np.arange(255)))

    # m = [[[0,1,255], [1,1,1], [1,1,1]],
    #      [[0,1,255], [1,1,1], [1,1,1]],
    #      [[0,1,255], [1,1,1], [1,1,1]]]

    N = 1000
    M = 300

    # m = np.zeros((50, N, 3), dtype=int)

    # for i in range(N):
    #     m[:, i, 0] = i
    #     m[:, i, 2] = 0

    #     for j in range(N):
    #         # for k in range(256):
    #             # print(i)
    #         m[:, j, 1] = i + N*j


    # print(m)

    # ["rainbow"]

    grad = np.linspace(0, 1, N)
    rainbow_spectra = np.empty((M, N), dtype=float)
    for i in range(M):
        rainbow_spectra[i, :] = grad

    m = cm.gist_rainbow(rainbow_spectra)[:,::-1]
    # print(m)

    plt.imshow(m)
    plt.show()


if __name__ == '__main__':
    main()