"""
Date: 2021-08-09 13:51:47
LastEditors: GodK
"""
import numpy as np
np.set_printoptions(precision=4)

def mylorenz_dynamic(N, t):
    time = 100
    stepsize = 0.02
    steps = round(time / stepsize)  # 总步数
    x = np.zeros((3 * N, steps))
    x[:, 0] = np.arange(-0.1, -0.1 + 0.003 * (3 * N), 0.003)

    # Lorenz system
    C = 0.1
    alpha = 10
    beta = 28
    gamma = -8/3

    for i in range(steps-1):
        if i >=t-1:
            sigma = alpha + 0.1*np.math.floor((i-t)/10)
        else:
            sigma = alpha
        
        x[0, i + 1] = x[0, i] + stepsize * (sigma * (x[1, i] - x[0, i]) + C * x[(N - 1) * 3, i])
        x[1, i + 1] = x[1, i] + stepsize * (beta * x[0, i] - x[1, i] - x[0, i] * x[2, i])
        x[2, i + 1] = x[2, i] + stepsize * (gamma * x[2, i] + x[0, i] * x[1, i])
        for j in range(N-1):
            x[3 * (j + 1), i + 1] = x[3 * (j + 1), i] + stepsize * (alpha * (x[1 + 3*(j + 1), i] - x[3 * (j + 1), i]) + C * x[3 * j, i])
            x[1 + 3 * (j + 1), i + 1] = x[1 + 3 * (j + 1), i] + stepsize * (beta * x[3 * (j + 1), i] - x[1 + 3 * (j + 1), i] - x[3 * (j + 1), i] * x[2 + 3 * (j + 1), i])
            x[2 + 3 * (j + 1), i + 1] = x[2 + 3 * (j + 1), i] + stepsize * (gamma * x[2 + 3 * (j + 1), i] + x[3 * (j + 1), i] * x[1 + 3 * (j + 1), i])
    
    Y = x.T
    return Y
