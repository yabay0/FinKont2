import numpy as np
from Assignment_3.question_1g_code import Heston_Fourier

a = 0.5
S0 = 1.0
A0 = 1.0
r = 0.02
T = 30.0
sigma = 0.2
K = np.exp(r * T)
gT = np.exp(((1 - a) * r + 0.5 * a * (1 - a) * sigma**2) * T)
S_star = (K / gT) ** (1 / a)

theta = sigma**2
kappa = 2.0
epsilon = 1.0
rho = -0.5
v0 = sigma**2

n = 1000
K_grid = np.linspace(0.001, S_star, n)
dK = K_grid[1] - K_grid[0]

def fpp(K):
    return a * (1 - a) * gT * K ** (a - 2)

weights = a * (1 - a) * gT * K_grid ** (a - 2)

put_prices = np.array([
    Heston_Fourier(S0, T, Ki, r, 0, v0, theta, kappa, epsilon, rho, greek=1) + Ki * np.exp(-r * T) - S0
    for Ki in K_grid
])

total_price = np.sum(weights * put_prices * dK)
dirac_weight = a * gT * S_star ** (a - 1)
dirac_put_price = (Heston_Fourier(S0, T, S_star, r, 0, v0, theta, kappa, epsilon, rho, greek=1) +
                   S_star * np.exp(-r * T) - S0)
hedge_price = total_price + dirac_weight * dirac_put_price

round(hedge_price, 4)