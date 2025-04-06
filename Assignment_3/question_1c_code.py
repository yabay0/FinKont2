import numpy as np
from scipy.stats import norm

r = 0.02
mu = 0.07
sigma = 0.20
S0 = 1
A0 = 1
T = 30
a = 0.5
K = np.exp(r * T)

# --- Question 1.b ---
# Price of portfolio insurance under Q: ae^{-rT} E^Q[(K - A_T^a)^+]
a_sigma = a * sigma
d1_q1b = (np.log(A0 / K) + (r + 0.5 * a_sigma**2) * T) / (a_sigma * np.sqrt(T))
d2_q1b = d1_q1b - a_sigma * np.sqrt(T)
price_q1b = K * np.exp(-r * T) * norm.cdf(-d2_q1b) - A0 * norm.cdf(-d1_q1b)

# --- Question 1.c ---
# 1st quantity: e^{-rT} E^P[(K - A_T^a)^+]
d2_1c_1 = (np.log(A0 / K) + (a * (mu - r) + r - 0.5 * a_sigma**2) * T) / (a_sigma * np.sqrt(T))
d1_1c_1 = d2_1c_1 + a_sigma * np.sqrt(T)
price_1c_1 = np.exp(-r * T) * K * norm.cdf(-d2_1c_1) - A0 * np.exp((a * mu - a * r) * T) * norm.cdf(-d1_1c_1)

# 2nd quantity: e^{-rT} E^Q[(K - S_T^a)^+]
d2_1c_2 = (np.log(S0**a / K) + a * (r - 0.5 * sigma**2) * T) / (a * sigma * np.sqrt(T))
d1_1c_2 = d2_1c_2 + a * sigma * np.sqrt(T)
price_1c_2 = np.exp(-r * T) * (K * norm.cdf(-d2_1c_2) - S0**a * np.exp((a * r - 0.5 * a * sigma**2 + 0.5 * a**2 * sigma**2) * T) * norm.cdf(-d1_1c_2))

# 3rd quantity: a * e^{-rT} * E^Q[(K - S_T)^+]
d1_1c_3 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2_1c_3 = d1_1c_3 - sigma * np.sqrt(T)
price_1c_3 = a * (K * np.exp(-r * T) * norm.cdf(-d2_1c_3) - S0 * norm.cdf(-d1_1c_3))

round(price_q1b, 4), round(price_1c_1, 4), round(price_1c_2, 4), round(price_1c_3, 4)