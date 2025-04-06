import numpy as np
from scipy.integrate import quad

def Heston_Fourier(spot, timetoexp, strike, r, divyield, V, theta, kappa, epsilon, rho, greek=1):
    X = np.log(spot/strike) + (r - divyield) * timetoexp
    kappahat = kappa - 0.5 * rho * epsilon
    xiDummy = kappahat**2 + 0.25 * epsilon**2

    def integrand(k):
        xi = np.sqrt(k**2 * epsilon**2 * (1 - rho**2) + 2j * k * epsilon * rho * kappahat + xiDummy)
        Psi_P = - (1j * k * rho * epsilon + kappahat) + xi
        Psi_M = (1j * k * rho * epsilon + kappahat) + xi
        arg_log = (Psi_M + Psi_P * np.exp(-xi * timetoexp)) / (2 * xi)
        alpha = -kappa * theta * (Psi_P * timetoexp + 2 * np.log(arg_log)) / epsilon**2
        beta = - (1 - np.exp(-xi * timetoexp)) / (Psi_M + Psi_P * np.exp(-xi * timetoexp))
        numerator = np.exp(( -1j * k + 0.5 ) * X + alpha + (k**2 + 0.25) * beta * V)

        if greek == 1:
            dummy = np.real(numerator / (k**2 + 0.25))
        elif greek == 2:
            dummy = np.real((0.5 - 1j*k) * numerator / (spot * (k**2 + 0.25)))
        elif greek == 3:
            dummy = -np.real(numerator / spot**2)
        elif greek == 4:
            dummy = np.real(numerator * beta)
        else:
            raise ValueError("Invalid greek value. Use 1, 2, 3, or 4.")
        return dummy

    integral_value, _ = quad(integrand, -100, 100, limit=200)

    if greek == 1:
        dummy = np.exp(-divyield * timetoexp) * spot - strike * np.exp(-r * timetoexp) * integral_value / (2 * np.pi)
    elif greek == 2:
        dummy = np.exp(-divyield * timetoexp) - strike * np.exp(-r * timetoexp) * integral_value / (2 * np.pi)
    elif greek == 3:
        dummy = -strike * np.exp(-r * timetoexp) * integral_value / (2 * np.pi)
    elif greek == 4:
        dummy = -strike * np.exp(-r * timetoexp) * integral_value / (2 * np.pi)

    return dummy

if __name__ == '__main__':
    spot = 1.0
    T = 30.0
    r = 0.02
    divyield = 0.0
    strike = np.exp(r * T)

    a = 0.5

    V_eff = a**2 * 0.04
    theta_eff = a**2 * 0.04
    kappa = 2.0
    epsilon_eff = a * 1.0
    rho = -0.5

    price = Heston_Fourier(spot, T, strike, r, divyield, V_eff, theta_eff, kappa, epsilon_eff, rho, greek=1)
    print(f"Call Price: {price:.4f}")