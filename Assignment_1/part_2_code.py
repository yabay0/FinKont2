import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

class BachelierImpliedVolatility:
    """A class to calculate implied volatility using both the Black–Scholes and Bachelier models."""
    
    def _d(spot, strike, sigma, T):
        """Compute d1 and d2 for the Black–Scholes formula."""
        d1 = (np.log(spot / strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def black_scholes_price(spot, strike, T, sigma, option="Call", r=0.0, q=0.0):
        """Calculate the Black–Scholes price for a European option."""
        d1, d2 = BachelierImpliedVolatility._d(spot, strike, sigma, T)
        if option.lower() == "call":
            return spot * np.exp(-q * T) * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        elif option.lower() == "put":
            return strike * np.exp(-r * T) * norm.cdf(-d2) - spot * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be either 'Call' or 'Put'.")

    def black_scholes_implied_vol(obs_price, spot, strike, T, r, q, sigma_bach, option="Call", low_vol=1e-6, high_vol=10.0):
        """Calculate the Black–Scholes implied volatility.
        
        It solves for sigma_bs such that:
            BS(spot, strike, T, sigma_bs, option, r, q) = Bach(spot, strike, T, sigma_bach)
        """
        def difference(sigma_bs):
            return (BachelierImpliedVolatility.black_scholes_price(spot, strike, T, sigma_bs, option, r, q)
                    - BachelierImpliedVolatility.bachelier_price(spot, strike, T, sigma_bach))
        return brentq(difference, low_vol, high_vol)

    def bachelier_price(spot, strike, T, sigma):
        """Calculate the option price using the Bachelier model."""
        d = (spot - strike) / (sigma * np.sqrt(T))
        return (spot - strike) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d)
    
    def plot_implied_volatility(S0, T, sigma_bach, strikes, option="Call", r=0.0, save_path=None):
        """
        Plot the Black–Scholes implied volatility across strikes using the Bachelier model price."""
        vols = np.array([
            BachelierImpliedVolatility.black_scholes_implied_vol(
                BachelierImpliedVolatility.bachelier_price(S0, k, T, sigma_bach),
                S0, k, T, r, 0.0, sigma_bach, option
            ) for k in strikes
        ])
        
        atm_strike = S0
        atm_price = BachelierImpliedVolatility.bachelier_price(S0, atm_strike, T, sigma_bach)
        atm_vol = BachelierImpliedVolatility.black_scholes_implied_vol(atm_price, S0, atm_strike, T, r, 0.0, sigma_bach, option)
    
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, vols, color="black", linewidth=2, label="Implied Volatility", zorder=1)
        plt.scatter(atm_strike, atm_vol, color="blue", s=80, label="ATM", zorder=2)
        plt.text(atm_strike + 4, atm_vol + 0.01, f"({atm_strike}, {round(atm_vol, 2)})", fontsize=11, color="blue")
        plt.xlabel("Strike ($K$)", fontsize=10)
        plt.ylabel("Implied Volatility ($\\sigma_{imp}$)", fontsize=10)
        plt.title(f"Implied Volatility Across Strikes in the Bachelier Model ($S(0)$={S0})", fontsize=12)
        plt.legend(loc="upper right", fontsize=10, ncol=4)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

if __name__ == "__main__":
    S0 = 100
    T = 0.25
    sigma_bach = 15.0
    r = 0.0
    strikes = np.arange(50, 300, 10)
    option_type = "Call"
    
    BachelierImpliedVolatility.plot_implied_volatility(
        S0=S0,
        T=T,
        sigma_bach=sigma_bach,
        strikes=strikes,
        option=option_type,
        r=r,
        save_path="Assignment_1/question_2b_s0_100.png"
    )