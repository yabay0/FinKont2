import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class StaticHedgePricer:
    def __init__(self, a, S0, r, T, sigma):
        self.a = a
        self.S0 = S0
        self.r = r
        self.T = T
        self.sigma = sigma
        self.K = np.exp(r * T)
        self.gT = np.exp(((1 - a) * r + 0.5 * a * (1 - a) * sigma**2) * T)
        self.S_star = (self.K / self.gT)**(1 / a)
    
    def black_scholes_put(self, S0, strike, T, r, sigma):
        d1 = (np.log(S0 / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = strike * np.exp(-r * T) * (1 - norm.cdf(d2)) + S0 * (norm.cdf(d1) - 1)
        return put_price
    
    def compute_static_hedge(self, n):
        K_grid = np.linspace(0.001, self.S_star, n)
        dK = K_grid[1] - K_grid[0]
        
        weights = self.a * (1 - self.a) * self.gT * (K_grid ** (self.a - 2))
        put_prices = np.array([
            self.black_scholes_put(self.S0, Ki, self.T, self.r, self.sigma)
            for Ki in K_grid
        ])
        
        integral_contribution = np.sum(weights * put_prices * dK)
        
        dirac_weight = self.a * self.gT * (self.S_star ** (self.a - 1))
        dirac_put = self.black_scholes_put(self.S0, self.S_star, self.T, self.r, self.sigma)
        dirac_contribution = dirac_weight * dirac_put
        
        hedge_price = integral_contribution + dirac_contribution
        return hedge_price

    def convergence_plot(self, n_min=50, n_max=10000, step=25, save_path=None):
        n_values = np.arange(n_min, n_max + 1, step)
        hedge_prices = [self.compute_static_hedge(n) for n in n_values]
        
        plt.style.use('ggplot')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#E5E5E5')
        
        ax.plot(n_values, hedge_prices, linewidth=2, color='blue')
        ax.set_xlabel("Number of Strikes", fontsize=10)
        ax.set_ylabel("Static Hedge Price", fontsize=10)
        ax.set_title("Convergence of Static Hedge Price", fontsize=12, color='black')
        
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')
        else:
            plt.show()

if __name__ == "__main__":
    a = 0.5
    S0 = 1
    r = 0.02
    T = 30
    sigma = 0.2
    
    pricer = StaticHedgePricer(a, S0, r, T, sigma)
    pricer.convergence_plot(save_path="Assignment_3/question_1e_static_hedge_convergence.png")