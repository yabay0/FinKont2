import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class PortfolioInsuranceHedgeBase:
    def __init__(self, r, S0, A0, T, K, a, sigma, Nhedges, Nreps):
        self.r = r
        self.S0 = S0
        self.A0 = A0
        self.T = T
        self.K = K
        self.a = a
        self.sigma = sigma
        self.Nhedges = Nhedges
        self.Nreps = Nreps
        self.dt = T / Nhedges
        self.exp_r_dt = np.exp(r * self.dt)
        self.g0 = A0 / (S0 ** a)
        
        self.S = None
        self.A = None
        self.V = None
        self.hedge_error = None

    def compute_g(self, t):
        return self.g0 * np.exp((1 - self.a) * (self.r + 0.5 * self.a * self.sigma ** 2) * t)
    
    def A_from_S(self, S, t):
        return self.compute_g(t) * (S ** self.a)
    
    def black_scholes_put_price(self, A, T_rem):
        if T_rem <= 0:
            return np.maximum(self.K - A, 0)
        vol = self.a * self.sigma
        d1 = (np.log(A / self.K) + (self.r + 0.5 * vol ** 2) * T_rem) / (vol * np.sqrt(T_rem))
        d2 = d1 - vol * np.sqrt(T_rem)
        price = self.K * np.exp(-self.r * T_rem) * norm.cdf(-d2) - A * norm.cdf(-d1)
        return price
    
    def delta(self, A, S, T_rem):
        if T_rem <= 0:
            return 0
        vol = self.a * self.sigma
        d1 = (np.log(A / self.K) + (self.r + 0.5 * vol ** 2) * T_rem) / (vol * np.sqrt(T_rem))
        delta = -norm.cdf(-d1) * self.a * A / S
        return delta
    
    def hedge_experiment(self):
        self.S = np.full(self.Nreps, self.S0, dtype=float)
        self.A = np.full(self.Nreps, self.A0, dtype=float)
        
        price0 = self.black_scholes_put_price(self.A0, self.T)
        delta0 = self.delta(self.A0, self.S0, self.T)
        self.V = np.full(self.Nreps, price0, dtype=float)
        current_delta = np.full(self.Nreps, delta0, dtype=float)
        cash = self.V - current_delta * self.S
        
        for i in range(1, self.Nhedges + 1):
            t = i * self.dt
            T_rem = self.T - t
            Z = np.random.normal(size=self.Nreps)
            self.S *= np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)
            self.A = self.A_from_S(self.S, t)
            
            cash = cash * np.exp(self.r * self.dt)
            self.V = current_delta * self.S + cash
            
            new_delta = self.delta(self.A, self.S, T_rem) if T_rem > 0 else 0
            cash = self.V - new_delta * self.S
            current_delta = new_delta
        
        payoff = np.maximum(self.K - self.A, 0)
        self.hedge_error = self.V - payoff
        return self.hedge_error
    
    def final_payoff_error(self):
        return self.hedge_error
    
    @classmethod
    def convergence_of_hedge_error(cls, ax, r, S0, A0, T, K, a, sigma, Nreps, max_hedges=1000):
        std_devs = []
        for Nhedges in range(2, max_hedges + 1):
            hedge = cls(r, S0, A0, T, K, a, sigma, Nhedges, Nreps)
            hedge.hedge_experiment()
            error = hedge.final_payoff_error()
            discounted_error = np.exp(-r * T) * error
            std_devs.append(np.std(discounted_error))
        ax.plot(range(2, max_hedges + 1), std_devs, linewidth=1, color='red')
        ax.set_xlabel("Number of Hedge Points", fontsize=10)
        ax.set_ylabel("Std Dev of Discounted Error", fontsize=10)
        ax.set_title("Convergence of Hedge Error", fontsize=12)
        ax.grid(True)
    
    @classmethod
    def plot_hedge_experiment(cls, r, S0, A0, T, K, a, sigma, Nreps, Nhedges, save_path=None):
        hedge = cls(r, S0, A0, T, K, a, sigma, Nhedges, Nreps)
        hedge.hedge_experiment()
        final_errors = hedge.final_payoff_error()
        S_final = hedge.S
        A_final = hedge.A
        V_final = hedge.V
        payoff = np.maximum(K - A_final, 0)
        
        fig = plt.figure(figsize=(10, 8))
        plt.style.use('ggplot')
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        
        ax1.scatter(A_final, V_final, s=10, alpha=0.2, label="Hedge Portfolio Payoff", zorder=2)
        A_range = np.linspace(np.min(A_final), np.max(A_final), 200)
        payoff_range = np.maximum(K - A_range, 0)
        ax1.plot(A_range, payoff_range, 'k-', linewidth=2, label="Insurance Contract Payoff", zorder=1)
        ax1.set_title("Discrete Delta Hedging of a Portfolio Insurance Contract", fontsize=12)
        ax1.text(0.35, 0.85, f"# Hedge Points = {Nhedges}", transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right')
        ax1.set_xlabel("A(T)", fontsize=10)
        ax1.set_ylabel("Hedge Portfolio Value", fontsize=10)
        ax1.legend()
        
        ax2.hist(final_errors, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_title("Histogram of Final Hedge Errors", fontsize=12)
        ax2.set_xlabel("Hedge Error", fontsize=10)
        ax2.set_ylabel("Frequency", fontsize=10)
        
        cls.convergence_of_hedge_error(ax=ax3, r=r, S0=S0, A0=A0, T=T, K=K, a=a, sigma=sigma, Nreps=Nreps, max_hedges=1000)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    @classmethod
    def plot_multiple_hedge_experiments(cls, r, S0, A0, T, K, a, sigma, Nreps, hedge_dict, save_path=None):
        ncols = len(hedge_dict)
        fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
        plt.style.use('ggplot')
        
        for i, (label, Nhedges) in enumerate(hedge_dict.items()):
            hedge = cls(r, S0, A0, T, K, a, sigma, Nhedges, Nreps)
            hedge.hedge_experiment()
            A_final = hedge.A
            V_final = hedge.V
            hedge_error = hedge.final_payoff_error()
            
            ax_scatter = axes[0, i]
            ax_scatter.scatter(A_final, V_final, s=10, alpha=0.2, label="Hedge Portfolio", zorder=2)
            A_range = np.linspace(np.min(A_final), np.max(A_final), 200)
            payoff_range = np.maximum(K - A_range, 0)
            ax_scatter.plot(A_range, payoff_range, 'k-', linewidth=2, label="Insurance Contract Payoff", zorder=1)
            ax_scatter.set_title(f"{label} Hedge (n={Nhedges})", fontsize=12)
            ax_scatter.set_xlabel("A(T)", fontsize=10)
            ax_scatter.set_ylabel("Hedge Portfolio Value", fontsize=10)
            ax_scatter.legend()
            
            ax_hist = axes[1, i]
            ax_hist.hist(hedge_error, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
            ax_hist.set_title(f"Hedge Error: {label}", fontsize=12)
            ax_hist.set_xlabel("Hedge Error", fontsize=10)
            ax_hist.set_ylabel("Frequency", fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

if __name__ == "__main__":
    S0 = 1
    A0 = 1
    T = 30 
    K = np.exp(0.6)
    r = 0.02
    a = 0.5
    sigma = 0.20
    Nreps = 1000
    
    Nhedges_daily = 7560
    PortfolioInsuranceHedgeBase.plot_hedge_experiment(r, S0, A0, T, K, a, sigma, Nreps, Nhedges_daily,
                                                        save_path="Assignment_3/question_1b_portfolio_insurance_hedge_experiment.png")
    
    hedge_dict = {
        "Yearly": 30,
        "Monthly": 360,
        "Weekly": 1560,
        "Daily": 7560
    }
    PortfolioInsuranceHedgeBase.plot_multiple_hedge_experiments(r, S0, A0, T, K, a, sigma, Nreps, hedge_dict,
                                                                save_path="Assignment_3/multiple_hedge_experiments.png")