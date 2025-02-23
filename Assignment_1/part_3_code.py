import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class QuantoOptionHedgeBase:
    def __init__(self, r_US, r_J, S_J_start, X_0, K, sigma_X, sigma_J, Y_0, T, Nhedges, Nreps):
        self.r_US = r_US
        self.r_J = r_J
        self.S_J_start = S_J_start
        self.X_0 = X_0
        self.K = K
        self.sigma_X = sigma_X
        self.sigma_J = sigma_J
        self.Y_0 = Y_0
        self.T = T
        self.Nhedges = Nhedges
        self.Nreps = Nreps
        self.dt = T / Nhedges
        self.exp_r_dt = np.exp(self.r_US * self.dt)

        self.q_const = self.r_US - self.r_J + np.dot(self.sigma_X.T, self.sigma_J)
        self.norm_sigma_J = np.linalg.norm(self.sigma_J)
        self.norm_sigma_X = np.linalg.norm(self.sigma_X)

        self.drift_SJ = ((self.r_J - np.dot(self.sigma_X.T, self.sigma_J))
                         - 0.5 * self.norm_sigma_J**2) * self.dt
        self.drift_X = ((self.r_US - self.r_J)
                        - 0.5 * self.norm_sigma_X**2) * self.dt
        self.vol_factor = np.sqrt(self.dt)

        self.pf_value = None
        self.S_J = None
        self.X = None

    def _compute_d(self, spot, strike, r, q, sigma, T, d_type=1):
        denom = sigma * np.sqrt(T)
        if d_type == 1:
            return (np.log(spot / strike) + (r - q) * T + 0.5 * sigma**2 * T) / denom
        else:
            return (np.log(spot / strike) + (r - q) * T - 0.5 * sigma**2 * T) / denom

    def black_scholes_price(self, spot, strike, T, r, q, sigma):
        d1 = self._compute_d(spot, strike, r, q, sigma, T, d_type=1)
        d2 = self._compute_d(spot, strike, r, q, sigma, T, d_type=2)
        return (strike * np.exp(-r * T) * norm.cdf(-d2)
                - np.exp(-q * T) * spot * norm.cdf(-d1))

    def _g(self, Y_0, spot, strike, T, r, sigma):
        d1 = self._compute_d(spot, strike, r, self.q_const, sigma, T, d_type=1)
        return Y_0 * np.exp(-self.q_const * T) * (norm.cdf(d1) - 1)

    def hedge_experiment(self):
        raise NotImplementedError()

    def final_payoff_error(self):
        raise NotImplementedError()

    @classmethod
    def convergence_of_hedge_error(cls, ax, r_US, r_J, S_J_start, X_0, K,
                                   sigma_X, sigma_J, Y_0, T, Nreps, max_hedges=1000):
        std_devs = []
        for Nhedges in range(2, max_hedges + 1):
            hedge = cls(r_US, r_J, S_J_start, X_0, K, sigma_X, sigma_J, Y_0, T, Nhedges, Nreps)
            hedge.hedge_experiment()
            error = hedge.final_payoff_error()
            discounted_error = np.exp(-r_US * T) * error
            std_devs.append(np.std(discounted_error))
        ax.plot(range(2, max_hedges + 1), std_devs, linewidth=1, color='red')
        ax.set_xlabel("Number of Hedge Points", fontsize=10)
        ax.set_ylabel("Std Dev of Discounted Error", fontsize=10)
        ax.set_title("Convergence of Hedge Error", fontsize=12)
        ax.grid(True)

    @classmethod
    def plot_hedge_experiment(cls, r_US, r_J, S_J_start, X_0, K,
                              sigma_X, sigma_J, Y_0, T, Nreps, Nhedges, save_path=None):
        hedge = cls(r_US, r_J, S_J_start, X_0, K, sigma_X, sigma_J, Y_0, T, Nhedges, Nreps)
        hedge.hedge_experiment()
        final_errors = hedge.final_payoff_error()
        S_final = hedge.S_J
        pf_value_final = hedge.pf_value

        fig = plt.figure(figsize=(10, 8))
        plt.style.use('ggplot')
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        ax1.scatter(S_final, pf_value_final, s=10, alpha=0.2, label="Hedge Portfolio", color='red')
        S_range = np.linspace(10000, 100000, 200)
        payoff_range = Y_0 * np.maximum(K - S_range, 0)
        ax1.plot(S_range, payoff_range, 'k-', linewidth=2, label="Put Payoff")
        ax1.set_title("Discrete Hedging of a Quanto Put Option", fontsize=12)
        ax1.text(0.35, 0.85, f"# Hedge Points = {Nhedges}",
                 transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right')
        ax1.set_xlabel("$S_J(T)$", fontsize=10)
        ax1.set_ylabel("Value of Hedge Portfolio", fontsize=10)
        ax1.legend()

        ax2.hist(final_errors, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_title("Histogram of Final Hedge Errors", fontsize=12)
        ax2.set_xlabel("Payoff - Portfolio", fontsize=10)
        ax2.set_ylabel("Frequency", fontsize=10)

        cls.convergence_of_hedge_error(ax=ax3, r_US=r_US, r_J=r_J,
                                       S_J_start=S_J_start, X_0=X_0, K=K,
                                       sigma_X=sigma_X, sigma_J=sigma_J,
                                       Y_0=Y_0, T=T, Nreps=Nreps, max_hedges=1000)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class QuantoOptionHedge3b(QuantoOptionHedgeBase):
    def hedge_experiment(self):
        option_price = self.black_scholes_price(
            self.S_J_start, self.K, self.T, self.r_US, self.q_const, self.norm_sigma_J
        )
        initial_outlay = self.Y_0 * np.exp(-self.r_US * self.T) * option_price

        self.pf_value = np.full(self.Nreps, initial_outlay)
        self.S_J = np.full(self.Nreps, self.S_J_start, dtype=float)
        self.X = np.full(self.Nreps, self.X_0, dtype=float)

        self.a = self._g(self.Y_0, self.S_J, self.K, self.T, self.r_US, self.norm_sigma_J) / self.X_0
        self.b = self.pf_value - self.X_0 * self.a * self.S_J

        for i in range(1, self.Nhedges):
            Z = np.random.normal(size=(self.Nreps, 2))
            self.S_J *= np.exp(self.drift_SJ + (Z @ self.sigma_J) * self.vol_factor)
            self.X   *= np.exp(self.drift_X  + (Z @ self.sigma_X) * self.vol_factor)

            self.pf_value = self.a * self.X * self.S_J + self.b * self.exp_r_dt
            remaining_T = self.T - (i - 1) * self.dt
            self.a = self._g(self.Y_0, self.S_J, self.K, remaining_T, self.r_US, self.norm_sigma_J) / self.X
            self.b = self.pf_value - self.X * self.a * self.S_J

    def final_payoff_error(self):
        payoff = self.Y_0 * np.maximum(self.K - self.S_J, 0)
        return payoff - self.pf_value


class QuantoOptionHedge3c(QuantoOptionHedgeBase):
    def hedge_experiment(self):
        option_price = self.black_scholes_price(
            self.S_J_start, self.K, self.T, self.r_US, self.q_const, self.norm_sigma_J
        )
        initial_outlay = self.Y_0 * np.exp(-self.r_US * self.T) * option_price

        self.S_J = np.full(self.Nreps, self.S_J_start, dtype=float)
        self.X   = np.full(self.Nreps, self.X_0, dtype=float)

        self.a = np.zeros(self.Nreps)
        self.c = np.zeros(self.Nreps)
        self.b = np.zeros(self.Nreps)

        a0 = self._g(self.Y_0, self.S_J, self.K, self.T, self.r_US, self.norm_sigma_J) / self.X_0
        c0 = - a0 * self.S_J
        b0 = initial_outlay - (a0 * self.X_0 * self.S_J) - (c0 * self.X_0)

        self.a[:] = a0
        self.c[:] = c0
        self.b[:] = b0

        for i in range(1, self.Nhedges):
            Z = np.random.normal(size=(self.Nreps, 2))
            self.S_J *= np.exp(self.drift_SJ + (Z @ self.sigma_J) * self.vol_factor)
            self.X   *= np.exp(self.drift_X  + (Z @ self.sigma_X) * self.vol_factor)

            old_a, old_c, old_b = self.a, self.c, self.b
            c_grown = old_c * np.exp(self.r_J * self.dt)
            b_grown = old_b * np.exp(self.r_US * self.dt)
            V_before = old_a * self.X * self.S_J + c_grown * self.X + b_grown

            remaining_T = self.T - (i - 1) * self.dt
            new_a = self._g(self.Y_0, self.S_J, self.K, remaining_T, self.r_US, self.norm_sigma_J) / self.X
            new_c = - new_a * self.S_J
            new_b = V_before - (new_a * self.X * self.S_J) - (new_c * self.X)

            self.a, self.c, self.b = new_a, new_c, new_b

        c_final = self.c * np.exp(self.r_J * self.dt)
        b_final = self.b * np.exp(self.r_US * self.dt)
        self.pf_value = self.a * self.X * self.S_J + c_final * self.X + b_final

    def final_payoff_error(self):
        payoff = self.Y_0 * np.maximum(self.K - self.S_J, 0)
        return payoff - self.pf_value


if __name__ == "__main__":
    r_US = 0.03
    r_J = 0.00
    S_J_start = 30000
    X_0 = 0.01
    K = 30000
    sigma_X = np.array([0.1, 0.02])
    sigma_J = np.array([0.0, 0.25])
    Y_0 = 0.01
    T = 2
    Nreps = 1000
    Nhedges = 504

    # Experiment 3.b
    QuantoOptionHedge3b.plot_hedge_experiment(
        r_US, r_J, S_J_start, X_0, K, sigma_X, sigma_J, Y_0, T, Nreps, Nhedges,
        save_path="Assignment_1/qquestion_3b_quanto_option_hedge_experiment.png"
    )

    # Experiment 3.c
    QuantoOptionHedge3c.plot_hedge_experiment(
        r_US, r_J, S_J_start, X_0, K, sigma_X, sigma_J, Y_0, T, Nreps, Nhedges,
        save_path="Assignment_1/qquestion_3c_quanto_option_hedge_experiment.png"
    )