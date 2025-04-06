import numpy as np
import matplotlib.pyplot as plt
from Assignment_3.question_1g_code import Heston_Fourier

def simulate_hedging(r, sigma, spot, rho, S0, capT, V0, theta, kappa, epsilon, Nhedge, Nrep):
    dt = capT / Nhedge
    strike = np.exp(r * capT)
    S = np.full(Nrep, S0)
    V = np.full(Nrep, V0)
    initial_outlay = Heston_Fourier(spot, capT, strike, r, 0, V0, theta, kappa, epsilon, rho, greek=1)
    Vpf = np.full(Nrep, initial_outlay)
    delta_initial = Heston_Fourier(S0, capT, strike, r, 0, V0, theta, kappa, epsilon, rho, greek=2) - 1
    delta = np.full(Nrep, delta_initial)
    a_pos = delta.copy()
    b_pos = Vpf - a_pos * S

    for i in range(1, Nhedge):
        print(f"Step {i}/{Nhedge}")
        timetomat = capT - i * dt
        dW1 = np.sqrt(dt) * np.random.randn(Nrep)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.sqrt(dt) * np.random.randn(Nrep)
        X = np.log(S)
        X = X + (r - 0.5 * np.maximum(V, 0)) * dt + np.sqrt(np.maximum(V, 0)) * dW1
        S = np.exp(X)
        V = V + kappa * (theta - np.maximum(V, 0)) * dt + epsilon * np.sqrt(np.maximum(V, 0)) * dW2
        Vpf = a_pos * S + b_pos * np.exp(r * dt)
        new_delta = np.empty(Nrep)
        for j in range(Nrep):
            new_delta[j] = Heston_Fourier(S[j], timetomat, strike, r, 0, V[j], theta, kappa, epsilon, rho, greek=2) - 1
        delta = new_delta.copy()
        a_pos = delta.copy()
        b_pos = Vpf - a_pos * S

    dW1 = np.sqrt(dt) * np.random.randn(Nrep)
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.sqrt(dt) * np.random.randn(Nrep)
    X = np.log(S)
    X = X + (r - 0.5 * np.maximum(V, 0)) * dt + np.sqrt(np.maximum(V, 0)) * dW1
    S = np.exp(X)
    V = V + kappa * (theta - np.maximum(V, 0)) * dt + epsilon * np.sqrt(np.maximum(V, 0)) * dW2
    Vpf = a_pos * S + b_pos * np.exp(r * dt)
    
    return S, Vpf, strike, Nhedge

def plot_results(S, Vpf, strike, Nhedge, save_path=None):
    fig, ax = plt.subplots()
    ax.scatter(S, Vpf, color='blue', label='Hedge Portfolio')
    ax.set_xlabel("S(T)")
    ax.set_ylabel("Value of hedge portfolio")
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 2)
    ax.set_title("Discrete Hedging Experiment of a Portfolio Insurance Contract")
    ax.text(9, 1.8, f"# hedge points = {Nhedge}", horizontalalignment='right')
    x_vals = np.linspace(0, 10, 400)
    payoff = np.maximum(strike - x_vals, 0)
    ax.plot(x_vals, payoff, linewidth=3, label='Payoff')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none')  # Use fig.get_facecolor()
    else:
        plt.show()

if __name__ == "__main__":
    r = 0.02
    sigma = 0.2
    spot = 1.0
    rho = -0.5
    S0 = 1.0
    capT = 30.0
    V0 = 0.2 ** 2
    theta = 0.2 ** 2
    kappa = 2.0
    epsilon = 1.0
    Nhedge = int(252 * 30)
    Nrep = 100
    S, Vpf, strike, Nhedge = simulate_hedging(r, sigma, spot, rho, S0, capT, V0, theta, kappa, epsilon, Nhedge, Nrep)
    plot_results(S, Vpf, strike, Nhedge, save_path="Assignment_3/question_1i_hedging_results.png")