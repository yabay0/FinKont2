binomial_bermudan <- function(S0, capT, strike, r, sigma, n, m, 
                               divyield = 0.0, opttype = 2) {
  dt <- capT / n
  u  <- exp(sigma * sqrt(dt))
  d  <- exp(-sigma * sqrt(dt))
  R  <- exp(r * dt)
  q  <- (R - (d + divyield * dt)) / (u - d)
  
  if (opttype == 1) {
    payoff <- function(x) pmax(x - strike, 0)
  } else {
    payoff <- function(x) pmax(strike - x, 0)
  }
  
  exponents <- 0:n
  prices <- payoff(S0 * (u^exponents) * (d^(n - exponents)))
  
  for (i in n:2) {
    exponents_i <- 0:(i - 1)
    St <- S0 * (u^exponents_i) * (d^((i - 1) - exponents_i))
    
    exercise_value <- if ((i - 1) %% m == 0) payoff(St) else numeric(length(St))
    cont <- (q * prices[2:(i + 1)] + (1 - q) * prices[1:i]) / R
    prices <- pmax(cont, exercise_value)
  }
  
  return((q * prices[2] + (1 - q) * prices[1]) / R)
}

S0     <- 1
strike <- 1.1
capT   <- 3
n_steps <- 3 * 252

estimator <- VolatilityEstimator$new(S, r)
rv_est  <- estimator$estimate_volatility_RV()
ml_est  <- estimator$estimate_volatility_ML()

sigma_RV <- rv_est$sigma_RV
sigma_ML <- ml_est$sigma_ML

sigmas <- c(sigma_ML, 0.15, sigma_RV)
prices <- matrix(0, nrow = 3, ncol = 4)
colnames(prices) <- c("Sigma", "American", "Bermudan", "European")
prices[,1] <- sigmas

for (i in seq_along(sigmas)) {
  vol_i <- sigmas[i]
  am_price <- binomial_bermudan(S0, capT, strike, r, vol_i, n_steps, 1)
  bm_price <- binomial_bermudan(S0, capT, strike, r, vol_i, n_steps, 252)
  eu_price <- binomial_bermudan(S0, capT, strike, r, vol_i, n_steps, n_steps)
  
  prices[i, 2] <- am_price
  prices[i, 3] <- bm_price
  prices[i, 4] <- eu_price
}

print(round(prices, 4))