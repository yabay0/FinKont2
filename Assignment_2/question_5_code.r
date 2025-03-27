B_fixed <- matrix(c(2.037512, -3.335443, 1.356457, -1.069988,  2.983411, -1.813576), nrow = 3, ncol = 2)

LSM_FixedAmericanPut <- function(S, strike, r) {
  paths <- nrow(S)
  steps <- ncol(S)
  P <- matrix(0, nrow = paths, ncol = steps)
  P[, steps] <- pmax(strike - S[, steps], 0)
  
  for (h in (steps - 1):2) {
    immediate <- pmax(strike - S[, h], 0)
    X <- cbind(1, S[, h], S[, h]^2)
    
    if (h == steps - 1) {
      b_fixed <- B_fixed[, 2]
    } else if (h == 2) {
      b_fixed <- B_fixed[, 1]
    } else {
      stop("Unexpected time step.")
    }
    
    predicted <- as.vector(X %*% b_fixed)
    
    for (i in 1:paths) {
      if (immediate[i] > predicted[i]) {
        P[i, h] <- immediate[i]
        if (h < steps) {
          P[i, (h + 1):steps] <- 0
        }
      }
    }
  }
  
  payoff <- numeric(paths)
  for (i in 1:paths) {
    exer_time <- which(P[i, ] > 0)[1]
    if (is.na(exer_time)) exer_time <- steps
    payoff[i] <- P[i, exer_time] * exp(-r * (exer_time - 1))
  }
  
  price <- mean(payoff)
  return(price)
}

set.seed(123)
n_experiments <- 10000
fixed_prices <- numeric(n_experiments)

for (exp in 1:n_experiments) {
  S <- matrix(0, nrow = n_paths, ncol = n_steps)
  S[, 1] <- S0
  for (j in 1:(n_steps - 1)) {
    Z <- rnorm(n_paths)
    S[, j + 1] <- S[, j] * exp((r - 0.5 * sigma_true^2) * dt + sigma_true * sqrt(dt) * Z)
  }
  fixed_prices[exp] <- LSM_FixedAmericanPut(S, strike, r)
}

mean_price_fixed_B <- mean(fixed_prices)
sd_price_fixed_B <- sd(fixed_prices)

png("/Users/yasinbaysal/Desktop/FinKont2/FinKont2/Assignment_2/question_4_LSM_american_put_fixed.png", width = 800, height = 600)

par(cex = 0.7)
hist(fixed_prices, breaks = 50, probability = FALSE,
     main = "Distribution of LSM American Put Prices using Fixed Strategy (10,000 experiments)",
     xlab = "LSM American Put Price",
     col = "lightblue", border = "black")

abline(v = mean_price_fixed_B, col = "red", lwd = 2)

legend("topright", legend = c(paste("Mean =", round(mean_price_fixed_B, 4)),
                              paste("Standard deviation =", round(sd_price_fixed_B, 4))), col = c("red", "white"), lwd = 2)

dev.off()