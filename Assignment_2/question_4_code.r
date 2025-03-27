set.seed(123)
n_experiments <- 10000
LSM_prices <- numeric(n_experiments)

S0 <- 1
strike <- 1.1
r <- 0.06
sigma_true <- 0.2
n_paths <- 8
n_steps <- 4
dt <- 1

for(exp in 1:n_experiments){
  S <- matrix(0, nrow = n_paths, ncol = n_steps)
  S[,1] <- S0
  for(j in 1:(n_steps - 1)){
    Z <- rnorm(n_paths)
    S[, j+1] <- S[, j] * exp((r - 0.5 * sigma_true^2) * dt + sigma_true * sqrt(dt) * Z)
  }
  LSM_prices[exp] <- LSM_AmericanPut(S, strike, r, degree = 2, method = "lm")
}

mean_price <- mean(LSM_prices)
sd_price <- sd(LSM_prices)

png("/Users/yasinbaysal/Desktop/FinKont2/FinKont2/Assignment_2/question_4_LSM_american_put.png", width = 800, height = 600)

par(cex = 0.7)
hist(LSM_prices, breaks = 50, probability = FALSE,
    main = "Distribution of LSM American Put Prices (10,000 experiments)",
    xlab = "LSM American Put Price",
    col = "lightblue", border = "black")

abline(v = mean_price, col = "red", lwd = 2)

legend("topright", legend = c(paste("Mean =", round(mean_price, 4)),
                        paste("Standard deviation =", round(sd_price, 4))),
      col = c("red", "white"), lwd = 2)

dev.off()