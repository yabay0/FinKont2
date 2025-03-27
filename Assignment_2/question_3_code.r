LSM_AmericanPut <- function(S, strike, r, degree = 5, method = "lm") {
  paths <- nrow(S)
  steps <- ncol(S)
  disc <- exp(r * (-(1:(steps - 1))))
  
  P <- matrix(0, nrow = paths, ncol = steps)
  
  for (j in 1:paths) {
    P[j, steps] <- max(strike - S[j, steps], 0)
  }
  
  for (h in (steps - 1):2) {
    Y <- rep(0, paths)
    for (j in 1:paths) {
      Y[j] <- sum(disc[1:(steps - h)] * P[j, (h + 1):steps])
    }
    
    dummy <- strike - S[, h]
    pick <- dummy > 0
    if (sum(pick) == 0) next
    
    if (method == "matrix") {
      X <- cbind(1, sapply(1:degree, function(k) S[, h]^k))
      X_pick <- X[pick, ]
      Y_pick <- Y[pick]
      b <- solve(t(X_pick) %*% X_pick, t(X_pick) %*% Y_pick)
      predicted <- X %*% b
    } else if (method == "lm") {
      data_pick <- data.frame(Y = Y[pick], S = S[pick, h])
      model <- lm(Y ~ poly(S, degree, raw = TRUE), data = data_pick)
      predicted <- predict(model, newdata = data.frame(S = S[, h]))
    } else {
      stop("Invalid method specified. Choose 'matrix' or 'lm'.")
    }
    
    for (j in 1:paths) {
      if (pick[j] && (predicted[j] < max(dummy[j], 0))) {
        P[j, h] <- max(dummy[j], 0)
        P[j, (h + 1):steps] <- 0 
      }
    }
  }
  
  temp <- rep(0, paths)
  for (j in 1:paths) {
    temp[j] <- sum(disc * P[j, 2:steps])
  }
  
  LSprice <- mean(temp)
  return(LSprice)
}

S <- matrix(c(1, 1.09, 1.08, 1.34,
              1, 1.16, 1.26, 1.54,
              1, 1.22, 1.07, 1.03,
              1, 0.93, 0.97, 0.92,
              1, 1.11, 1.56, 1.52,
              1, 0.76, 0.77, 0.90,
              1, 0.92, 0.84, 1.01,
              1, 0.88, 1.22, 1.34), 
            nrow = 8, ncol = 4, byrow = TRUE)

degrees <- c(2, 3, 4, 5)
price_results <- data.frame(Degree = degrees, lm = NA, matrix = NA)

for (i in 1:length(degrees)) {
  d <- degrees[i]
  price_results$lm[i] <- LSM_AmericanPut(S, strike = 1.1, r = 0.06, degree = d, method = "lm")
  price_results$matrix[i] <- LSM_AmericanPut(S, strike = 1.1, r = 0.06, degree = d, method = "matrix")
}

print(price_results)

degree <- 5
h <- 3
dummy <- strike - S[, h]
pick <- dummy > 0

cat("At time step h =", h)
cat("Asset values:", S[, h])
cat("Immediate payoff (strike - S):", dummy)
cat("Number of in-the-money paths:", sum(pick))

X <- cbind(1, sapply(1:degree, function(k) S[, h]^k))
cat("Full design matrix X (for all paths):")
print(X)

X_pick <- X[pick, ]
cat("Design matrix X_pick (in-the-money paths):")
print(X_pick)

rank_X_pick <- qr(X_pick)$rank
cat("Rank of X_pick:", rank_X_pick, "out of", ncol(X_pick), "columns\n")

cond_number <- kappa(X_pick)
cat("Condition number of X_pick:", cond_number, "\n")