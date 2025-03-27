library(R6)

VolatilityEstimator <- R6Class("VolatilityEstimator",
  public = list(
    S = NULL,
    r = NULL,
    paths = NULL,
    steps = NULL,
    log_returns = NULL,
    M = NULL,
    
    initialize = function(S, r) {
      self$S <- as.matrix(S)
      self$r <- r
      dims <- dim(self$S)
      self$paths <- dims[1]
      self$steps <- dims[2]
      self$log_returns <- self$compute_log_returns()
      self$M <- length(self$log_returns)
    },
    
    compute_log_returns = function() {
      log_rets <- c()
      for (i in 1:self$paths) {
        for (j in 1:(self$steps - 1)) {
          log_rets <- c(log_rets, log(self$S[i, j+1] / self$S[i, j]))
        }
      }
      return(log_rets)
    },
    
    estimate_volatility_RV = function() {
      sigma_RV <- sd(self$log_returns)
      sigma_RV_se <- sigma_RV / sqrt(2 * (self$M - 1))
      return(list(sigma_RV = sigma_RV, sigma_RV_se = sigma_RV_se))
    },
    
    log_likelihood = function(sigma) {
      x <- self$log_returns
      mu <- self$r - 0.5 * sigma^2
      N <- length(x)
      ll <- -N * log(sigma) - 0.5 * N * log(2 * pi) - 
            sum((x - mu)^2) / (2 * sigma^2)
      return(ll)
    },
    
    estimate_volatility_ML = function() {
      nll <- function(sigma) {
        -self$log_likelihood(sigma)
      }
      opt <- optimize(nll, interval = c(0.1, 0.3))
      sigma_ML <- opt$minimum
      eps <- 1e-3
      ll_plus <- self$log_likelihood(sigma_ML + eps)
      ll_minus <- self$log_likelihood(sigma_ML - eps)
      ll_current <- self$log_likelihood(sigma_ML)
      second_deriv <- -(ll_plus - 2 * ll_current + ll_minus) / (eps^2)
      sigma_ML_se <- sqrt(1 / second_deriv)
      return(list(sigma_ML = sigma_ML, sigma_ML_se = sigma_ML_se))
    }
  )
)

S <- matrix(c(
  1.00, 1.09, 1.08, 1.34,
  1.00, 1.16, 1.26, 1.54,
  1.00, 1.22, 1.07, 1.03,
  1.00, 0.93, 0.97, 0.92,
  1.00, 1.11, 1.56, 1.52,
  1.00, 0.76, 0.77, 0.90,
  1.00, 0.92, 0.84, 1.01,
  1.00, 0.88, 1.22, 1.34
), nrow = 8, byrow = TRUE)

r <- 0.06
estimator <- VolatilityEstimator$new(S, r)
rv_results <- estimator$estimate_volatility_RV()
ml_results <- estimator$estimate_volatility_ML()

(results <- data.frame(
  Method           = c("sigma_{ML}", "sigma_{RV}"),
  Estimate         = c(sprintf("%.4f", ml_results$sigma_ML), 
                       sprintf("%.4f", rv_results$sigma_RV)),
  "Standard Error" = c(sprintf("%.4f", ml_results$sigma_ML_se), 
                       sprintf("%.4f", rv_results$sigma_RV_se))
))