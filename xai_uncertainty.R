# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Load libraries, register cores
library(data.table)
library(mlbench)
library(xgboost)
library(precrec)
library(Rfast)
library(ggplot2)
library(viridis)
library(doMC)
registerDoMC(8)

# Set seed
set.seed(123)

### Friedman benchmark ###

# Simulate data
n <- 2000
dat1 <- mlbench.friedman1(n)
dat2 <- mlbench.friedman1(n)
dat3 <- mlbench.friedman1(n/2)
x1 <- dat1$x
x2 <- dat2$x
x3 <- dat3$x
colnames(x1) <- colnames(x2) <- colnames(x3) <- paste0('x', 1:10)
y1 <- dat1$y
y2 <- dat2$y
y3 <- dat3$y
y1 <- scale(y1)^2
y2 <- scale(y2)^2
y3 <- scale(y3)^2
z_fn <- function(x, y) {
  10 * sin(pi * x[, 6] * x[, 7]) + 20 * (x[, 8] - 1/2)^2 + 10 * x[, 9] + 
    5 * x[, 10] + rnorm(nrow(x), sd = sqrt(y))
}
z1 <- z_fn(x1, y1)
z2 <- z_fn(x2, y2)
z3 <- z_fn(x3, y3)

# Train XGBoost model for conditional mean, compute Shapley values on test set
f <- xgboost(data = x1, label = z1, nrounds = 25, verbose = 0)
phi_f <- predict(f, x3, predcontrib = TRUE)[, 1:10]
colnames(phi_f) <- paste0('x', 1:10)

# Train XGBoost model for conditional variance, compute Shapley values on test set
eps <- z2 - predict(f, x2)
#h <- xgboost(data = x2, label = log(eps^2), nrounds = 25)
h <- xgboost(data = x2, label = y2, nrounds = 25, verbose = 0)
phi_h <- predict(h, x3, predcontrib = TRUE)[, 1:10]
colnames(phi_h) <- paste0('x', 1:10)

# Plot results
df <- melt(data.table(x3), measure.vars = 1:10, variable.name = 'feature')
tmp1 <- melt(data.table(phi_f), measure.vars = 1:10, variable.name = 'feature',
             value.name = 'phi')[, 'moment' := 'Mean'][, feature := NULL]
tmp2 <- melt(data.table(phi_h), measure.vars = 1:10, variable.name = 'feature',
             value.name = 'phi')[, 'moment' := 'Variance'][, feature := NULL]
df <- rbind(cbind(df, tmp1), cbind(df, tmp2))
ggplot(df, aes(phi, feature, color = value)) + 
  geom_jitter(size = .5, width = 0, height = 0.1, alpha = 0.25) + 
  geom_vline(xintercept = 0, color = 'red', linetype = 'dashed') +
  scale_color_viridis('Feature\nValue', option = 'C') +
  labs(x = 'Shapley Value', y = 'Feature') +
  theme_bw() + 
  theme(text = element_text(size = 8), legend.key.size = unit(0.4, 'cm')) + 
  facet_wrap(~ moment, scales = 'free_x')
ggsave('friedman.pdf', width = 5, height = 3)


# Coverage
tst_n <- 1000
alpha <- 0.1
inner_loop <- function(b, j, moment) {
  if (moment == 1) {
    phi <- phi_f
  } else if (moment == 2) {
    phi <- phi_h
  }
  i <- sample(tst_n, 1)
  s <- phi[i, j]
  x <- phi[-i, j]
  #mu <- mean(x)
  #R <- abs(x - mu)
  #q <- ceiling(tst_n * (1 - alpha))
  #r <- sort(R)[q]
  #lo <- mu - r
  #hi <- mu + r
  q_lo <- floor(tst_n * alpha / 2)
  q_hi <- ceiling(tst_n * (1 - alpha / 2))
  lo <- sort(x)[q_lo]
  hi <- sort(x)[q_hi]
  cvg <- 1 - ((sum(x < lo) + sum(x > hi)) / (tst_n - 1))
  data.table(
    'Feature' = paste0('x', j), 
    'Moment' = moment,
    'Covered' = ifelse(s >= lo & s <= hi, 1, 0),
    'hi' = hi, 'lo' = lo
  )
}
outer_loop <- function(m) {
  df <- foreach(bb = 1:1000, .combine = rbind) %:%
    foreach(jj = 1:10, .combine = rbind) %:%
    foreach(mm = c(1, 2), .combine = rbind) %do% inner_loop(bb, jj, mm)
  df[, Cvg := mean(Covered), by = .(Feature, Moment)]
  df[, hi_avg := mean(hi), by = .(Feature, Moment)]
  df[, lo_avg := mean(lo), by = .(Feature, Moment)]
  unique(df[, .(Feature, Moment, Cvg, hi_avg, lo_avg)])
}
df <- foreach(mm = 1:20, .combine = rbind) %dopar%
  outer_loop(mm)

df1 <- df[Moment == 1, mean(Cvg), by = Feature]
setnames(df1, 'V1', 'Coverage')
df1[, hi := df[Moment == 1, mean(hi_avg), by = Feature]$V1]
df1[, lo := df[Moment == 1, mean(lo_avg), by = Feature]$V1]
df2 <- df[Moment == 2, mean(Cvg), by = Feature]
setnames(df2, 'V1', 'Coverage')
df2[, hi := df[Moment == 2, mean(hi_avg), by = Feature]$V1]
df2[, lo := df[Moment == 2, mean(lo_avg), by = Feature]$V1]


# Feature value acquisition
zero_out <- function(x, frac = 0.1) {
  n <- nrow(x)
  out <- foreach(j = 1:10, .combine = cbind) %do% {
    tmp <- x[, j]
    tmp[sample(n, frac * n)] <- NA_real_
    return(tmp)
  }
  colnames(out) <- colnames(x)
  return(out)
}
df <- foreach(ff = c(0, 0.05, 0.1, 0.2, 0.5), .combine = rbind) %dopar% {
  x2_tilde <- zero_out(x2, ff)
  h <- xgboost(data = x2_tilde, label = y2, nrounds = 10, verbose = 0)
  x3_tilde <- zero_out(x3)
  phi_h <- predict(h, x3_tilde, predcontrib = TRUE)[, 1:10]
  phi <- foreach(i = 1:nrow(phi_h), .combine = rbind) %do% {
    phi_h[i, ] / sum(phi_h[i, ])
  }
  df <- rbindlist(
    lapply(1:10, function(j) {
      na_idx <- is.na(x3_tilde[, j])
      data.table('feature' = paste0('x', j), 
                 'phi' = abs(phi[na_idx, j]), 
                 'impt' = ifelse(j <= 5L, 1L, 0L))
    })
  )
  rocs <- evalmod(scores = df$phi, labels = df$impt)$rocs
  out <- data.table(FPR = rocs[[1]]$x, TPR = rocs[[1]]$y, frac = ff * 100,
                    'AUC' = attr(rocs[[1]], 'auc'))
  return(out)
}
ggplot(df, aes(FPR, TPR, color = as.factor(frac))) +
  geom_line(linewidth = 0.75) + 
  geom_abline(intercept = 0L, slope = 1L,
              linetype = 'dashed', color = 'grey') +
  lims(x = c(0L, 1L), y = c(0L, 1L)) +
  labs(x = 'False Positive Rate', y = 'True Positive Rate') +
  scale_color_d3(name = 'Percent\nMissing') +
  theme_bw() + 
  theme(text = element_text(size = 8), legend.key.size = unit(0.4, 'cm')) 
ggsave('fva.pdf', width = 5, height = 3)




### COVARIATE SHIFT ###

## CASE 1: BREASTCANCER ##
df <- data.table(BreastCancer)[, c('Id', 'Class') := NULL]
df <- as.data.table(lapply(df, as.numeric))
df[, Class := BreastCancer$Class]
x <- df[, -c('Class')]
x <- as.data.table(lapply(x, as.numeric))
n <- x[, .N]
idx <- sample(1:n, n/5)
trn_x <- as.matrix(x[-idx, ])
tst_x <- as.matrix(x[idx, ])
trn_y <- df$Class[-idx]
trn_y <- ifelse(trn_y == 'malignant', 1L, 0L)
tst_y <- df$Class[idx]
f0 <- xgboost(data = trn_x, label = trn_y, nrounds = 20, 
              objective = 'binary:logistic', verbose = 0)
y1_hat <- predict(f0, trn_x)
y0_hat <- 1 - y1_hat
entropy <- -(y1_hat * log(y1_hat) + y0_hat * log(y0_hat))
f1 <- xgboost(data = trn_x, label = entropy, nrounds = 20, verbose = 0)
phi0 <- as.data.table(predict(f1, tst_x, predcontrib = TRUE))
xgb.importance(colnames(trn_x), model = f0)

# Perturb cell size
x_tilde <- x
x_tilde[, Cell.size := Cell.size + rnorm(.N)]
x_tilde_mat <- as.matrix(x_tilde)
y1_tilde <- predict(f0, x_tilde_mat)
y0_tilde <- 1 - y1_tilde
h_tilde <- -(y1_tilde * log(y1_tilde) + y1_tilde * log(y0_tilde))
f1_tilde <- xgboost(data = x_tilde_mat[-idx, ], label = h_tilde[-idx], nrounds = 20, 
                    verbose = 0)
phi1 <- as.data.table(predict(f1_tilde, x_tilde_mat[idx, ], predcontrib = TRUE))

# Plot: feature value vs. Shapley value, colored by shift indicator
tmp1 <- data.table('phi' = c(phi0$Cell.size, phi1$Cell.size), 
                  'value' = c(df[idx, Cell.size], x_tilde[idx, Cell.size]))
tmp1[, Data := rep(c('Original', 'Perturbed'), each = length(idx))]
tmp1[, Dataset := 'BreastCancer']

## CASE 2: Diabetes
data(PimaIndiansDiabetes2)
df <- data.table(PimaIndiansDiabetes2)
x <- df[, -c('diabetes')]
n <- x[, .N]
idx <- sample(1:n, n/5)
trn_x <- as.matrix(x[-idx, ])
tst_x <- as.matrix(x[idx, ])
trn_y <- df$diabetes[-idx]
trn_y <- ifelse(trn_y == 'pos', 1L, 0L)
f0 <- xgboost(data = trn_x, label = trn_y, nrounds = 20, 
              objective = 'binary:logistic', verbose = 0)
y1_hat <- predict(f0, trn_x)
y0_hat <- 1 - y1_hat
entropy <- -(y1_hat * log(y1_hat) + y0_hat * log(y0_hat))
f1 <- xgboost(data = trn_x, label = entropy, nrounds = 20, verbose = 0)
phi0 <- as.data.table(predict(f1, tst_x, predcontrib = TRUE))
xgb.importance(colnames(trn_x), model = f0)

# Perturb glucose
x_tilde <- x
x_tilde[, glucose := glucose + rnorm(.N)]
x_tilde_mat <- as.matrix(x_tilde)
y1_tilde <- predict(f0, x_tilde_mat)
y0_tilde <- 1 - y1_tilde
h_tilde <- -(y1_tilde * log(y1_tilde) + y1_tilde * log(y0_tilde))
f1_tilde <- xgboost(data = x_tilde_mat[-idx, ], label = h_tilde[-idx], nrounds = 20, 
                    verbose = 0)
phi1 <- as.data.table(predict(f1_tilde, x_tilde_mat[idx, ], predcontrib = TRUE))

# Plot: feature value vs. Shapley value, colored by shift indicator
tmp2 <- data.table('phi' = c(phi0$glucose, phi1$glucose), 
                  'value' = c(df[idx, glucose], x_tilde[idx, glucose]))
tmp2[, Data := rep(c('Original', 'Perturbed'), each = length(idx))]
tmp2[, Dataset := 'Diabetes']

## CASE 3: ##
data(Sonar)
df <- data.table(Sonar)
x <- df[, -c('Class')]
n <- x[, .N]
idx <- sample(1:n, n/5)
trn_x <- as.matrix(x[-idx, ])
tst_x <- as.matrix(x[idx, ])
trn_y <- df$Class[-idx]
trn_y <- ifelse(trn_y == 'M', 1L, 0L)
f0 <- xgboost(data = trn_x, label = trn_y, nrounds = 20, 
              objective = 'binary:logistic', verbose = 0)
y1_hat <- predict(f0, trn_x)
y0_hat <- 1 - y1_hat
entropy <- -(y1_hat * log(y1_hat) + y0_hat * log(y0_hat))
f1 <- xgboost(data = trn_x, label = entropy, nrounds = 20, verbose = 0)
phi0 <- as.data.table(predict(f1, tst_x, predcontrib = TRUE))
xgb.importance(colnames(trn_x), model = f0)

# Perturb V12
x_tilde <- x
x_tilde[, V12 := V12 + rnorm(.N, sd = 0.01)]
x_tilde_mat <- as.matrix(x_tilde)
y1_tilde <- predict(f0, x_tilde_mat)
y0_tilde <- 1 - y1_tilde
h_tilde <- -(y1_tilde * log(y1_tilde) + y1_tilde * log(y0_tilde))
f1_tilde <- xgboost(data = x_tilde_mat[-idx, ], label = h_tilde[-idx], nrounds = 20, 
                    verbose = 0)
phi1 <- as.data.table(predict(f1_tilde, x_tilde_mat[idx, ], predcontrib = TRUE))

# Plot: feature value vs. Shapley value, colored by shift indicator
tmp3 <- data.table('phi' = c(phi0$V12, phi1$V12), 
                  'value' = c(df[idx, V12], x_tilde[idx, V12]))
tmp3[, Data := rep(c('Original', 'Perturbed'), each = length(idx))]
tmp3[, Dataset := 'Sonar']


## CASE 4: Ionosphere ###

data(Ionosphere)
df <- data.table(Ionosphere)[, V2 := NULL][, V1 := as.numeric(V1)]
x <- df[, -c('Class')]
n <- x[, .N]
idx <- sample(1:n, n/5)
trn_x <- as.matrix(x[-idx, ])
tst_x <- as.matrix(x[idx, ])
trn_y <- df$Class[-idx]
trn_y <- ifelse(trn_y == 'good', 1L, 0L)
f0 <- xgboost(data = trn_x, label = trn_y, nrounds = 20, 
              objective = 'binary:logistic', verbose = 0)
y1_hat <- predict(f0, trn_x)
y0_hat <- 1 - y1_hat
entropy <- -(y1_hat * log(y1_hat) + y0_hat * log(y0_hat))
f1 <- xgboost(data = trn_x, label = entropy, nrounds = 20, verbose = 0)
phi0 <- as.data.table(predict(f1, tst_x, predcontrib = TRUE))
xgb.importance(colnames(trn_x), model = f0)

# Perturb V5
x_tilde <- x
x_tilde[, V5 := V5 + rnorm(.N, sd = 0.25)]
x_tilde_mat <- as.matrix(x_tilde)
y1_tilde <- predict(f0, x_tilde_mat)
y0_tilde <- 1 - y1_tilde
h_tilde <- -(y1_tilde * log(y1_tilde) + y1_tilde * log(y0_tilde))
f1_tilde <- xgboost(data = x_tilde_mat[-idx, ], label = h_tilde[-idx], nrounds = 20, 
                    verbose = 0)
phi1 <- as.data.table(predict(f1_tilde, x_tilde_mat[idx, ], predcontrib = TRUE))

# Plot: feature value vs. Shapley value, colored by shift indicator
tmp4 <- data.table('phi' = c(phi0$V5, phi1$V5), 
                   'value' = c(df[idx, V5], x_tilde[idx, V5]))
tmp4[, Data := rep(c('Original', 'Perturbed'), each = length(idx))]
tmp4[, Dataset := 'Ionosphere']

tmp <- rbind(tmp1, tmp2, tmp3, tmp4)
ggplot(tmp, aes(value, phi, color = Data, shape = Data)) + 
  geom_point(size = 0.5, alpha = 0.75) + 
  scale_color_d3() + 
  labs(x = 'Feature Value', y = 'Shapley Value') +
  facet_wrap(~ Dataset, nrow = 2, scales = 'free') +
  theme_bw() + 
  theme(text = element_text(size = 8), legend.key.size = unit(0.4, 'cm')) 
ggsave('ood.pdf', width = 5, height = 3)


### Convergence experiments ###

# Hyperparameters
n <- 2000
d <- 4
sparsity <- 0
rho <- 0.5
mu <- rep(0, d + 1)
sigma <- toeplitz(rho^(0:(d-1)))

# Effects 
beta <- double(length = d)
k <- round((1 - sparsity) * d)
if (k > 0) {
  beta[1:k] <- 1L
}

# Create data
x <- matrix(rmvnorm(n + 1000, mu = mu, sigma = sigma), ncol = d + 1,
            dimnames = list(NULL, paste0('x', 1:d)))
log_var <- x %*% beta
s2 <- exp(log_var)
y <- rnorm(n + 1000, sd = sqrt(s2))
y2 <- log(y^2)

# Split train/test
trn_x <- x[1:n, ]
trn_y <- y2[1:n]
tst_x <- x[(n + 1):(n + 1000), ]
tst_y <- y2[(n + 1):(n + 1000)]

# Fit heteroskedastic error model
h <- lm(trn_y ~ trn_x)
summary(h)

# Plot
ggplot(df1, aes(n, MAE, color = Method, fill = Method)) + 
  geom_point() + 
  geom_line() + 
  geom_ribbon(aes(ymin = MAE - std_dev, ymax = MAE + std_dev), alpha = 0.4) + 
  scale_y_log10() +
  scale_color_d3() + 
  scale_fill_d3() +
  labs(x = 'Sample Size', y = 'MAE (log scale)') +
  theme_bw() +
  theme(text = element_text(size = 8), legend.key.size = unit(0.4, 'cm')) 
ggsave('n_vs_mae.pdf', width = 5, height = 3)

ggplot(df2, aes(rho, MAE, color = Method, fill = Method)) + 
  geom_point() + 
  geom_line() + 
  geom_ribbon(aes(ymin = MAE - std_dev, ymax = MAE + std_dev), alpha = 0.4) + 
  scale_y_log10() +
  scale_color_d3() + 
  scale_fill_d3() +
  labs(x = 'Correlation', y = 'MAE (log scale)') +
  theme_bw() +
  theme(text = element_text(size = 8), legend.key.size = unit(0.4, 'cm')) 
ggsave('rho_vs_mae.pdf', width = 5, height = 3)










































