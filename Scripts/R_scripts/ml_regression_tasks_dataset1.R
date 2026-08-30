#!/usr/bin/env Rscript
#
# ml_regression_tasks_dataset1_rerun.R
#
# Dataset1 (Adult Census Income) regression energy measurement, R side.
# Mirrors ml_regression_tasks_dataset1_rerun.py: same data path, same
# feature/target selection, same split proportions, same standardisation of
# both X and Y, same kernel caps, same five algorithms in the paper's package
# mapping, same ten measured functions, same repetition/shuffle protocol.
#
# Model -> CRAN package mapping (fixed by the paper, do not change):
#   Linear Regression      -> glmnet
#   Gaussian Regression     -> MASS
#   Decision Tree            -> tree
#   Support Vector Machine   -> e1071
#   Neural Network            -> neuralnet
#
# [R-DECISION-GAUSSIAN] MASS itself has no Gaussian Process regressor (nor
# does any function in it implement one); the only concrete precedent for
# "Gaussian Regression via MASS" in this codebase is Scripts/R_scripts/
# ml_regression_tasks_dataset1.R, which loads MASS and fits with base
# stats::glm(family = gaussian()) -- i.e. "Gaussian regression" read as
# regression under a Gaussian error assumption, not as a literal port of
# sklearn's kernel-based GaussianProcessRegressor. That precedent is kept
# here rather than inventing a different MASS-based approximation. This
# should be confirmed with Ch and stated in Section 5.

RANDOM_STATE <- 42
N_REPETITIONS <- 10
SLEEP_SECONDS <- 30
KERNEL_TRAIN_CAP <- 20000
KERNEL_PRED_CAP <- 20000

# [FIX-DATA] Combined with adult.test (the UCI test split, downloaded
# separately) so the full dataset matches the paper's reported 48,842 rows
# (32,561 + 16,281) rather than the 32,561-row train-only file alone. See
# FIXES.md Section 1.
DATA_PATH <- "/home/ug/RvsPython/ver/adult.data"
DATA_PATH_TEST <- "/home/ug/RvsPython/ver/adult.test"
RJOULES_OUTPUT_CSV <- "output_ml_regression_adult_rerun_r.csv"
if (file.exists(RJOULES_OUTPUT_CSV)) file.remove(RJOULES_OUTPUT_CSV)

.script_args <- commandArgs(trailingOnly = FALSE)
.script_dir <- dirname(sub("--file=", "", grep("--file=", .script_args, value = TRUE)))
if (length(.script_dir) == 0 || .script_dir == "") .script_dir <- "."
source(file.path(.script_dir, "energy_measurement.R"))
sleep <- function() Sys.sleep(SLEEP_SECONDS)

suppressMessages({
  library(glmnet)
  library(MASS)
  library(tree)
  library(e1071)
  library(neuralnet)
})

# ---------------------------------------------------------------------------
# Data preparation -- identical column selection to the Python script.
# ---------------------------------------------------------------------------

columns <- c("age", "workclass", "fnlwgt", "education", "education.num",
             "marital.status", "occupation", "relationship", "race", "sex",
             "capital.gain", "capital.loss", "hours.per.week",
             "native.country", "income")

# adult.test has a bogus first line ("|1x3 Cross validator") and encodes the
# income label with a trailing period ("<=50K." / ">50K."), unlike adult.data
# ("<=50K" / ">50K"); stripped so the two files share the same two classes
# instead of silently becoming four.
train_df <- read.csv(DATA_PATH, header = FALSE, col.names = columns,
                      strip.white = TRUE, stringsAsFactors = FALSE)
test_df <- read.csv(DATA_PATH_TEST, header = FALSE, col.names = columns,
                     strip.white = TRUE, stringsAsFactors = FALSE, skip = 1)
test_df$income <- sub("\\.$", "", test_df$income)
dataframe <- rbind(train_df, test_df)
dataframe[dataframe == "?"] <- NA

# [R-DEV] as.integer(factor(x)) - 1 reproduces sklearn's LabelEncoder: a
# 0-based integer code per unique level, sorted the way np.unique would sort
# it -- numerically for a numeric column, lexicographically for a string
# column. R's factor() alone always sorts by the string representation, which
# silently miscodes a numeric categorical column (e.g. rating_n below: 1..10
# would sort as "1","10","2",...,"9"), so numeric columns are handled
# separately. NA ("?" in the string columns) gets its own explicit trailing
# level rather than propagating, matching sklearn's LabelEncoder (which also
# fits successfully on NaN) and avoiding NA in the feature matrix, which
# several of the R model fitters below (glmnet, e1071::svm, randomForest by
# default) do not accept.
label_encode <- function(x) {
  if (is.numeric(x)) {
    lvls <- sort(unique(x[!is.na(x)]))
    codes <- match(x, lvls) - 1L
    codes[is.na(x)] <- length(lvls)
    return(codes)
  }
  x[is.na(x)] <- "__NA__"
  as.integer(factor(x)) - 1L
}
dataframe$income_n <- label_encode(dataframe$income)
dataframe$sex_n <- label_encode(dataframe$sex)
dataframe$occupation_n <- label_encode(dataframe$occupation)
dataframe$marital.status_n <- label_encode(dataframe$marital.status)
dataframe$workclass_n <- label_encode(dataframe$workclass)
dataframe$education_n <- label_encode(dataframe$education)

training_features <- c("workclass_n", "sex_n", "fnlwgt", "occupation_n",
                        "marital.status_n", "education_n", "education.num",
                        "capital.gain", "hours.per.week", "age", "capital.loss")
target <- "income_n"

split_indices <- function(n, seed) {
  set.seed(seed)
  idx <- sample.int(n)
  n_test <- floor(0.2 * n)
  test_idx <- idx[seq_len(n_test)]
  rest_idx <- idx[(n_test + 1):n]
  n_pred <- floor(0.25 * length(rest_idx))
  pred_idx <- rest_idx[seq_len(n_pred)]
  train_idx <- rest_idx[(n_pred + 1):length(rest_idx)]
  list(train = train_idx, pred = pred_idx, test = test_idx)
}

idx <- split_indices(nrow(dataframe), RANDOM_STATE)
X_train_raw <- as.matrix(dataframe[idx$train, training_features])
X_pred_raw <- as.matrix(dataframe[idx$pred, training_features])
X_test_raw <- as.matrix(dataframe[idx$test, training_features])
Y_train_raw <- dataframe[idx$train, target]
Y_pred_raw <- dataframe[idx$pred, target]
Y_test_raw <- dataframe[idx$test, target]

# All five algorithms fit on the same standardised X and Y, matching [DEV-2]
# in the Python script (every original inconsistency in scaling is avoided).
mu_x <- colMeans(X_train_raw); sd_x <- apply(X_train_raw, 2, sd); sd_x[sd_x == 0] <- 1
mu_y <- mean(Y_train_raw); sd_y <- sd(Y_train_raw); if (sd_y == 0) sd_y <- 1
standardize_x <- function(X) sweep(sweep(X, 2, mu_x, "-"), 2, sd_x, "/")
standardize_y <- function(y) (y - mu_y) / sd_y
unstandardize_y <- function(y) y * sd_y + mu_y

X1 <- standardize_x(X_train_raw)
X_pred_s <- standardize_x(X_pred_raw)
X_test_s <- standardize_x(X_test_raw)
Y1 <- standardize_y(Y_train_raw)
Y_test <- Y_test_raw

cap_rows <- function(X, cap, seed) {
  if (!is.null(cap) && nrow(X) > cap) {
    set.seed(seed)
    idx <- sample.int(nrow(X), cap)
    list(X = X[idx, , drop = FALSE], idx = idx)
  } else {
    list(X = X, idx = seq_len(nrow(X)))
  }
}

kernel_train <- cap_rows(X1, KERNEL_TRAIN_CAP, RANDOM_STATE)
X1_kernel <- kernel_train$X
Y1_kernel <- Y1[kernel_train$idx]
if (nrow(X1_kernel) < nrow(X1)) {
  cat(sprintf("[note] kernel methods trained on %d of %d rows\n", nrow(X1_kernel), nrow(X1)))
}

# [FIX-OOM] Same fix as the Python scripts: capped query side for the kernel
# models (Gaussian, SVR) only.
kernel_pred <- cap_rows(X_pred_s, KERNEL_PRED_CAP, RANDOM_STATE)
X_pred_kernel_s <- kernel_pred$X
kernel_test <- cap_rows(X_test_s, KERNEL_PRED_CAP, RANDOM_STATE)
X_test_kernel_s <- kernel_test$X
Y_test_kernel <- Y_test[kernel_test$idx]
if (nrow(X_test_kernel_s) < nrow(X_test_s)) {
  cat(sprintf("[note] kernel methods scored on %d of %d test rows\n", nrow(X_test_kernel_s), nrow(X_test_s)))
}

report <- function(name, predicted_scaled, kernel = FALSE) {
  actual <- if (kernel) Y_test_kernel else Y_test
  predicted <- unstandardize_y(predicted_scaled)
  cat("The below details are for", name, "..\n")
  resid <- actual - predicted
  mse <- mean(resid^2)
  mae <- mean(abs(resid))
  ss_res <- sum(resid^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r2 <- 1 - ss_res / ss_tot
  cat("r2 value= ", r2, "\n")
  cat("MAE value= ", mae, "\n")
  cat("MSE value= ", mse, "\n")
  cat("RMSE value= ", sqrt(mse), "\n")
}

# ---------------------------------------------------------------------------
# Linear Regression (glmnet)
# ---------------------------------------------------------------------------

linear_regression <- function() {
  glmnet(X1, Y1, family = "gaussian")
}

test_linear_regression <- function() {
  invisible(linear_regression())
}

linear_regression_model <- linear_regression()

test_linear_regression_inference <- function() {
  predict(linear_regression_model, newx = X_pred_s, s = min(linear_regression_model$lambda))
}

# ---------------------------------------------------------------------------
# Gaussian Regression (MASS precedent: glm(family = gaussian()))
# ---------------------------------------------------------------------------

gaussian_regression <- function() {
  train_df <- as.data.frame(X1)
  train_df$y <- as.numeric(Y1)
  glm(y ~ ., data = train_df, family = gaussian())
}

test_gaussian_regression <- function() {
  invisible(gaussian_regression())
}

gaussian_regression_model <- gaussian_regression()

test_gaussian_regression_inference <- function() {
  predict(gaussian_regression_model, newdata = as.data.frame(X_pred_s))
}

# ---------------------------------------------------------------------------
# Decision Tree Regression (tree)
# ---------------------------------------------------------------------------

decision_tree_regression <- function() {
  train_df <- as.data.frame(X1)
  train_df$y <- as.numeric(Y1)
  tree(y ~ ., data = train_df)
}

test_decision_tree_regression <- function() {
  invisible(decision_tree_regression())
}

decision_tree_regression_model <- decision_tree_regression()

test_decision_tree_regression_inference <- function() {
  predict(decision_tree_regression_model, newdata = as.data.frame(X_pred_s))
}

# ---------------------------------------------------------------------------
# Support Vector Regression (e1071)
# ---------------------------------------------------------------------------

support_vector_regression <- function() {
  svm(x = X1_kernel, y = as.numeric(Y1_kernel), type = "eps-regression")
}

test_support_vector_regression <- function() {
  invisible(support_vector_regression())
}

svr_regression_model <- support_vector_regression()

test_support_vector_regression_inference <- function() {
  # [FIX-OOM] Capped query set; see KERNEL_PRED_CAP above.
  predict(svr_regression_model, newdata = X_pred_kernel_s)
}

# ---------------------------------------------------------------------------
# Neural Network Regression (neuralnet)
# ---------------------------------------------------------------------------
# [R-FIX-5] neuralnet()'s own default threshold (0.01, the max allowed
# partial-derivative sum before it calls the fit converged) very often is not
# reached within stepmax at this scale, and unlike sklearn's MLPRegressor --
# which still predicts fine from whatever weights exist after max_iter -- a
# non-converged neuralnet::neuralnet fit has NO usable weights at all
# (`nn$weights` is completely absent, not partial), so predict() throws
# immediately with "requires numeric/complex matrix/vector arguments".
# Reproduced directly: this crashed the real Dataset2 regression run.
# The threshold this needs to reach convergence scales with the training set
# size (it is an unnormalised sum over all training rows' partial
# derivatives, not averaged), so no single fixed threshold works across
# Dataset1/2/3's very different row counts. Fixed with a bounded, increasing
# threshold ladder and a modest per-attempt stepmax: cheap early attempts are
# tried first, each bounded, and the first one that produces usable weights
# is kept. Confirmed empirically at Dataset2 scale (~94,000 training rows):
# threshold 0.01 and 1 do not converge within stepmax = 3000, threshold 50
# does, in a few seconds.
fit_neuralnet <- function(formula, data,
                           thresholds = c(0.01, 1, 10, 50, 100, 500, 1000, 5000, 10000),
                           stepmax = 3000, ...) {
  for (th in thresholds) {
    nn <- suppressWarnings(neuralnet(formula, data = data, threshold = th, stepmax = stepmax, ...))
    if (!is.null(nn$weights)) return(nn)
  }
  stop("neuralnet did not produce usable weights within the threshold schedule")
}

neural_network_regression <- function() {
  train_df <- as.data.frame(X1)
  train_df$y <- as.numeric(Y1)
  fit_neuralnet(y ~ ., data = train_df, linear.output = TRUE)
}

test_neural_network_regression <- function() {
  invisible(neural_network_regression())
}

neural_network_model <- neural_network_regression()

test_neural_network_regression_inference <- function() {
  predict(neural_network_model, newdata = as.data.frame(X_pred_s))
}

# ---------------------------------------------------------------------------
# Accuracy reporting, then the measurement loop
# ---------------------------------------------------------------------------

report("Linear Regression",
       as.numeric(predict(linear_regression_model, newx = X_test_s, s = min(linear_regression_model$lambda))))
report("Gaussian Regression",
       as.numeric(predict(gaussian_regression_model, newdata = as.data.frame(X_test_kernel_s))), kernel = TRUE)
report("Decision tree Regression",
       as.numeric(predict(decision_tree_regression_model, newdata = as.data.frame(X_test_s))))
report("Support vector Regression",
       as.numeric(predict(svr_regression_model, newdata = X_test_kernel_s)), kernel = TRUE)
report("Neural Network Regression",
       as.numeric(predict(neural_network_model, newdata = as.data.frame(X_test_s))))

function_list <- list(
  test_linear_regression = test_linear_regression,
  test_gaussian_regression = test_gaussian_regression,
  test_decision_tree_regression = test_decision_tree_regression,
  test_support_vector_regression = test_support_vector_regression,
  test_neural_network_regression = test_neural_network_regression,
  test_linear_regression_inference = test_linear_regression_inference,
  test_gaussian_regression_inference = test_gaussian_regression_inference,
  test_decision_tree_regression_inference = test_decision_tree_regression_inference,
  test_support_vector_regression_inference = test_support_vector_regression_inference,
  test_neural_network_regression_inference = test_neural_network_regression_inference
)

for (i in 1:N_REPETITIONS) {
  cat("This is iteration no:", i - 1, "\n")
  order <- sample(names(function_list))
  for (nm in order) {
    sleep()
    measure_energy(function_list[[nm]], name = nm)()
  }
}

cat("Process complete\n")
