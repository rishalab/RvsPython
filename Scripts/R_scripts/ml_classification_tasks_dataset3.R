#!/usr/bin/env Rscript
#
# ml_classification_tasks_dataset3.R
#
# Dataset3 (NYC Taxi Trip Duration) classification energy measurement, R side.
# Mirrors ml_classification_tasks_dataset3.py: same data path, same feature
# selection, same [DECISION-B] target-leakage exclusion of dropoff_datetime,
# same [DECISION-B]-adjacent classification target (median split on
# trip_duration, since the same StandardScaler+astype(int) trick as Dataset1
# and Dataset2 would place almost every row in one class here), same kernel
# caps, same five algorithms, same ten measured functions, same
# repetition/shuffle protocol.
#
# Model -> CRAN package mapping (fixed by the paper, do not change):
#   Logistic Regression  -> glmnet
#   Gaussian Naive Bayes  -> naivebayes
#   Decision Tree         -> rpart
#   Support Vector Machine -> e1071
#   Random Forest          -> randomForest

RANDOM_STATE <- 42
N_REPETITIONS <- 10
SLEEP_SECONDS <- 30

# [DECISION-A] Same cap as all five other scripts; must not be changed here
# in isolation. See ml_classification_tasks_dataset3.py.
KERNEL_TRAIN_CAP <- 20000
KERNEL_PRED_CAP <- 20000

DATA_PATH <- "/home/ug/RvsPython/ver/RvsPython/D3.csv"
RJOULES_OUTPUT_CSV <- "output_ml_classification_taxi_r.csv"
if (file.exists(RJOULES_OUTPUT_CSV)) file.remove(RJOULES_OUTPUT_CSV)

.script_args <- commandArgs(trailingOnly = FALSE)
.script_dir <- dirname(sub("--file=", "", grep("--file=", .script_args, value = TRUE)))
if (length(.script_dir) == 0 || .script_dir == "") .script_dir <- "."
source(file.path(.script_dir, "energy_measurement.R"))
sleep <- function() Sys.sleep(SLEEP_SECONDS)

suppressMessages({
  library(glmnet)
  library(naivebayes)
  library(rpart)
  library(e1071)
  library(randomForest)
})

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

dataframe <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
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
dataframe$pickup_datetime_n <- label_encode(dataframe$pickup_datetime)
dataframe$store_and_fwd_flag_n <- label_encode(dataframe$store_and_fwd_flag)

training_features <- c("vendor_id", "pickup_datetime_n", "store_and_fwd_flag_n",
                        "passenger_count", "pickup_longitude", "pickup_latitude",
                        "dropoff_longitude", "dropoff_latitude")

dataframe <- dataframe[stats::complete.cases(dataframe[, c(training_features, "trip_duration")]), ]
median_duration <- median(dataframe$trip_duration)
dataframe$long_trip <- as.integer(dataframe$trip_duration >= median_duration)
cat(sprintf("[note] median trip_duration = %s s; class balance = %.3f\n",
            median_duration, mean(dataframe$long_trip)))
target <- "long_trip"

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
Y_train <- dataframe[idx$train, target]
Y_test <- dataframe[idx$test, target]

mu <- colMeans(X_train_raw)
sdv <- apply(X_train_raw, 2, sd)
sdv[sdv == 0] <- 1
standardize <- function(X) sweep(sweep(X, 2, mu, "-"), 2, sdv, "/")
X1 <- standardize(X_train_raw)
X_pred_s <- standardize(X_pred_raw)
X_test_s <- standardize(X_test_raw)
Y1 <- Y_train

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
  cat(sprintf("[note] SVM trained on %d of %d rows\n", nrow(X1_kernel), nrow(X1)))
}

kernel_pred <- cap_rows(X_pred_s, KERNEL_PRED_CAP, RANDOM_STATE)
X_pred_kernel_s <- kernel_pred$X
kernel_test <- cap_rows(X_test_s, KERNEL_PRED_CAP, RANDOM_STATE)
X_test_kernel_s <- kernel_test$X
Y_test_kernel <- Y_test[kernel_test$idx]
if (nrow(X_test_kernel_s) < nrow(X_test_s)) {
  cat(sprintf("[note] SVM scored on %d of %d test rows\n", nrow(X_test_kernel_s), nrow(X_test_s)))
}

weighted_prf <- function(actual, predicted) {
  levels_all <- sort(unique(c(actual, predicted)))
  n <- length(actual)
  precision <- 0; recall <- 0; f1 <- 0
  for (cls in levels_all) {
    tp <- sum(predicted == cls & actual == cls)
    fp <- sum(predicted == cls & actual != cls)
    fn <- sum(predicted != cls & actual == cls)
    w <- sum(actual == cls) / n
    p_c <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    r_c <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    f_c <- if ((p_c + r_c) > 0) 2 * p_c * r_c / (p_c + r_c) else 0
    precision <- precision + w * p_c
    recall <- recall + w * r_c
    f1 <- f1 + w * f_c
  }
  list(precision = precision, recall = recall, f1 = f1)
}

report <- function(name, predicted, kernel = FALSE) {
  actual <- if (kernel) Y_test_kernel else Y_test
  cat("The below details are for", name, "..\n")
  acc <- mean(actual == predicted)
  prf <- weighted_prf(actual, predicted)
  cat("Accuracy= ", acc, "\n")
  cat("Recall= ", prf$recall, "\n")
  cat("Precision= ", prf$precision, "\n")
  cat("f1 score= ", prf$f1, "\n")
}

# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

random_forest_classification <- function() {
  randomForest(x = X1, y = factor(Y1))
}

test_random_forest_classification <- function() {
  invisible(random_forest_classification())
}

random_forest_model <- random_forest_classification()

test_random_forest_classification_inference <- function() {
  predict(random_forest_model, newdata = X_pred_s)
}

# ---------------------------------------------------------------------------
# Logistic Regression (glmnet)
# ---------------------------------------------------------------------------

logistic_regression_classification <- function() {
  glmnet(X1, factor(Y1), family = "binomial")
}

test_logistic_regression_classification <- function() {
  invisible(logistic_regression_classification())
}

logistic_regression_model <- logistic_regression_classification()

test_logistic_regression_classification_inference <- function() {
  as.integer(predict(logistic_regression_model, newx = X_pred_s,
                      s = min(logistic_regression_model$lambda), type = "class"))
}

# ---------------------------------------------------------------------------
# Gaussian Naive Bayes (naivebayes)
# ---------------------------------------------------------------------------

gaussian_NB_classification <- function() {
  gaussian_naive_bayes(x = X1, y = factor(Y1))
}

test_gaussian_NB_classification <- function() {
  invisible(gaussian_NB_classification())
}

naive_bayes_model <- gaussian_NB_classification()

test_gaussian_NB_classification_inference <- function() {
  predict(naive_bayes_model, newdata = X_pred_s)
}

# ---------------------------------------------------------------------------
# Support Vector Machine (e1071)
# ---------------------------------------------------------------------------

SVM_classification <- function() {
  svm(x = X1_kernel, y = factor(Y1_kernel), type = "C-classification")
}

test_SVM_classification <- function() {
  invisible(SVM_classification())
}

svm_classifier_model <- SVM_classification()

test_SVM_classification_inference <- function() {
  # [FIX-OOM] Capped query set; see KERNEL_PRED_CAP above.
  predict(svm_classifier_model, newdata = X_pred_kernel_s)
}

# ---------------------------------------------------------------------------
# Decision Tree (rpart)
# ---------------------------------------------------------------------------

decision_tree_classification <- function() {
  train_df <- as.data.frame(X1)
  train_df$y <- factor(Y1)
  rpart(y ~ ., data = train_df, method = "class")
}

test_decision_tree_classification <- function() {
  invisible(decision_tree_classification())
}

decision_tree_classifier_model <- decision_tree_classification()

test_decision_tree_classification_inference <- function() {
  pred_df <- as.data.frame(X_pred_s)
  predict(decision_tree_classifier_model, newdata = pred_df, type = "class")
}

# ---------------------------------------------------------------------------
# Accuracy reporting, then the measurement loop
# ---------------------------------------------------------------------------

report("Random forest classification",
       predict(random_forest_model, newdata = as.data.frame(X_test_s)))
report("Logistic Regression classification",
       as.integer(predict(logistic_regression_model, newx = X_test_s,
                           s = min(logistic_regression_model$lambda), type = "class")))
report("Naive Bayes classification",
       predict(naive_bayes_model, newdata = X_test_s))
report("SVM classification",
       predict(svm_classifier_model, newdata = X_test_kernel_s), kernel = TRUE)
report("Decision tree classification",
       predict(decision_tree_classifier_model, newdata = as.data.frame(X_test_s), type = "class"))

function_list <- list(
  test_random_forest_classification = test_random_forest_classification,
  test_logistic_regression_classification = test_logistic_regression_classification,
  test_gaussian_NB_classification = test_gaussian_NB_classification,
  test_SVM_classification = test_SVM_classification,
  test_decision_tree_classification = test_decision_tree_classification,
  test_random_forest_classification_inference = test_random_forest_classification_inference,
  test_logistic_regression_classification_inference = test_logistic_regression_classification_inference,
  test_gaussian_NB_classification_inference = test_gaussian_NB_classification_inference,
  test_SVM_classification_inference = test_SVM_classification_inference,
  test_decision_tree_classification_inference = test_decision_tree_classification_inference
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
