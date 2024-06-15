# Building ensembles of classification models

In this section we will be building two ensembles of classification models. We will use six classification models, and six ensembles, for a total of 12 results.

### Let's start at the end and work backwards

We know we want to finish with predictions from an ensemble of classification models. Therefore we need an ensemble of classification models. Therefore we need classification models. Let's choose five classification models for our individual models, and use same five for our ensemble. Note you may use any modeling method you wish for the ensemble, since it's data.

Our final result will look something like this:

> Predictions on the holdout data from classification model 1
>
> Predictions on the holdout data from classification model 2
>
> Predictions on the holdout data from classification model 3
>
> Predictions on the holdout data from classification model 4
>
> Predictions on the holdout data from classification model 5
>
> Use those predictions to make an ensemble
>
> Use the ensemble models to make predictions on the ensemble holdout data
>
> Report the results

Let's come up with a list of five classification models we will use:

Bagged Random Forest

C50

Ranger

Support Vector Machines

XGBoost

Note that there is nothing special about using five models. Any number of models may be used, but this set will be a very good start.

For this solution we will also add a mean duration by model to the finished report.

Since we have our basic outline, and we know where we want to end, we are ready to begin.


``` r

# Load libraries
library(randomForest)
#> randomForest 4.7-1.1
#> Type rfNews() to see new features/changes/bug fixes.
```

``` r
library(tidyverse)
#> ── Attaching core tidyverse packages ──── tidyverse 2.0.0 ──
#> ✔ dplyr     1.1.4     ✔ readr     2.1.5
#> ✔ forcats   1.0.0     ✔ stringr   1.5.1
#> ✔ ggplot2   3.5.1     ✔ tibble    3.2.1
#> ✔ lubridate 1.9.3     ✔ tidyr     1.3.1
#> ✔ purrr     1.0.2
#> ── Conflicts ────────────────────── tidyverse_conflicts() ──
#> ✖ dplyr::combine()  masks randomForest::combine()
#> ✖ dplyr::filter()   masks stats::filter()
#> ✖ dplyr::lag()      masks stats::lag()
#> ✖ ggplot2::margin() masks randomForest::margin()
#> ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
```

``` r
library(tree)

# Set initial values to 0
bag_rf_train_accuracy <- 0
bag_rf_test_accuracy <- 0
bag_rf_validation_accuracy <- 0
bag_rf_overfitting <- 0
bag_rf_holdout <- 0
bag_rf_table_total <- 0
bag_rf_duration <- 0

C50_train_accuracy <- 0
C50_test_accuracy <- 0
C50_validation_accuracy <- 0
C50_overfitting <- 0
C50_holdout <- 0
C50_table_total <- 0
C50_duration <- 0

ranger_train_accuracy <- 0
ranger_test_accuracy <- 0
ranger_accuracy <- 0
ranger_test_accuracy_mean <- 0
ranger_holdout <- 0
ranger_table_total <- 0
ranger_duration <- 0

svm_train_accuracy <- 0
svm_test_accuracy <- 0
svm_accuracy <- 0
svm_test_accuracy_mean <- 0
svm_holdout <- 0
svm_table_total <- 0
svm_duration <- 0

ensemble_bag_rf_train_accuracy <- 0
ensemble_bag_rf_test_accuracy <- 0
ensemble_bag_rf_validation_accuracy <- 0
ensemble_bag_rf_overfitting <- 0
ensemble_bag_rf_holdout <- 0
ensemble_bag_rf_table_total <- 0
ensemble_bag_rf_duration <- 0

ensemble_C50_train_accuracy <- 0
ensemble_C50_test_accuracy <- 0
ensemble_C50_validation_accuracy <- 0
ensemble_C50_overfitting <- 0
ensemble_C50_holdout <- 0
ensemble_C50_table_total <- 0
ensemble_C50_duration <- 0

ensemble_ranger_train_accuracy <- 0
ensemble_ranger_test_accuracy <- 0
ensemble_ranger_validation_accuracy <- 0
ensemble_ranger_overfitting <- 0
ensemble_ranger_holdout <- 0
ensemble_ranger_table_total <- 0
ensemble_ranger_duration <- 0

ensemble_rf_train_accuracy <- 0
ensemble_rf_test_accuracy <- 0
ensemble_rf_validation_accuracy <- 0
ensemble_rf_overfitting <- 0
ensemble_rf_holdout <- 0
ensemble_rf_table_total <- 0
ensemble_rf_duration <- 0

ensemble_svm_train_accuracy <- 0
ensemble_svm_test_accuracy <- 0
ensemble_svm_validation_accuracy <- 0
ensemble_svm_overfitting <- 0
ensemble_svm_holdout <- 0
ensemble_svm_table_total <- 0
ensemble_svm_duration <- 0


# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
classification_1 <- function(data, colnum, numresamples, train_amount, test_amount){

# Change target column name to y
y <- 0
colnames(data)[colnum] <- "y"
# 
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
df <- df[sample(nrow(df)), ]

# Set up resamples:
for (i in 1:numresamples) {
 
# Randomize the rows  
df <- df[sample(nrow(df)), ]

# Split the data into train and test sets
index <- sample(c(1:2), nrow(df), replace = TRUE, prob = c(train_amount, test_amount))

train <- df[index == 1, ]
test <- df[index == 2, ]

train01 <- train
test01 <- test

y_train <- train$y
y_test <- test$y

train <- df[index == 1, ] %>% dplyr::select(-y)
test <- df[index == 2, ] %>% dplyr::select(-y)

# Fit the model on the training data
bag_rf_start <- Sys.time()
bag_rf_train_fit <- randomForest::randomForest(y ~ ., data = train01, mtry = ncol(train))
bag_rf_train_pred <- predict(bag_rf_train_fit, train, type = "class")
bag_rf_train_table <- table(bag_rf_train_pred, y_train)
bag_rf_train_accuracy[i] <- sum(diag(bag_rf_train_table)) / sum(bag_rf_train_table)
bag_rf_train_accuracy_mean <- mean(bag_rf_train_accuracy)

# Check accuracy and make predictions from the model, applied to the test data
bag_rf_test_pred <- predict(bag_rf_train_fit, test, type = "class")
bag_rf_test_table <- table(bag_rf_test_pred, y_test)
bag_rf_test_accuracy[i] <- sum(diag(bag_rf_test_table)) / sum(bag_rf_test_table)
bag_rf_test_accuracy_mean <- mean(bag_rf_test_accuracy)

# Calculate model accuracy
bag_rf_holdout[i] <- mean(c(bag_rf_test_accuracy_mean))
bag_rf_holdout_mean <- mean(bag_rf_holdout)

# Calculate table
bag_rf_table <- bag_rf_test_table
bag_rf_table_total <- bag_rf_table_total + bag_rf_table

bag_rf_end <- Sys.time()
bag_rf_duration[i] <- bag_rf_end - bag_rf_start
bag_rf_duration_mean <- mean(bag_rf_duration)

# C50 model
C50_start <- Sys.time()
# Fit the model on the training data
C50_train_fit <- C50::C5.0(as.factor(y_train) ~ ., data = train)
C50_train_pred <- predict(C50_train_fit, train)
C50_train_table <- table(C50_train_pred, y_train)
C50_train_accuracy[i] <- sum(diag(C50_train_table)) / sum(C50_train_table)
C50_train_accuracy_mean <- mean(C50_train_accuracy)
C50_train_mean <- mean(diag(C50_train_table)) / mean(C50_train_table)

# Check accuracy and make predictions from the model, applied to the test data
C50_test_pred <- predict(C50_train_fit, test)
C50_test_table <- table(C50_test_pred, y_test)
C50_test_accuracy[i] <- sum(diag(C50_test_table)) / sum(C50_test_table)
C50_test_accuracy_mean <- mean(C50_test_accuracy)
C50_test_mean <- mean(diag(C50_test_table)) / mean(C50_test_table)

# Calculate accuracy
C50_holdout[i] <- mean(c(C50_test_accuracy_mean))
C50_holdout_mean <- mean(C50_holdout)

C50_end <- Sys.time()
C50_duration[i] <- C50_end - C50_start
C50_duration_mean <- mean(C50_duration)

# Ranger model

ranger_start <- Sys.time()

# Fit the model on the training data
ranger_train_fit <- MachineShop::fit(y ~ ., data = train01, model = "RangerModel")
ranger_train_predict <- predict(object = ranger_train_fit, newdata = train01)
ranger_train_table <- table(ranger_train_predict, y_train)
ranger_train_accuracy[i] <- sum(diag(ranger_train_table)) / sum(ranger_train_table)
ranger_train_accuracy_mean <- mean(ranger_train_accuracy)

# Check accuracy and make predictions from the model, applied to the test data
ranger_test_predict <- predict(object = ranger_train_fit, newdata = test01)
ranger_test_table <- table(ranger_test_predict, y_test)
ranger_test_accuracy[i] <- sum(diag(ranger_test_table)) / sum(ranger_test_table)
ranger_test_accuracy_mean <- mean(ranger_test_accuracy)
ranger_test_pred <- ranger_test_predict

# Calculate overall model accuracy
ranger_holdout[i] <- mean(c(ranger_test_accuracy_mean))
ranger_holdout_mean <- mean(ranger_holdout)

ranger_end <- Sys.time()
ranger_duration[i] <- ranger_end - ranger_start
ranger_duration_mean <- mean(ranger_duration)

# Support vector machines
svm_start <- Sys.time()

svm_train_fit <- e1071::svm(y_train ~ ., data = train, kernel = "radial", gamma = 1, cost = 1)
svm_train_pred <- predict(svm_train_fit, train, type = "class")
svm_train_table <- table(svm_train_pred, y_train)
svm_train_accuracy[i] <- sum(diag(svm_train_table)) / sum(svm_train_table)
svm_train_accuracy_mean <- mean(svm_train_accuracy)

# Check accuracy and make predictions from the model, applied to the test data
svm_test_pred <- predict(svm_train_fit, test, type = "class")
svm_test_table <- table(svm_test_pred, y_test)
svm_test_accuracy[i] <- sum(diag(svm_test_table)) / sum(svm_test_table)
svm_test_accuracy_mean <- mean(svm_test_accuracy)

# Calculate overall model accuracy
svm_holdout[i] <- mean(c(svm_test_accuracy_mean))
svm_holdout_mean <- mean(svm_holdout)

svm_end <- Sys.time()
svm_duration[i] <- svm_end - svm_start
svm_duration_mean <- mean(svm_duration)

# XGBoost
xgb_start <- Sys.time()

xgb_train_accuracy <- 0
xgb_test_accuracy <- 0
xgb_accuracy <- 0
xgb_test_accuracy_mean <- 0
xgb_holdout <- 0
xgb_table_total <- 0
xgb_duration <- 0

y_train <- as.integer(train01$y) - 1
y_test <- as.integer(test01$y) - 1
X_train <- train %>% dplyr::select(dplyr::where(is.numeric))
X_test <- test %>% dplyr::select(dplyr::where(is.numeric))

xgb_train <- xgboost::xgb.DMatrix(data = as.matrix(X_train), label = y_train)
xgb_test <- xgboost::xgb.DMatrix(data = as.matrix(X_test), label = y_test)
xgb_params <- list(
    booster = "gbtree",
    eta = 0.01,
    max_depth = 8,
    gamma = 4,
    subsample = 0.75,
    colsample_bytree = 1,
    objective = "multi:softprob",
    eval_metric = "mlogloss",
    num_class = length(levels(df$y))
  )
xgb_model <- xgboost::xgb.train(
    params = xgb_params,
    data = xgb_train,
    nrounds = 5000,
    verbose = 1
  )

xgb_preds <- predict(xgb_model, as.matrix(X_train), reshape = TRUE)
xgb_preds <- as.data.frame(xgb_preds)
colnames(xgb_preds) <- levels(df$y)

xgb_preds$PredictedClass <- apply(xgb_preds, 1, function(y) colnames(xgb_preds)[which.max(y)])
xgb_preds$ActualClass <- levels(df$y)[y_train + 1]
xgb_train_table <- table(xgb_preds$PredictedClass, xgb_preds$ActualClass)
xgb_train_accuracy[i] <- sum(diag(xgb_train_table)) / sum(xgb_train_table)
xgb_train_accuracy_mean <- mean(xgb_train_accuracy)

# Check accuracy and make predictions from the model, applied to the test data
y_train <- as.integer(train01$y) - 1
y_test <- as.integer(test01$y) - 1
X_train <- train %>% dplyr::select(dplyr::where(is.numeric))
X_test <- test %>% dplyr::select(dplyr::where(is.numeric))

xgb_train <- xgboost::xgb.DMatrix(data = as.matrix(X_train), label = y_train)
xgb_test <- xgboost::xgb.DMatrix(data = as.matrix(X_test), label = y_test)
xgb_params <- list(
    booster = "gbtree",
    eta = 0.01,
    max_depth = 8,
    gamma = 4,
    subsample = 0.75,
    colsample_bytree = 1,
    objective = "multi:softprob",
    eval_metric = "mlogloss",
    num_class = length(levels(df$y))
  )
xgb_model <- xgboost::xgb.train(
    params = xgb_params,
    data = xgb_train,
    nrounds = 5000,
    verbose = 1
  )

xgb_preds <- predict(xgb_model, as.matrix(X_test), reshape = TRUE)
xgb_preds <- as.data.frame(xgb_preds)
colnames(xgb_preds) <- levels(df$y)

xgb_preds$PredictedClass <- apply(xgb_preds, 1, function(y) colnames(xgb_preds)[which.max(y)])
xgb_preds$ActualClass <- levels(df$y)[y_test + 1]
xgb_test_table <- table(xgb_preds$PredictedClass, xgb_preds$ActualClass)
xgb_test_accuracy[i] <- sum(diag(xgb_test_table)) / sum(xgb_test_table)
xgb_test_accuracy_mean <- mean(xgb_test_accuracy)

# Calculate overall model accuracy
xgb_holdout[i] <- mean(c(xgb_test_accuracy_mean))
xgb_holdout_mean <- mean(xgb_holdout)

xgb_end <- Sys.time()
xgb_duration[i] <- xgb_end - xgb_start
xgb_duration_mean <- mean(xgb_duration)

# Build the ensemble of predictions

ensemble1 <- data.frame(
  'Bag_rf' = bag_rf_test_pred,
  'C50' = C50_test_pred,
  'Ranger' = ranger_test_predict,
  'SVM' = svm_test_pred,
  'XGBoost' = xgb_preds
)

ensemble_row_numbers <- as.numeric(row.names(ensemble1))
ensemble1$y <- df[ensemble_row_numbers, "y"]

ensemble_index <- sample(c(1:2), nrow(ensemble1), replace = TRUE, prob = c(train_amount, test_amount))
ensemble_train <- ensemble1[ensemble_index == 1, ]
ensemble_test <- ensemble1[ensemble_index == 2, ]
ensemble_y_train <- ensemble_train$y
ensemble_y_test <- ensemble_test$y


# Ensemble bagged random forest
ensemble_bag_rf_start <- Sys.time()

ensemble_bag_train_rf <- randomForest::randomForest(ensemble_y_train ~ ., data = ensemble_train, mtry = ncol(ensemble_train) - 1)
ensemble_bag_rf_train_pred <- predict(ensemble_bag_train_rf, ensemble_train, type = "class")
ensemble_bag_rf_train_table <- table(ensemble_bag_rf_train_pred, ensemble_train$y)
ensemble_bag_rf_train_accuracy[i] <- sum(diag(ensemble_bag_rf_train_table)) / sum(ensemble_bag_rf_train_table)
ensemble_bag_rf_train_accuracy_mean <- mean(ensemble_bag_rf_train_accuracy)

ensemble_bag_rf_test_pred <- predict(ensemble_bag_train_rf, ensemble_test, type = "class")
ensemble_bag_rf_test_table <- table(ensemble_bag_rf_test_pred, ensemble_test$y)
ensemble_bag_rf_test_accuracy[i] <- sum(diag(ensemble_bag_rf_test_table)) / sum(ensemble_bag_rf_test_table)
ensemble_bag_rf_test_accuracy_mean <- mean(ensemble_bag_rf_test_accuracy)

ensemble_bag_rf_holdout[i] <- mean(c(ensemble_bag_rf_test_accuracy_mean))
ensemble_bag_rf_holdout_mean <- mean(ensemble_bag_rf_holdout)

ensemble_bag_rf_end <- Sys.time()
ensemble_bag_rf_duration[i] <- ensemble_bag_rf_end - ensemble_bag_rf_start
ensemble_bag_rf_duration_mean <- mean(ensemble_bag_rf_duration)

# Ensemble C50

ensemble_C50_start <- Sys.time()

ensemble_C50_train_fit <- C50::C5.0(ensemble_y_train ~ ., data = ensemble_train)
ensemble_C50_train_pred <- predict(ensemble_C50_train_fit, ensemble_train)
ensemble_C50_train_table <- table(ensemble_C50_train_pred, ensemble_y_train)
ensemble_C50_train_accuracy[i] <- sum(diag(ensemble_C50_train_table)) / sum(ensemble_C50_train_table)
ensemble_C50_train_accuracy_mean <- mean(ensemble_C50_train_accuracy)

ensemble_C50_test_pred <- predict(ensemble_C50_train_fit, ensemble_test)
ensemble_C50_test_table <- table(ensemble_C50_test_pred, ensemble_y_test)
ensemble_C50_test_accuracy[i] <- sum(diag(ensemble_C50_test_table)) / sum(ensemble_C50_test_table)
ensemble_C50_test_accuracy_mean <- mean(ensemble_C50_test_accuracy)

ensemble_C50_holdout[i] <- mean(c(ensemble_C50_test_accuracy_mean))
ensemble_C50_holdout_mean <- mean(ensemble_C50_holdout)

ensemble_C50_end <- Sys.time()
ensemble_C50_duration[i] <- ensemble_C50_end - ensemble_C50_start
ensemble_C50_duration_mean <- mean(ensemble_C50_duration)

# Ensemble using Ranger

ensemble_ranger_start <- Sys.time()

ensemble_ranger_train_fit <- MachineShop::fit(y ~ ., data = ensemble_train, model = "RangerModel")
ensemble_ranger_train_pred <- predict(ensemble_ranger_train_fit, newdata = ensemble_train)
ensemble_ranger_train_table <- table(ensemble_ranger_train_pred, ensemble_y_train)
ensemble_ranger_train_accuracy[i] <- sum(diag(ensemble_ranger_train_table)) / sum(ensemble_ranger_train_table)
ensemble_ranger_train_accuracy_mean <- mean(ensemble_ranger_train_accuracy)

ensemble_ranger_test_fit <- MachineShop::fit(y ~ ., data = ensemble_train, model = "RangerModel")
ensemble_ranger_test_pred <- predict(ensemble_ranger_test_fit, newdata = ensemble_test)
ensemble_ranger_test_table <- table(ensemble_ranger_test_pred, ensemble_y_test)
ensemble_ranger_test_accuracy[i] <- sum(diag(ensemble_ranger_test_table)) / sum(ensemble_ranger_test_table)
ensemble_ranger_test_accuracy_mean <- mean(ensemble_ranger_test_accuracy)

ensemble_ranger_holdout[i] <- mean(c(ensemble_ranger_test_accuracy_mean))
ensemble_ranger_holdout_mean <- mean(ensemble_ranger_holdout)

ensemble_ranger_end <- Sys.time()
ensemble_ranger_duration[i] <- ensemble_ranger_end - ensemble_ranger_start
ensemble_ranger_duration_mean <- mean(ensemble_ranger_duration)

# Ensemble Random Forest

ensemble_rf_start <- Sys.time()

ensemble_train_rf_fit <- randomForest::randomForest(x = ensemble_train, y = ensemble_y_train)
ensemble_rf_train_pred <- predict(ensemble_train_rf_fit, ensemble_train, type = "class")
ensemble_rf_train_table <- table(ensemble_rf_train_pred, ensemble_y_train)
ensemble_rf_train_accuracy[i] <- sum(diag(ensemble_rf_train_table)) / sum(ensemble_rf_train_table)
ensemble_rf_train_accuracy_mean <- mean(ensemble_rf_train_accuracy)

ensemble_rf_test_pred <- predict(ensemble_train_rf_fit, ensemble_test, type = "class")
ensemble_rf_test_table <- table(ensemble_rf_test_pred, ensemble_y_test)
ensemble_rf_test_accuracy[i] <- sum(diag(ensemble_rf_test_table)) / sum(ensemble_rf_test_table)
ensemble_rf_test_accuracy_mean <- mean(ensemble_rf_test_accuracy)

ensemble_rf_holdout[i] <- mean(c(ensemble_rf_test_accuracy_mean))
ensemble_rf_holdout_mean <- mean(ensemble_rf_holdout)

ensemble_rf_end <- Sys.time()
ensemble_rf_duration[i] <- ensemble_rf_end -ensemble_rf_start
ensemble_rf_duration_mean <- mean(ensemble_rf_duration)


# Ensemble Support Vector Machines

ensemble_svm_start <- Sys.time()

ensemble_svm_train_fit <- e1071::svm(ensemble_y_train ~ ., data = ensemble_train, kernel = "radial", gamma = 1, cost = 1)
ensemble_svm_train_pred <- predict(ensemble_svm_train_fit, ensemble_train, type = "class")
ensemble_svm_train_table <- table(ensemble_svm_train_pred, ensemble_y_train)
ensemble_svm_train_accuracy[i] <- sum(diag(ensemble_svm_train_table)) / sum(ensemble_svm_train_table)
ensemble_svm_train_accuracy_mean <- mean(ensemble_svm_train_accuracy)

ensemble_svm_test_fit <- e1071::svm(ensemble_y_train ~ ., data = ensemble_train, kernel = "radial", gamma = 1, cost = 1)
ensemble_svm_test_pred <- predict(ensemble_svm_test_fit, ensemble_test, type = "class")
ensemble_svm_test_table <- table(ensemble_svm_test_pred, ensemble_y_test)
ensemble_svm_test_accuracy[i] <- sum(diag(ensemble_svm_test_table)) / sum(ensemble_svm_test_table)
ensemble_svm_test_accuracy_mean <- mean(ensemble_svm_test_accuracy)

ensemble_svm_holdout[i] <- mean(c(ensemble_svm_test_accuracy_mean))
ensemble_svm_holdout_mean <- mean(ensemble_svm_holdout)

ensemble_svm_end <- Sys.time()
ensemble_svm_duration[i] <-  ensemble_svm_end - ensemble_svm_start
ensemble_svm_duration_mean <- mean(ensemble_svm_duration)

# Return accuracy results

results <- data.frame(
  'Model' = c('Bagged_Random_Forest', 'C50', 'Ranger', 'Support_Vector_Machines', 'XGBoost', 'Ensemble_Bag_RF', 'Ensemble_C50', 'Ensemble_Ranger', 'Ensemble_RF', 'Ensemble_SVM'),
  'Accuracy' = c(bag_rf_holdout_mean, C50_holdout_mean, ranger_holdout_mean, svm_holdout_mean, xgb_holdout_mean, ensemble_bag_rf_holdout_mean, ensemble_C50_holdout_mean, ensemble_ranger_holdout_mean, ensemble_rf_holdout_mean, ensemble_svm_holdout_mean),
  'Duration' = c(bag_rf_duration_mean, C50_duration_mean, ranger_duration_mean, svm_duration_mean, xgb_duration_mean, ensemble_bag_rf_duration_mean, ensemble_C50_duration_mean, ensemble_ranger_duration_mean, ensemble_rf_duration_mean, ensemble_svm_duration_mean)
)

results <- results %>% arrange(desc(Accuracy))

return(results)

}# Closing braces for numresamples loop
}# Closing brace for classification1 function

classification_1(data = ISLR::Carseats,colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#>                      Model  Accuracy    Duration
#> 1          Ensemble_Bag_RF 1.0000000 0.129196167
#> 2             Ensemble_C50 1.0000000 0.007623911
#> 3              Ensemble_RF 0.9545455 0.018841982
#> 4             Ensemble_SVM 0.8484848 0.005934000
#> 5     Bagged_Random_Forest 0.6829268 0.090348959
#> 6                  XGBoost 0.6768293 6.853636026
#> 7                   Ranger 0.6646341 0.738637924
#> 8                      C50 0.6463415 0.422610044
#> 9  Support_Vector_Machines 0.5792683 0.013298988
#> 10         Ensemble_Ranger 0.4696970 0.033684015
```

``` r

warnings()
```


``` r

df1 <- Ensembles::dry_beans_small
#> Registered S3 method overwritten by 'tsibble':
#>   method          from
#>   format.interval inum
#> Registered S3 method overwritten by 'GGally':
#>   method from   
#>   +.gg   ggplot2
```

``` r

classification_1(data = df1, colnum = 17, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#>                      Model  Accuracy    Duration
#> 1          Ensemble_Bag_RF 1.0000000  0.12194419
#> 2             Ensemble_C50 1.0000000  0.01354599
#> 3     Bagged_Random_Forest 0.9081633  0.14628696
#> 4                   Ranger 0.9047619  0.04827619
#> 5                  XGBoost 0.8945578 21.21282005
#> 6                      C50 0.8809524  0.02526808
#> 7  Support_Vector_Machines 0.8571429  0.04564309
#> 8             Ensemble_SVM 0.7520661  0.01254821
#> 9              Ensemble_RF 0.6776860  0.16191602
#> 10         Ensemble_Ranger 0.1239669  0.04927993
```

``` r
warnings()
```
