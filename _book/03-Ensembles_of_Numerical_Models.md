---
editor_options: 
  markdown: 
    wrap: 72
---

# Building weighted ensembles to model numerical data

In the last chapter we learned how to make 23 individual models,
including calculating the error rate (as root mean squared error), and
predictions from the holdout data (test and validation).

This chapter will show how to use those results to make weighted
ensembles that can be used to model numerical data.

Let's start at the end. Let's imagine the finished product. A list of
ensembles and individual models, with error rates sorted in decreasing
order (best result on the top of the list)

Therefore we're going to need a weighted ensemble. But what weight to
use?

It turns out there is an excellent answer available for virtually no
work on our part (which is great!). Each model has a mean error score.
The weight we will use will be the reciprocal of the error.

Let's say we have two models. One has an error rate of 5.0 and the other
has an error rate of 2.0. Clearly the model with the error rate of 2.0
is superior to the model with the error rate of 5.0.

What we will do in building our ensemble is multiply the values in the
ensemble by 1/(error rate). This will give higher weights to models with
higher accuracy.

Let's see how this works with an extremely simple ensemble.


``` r
library(tree) # Allows us to use tree models
library(MASS) # For the Boston Housing data set library(Metrics)
library(tidyverse)
#> ── Attaching core tidyverse packages ──── tidyverse 2.0.0 ──
#> ✔ dplyr     1.1.4     ✔ readr     2.1.5
#> ✔ forcats   1.0.0     ✔ stringr   1.5.1
#> ✔ ggplot2   3.5.1     ✔ tibble    3.2.1
#> ✔ lubridate 1.9.3     ✔ tidyr     1.3.1
#> ✔ purrr     1.0.2     
#> ── Conflicts ────────────────────── tidyverse_conflicts() ──
#> ✖ dplyr::filter() masks stats::filter()
#> ✖ dplyr::lag()    masks stats::lag()
#> ✖ dplyr::select() masks MASS::select()
#> ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
```

``` r

# Set initial values to 0
linear_train_RMSE <- 0
linear_test_RMSE <- 0
linear_RMSE <- 0
linear_test_predict_value <- 0

tree_train_RMSE <- 0
tree_test_RMSE <- 0
tree_RMSE <- 0
tree_holdout_RMSE <- 0
tree_test_predict_value <- 0

ensemble_linear_RMSE <- 0
ensemble_linear_RMSE_mean <- 0
ensemble_tree_RMSE <- 0
ensemble_tree_RMSE_mean <- 0

numerical_1 <- function(data, colnum, train_amount, test_amount, numresamples){

# Move target column to far right
y <- 0
colnames(data)[colnum] <- "y"

# Set up resampling
for (i in 1:numresamples) {
  idx <- sample(seq(1, 2), size = nrow(data), replace = TRUE, prob = c(train_amount, test_amount))
  train <- data[idx == 1, ]
  test <- data[idx == 2, ]

# Fit linear model on the training data, make predictions on the test data
linear_model <- lm(y ~ ., data = train)
linear_predictions <- predict(object = linear_model, newdata = test)
linear_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = linear_predictions)
linear_RMSE_mean <- mean(linear_RMSE)

# Fit tree model on the training data, make predictions on the test data
tree_model <- tree(y ~ ., data = train)
tree_predictions <- predict(object = tree_model, newdata = test)
tree_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = tree_predictions)
tree_RMSE_mean <- mean(tree_RMSE)

# Make the weighted ensemble
ensemble <- data.frame(
  'linear' = linear_predictions / linear_RMSE_mean,
  'tree' = tree_predictions / tree_RMSE_mean,
  'y_ensemble' = test$y)

# Split ensemble between train and test
ensemble_idx <- sample(seq(1, 2), size = nrow(ensemble), replace = TRUE, prob = c(train_amount, test_amount))
ensemble_train <- ensemble[ensemble_idx == 1, ]
ensemble_test <- ensemble[ensemble_idx == 2, ]

# Fit the ensemble data on the ensemble training data, predict on ensemble test data
ensemble_linear_model <- lm(y_ensemble ~ ., data = ensemble_train)

ensemble_linear_predictions <- predict(object = ensemble_linear_model, newdata = ensemble_test)

ensemble_linear_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y, predicted = ensemble_linear_predictions)

ensemble_linear_RMSE_mean <- mean(ensemble_linear_RMSE)

# Fit the tree model on the ensemble training data, predict on ensemble test data
ensemble_tree_model <- tree(y_ensemble ~ ., data = ensemble_train)

ensemble_tree_predictions <- predict(object = ensemble_tree_model, newdata = ensemble_test) 

ensemble_tree_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y, predicted = ensemble_tree_predictions)

ensemble_tree_RMSE_mean <- mean(ensemble_tree_RMSE)

results <- data.frame(
  'Model' = c('Linear', 'Tree', 'Ensemble_Linear', 'Ensemble_tree'),
  'Error_Rate' = c(linear_RMSE_mean, tree_RMSE_mean, ensemble_linear_RMSE_mean, ensemble_tree_RMSE_mean)
)

results <- results %>% arrange(Error_Rate)

} # Closing brace for numresamples
return(list(results))

} # Closing brace for the function

numerical_1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [[1]]
#>             Model Error_Rate
#> 1 Ensemble_Linear   4.142535
#> 2   Ensemble_tree   4.728397
#> 3          Linear   4.829952
#> 4            Tree   4.940936
```

``` r

warnings()
```

What we will be doing in this section is making a weighted ensemble
using 17 models of numerical data, then using that ensemble to measure
the accuracy of the models on the holdout (test) data.

## Think before you do something. This will help when we start at the end and work backwards toward the beginning.

We are going to make an ensemble. The ensemble is going to a made of
predictions from numerical models. You've already seen a couple of
ensembles. This one will be extremely similar, but will involve more
models.

Starting at the end, we want the error rate (root mean squared error)
for the ensemble and prediction models.

That means we'll need to make an ensemble. That means we'll need
individual model predictions, because that's how an ensemble is made. If
an ensemble is made of individual model predictions, that means we'll
need individual models. We already know how to do that, because we did
it in the last chapter.

We're going to build a simple ensemble with seven models, and then use
that ensemble with four very different methods. The Ensembles package
actually works with a total of 40 different models. The process is
exactly the same, whether we are working with seven individual models or
23 individual models, five ensemble models or 17 ensemble models. The
structre and methods are the same.

So let's get started!

The seven individual models we will be building are:

-   Linear (tuned)

-   Bayesglm

-   Bayesrnn

-   Gradient Boosted

-   RandomForest

-   Trees

It's important to understand that many other options are possible. Your
are encouraged to add at least one more modeling method to the ensemble,
and see how it impacts the results.

## One of your own: Add one model to the list of seven individual models, see how it impacts results.

## Plan ahead as much as you can, that makes the entire model building process much easier.

Here is the code to build ensembles. I very strongly recommend doing
this yourself, and checking every 5-10 lines to make sure there are no
errors.


``` r

#Load packages we will need

library(arm) # Allows us to run bayesglm
#> Loading required package: Matrix
#> 
#> Attaching package: 'Matrix'
#> The following objects are masked from 'package:tidyr':
#> 
#>     expand, pack, unpack
#> Loading required package: lme4
#> 
#> arm (Version 1.14-4, built: 2024-4-1)
#> Working directory is /Users/russellconte/Library/Mobile Documents/com~apple~CloudDocs/Documents/Machine Learning templates in R/EnsemblesBook
```

``` r
library(brnn) # Allows us to run brnn
#> Loading required package: Formula
#> Loading required package: truncnorm
```

``` r
library(e1071) # Allows us to run several tuned model, such as linear and KNN
library(randomForest) # Allows us to run random forest models
#> randomForest 4.7-1.1
#> Type rfNews() to see new features/changes/bug fixes.
#> 
#> Attaching package: 'randomForest'
#> The following object is masked from 'package:dplyr':
#> 
#>     combine
#> The following object is masked from 'package:ggplot2':
#> 
#>     margin
```

``` r
library(tree) # Allows us to run tree models
```

### A few other packages we will need to keep everything running smoothly


``` r
library(tidyverse) # Amazing set of tools for data science
library(MASS) # Gives us the Boston Housing data set
library(Metrics) # Allows us to calculate accuracy or error rates
```

### Build the function that will build the individual and ensemble models


``` r

numerical <- function(data, colnum, numresamples, train_amount, test_amount){

# Make the target column the right most column, change the column name to y:

y <- 0
colnames(data)[colnum] <- "y"

df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ]

# Set initial values to 0 for both individual and ensemble methods:

bayesglm_train_RMSE <- 0
bayesglm_test_RMSE <- 0
bayesglm_validation_RMSE <- 0
bayesglm_sd <- 0
bayesglm_overfitting <- 0
bayesglm_duration <- 0
bayesglm_duration_mean <- 0
bayesglm_holdout_mean <- 0
bayesglm_holdout_RMSE <- 0
bayesglm_holdout_RMSE_mean <- 0

bayesrnn_train_RMSE <- 0
bayesrnn_test_RMSE <- 0
bayesrnn_validation_RMSE <- 0
bayesrnn_sd <- 0
bayesrnn_overfitting <- 0
bayesrnn_duration <- 0
bayesrnn_duration_mean <- 0
bayesrnn_holdout_mean <- 0
bayesrnn_holdout_RMSE <- 0
bayesrnn_holdout_RMSE_mean <- 0

gb_train_RMSE <- 0
gb_test_RMSE <- 0
gb_validation_RMSE <- 0
gb_sd <- 0
gb_overfitting <- 0
gb_duration <- 0
gb_duration_mean <- 0
gb_holdout_mean <- 0
gb_holdout_RMSE <- 0
gb_holdout_RMSE_mean <- 0

linear_train_RMSE <- 0
linear_test_RMSE <- 0
linear_validation_RMSE <- 0
linear_sd <- 0
linear_overfitting <- 0
linear_duration <- 0
linear_holdout_RMSE <- 0
linear_holdout_RMSE_mean <- 0

rf_train_RMSE <- 0
rf_test_RMSE <- 0
rf_validation_RMSE <- 0
rf_sd <- 0
rf_overfitting <- 0
rf_duration <- 0
rf_duration_mean <- 0
rf_holdout_mean <- 0
rf_holdout_RMSE <- 0
rf_holdout_RMSE_mean <- 0

tree_train_RMSE <- 0
tree_test_RMSE <- 0
tree_validation_RMSE <- 0
tree_sd <- 0
tree_overfitting <- 0
tree_duration <- 0
tree_duration_mean <- 0
tree_holdout_mean <- 0
tree_holdout_RMSE <- 0
tree_holdout_RMSE_mean <- 0

ensemble_bayesglm_train_RMSE <- 0
ensemble_bayesglm_test_RMSE <- 0
ensemble_bayesglm_validation_RMSE <- 0
ensemble_bayesglm_sd <- 0
ensemble_bayesglm_overfitting <- 0
ensemble_bayesglm_duration <- 0
ensemble_bayesglm_holdout_RMSE <- 0
ensemble_bayesglm_holdout_RMSE_mean <- 0
ensemble_bayesglm_predict_value_mean <- 0

ensemble_bayesrnn_train_RMSE <- 0
ensemble_bayesrnn_test_RMSE <- 0
ensemble_bayesrnn_validation_RMSE <- 0
ensemble_bayesrnn_sd <- 0
ensemble_bayesrnn_overfitting <- 0
ensemble_bayesrnn_duration <- 0
ensemble_bayesrnn_holdout_RMSE <- 0
ensemble_bayesrnn_holdout_RMSE_mean <- 0
ensemble_bayesrnn_predict_value_mean <- 0

ensemble_gb_train_RMSE <- 0
ensemble_gb_test_RMSE <- 0
ensemble_gb_validation_RMSE <- 0
ensemble_gb_sd <- 0
ensemble_gb_overfitting <- 0
ensemble_gb_duration <- 0
ensemble_gb_holdout_RMSE <- 0
ensemble_gb_holdout_RMSE_mean <- 0
ensemble_gb_predict_value_mean <- 0

ensemble_linear_train_RMSE <- 0
ensemble_linear_test_RMSE <- 0
ensemble_linear_validation_RMSE <- 0
ensemble_linear_sd <- 0
ensemble_linear_overfitting <- 0
ensemble_linear_duration <- 0
ensemble_linear_holdout_RMSE <- 0
ensemble_linear_holdout_RMSE_mean <- 0

ensemble_rf_train_RMSE <- 0
ensemble_rf_test_RMSE <- 0
ensemble_rf_test_RMSE_mean <- 0
ensemble_rf_validation_RMSE <- 0
ensemble_rf_sd <- 0
ensemble_rf_overfitting <- 0
ensemble_rf_duration <- 0
ensemble_rf_holdout_RMSE <- 0
ensemble_rf_holdout_RMSE_mean <- 0

ensemble_tree_train_RMSE <- 0
ensemble_tree_test_RMSE <- 0
ensemble_tree_validation_RMSE <- 0
ensemble_tree_sd <- 0
ensemble_tree_overfitting <- 0
ensemble_tree_duration <- 0
ensemble_tree_holdout_RMSE <- 0
ensemble_tree_holdout_RMSE_mean <- 0

#Let's build the function that does all the resampling and puts everything together:

for (i in 1:numresamples) {

# Randomly split the data between train and test
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1, ]
test <- df[idx == 2, ]

# Bayesglm

bayesglm_train_fit <- arm::bayesglm(y ~ ., data = train, family = gaussian(link = "identity"))
bayesglm_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = bayesglm_train_fit, newdata = train))
bayesglm_train_RMSE_mean <- mean(bayesglm_train_RMSE)
bayesglm_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = bayesglm_train_fit, newdata = test))
bayesglm_test_RMSE_mean <- mean(bayesglm_test_RMSE)
bayesglm_holdout_RMSE[i] <- mean(bayesglm_test_RMSE_mean)
bayesglm_holdout_RMSE_mean <- mean(bayesglm_holdout_RMSE)
bayesglm_test_predict_value <- as.numeric(predict(object = bayesglm_train_fit, newdata = test))
y_hat_bayesglm <- c(bayesglm_test_predict_value)

# Bayesrnn

bayesrnn_train_fit <- brnn::brnn(x = as.matrix(train), y = train$y)
bayesrnn_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = bayesrnn_train_fit, newdata = train))
bayesrnn_train_RMSE_mean <- mean(bayesrnn_train_RMSE)
bayesrnn_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = bayesrnn_train_fit, newdata = test))
bayesrnn_test_RMSE_mean <- mean(bayesrnn_test_RMSE)
bayesrnn_holdout_RMSE[i] <- mean(c(bayesrnn_test_RMSE_mean))
bayesrnn_holdout_RMSE_mean <- mean(bayesrnn_holdout_RMSE)
bayesrnn_train_predict_value <- as.numeric(predict(object = bayesrnn_train_fit, newdata = train))
bayesrnn_test_predict_value <- as.numeric(predict(object = bayesrnn_train_fit, newdata = test))
bayesrnn_predict_value_mean <- mean(c(bayesrnn_test_predict_value))
y_hat_bayesrnn <- c(bayesrnn_test_predict_value)

# Gradient boosted

gb_train_fit <- gbm::gbm(train$y ~ ., data = train, distribution = "gaussian", n.trees = 100, shrinkage = 0.1, interaction.depth = 10)
gb_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = gb_train_fit, newdata = train))
gb_train_RMSE_mean <- mean(gb_train_RMSE)
gb_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = gb_train_fit, newdata = test))
gb_test_RMSE_mean <- mean(gb_test_RMSE)
gb_holdout_RMSE[i] <- mean(c(gb_test_RMSE_mean))
gb_holdout_RMSE_mean <- mean(gb_holdout_RMSE)
gb_train_predict_value <- as.numeric(predict(object = gb_train_fit, newdata = train))
gb_test_predict_value <- as.numeric(predict(object = gb_train_fit, newdata = test)) 
gb_predict_value_mean <- mean(c(gb_test_predict_value))
y_hat_gb <- c(gb_test_predict_value)

# Tuned linear models

linear_train_fit <- e1071::tune.rpart(formula = y ~ ., data = train)
linear_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = linear_train_fit$best.model, newdata = train))
linear_train_RMSE_mean <- mean(linear_train_RMSE)
linear_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = linear_train_fit$best.model, newdata = test))
linear_test_RMSE_mean <- mean(linear_test_RMSE)
linear_holdout_RMSE[i] <- mean(c(linear_test_RMSE_mean))
linear_holdout_RMSE_mean <- mean(linear_holdout_RMSE)
linear_train_predict_value <- as.numeric(predict(object = linear_train_fit$best.model, newdata = train))
linear_test_predict_value <- as.numeric(predict(object = linear_train_fit$best.model, newdata = test))
y_hat_linear <- c(linear_test_predict_value)

# RandomForest

rf_train_fit <- e1071::tune.randomForest(x = train, y = train$y, data = train)
rf_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = rf_train_fit$best.model, newdata = train))
rf_train_RMSE_mean <- mean(rf_train_RMSE)
rf_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = rf_train_fit$best.model, newdata = test))
rf_test_RMSE_mean <- mean(rf_test_RMSE)
rf_holdout_RMSE[i] <- mean(c(rf_test_RMSE_mean))
rf_holdout_RMSE_mean <- mean(rf_holdout_RMSE)
rf_train_predict_value <- predict(object = rf_train_fit$best.model, newdata = train)
rf_test_predict_value <- predict(object = rf_train_fit$best.model, newdata = test)
y_hat_rf <- c(rf_test_predict_value)

# Trees

tree_train_fit <- tree::tree(train$y ~ ., data = train)
tree_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = tree_train_fit, newdata = train))
tree_train_RMSE_mean <- mean(tree_train_RMSE)
tree_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = tree_train_fit, newdata = test))
tree_test_RMSE_mean <- mean(tree_test_RMSE)
tree_holdout_RMSE[i] <- mean(c(tree_test_RMSE_mean))
tree_holdout_RMSE_mean <- mean(tree_holdout_RMSE)
tree_train_predict_value <- as.numeric(predict(object = tree::tree(y ~ ., data = train), newdata = train))
tree_test_predict_value <- as.numeric(predict(object = tree::tree(y ~ ., data = train), newdata = test))
y_hat_tree <- c(tree_test_predict_value)

# Make the weighted ensemble:

ensemble <- data.frame(
  "BayesGLM" = y_hat_bayesglm * 1 / bayesglm_holdout_RMSE_mean,
  "BayesRNN" = y_hat_bayesrnn * 1 / bayesrnn_holdout_RMSE_mean,
  "GBM" = y_hat_gb * 1 / gb_holdout_RMSE_mean,
  "Linear" = y_hat_linear * 1 / linear_holdout_RMSE_mean,
  "RandomForest" = y_hat_rf * 1 / rf_holdout_RMSE_mean,
  "Tree" = y_hat_tree * 1 / tree_holdout_RMSE_mean
  )

ensemble$Row_mean <- rowMeans(ensemble)
  ensemble$y_ensemble <- c(test$y)
  y_ensemble <- c(test$y)

# Split the ensemble into train and test, according to user choices:

ensemble_idx <- sample(seq(1, 2), size = nrow(ensemble), replace = TRUE, prob = c(train_amount, test_amount))
ensemble_train <- ensemble[ensemble_idx == 1, ]
ensemble_test <- ensemble[ensemble_idx == 2, ]

# Ensemble BayesGLM

ensemble_bayesglm_train_fit <- arm::bayesglm(y_ensemble ~ ., data = ensemble_train, family = gaussian(link = "identity"))
ensemble_bayesglm_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_bayesglm_train_fit, newdata = ensemble_train))
ensemble_bayesglm_train_RMSE_mean <- mean(ensemble_bayesglm_train_RMSE)
ensemble_bayesglm_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_bayesglm_train_fit, newdata = ensemble_test))
ensemble_bayesglm_test_RMSE_mean <- mean(ensemble_bayesglm_test_RMSE)
ensemble_bayesglm_holdout_RMSE[i] <- mean(c(ensemble_bayesglm_test_RMSE_mean))
ensemble_bayesglm_holdout_RMSE_mean <- mean(ensemble_bayesglm_holdout_RMSE)

# Ensemble BayesRNN

ensemble_bayesrnn_train_fit <- brnn::brnn(x = as.matrix(ensemble_train), y = ensemble_train$y_ensemble)
ensemble_bayesrnn_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_bayesrnn_train_fit, newdata = ensemble_train))
ensemble_bayesrnn_train_RMSE_mean <- mean(ensemble_bayesrnn_train_RMSE)
ensemble_bayesrnn_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_bayesrnn_train_fit, newdata = ensemble_test))
ensemble_bayesrnn_test_RMSE_mean <- mean(ensemble_bayesrnn_test_RMSE)
ensemble_bayesrnn_holdout_RMSE[i] <- mean(c(ensemble_bayesrnn_test_RMSE_mean))
ensemble_bayesrnn_holdout_RMSE_mean <- mean(ensemble_bayesrnn_holdout_RMSE)

# Ensemble Graident Boosted

ensemble_gb_train_fit <- gbm::gbm(ensemble_train$y_ensemble ~ ., data = ensemble_train, distribution = "gaussian", n.trees = 100, shrinkage = 0.1, interaction.depth = 10)
ensemble_gb_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_gb_train_fit, newdata = ensemble_train))
ensemble_gb_train_RMSE_mean <- mean(ensemble_gb_train_RMSE)
ensemble_gb_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_gb_train_fit, newdata = ensemble_test))
ensemble_gb_test_RMSE_mean <- mean(ensemble_gb_test_RMSE)
ensemble_gb_holdout_RMSE[i] <- mean(c(ensemble_gb_test_RMSE_mean))
ensemble_gb_holdout_RMSE_mean <- mean(ensemble_gb_holdout_RMSE)

# Ensemble using Tuned Random Forest

ensemble_rf_train_fit <- e1071::tune.randomForest(x = ensemble_train, y = ensemble_train$y_ensemble, data = ensemble_train)
ensemble_rf_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_rf_train_fit$best.model, newdata = ensemble_train))
ensemble_rf_train_RMSE_mean <- mean(ensemble_rf_train_RMSE)
ensemble_rf_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_rf_train_fit$best.model, newdata = ensemble_test))
ensemble_rf_test_RMSE_mean <- mean(ensemble_rf_test_RMSE)
ensemble_rf_holdout_RMSE[i] <- mean(c(ensemble_rf_test_RMSE_mean))
ensemble_rf_holdout_RMSE_mean <- mean(ensemble_rf_holdout_RMSE)

# Trees

ensemble_tree_train_fit <- tree::tree(ensemble_train$y_ensemble ~ ., data = ensemble_train)
ensemble_tree_train_RMSE[i] <- Metrics::rmse(actual = ensemble_train$y_ensemble, predicted = predict(object = ensemble_tree_train_fit, newdata = ensemble_train))
ensemble_tree_train_RMSE_mean <- mean(ensemble_tree_train_RMSE)
ensemble_tree_test_RMSE[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = predict(object = ensemble_tree_train_fit, newdata = ensemble_test))
ensemble_tree_test_RMSE_mean <- mean(ensemble_tree_test_RMSE)
ensemble_tree_holdout_RMSE[i] <- mean(c(ensemble_tree_test_RMSE_mean))
ensemble_tree_holdout_RMSE_mean <- mean(ensemble_tree_holdout_RMSE)

summary_results <- data.frame(
  'Model' = c('BayesGLM', 'BayesRNN', 'Gradient_Boosted', 'Linear', 'Random_Forest', 'Trees', 'Ensemble_BayesGLM', 'Ensemble_BayesRNN', 'Ensemble_Gradient_Boosted', 'Ensemble_Random_Forest', 'Ensemble_Trees'), 
  'Error' = c(bayesglm_holdout_RMSE_mean, bayesrnn_holdout_RMSE_mean, gb_holdout_RMSE_mean, linear_holdout_RMSE_mean, rf_holdout_RMSE_mean, tree_holdout_RMSE_mean, ensemble_bayesglm_holdout_RMSE_mean, ensemble_bayesrnn_holdout_RMSE_mean, ensemble_gb_holdout_RMSE_mean, ensemble_rf_holdout_RMSE_mean, ensemble_tree_holdout_RMSE_mean) )

summary_results <- summary_results %>% arrange(Error)

} # closing brace for numresamples
return(summary_results)

} # closing brace for numerical function

numerical(data = MASS::Boston, colnum = 14, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015979 
#> gamma= 31.0592 	 alpha= 5.2255 	 beta= 16345.73
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7043456 
#> gamma= 13.0833 	 alpha= 2.0427 	 beta= 5622.241
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015132 
#> gamma= 30.7103 	 alpha= 4.8028 	 beta= 21458.26
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7046363 
#> gamma= 14.3717 	 alpha= 2.2275 	 beta= 7483.324
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015926 
#> gamma= 31.3551 	 alpha= 5.3838 	 beta= 14651.49
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.704124 
#> gamma= 12.9512 	 alpha= 1.9297 	 beta= 8531.317
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015669 
#> gamma= 28.7269 	 alpha= 4.3173 	 beta= 14167.72
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7038309 
#> gamma= 14.1897 	 alpha= 2.1638 	 beta= 11022.44
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016085 
#> gamma= 30.9911 	 alpha= 4.6469 	 beta= 13982.35
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7041953 
#> gamma= 13.1431 	 alpha= 1.9324 	 beta= 5765.634
#> Using 100 trees...
#> 
#> Using 100 trees...
#>                        Model     Error
#> 1          Ensemble_BayesGLM 0.1321454
#> 2                   BayesRNN 0.1408040
#> 3          Ensemble_BayesRNN 0.2389492
#> 4     Ensemble_Random_Forest 0.9385837
#> 5              Random_Forest 1.6994395
#> 6             Ensemble_Trees 1.8551031
#> 7  Ensemble_Gradient_Boosted 2.8566052
#> 8           Gradient_Boosted 3.1890256
#> 9                     Linear 4.7726583
#> 10                     Trees 4.8282863
#> 11                  BayesGLM 4.8738526
```

``` r

warnings()
```

Here's the very cool part of setting it up this way. If you have a
totally different data set, all you need to do is put the information
into the function, and everything runs. Check this out:


``` r

numerical(data = ISLR::Auto[, 1:ncol(ISLR::Auto)-1], colnum = 1, numresamples = 25, train_amount = 0.50, test_amount = 0.50)
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024926 
#> gamma= 18.4162 	 alpha= 2.7514 	 beta= 13109.59
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.70502 
#> gamma= 13.9302 	 alpha= 2.5764 	 beta= 4233.778
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024799 
#> gamma= 18.7396 	 alpha= 2.5691 	 beta= 11432.25
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7049182 
#> gamma= 13.1686 	 alpha= 2.4255 	 beta= 4584.367
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7026276 
#> gamma= 17.5839 	 alpha= 3.1297 	 beta= 14069.74
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7043456 
#> gamma= 12.4475 	 alpha= 2.4651 	 beta= 5270.222
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025856 
#> gamma= 18.8981 	 alpha= 3.1775 	 beta= 9580.519
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7044656 
#> gamma= 12.8729 	 alpha= 2.072 	 beta= 11330.64
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7021313 
#> gamma= 18.7676 	 alpha= 3.5901 	 beta= 12531.76
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7061688 
#> gamma= 11.3326 	 alpha= 2.4941 	 beta= 3858.152
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024673 
#> gamma= 18.274 	 alpha= 1.9759 	 beta= 17960.72
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.70502 
#> gamma= 12.6674 	 alpha= 2.0891 	 beta= 6155.332
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024302 
#> gamma= 18.5515 	 alpha= 3.4186 	 beta= 14243.8
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7055354 
#> gamma= 13.8508 	 alpha= 2.5064 	 beta= 4256.149
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7026858 
#> gamma= 19.3318 	 alpha= 3.6784 	 beta= 8425.08
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.704124 
#> gamma= 13.3745 	 alpha= 2.593 	 beta= 5848.52
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024548 
#> gamma= 19.4772 	 alpha= 3.7208 	 beta= 9429.915
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7045924 
#> gamma= 13.0337 	 alpha= 2.5198 	 beta= 4784.62
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025719 
#> gamma= 19.0825 	 alpha= 3.3722 	 beta= 9364.237
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7044656 
#> gamma= 14.1169 	 alpha= 2.3046 	 beta= 5773.569
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025856 
#> gamma= 19.3196 	 alpha= 3.4654 	 beta= 9802.053
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.704681 
#> gamma= 13.2377 	 alpha= 2.2026 	 beta= 5345.472
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025185 
#> gamma= 18.6502 	 alpha= 3.5727 	 beta= 10209.71
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7049686 
#> gamma= 12.381 	 alpha= 2.1155 	 beta= 5665.373
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025449 
#> gamma= 18.6783 	 alpha= 2.9818 	 beta= 12973.07
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7044656 
#> gamma= 13.8315 	 alpha= 2.0513 	 beta= 7035.462
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025584 
#> gamma= 18.9933 	 alpha= 3.2416 	 beta= 9231.745
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7047266 
#> gamma= 11.8112 	 alpha= 1.9485 	 beta= 6097.915
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024548 
#> gamma= 19.1122 	 alpha= 3.1686 	 beta= 9241.642
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7048205 
#> gamma= 12.8979 	 alpha= 2.1488 	 beta= 6987.878
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024673 
#> gamma= 19.4256 	 alpha= 3.698 	 beta= 9353.561
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7055354 
#> gamma= 10.7933 	 alpha= 1.7317 	 beta= 5212.225
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024799 
#> gamma= 17.015 	 alpha= 2.9886 	 beta= 10479.61
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7053523 
#> gamma= 13.0587 	 alpha= 2.2748 	 beta= 5016.278
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024673 
#> gamma= 18.9127 	 alpha= 2.9877 	 beta= 16940.02
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7049182 
#> gamma= 13.3138 	 alpha= 2.7722 	 beta= 5059.017
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024926 
#> gamma= 19.3027 	 alpha= 2.8318 	 beta= 9719.859
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7057316 
#> gamma= 11.9986 	 alpha= 2.0843 	 beta= 3711.19
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7026419 
#> gamma= 18.7109 	 alpha= 3.1853 	 beta= 9516.249
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7052367 
#> gamma= 13.4531 	 alpha= 2.4635 	 beta= 5533.873
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025317 
#> gamma= 18.9174 	 alpha= 3.0516 	 beta= 9411.772
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7049182 
#> gamma= 13.1981 	 alpha= 2.1471 	 beta= 7517.678
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025856 
#> gamma= 18.6904 	 alpha= 2.8206 	 beta= 13973.04
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7043456 
#> gamma= 12.1392 	 alpha= 2.1957 	 beta= 8680.461
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025995 
#> gamma= 19.0408 	 alpha= 3.3839 	 beta= 11170.28
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7049686 
#> gamma= 13.169 	 alpha= 2.0876 	 beta= 7091.306
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024673 
#> gamma= 18.9979 	 alpha= 2.8482 	 beta= 10179.61
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7045071 
#> gamma= 12.4243 	 alpha= 2.6426 	 beta= 4855.048
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7023942 
#> gamma= 19.4829 	 alpha= 3.7004 	 beta= 9447.426
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7052367 
#> gamma= 12.5163 	 alpha= 2.1979 	 beta= 4306.634
#> Using 100 trees...
#> 
#> Using 100 trees...
#>                        Model     Error
#> 1          Ensemble_BayesGLM 0.1293269
#> 2                   BayesRNN 0.1310772
#> 3          Ensemble_BayesRNN 0.2064510
#> 4     Ensemble_Random_Forest 0.8823964
#> 5  Ensemble_Gradient_Boosted 1.4839729
#> 6             Ensemble_Trees 1.6717371
#> 7              Random_Forest 1.6945471
#> 8           Gradient_Boosted 2.9664228
#> 9                   BayesGLM 3.4312923
#> 10                     Trees 3.7725029
#> 11                    Linear 3.7907192
```

## One of your own: Add a model to the individual models, and a model to the ensemble of models

One of your own: Change the data, run it again, comment on the results

## Post your results on social media in a way that a non-technical person can understand them. For example:

"Just ran six individual and six ensemble models, very easy to do, no
errors or warnings. I plan to do ensembles with other data sets soon.
#AIEnsembles

## Exercises to help you improve your skills:

Build an individual numerical model using each of the following model
methods (it's perfectly OK to check prior sections of the book, this is
an example of delayed repetition):

Gradient Boosted (from the gmb library)

Rpart (from the rpart library)

Support Vector Machines (tuned from the e1071 library)

One model method of your own choosing

Build an ensemble using those four methods, test it using the Boston
Housing data set. Compare the results of this ensemble to the one made
in the text of this chapter.

Apply the function you made to a different numerical data set. This can
be done in one line of code, once the ensemble is set up.

## Post the results of your new ensemble on social media in a way that helps others understand the results or methods.

## Helping leaders make the best possible decisions: What are the options, benefits and costs? How strong are the results/recommendations?

## Helping leaders make the best possible decisions: Margins of error/reasons for error

## Helping leaders make the best possible decisions: Highest accuracy, strongest predictor(s)

## Helping your customer/manager/board to start thinking analytically
