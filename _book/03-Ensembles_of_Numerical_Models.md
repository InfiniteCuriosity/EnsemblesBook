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


```r
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
#> 1 Ensemble_Linear   4.127265
#> 2            Tree   4.750828
#> 3   Ensemble_tree   4.812045
#> 4          Linear   4.865660

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


```r

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
#> Working directory is /Users/russconte/Library/Mobile Documents/com~apple~CloudDocs/Documents/Machine Learning templates in R/EnsemblesBook
library(brnn) # Allows us to run brnn
#> Loading required package: Formula
#> Loading required package: truncnorm
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
library(tree) # Allows us to run tree models
```

### A few other packages we will need to keep everything running smoothly


```r
library(tidyverse) # Amazing set of tools for data science
library(MASS) # Gives us the Boston Housing data set
library(Metrics) # Allows us to calculate accuracy or error rates
```

### Build the function that will build the individual and ensemble models


```r

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
#> Scaling factor= 0.7015569 
#> gamma= 31.1545 	 alpha= 5.1034 	 beta= 14788.74
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7042319 
#> gamma= 14.5179 	 alpha= 2.1332 	 beta= 7029.778
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015619 
#> gamma= 31.0596 	 alpha= 4.216 	 beta= 30890.58
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7036303 
#> gamma= 13.6581 	 alpha= 2.1458 	 beta= 6053.645
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015874 
#> gamma= 31.4948 	 alpha= 5.4893 	 beta= 14943
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7041593 
#> gamma= 13.4907 	 alpha= 1.8086 	 beta= 6770.96
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016085 
#> gamma= 31.1358 	 alpha= 5.6648 	 beta= 16396.45
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7037423 
#> gamma= 13.7725 	 alpha= 1.7388 	 beta= 11998.36
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7017105 
#> gamma= 30.4814 	 alpha= 4.7672 	 beta= 16756.22
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7036577 
#> gamma= 12.6686 	 alpha= 1.8284 	 beta= 8985.648
#> Using 100 trees...
#> 
#> Using 100 trees...
#>                        Model     Error
#> 1          Ensemble_BayesGLM 0.1379912
#> 2                   BayesRNN 0.1507615
#> 3          Ensemble_BayesRNN 0.1855423
#> 4     Ensemble_Random_Forest 0.9268246
#> 5              Random_Forest 1.7842346
#> 6             Ensemble_Trees 2.0855036
#> 7  Ensemble_Gradient_Boosted 2.2402316
#> 8           Gradient_Boosted 3.5698647
#> 9                      Trees 4.8366919
#> 10                  BayesGLM 4.9877972
#> 11                    Linear 5.1775478

warnings()
```

Here's the very cool part of setting it up this way. If you have a
totally different data set, all you need to do is put the information
into the function, and everything runs. Check this out:


```r

numerical(data = ISLR::Auto[, 1:ncol(ISLR::Auto)-1], colnum = 1, numresamples = 25, train_amount = 0.50, test_amount = 0.50)
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7026564 
#> gamma= 17.0741 	 alpha= 2.6627 	 beta= 8713.611
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
#> gamma= 11.5889 	 alpha= 2.0474 	 beta= 4629.483
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025856 
#> gamma= 18.5517 	 alpha= 2.648 	 beta= 28556.25
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7050725 
#> gamma= 13.0326 	 alpha= 2.8908 	 beta= 4353.741
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7023942 
#> gamma= 19.3003 	 alpha= 3.4712 	 beta= 10171.52
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7048689 
#> gamma= 12.1648 	 alpha= 2.5587 	 beta= 4977.586
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7023593 
#> gamma= 19.5728 	 alpha= 3.8344 	 beta= 9479.29
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7055993 
#> gamma= 13.403 	 alpha= 2.3741 	 beta= 4139.822
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7023708 
#> gamma= 19.2389 	 alpha= 3.4001 	 beta= 11594.67
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7058001 
#> gamma= 14.2831 	 alpha= 1.8106 	 beta= 6708.337
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024673 
#> gamma= 18.8419 	 alpha= 2.8555 	 beta= 11166.83
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
#> gamma= 13.9749 	 alpha= 2.3209 	 beta= 4007.501
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7026276 
#> gamma= 18.6506 	 alpha= 3.2347 	 beta= 9230.53
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
#> gamma= 13.6663 	 alpha= 2.3144 	 beta= 5343.38
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7023254 
#> gamma= 19.1902 	 alpha= 3.5465 	 beta= 9493.475
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7068673 
#> gamma= 11.9701 	 alpha= 2.4945 	 beta= 3145.46
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7026135 
#> gamma= 18.831 	 alpha= 3.0418 	 beta= 11733.43
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7050725 
#> gamma= 13.0237 	 alpha= 2.2562 	 beta= 5798.846
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024061 
#> gamma= 18.7375 	 alpha= 3.5705 	 beta= 10077.21
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7050725 
#> gamma= 12.8282 	 alpha= 2.2718 	 beta= 4613.55
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7026135 
#> gamma= 19.3209 	 alpha= 3.5508 	 beta= 8750.865
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
#> gamma= 12.9429 	 alpha= 2.1176 	 beta= 5934.885
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024425 
#> gamma= 18.7662 	 alpha= 2.9538 	 beta= 22385.57
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7051261 
#> gamma= 12.0317 	 alpha= 2.2065 	 beta= 5723.925
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024061 
#> gamma= 18.5628 	 alpha= 1.7018 	 beta= 84334.34
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
#> gamma= 12.1488 	 alpha= 2.1287 	 beta= 4850.201
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025449 
#> gamma= 19.4594 	 alpha= 3.7409 	 beta= 9004.504
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7045493 
#> gamma= 12.953 	 alpha= 2.0884 	 beta= 5940.986
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024181 
#> gamma= 19.0158 	 alpha= 3.2166 	 beta= 13737.31
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7051261 
#> gamma= 15.683 	 alpha= 2.2925 	 beta= 9456.172
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7023593 
#> gamma= 18.689 	 alpha= 2.4452 	 beta= 9383.395
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.705473 
#> gamma= 12.9215 	 alpha= 1.0122 	 beta= 4870.5
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025055 
#> gamma= 19.0227 	 alpha= 3.4188 	 beta= 9603.117
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
#> gamma= 13.1365 	 alpha= 2.3345 	 beta= 4404.568
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7024799 
#> gamma= 18.5332 	 alpha= 3.4972 	 beta= 11495.23
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7055993 
#> gamma= 12.307 	 alpha= 1.39 	 beta= 78137.77
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025055 
#> gamma= 18.427 	 alpha= 2.1005 	 beta= 32867.57
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
#> gamma= 12.2244 	 alpha= 2.6623 	 beta= 4237.805
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7026564 
#> gamma= 18.691 	 alpha= 2.8504 	 beta= 17454.78
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
#> gamma= 13.4225 	 alpha= 3.0357 	 beta= 5413.544
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025584 
#> gamma= 18.9272 	 alpha= 3.1235 	 beta= 9202.181
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7051261 
#> gamma= 12.8962 	 alpha= 2.4382 	 beta= 4970.101
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7023708 
#> gamma= 18.9841 	 alpha= 3.0874 	 beta= 9927.468
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7051261 
#> gamma= 12.1086 	 alpha= 2.2211 	 beta= 5083.125
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7027941 
#> gamma= 19.3697 	 alpha= 3.8956 	 beta= 7783.408
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
#> gamma= 13.9015 	 alpha= 2.4818 	 beta= 5403.593
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7025449 
#> gamma= 13.3282 	 alpha= 0.9683 	 beta= 9062.74
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
#> gamma= 14.4531 	 alpha= 2.326 	 beta= 8345.213
#> Using 100 trees...
#> 
#> Using 100 trees...
#> Number of parameters (weights and biases) to estimate: 20 
#> Nguyen-Widrow method
#> Scaling factor= 0.7021989 
#> gamma= 18.0495 	 alpha= 3.4913 	 beta= 10339.51
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
#> gamma= 12.4934 	 alpha= 2.3563 	 beta= 4167.19
#> Using 100 trees...
#> 
#> Using 100 trees...
#>                        Model     Error
#> 1          Ensemble_BayesGLM 0.1270331
#> 2                   BayesRNN 0.1411679
#> 3          Ensemble_BayesRNN 0.1906438
#> 4     Ensemble_Random_Forest 0.9333546
#> 5              Random_Forest 1.6440270
#> 6  Ensemble_Gradient_Boosted 1.7233343
#> 7             Ensemble_Trees 1.7852984
#> 8           Gradient_Boosted 2.9255353
#> 9                   BayesGLM 3.4321977
#> 10                    Linear 3.6335479
#> 11                     Trees 3.6335709
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
