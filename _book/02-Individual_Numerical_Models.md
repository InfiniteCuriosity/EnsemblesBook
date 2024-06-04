---
editor_options: 
  markdown: 
    wrap: 72
---

# Numerical data: How to make 23 individual models, and basic skills with functions

This is where we will begin building the skills to make ensembles of
models of numerical data. However, this is going to be much easier than
it might appear at first. Let's see how we can make this as easy as
possible.

How to work backwards and make the function we need: Start from the end

We are going to start at the ending, not at the beginning, and work
backwards from there. This method is much, much easier than working
forward, as you will see throughout this book. While it might be a
little uncomfortable at first, this skill will allow you to complete
your work at a faster rate than if you work forward.

We'll use the Boston Housing data set, and we'll start with the Bagged
Random Forest function. For now we're only going to work with one
function, to keep everything simple. In essence, we are going to run
this like an assembly line.

We want the ending to be the error rate by model. Virtually any customer
you work with is going to want to know, "How accurate is it?" That's our
starting point.

How do we determine model accuracy? We already did this in the previous
chapter, finding the root mean squared error for the individual models
and the ensemble models. We're going to do the same steps here, so the
process is familiar to you.

To get the error rate by model on the holdout data sets (test and
validation), we're going to need a model (Bagged Random Forest in this
first example), fit to the training data, and use that model to make
predictions on the test data. We can then measure the error in the
predictions, just as we did before. These steps should be familiar to
you. If not, please re-read the previous chapter.

But what do we need to complete those steps? We're going to have to go
backward (a little) and make a function that will allow us to work with
any data set.

What does our function need? Let's make a list:

-   The data (such as Boston housing)

-   Column number (such as 14, the median value of the property)

-   Train amount

-   Test amount

-   Validation amount

-   Number of times to resample

One of the key steps here is to change the name of the target variable
to y. The initial name could be nearly anything, but this method changes
the name of the target variable to y. This allows us to make one small
change that will allow this to be the easiest possible solution:

### All our models will be structured the same way: y \~ ., data = train

This means that y (our target value) is a function of the other
features, and the data set is the training data set. While there will be
some variations on this in our 27 models, the basic structure is the
same.

### Having the same structure for all the models makes it much easier to build, debug, and deploy the completed models.

Then we only need to start with our initial values, and it will run.

One extremely nice part about creating models this way is the enormous
efficiency it gives us. Once we have the Bagged Random Forest model
working, we will be able to use very similar (and identical in many
cases!) processes with other models (such as Support Vector Machines).

The rock solid foundation we lay at the beginning will allow us to have
a smooth and easy experience once the foundation is solid and we use it
to build more models. The other models will mainly be almost exact
duplicates of our fist example.'

Here are the steps we will follow:

-   Load the library

-   Set initial values to 0

-   Create the function

-   Set up random resampling

-   Break the data into train and test

-   Fit the model on the training data, make predictions and measure
    error on the test data

-   Return the results

-   Check for errors or warnings

-   Test on a different data set

### Exercise: Re-read the steps above how we will work backwards to come up with the function we need.

### 1. Bagged Random Forest


```r
library(e1071) # will allow us to use a tuned random forest model
library(Metrics) # Will allow us to calculate the root mean squared error
library(randomForest) # To use the random forest function
#> randomForest 4.7-1.1
#> Type rfNews() to see new features/changes/bug fixes.
library(tidyverse) # Amazing set of tools for data science
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


```r
# Set initial values to 0. The function will return an error if any of these are left out.

bag_rf_holdout_RMSE <- 0
bag_rf_holdout_RMSE_mean <- 0
bag_rf_train_RMSE <- 0
bag_rf_test_RMSE <- 0
bag_rf_validation_RMSE <- 0
```


```r

# Define the function

numerical_1 <- function(data, colnum, train_amount, test_amount, numresamples){

#Set up random resampling

for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col())
df <- df[sample(nrow(df)), ] # randomizes the rows

#Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1, ]
test <- df[idx == 2, ]

# Fit the model to the training data, make predictions on the testing data, then calculate the error rates on the testing data sets.
bag_rf_train_fit <- e1071::tune.randomForest(x = train, y = train$y, mtry = ncol(train) - 1)
bag_rf_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict( object = bag_rf_train_fit$best.model, newdata = train))
bag_rf_train_RMSE_mean <- mean(bag_rf_train_RMSE)
bag_rf_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict( object = bag_rf_train_fit$best.model, newdata = test))
bag_rf_test_RMSE_mean <- mean(bag_rf_test_RMSE)

# Itemize the error on the holdout data sets, and calculate the mean of the results
bag_rf_holdout_RMSE[i] <- mean(bag_rf_test_RMSE_mean)
bag_rf_holdout_RMSE_mean <- mean(c(bag_rf_holdout_RMSE))

# These are the predictions we will need when we make the ensembles
bag_rf_test_predict_value <- as.numeric(predict(object = bag_rf_train_fit$best.model, newdata = test))


# Return the mean of the results to the user

} # closing brace for numresamples
  return(bag_rf_holdout_RMSE_mean)

} # closing brace for numerical_1 function

# Here is our first numerical function in actual use. We will use 25 resamples

numerical_1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 0.3023503
warnings() # no warnings, the best possible result
```

Exercise: Try it yourself: Change the values of train, test and
validation, and the number of resamples. See how those change the
result.

One of your own: Find any numerical data set, and make a bagged random
forest function for that data set. (For example, you may use the Auto
data set in the ISLR package. You will need to remove the last column,
vehicle name. Model mpg as a function of the other features using the
Bagged Random Forest function, but any numerical data set will work).

Post: Share on social your first results making a numerical function
(screen shot/video optional at this stage, we will be learning how to do
those later)

For example, "Did my first data science function building up to making
ensembles later on. Got everything to run, no errors. #AIEnsembles"

Now we will build the remaining 22 models for numerical data. They are
all built using the same structure, on the same foundation.

Now that we know how to build a basic function, let's build the 22 other
sets of tools we will need to make our ensemble, starting with bagging:

### 2. Bagging (bootstrap aggregating)


```r
library(ipred) #for the bagging function

# Set initial values to 0
bagging_train_RMSE <- 0
bagging_test_RMSE <- 0
bagging_validation_RMSE <- 0
bagging_holdout_RMSE <- 0
bagging_test_predict_value <- 0
bagging_validation_predict_value <- 0

#Create the function:

bagging_1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

#Set up random resampling
for (i in 1:numresamples) {

#Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets

idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1, ]
test <- df[idx == 2, ]

# Fit the model to the training data, calculate error, make predictions on the holdout data

bagging_train_fit <- ipred::bagging(formula = y ~ ., data = train)
bagging_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = bagging_train_fit, newdata = train))
bagging_train_RMSE_mean <- mean(bagging_train_RMSE)
bagging_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = bagging_train_fit, newdata = test))
bagging_test_RMSE_mean <- mean(bagging_test_RMSE)
bagging_holdout_RMSE[i] <- mean(bagging_test_RMSE_mean)
bagging_holdout_RMSE_mean <- mean(bagging_holdout_RMSE)
y_hat_bagging <- c(bagging_test_predict_value)

} # closing braces for the resampling function
  return(bagging_holdout_RMSE_mean)
  
} # closing braces for the bagging function

# Test the function:
bagging_1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.20, numresamples = 25)
#> [1] 4.378991
warnings() # no warnings
```

### 3. BayesGLM


```r
library(arm) # to use bayesglm function
#> Loading required package: MASS
#> 
#> Attaching package: 'MASS'
#> The following object is masked from 'package:dplyr':
#> 
#>     select
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

# Set initial values to 0
bayesglm_train_RMSE <- 0
bayesglm_test_RMSE <- 0
bayesglm_validation_RMSE <- 0
bayesglm_holdout_RMSE <- 0
bayesglm_test_predict_value <- 0
bayesglm_validation_predict_value <- 0

# Create the function:
bayesglm_1 <- function(data, colnum, train_amount, test_amount, numresamples){

#Set up random resampling
for (i in 1:numresamples) {

#Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

#Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

#Breaks the data into train, test and validation sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1, ]
test <- df[idx == 2, ]

bayesglm_train_fit <- arm::bayesglm(y ~ ., data = train, family = gaussian(link = "identity"))
bayesglm_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = bayesglm_train_fit, newdata = train))
bayesglm_train_RMSE_mean <- mean(bayesglm_train_RMSE)
bayesglm_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = bayesglm_train_fit, newdata = test))
bayesglm_test_RMSE_mean <- mean(bayesglm_test_RMSE) 
y_hat_bayesglm <- c(bayesglm_test_predict_value)

} # closing braces for resampling
  return(bayesglm_test_RMSE_mean)
  
} # closing braces for the function

bayesglm_1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.20, numresamples = 25)
#> [1] 4.843296
warnings() # no warnings
```

### 4. BayesRNN


```r
library(brnn) # so we can use the BayesRNN function
#> Loading required package: Formula
#> Loading required package: truncnorm

#Set initial values to 0

bayesrnn_train_RMSE <- 0
bayesrnn_test_RMSE <- 0
bayesrnn_validation_RMSE <- 0
bayesrnn_holdout_RMSE <- 0
bayesrnn_test_predict_value <- 0
bayesrnn_validation_predict_value <- 0

# Create the function:

bayesrnn_1 <- function(data, colnum, train_amount, test_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1, ]
test <- df[idx == 2, ]

# Fit the model on the training data, make predictions on the testing data
bayesrnn_train_fit <- brnn::brnn(x = as.matrix(train), y = train$y)
bayesrnn_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = bayesrnn_train_fit, newdata = train))
bayesrnn_train_RMSE_mean <- mean(bayesrnn_train_RMSE)
bayesrnn_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = bayesrnn_train_fit, newdata = test))
bayesrnn_test_RMSE_mean <- mean(bayesrnn_test_RMSE)

y_hat_bayesrnn <- c(bayesrnn_test_predict_value)

} # Closing brace for number of resamples 
  return(bayesrnn_test_RMSE_mean)

} # Closing brace for the function

bayesrnn_1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.701572 
#> gamma= 31.2244 	 alpha= 3.0586 	 beta= 38683.82 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015275 
#> gamma= 29.2045 	 alpha= 2.067 	 beta= 15308.9 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016356 
#> gamma= 30.3801 	 alpha= 2.9574 	 beta= 50623.19 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016411 
#> gamma= 29.6522 	 alpha= 2.1858 	 beta= 13789.78 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016411 
#> gamma= 30.9251 	 alpha= 3.9807 	 beta= 13284.05 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015619 
#> gamma= 30.4682 	 alpha= 5.0886 	 beta= 16331.64 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016085 
#> gamma= 31.192 	 alpha= 5.1146 	 beta= 19552.41 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015669 
#> gamma= 30.1789 	 alpha= 4.3 	 beta= 35285.68 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015669 
#> gamma= 31.3872 	 alpha= 3.8202 	 beta= 15992.69 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7017538 
#> gamma= 31.2203 	 alpha= 5.3491 	 beta= 13244.21 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015771 
#> gamma= 31.0916 	 alpha= 4.1637 	 beta= 15897.53 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016085 
#> gamma= 31.5276 	 alpha= 4.968 	 beta= 17564.28 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016636 
#> gamma= 31.0946 	 alpha= 5.251 	 beta= 15223.9 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015771 
#> gamma= 31.4853 	 alpha= 4.6814 	 beta= 15895.73 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015619 
#> gamma= 31.5488 	 alpha= 4.4819 	 beta= 14880.51 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.701542 
#> gamma= 31.2898 	 alpha= 3.9508 	 beta= 15560.15 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016926 
#> gamma= 31.6292 	 alpha= 4.1742 	 beta= 13535.13 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015275 
#> gamma= 29.9586 	 alpha= 3.9612 	 beta= 38416.76 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015227 
#> gamma= 31.4039 	 alpha= 5.6172 	 beta= 15691.66 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016246 
#> gamma= 31.3544 	 alpha= 5.3271 	 beta= 15780.51 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016138 
#> gamma= 31.3198 	 alpha= 5.7416 	 beta= 15094.68 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016246 
#> gamma= 31.0694 	 alpha= 5.2206 	 beta= 21175.37 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.701735 
#> gamma= 31.4069 	 alpha= 5.6578 	 beta= 14047.93 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016579 
#> gamma= 31.289 	 alpha= 5.201 	 beta= 13275.09 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016411 
#> gamma= 30.9163 	 alpha= 4.1354 	 beta= 25590.17
#> [1] 0.1331119

warnings() # no warnings for BayesRNN function
```

### 5. Boosted Random Forest


```r
library(e1071)
library(randomForest)
library(tidyverse)

#Set initial values to 0
boost_rf_train_RMSE <- 0
boost_rf_test_RMSE <- 0
boost_rf_validation_RMSE <- 0
boost_rf_holdout_RMSE <- 0
boost_rf_test_predict_value <- 0
boost_rf_validation_predict_value <- 0

#Create the function:
boost_rf_1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

#Set up random resampling
for (i in 1:numresamples) {

#Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

#Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1, ]
test <- df[idx == 2, ]

# Fit boosted random forest model on the training data, make predictions on holdout data

boost_rf_train_fit <- e1071::tune.randomForest(x = train, y = train$y, mtry = ncol(train) - 1)
boost_rf_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict( object = boost_rf_train_fit$best.model, newdata = train
  ))
boost_rf_train_RMSE_mean <- mean(boost_rf_train_RMSE)
boost_rf_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict( object = boost_rf_train_fit$best.model, newdata = test
  ))
boost_rf_test_RMSE_mean <- mean(boost_rf_test_RMSE)

} # closing brace for numresamples
  return(boost_rf_test_RMSE_mean)
  
} # closing brace for the function

boost_rf_1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 0.3040725
warnings() # no warnings for Boosted Random Forest function
```

### 6. Cubist


```r
library(Cubist)
#> Loading required package: lattice
library(tidyverse)

# Set initial values to 0

cubist_train_RMSE <- 0
cubist_test_RMSE <- 0
cubist_validation_RMSE <- 0
cubist_holdout_RMSE <- 0
cubist_test_predict_value <- 0

# Create the function:

cubist_1 <- function(data, colnum, train_amount, test_amount, numresamples){

#Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1, ]
test <- df[idx == 2, ]

# Fit the model on the training data, make predictions on the holdout data
cubist_train_fit <- Cubist::cubist(x = train[, 1:ncol(train) - 1], y = train$y)
cubist_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = cubist_train_fit, newdata = train))
cubist_train_RMSE_mean <- mean(cubist_train_RMSE)
cubist_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = cubist_train_fit, newdata = test))
cubist_test_RMSE_mean <- mean(cubist_test_RMSE)

} # closing braces for numresamples
  return(cubist_test_RMSE_mean)
  
} # closing braces for the function

cubist_1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 4.383964
warnings() # no warnings for individual cubist function
```

### 7. Elastic


```r

library(glmnet) # So we can run the elastic model
#> Loaded glmnet 4.1-8
library(tidyverse)

# Set initial values to 0

elastic_train_RMSE <- 0
elastic_test_RMSE <- 0
elastic_validation_RMSE <- 0
elastic_holdout_RMSE <- 0
elastic_test_predict_value <- 0
elastic_validation_predict_value <- 0
elastic_test_RMSE <- 0
elastic_test_RMSE_df <- data.frame(elastic_test_RMSE)
elastic_validation_RMSE <- 0
elastic_validation_RMSE_df <- data.frame(elastic_validation_RMSE)
elastic_holdout_RMSE <- 0
elastic_holdout_RMSE_df <- data.frame(elastic_holdout_RMSE)

# Create the function:
elastic_1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train, test and validation sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

# Set up the elastic model

y <- train$y
x <- data.matrix(train %>% dplyr::select(-y))
elastic_model <- glmnet::glmnet(x, y, alpha = 0.5)
elastic_cv <- cv.glmnet(x, y, alpha = 0.5)
best_elastic_lambda <- elastic_cv$lambda.min
best_elastic_model <- glmnet::glmnet(x, y, alpha = 0, lambda = best_elastic_lambda)
elastic_test_pred <- predict(best_elastic_model, s = best_elastic_lambda, newx = data.matrix(test %>% dplyr::select(-y)))

elastic_test_RMSE <- Metrics::rmse(actual = test$y, predicted = elastic_test_pred)
elastic_test_RMSE_df <- rbind(elastic_test_RMSE_df, elastic_test_RMSE)
elastic_test_RMSE_mean <- mean(elastic_test_RMSE_df$elastic_test_RMSE[2:nrow(elastic_test_RMSE_df)])

elastic_holdout_RMSE <- mean(elastic_test_RMSE_mean)
elastic_holdout_RMSE_df <- rbind(elastic_holdout_RMSE_df, elastic_holdout_RMSE)
elastic_holdout_RMSE_mean <- mean(elastic_holdout_RMSE_df$elastic_holdout_RMSE[2:nrow(elastic_holdout_RMSE_df)])

} # closing brace for numresample
  return(elastic_holdout_RMSE_mean)
  
} # closing brace for the elastic function

elastic_1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 4.926706
warnings() # no warnings for individual elastic function
```

### 8. Generalized Additive Models with smoothing splines


```r
library(gam) # for fitting generalized additive models
#> Loading required package: splines
#> Loading required package: foreach
#> 
#> Attaching package: 'foreach'
#> The following objects are masked from 'package:purrr':
#> 
#>     accumulate, when
#> Loaded gam 1.22-3

# Set initial values to 0

gam_train_RMSE <- 0
gam_test_RMSE <- 0
gam_holdout_RMSE <- 0
gam_test_predict_value <- 0

# Create the function:
gam1 <- function(data, colnum, train_amount, test_amount, numresamples){

# Set up random resampling

for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right

df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

# Set up to fit the model on the training data

n_unique_vals <- purrr::map_dbl(df, dplyr::n_distinct)

# Names of columns with >= 4 unique vals
keep <- names(n_unique_vals)[n_unique_vals >= 4]

gam_data <- df %>% dplyr::select(dplyr::all_of(keep))

# Model data

train1 <- train %>% dplyr::select(dplyr::all_of(keep))

test1 <- test %>% dplyr::select(dplyr::all_of(keep))

names_df <- names(gam_data[, 1:ncol(gam_data) - 1])
f2 <- stats::as.formula(paste0("y ~", paste0("gam::s(", names_df, ")", collapse = "+")))

gam_train_fit <- gam::gam(f2, data = train1)
gam_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = gam_train_fit, newdata = train))
gam_train_RMSE_mean <- mean(gam_train_RMSE)
gam_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = gam_train_fit, newdata = test))
gam_test_RMSE_mean <- mean(gam_test_RMSE)
gam_holdout_RMSE[i] <- mean(gam_test_RMSE_mean)
gam_holdout_RMSE_mean <- mean(gam_holdout_RMSE)

} # closing braces for numresamples
  return(gam_holdout_RMSE_mean)
  
} # closing braces for gam function

gam1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 4.880676
warnings() # no warnings for individual gam function
```

### 9. Gradient Boosted


```r
library(gbm) # to allow use of gradient boosted models
#> Loaded gbm 2.1.9
#> This version of gbm is no longer under development. Consider transitioning to gbm3, https://github.com/gbm-developers/gbm3

# Set initial values to 0
gb_train_RMSE <- 0
gb_test_RMSE <- 0
gb_validation_RMSE <- 0
gb_holdout_RMSE <- 0
gb_test_predict_value <- 0
gb_validation_predict_value <- 0

gb1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

gb_train_fit <- gbm::gbm(train$y ~ ., data = train, distribution = "gaussian", n.trees = 100, shrinkage = 0.1, interaction.depth = 10)
gb_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = gb_train_fit, newdata = train))
gb_train_RMSE_mean <- mean(gb_train_RMSE)
gb_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = gb_train_fit, newdata = test))
gb_test_RMSE_mean <- mean(gb_test_RMSE)

} # closing brace for numresamples
  return(gb_test_RMSE_mean)
  
} # closing brace for gb1 function

gb1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> Using 100 trees...
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> 
#> Using 100 trees...
#> [1] 3.527798
warnings() # no warnings for individual gradient boosted function
```

### 10. K-Nearest Neighbors (tuned)


```r

library(e1071)

# Set initial values to 0
knn_train_RMSE <- 0
knn_test_RMSE <- 0
knn_validation_RMSE <- 0
knn_holdout_RMSE <- 0
knn_test_predict_value <- 0
knn_validation_predict_value <- 0

knn1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y

y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right

df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

knn_train_fit <- e1071::tune.gknn(x = train[, 1:ncol(train) - 1], y = train$y, scale = TRUE, k = c(1:25))
knn_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict( object = knn_train_fit$best.model,
    newdata = train[, 1:ncol(train) - 1], k = knn_train_fit$best_model$k))
knn_train_RMSE_mean <- mean(knn_train_RMSE)
knn_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict( object = knn_train_fit$best.model,
    k = knn_train_fit$best_model$k, newdata = test[, 1:ncol(test) - 1]))
knn_test_RMSE_mean <- mean(knn_test_RMSE)
knn_holdout_RMSE[i] <- mean(c(knn_test_RMSE_mean))
knn_holdout_RMSE_mean <- mean(knn_holdout_RMSE)

} # closing brace for numresamples
  return(knn_holdout_RMSE_mean)
  
} # closing brace for knn1 function

knn1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 6.64997
warnings() # no warnings for individual knn function
```

### 11. Lasso


```r
library(glmnet) # So we can run the lasso model

# Set initial values to 0

lasso_train_RMSE <- 0
lasso_test_RMSE <- 0
lasso_validation_RMSE <- 0
lasso_holdout_RMSE <- 0
lasso_test_predict_value <- 0
lasso_validation_predict_value <- 0
lasso_test_RMSE <- 0
lasso_test_RMSE_df <- data.frame(lasso_test_RMSE)
lasso_validation_RMSE <- 0
lasso_validation_RMSE_df <- data.frame(lasso_validation_RMSE)
lasso_holdout_RMSE <- 0
lasso_holdout_RMSE_df <- data.frame(lasso_holdout_RMSE)

# Create the function:
lasso_1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1, ]
test <- df[idx == 2, ]

# Set up the lasso model
y <- train$y
x <- data.matrix(train %>% dplyr::select(-y))
lasso_model <- glmnet::glmnet(x, y, alpha = 1.0)
lasso_cv <- cv.glmnet(x, y, alpha = 1.0)
best_lasso_lambda <- lasso_cv$lambda.min
best_lasso_model <- glmnet::glmnet(x, y, alpha = 0, lambda = best_lasso_lambda)
lasso_test_pred <- predict(best_lasso_model, s = best_lasso_lambda, newx = data.matrix(test %>% dplyr::select(-y)))

lasso_test_RMSE <- Metrics::rmse(actual = test$y, predicted = lasso_test_pred)
lasso_test_RMSE_df <- rbind(lasso_test_RMSE_df, lasso_test_RMSE)
lasso_test_RMSE_mean <- mean(lasso_test_RMSE_df$lasso_test_RMSE[2:nrow(lasso_test_RMSE_df)])

lasso_holdout_RMSE <- mean(lasso_test_RMSE_mean)
lasso_holdout_RMSE_df <- rbind(lasso_holdout_RMSE_df, lasso_holdout_RMSE)
lasso_holdout_RMSE_mean <- mean(lasso_holdout_RMSE_df$lasso_holdout_RMSE[2:nrow(lasso_holdout_RMSE_df)])

} # closing brace for numresample
  return(lasso_holdout_RMSE_mean)
  
} # closing brace for the lasso_1 function

lasso_1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 4.843122
warnings() # no warnings for individual lasso function
```

### 12. Linear (tuned)


```r

library(e1071) # for tuned linear models

# Set initial values to 0
linear_train_RMSE <- 0
linear_test_RMSE <- 0
linear_holdout_RMSE <- 0

# Set up the function
linear1 <- function(data, colnum, train_amount, test_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

#Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

linear_train_fit <- e1071::tune.rpart(formula = y ~ ., data = train)
linear_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = linear_train_fit$best.model, newdata = train))
linear_train_RMSE_mean <- mean(linear_train_RMSE)
linear_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = linear_train_fit$best.model, newdata = test))
linear_holdout_RMSE_mean <- mean(linear_test_RMSE)

} # closing brace for numresamples
  return(linear_holdout_RMSE_mean)
  
} # closing brace for linear1 function

linear1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 4.798766
warnings() # no warnings for individual lasso function
```

### 13. LQS


```r

library(MASS) # to allow us to run LQS models

# Set initial values to 0

lqs_train_RMSE <- 0
lqs_test_RMSE <- 0
lqs_validation_RMSE <- 0
lqs_holdout_RMSE <- 0
lqs_test_predict_value <- 0
lqs_validation_predict_value <- 0

lqs1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

lqs_train_fit <- MASS::lqs(train$y ~ ., data = train)
lqs_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = lqs_train_fit, newdata = train))
lqs_train_RMSE_mean <- mean(lqs_train_RMSE)
lqs_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = lqs_train_fit, newdata = test))
lqs_test_RMSE_mean <- mean(lqs_test_RMSE)

y_hat_lqs <- c(lqs_test_predict_value, lqs_validation_predict_value)

} # Closing brace for numresamples
    return(lqs_test_RMSE_mean)

} # Closing brace for lqs1 function

lqs1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 7.011302
warnings() # no warnings for individual lqs function
```

### 14. Neuralnet


```r
library(neuralnet)
#> 
#> Attaching package: 'neuralnet'
#> The following object is masked from 'package:dplyr':
#> 
#>     compute

#Set initial values to 0

neuralnet_train_RMSE <- 0
neuralnet_test_RMSE <- 0
neuralnet_validation_RMSE <- 0
neuralnet_holdout_RMSE <- 0
neuralnet_test_predict_value <- 0
neuralnet_validation_predict_value <- 0

# Fit the model to the training data
neuralnet1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y

y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test data sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

maxs <- apply(df, 2, max)
mins <- apply(df, 2, min)
scaled <- as.data.frame(scale(df, center = mins, scale = maxs - mins))
train_ <- scaled[idx == 1, ]
test_ <- scaled[idx == 2, ]
n <- names(train_)
f <- as.formula(paste("y ~", paste(n[!n %in% "y"], collapse = " + ")))
nn <- neuralnet(f, data = train_, hidden = c(5, 3), linear.output = TRUE)
predict_test_nn <- neuralnet::compute(nn, test_[, 1:ncol(df) - 1])
predict_test_nn_ <- predict_test_nn$net.result * (max(df$y) - min(df$y)) + min(df$y)
predict_train_nn <- neuralnet::compute(nn, train_[, 1:ncol(df) - 1])
predict_train_nn_ <- predict_train_nn$net.result * (max(df$y) - min(df$y)) + min(df$y)
neuralnet_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict_train_nn_)
neuralnet_train_RMSE_mean <- mean(neuralnet_train_RMSE)
neuralnet_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict_test_nn_)
neuralnet_test_RMSE_mean <- mean(neuralnet_test_RMSE)

neuralnet_holdout_RMSE[i] <- mean(c(neuralnet_test_RMSE))
neuralnet_holdout_RMSE_mean <- mean(neuralnet_holdout_RMSE)

} # Closing brace for numresamples
  return(neuralnet_holdout_RMSE_mean)
  
} # closing brace for neuralnet1 function

neuralnet1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 4.140954
warnings() # no warnings for individual neuralnet function
```

### 15. Partial Least Squares


```r

library(pls)
#> 
#> Attaching package: 'pls'
#> The following objects are masked from 'package:arm':
#> 
#>     coefplot, corrplot
#> The following object is masked from 'package:stats':
#> 
#>     loadings

# Set initial values to 0
pls_train_RMSE <- 0
pls_test_RMSE <- 0
pls_validation_RMSE <- 0
pls_holdout_RMSE <- 0
pls_test_predict_value <- 0
pls_validation_predict_value <- 0

pls1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y

y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right

df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

pls_train_fit <- pls::plsr(train$y ~ ., data = train)
pls_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = pls_train_fit, newdata = train))
pls_train_RMSE_mean <- mean(pls_train_RMSE)
pls_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = pls_train_fit, newdata = test))
pls_test_RMSE_mean <- mean(pls_test_RMSE)

} # Closing brace for numresamples loop
  return( pls_test_RMSE_mean)
  
} # Closing brace for pls1 function

pls1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 6.106539
warnings() # no warnings for individual pls function
```

### 16. Principal Components Regression


```r

library(pls) # To run pcr models

#Set initial values to 0
pcr_train_RMSE <- 0
pcr_test_RMSE <- 0
pcr_validation_RMSE <- 0
pcr_holdout_RMSE <- 0
pcr_test_predict_value <- 0
pcr_validation_predict_value <- 0

pcr1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y

y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

pcr_train_fit <- pls::pcr(train$y ~ ., data = train)
pcr_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = pcr_train_fit, newdata = train))
pcr_train_RMSE_mean <- mean(pcr_train_RMSE)
pcr_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = pcr_train_fit, newdata = test))
pcr_test_RMSE_mean <- mean(pcr_test_RMSE)

} # Closing brace for numresamples loop
  return(pcr_test_RMSE_mean)
  
} # Closing brace for PCR function

pcr1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 6.715986
warnings() # no warnings for individual pls function
```

### 17. Random Forest


```r
library(randomForest)

# Set initial values to 0
rf_train_RMSE <- 0
rf_test_RMSE <- 0
rf_validation_RMSE <- 0
rf_holdout_RMSE <- 0
rf_test_predict_value <- 0
rf_validation_predict_value <- 0

# Set up the function
rf1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

#Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train, test and validation sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

rf_train_fit <- tune.randomForest(x = train, y = train$y, data = train)
rf_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = rf_train_fit$best.model, newdata = train))
rf_train_RMSE_mean <- mean(rf_train_RMSE)
rf_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = rf_train_fit$best.model, newdata = test))
rf_test_RMSE_mean <- mean(rf_test_RMSE)

} # Closing brace for numresamples loop
return(rf_test_RMSE_mean)
  
} # Closing brace for rf1 function

rf1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 1.779677
warnings() # no warnings for individual random forest function
```

### 18. Ridge Regression


```r

library(glmnet) # So we can run the ridge model

# Set initial values to 0
ridge_train_RMSE <- 0
ridge_test_RMSE <- 0
ridge_validation_RMSE <- 0
ridge_holdout_RMSE <- 0
ridge_test_predict_value <- 0
ridge_validation_predict_value <- 0
ridge_test_RMSE <- 0
ridge_test_RMSE_df <- data.frame(ridge_test_RMSE)
ridge_validation_RMSE <- 0
ridge_validation_RMSE_df <- data.frame(ridge_validation_RMSE)
ridge_holdout_RMSE <- 0
ridge_holdout_RMSE_df <- data.frame(ridge_holdout_RMSE)

# Create the function:
ridge1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train, test and validation sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1, ]
test <- df[idx == 2, ]

# Set up the ridge model
y <- train$y
x <- data.matrix(train %>% dplyr::select(-y))
ridge_model <- glmnet::glmnet(x, y, alpha = 0)
ridge_cv <- cv.glmnet(x, y, alpha = 0)
best_ridge_lambda <- ridge_cv$lambda.min
best_ridge_model <- glmnet::glmnet(x, y, alpha = 0, lambda = best_ridge_lambda)
ridge_test_pred <- predict(best_ridge_model, s = best_ridge_lambda, newx = data.matrix(test %>% dplyr::select(-y)))

ridge_test_RMSE <- Metrics::rmse(actual = test$y, predicted = ridge_test_pred)
ridge_test_RMSE_df <- rbind(ridge_test_RMSE_df, ridge_test_RMSE)
ridge_test_RMSE_mean <- mean(ridge_test_RMSE_df$ridge_test_RMSE[2:nrow(ridge_test_RMSE_df)])

ridge_holdout_RMSE <- mean(ridge_test_RMSE_mean)
ridge_holdout_RMSE_df <- rbind(ridge_holdout_RMSE_df, ridge_holdout_RMSE)
ridge_holdout_RMSE_mean <- mean(ridge_holdout_RMSE_df$ridge_holdout_RMSE[2:nrow(ridge_holdout_RMSE_df)])

} # closing brace for numresample
  return(ridge_holdout_RMSE_mean)
  
} # closing brace for the ridge function

ridge1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 5.043585
warnings() # no warnings for individual ridge function
```

### 19. Robust Regression


```r

library(MASS) # To run rlm function for robust regression

# Set initial values to 0
robust_train_RMSE <- 0
robust_test_RMSE <- 0
robust_validation_RMSE <- 0
robust_holdout_RMSE <- 0
robust_test_predict_value <- 0
robust_validation_predict_value <- 0

# Make the function
robust1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right

df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

robust_train_fit <- MASS::rlm(x = train[, 1:ncol(df) - 1], y = train$y)
robust_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = robust_train_fit$fitted.values)
robust_train_RMSE_mean <- mean(robust_train_RMSE)
robust_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = MASS::rlm(y ~ ., data = train), newdata = test))
robust_test_RMSE_mean <- mean(robust_test_RMSE) 

} # Closing brace for numresamples loop
return(robust_test_RMSE_mean)
  
} # Closing brace for robust1 function

robust1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> Warning in rlm.default(x = train[, 1:ncol(df) - 1], y =
#> train$y): 'rlm' failed to converge in 20 steps

#> Warning in rlm.default(x = train[, 1:ncol(df) - 1], y =
#> train$y): 'rlm' failed to converge in 20 steps
#> [1] 4.952534
warnings() # no warnings for individual robust function
```

### 20. Rpart


```r

library(rpart)

# Set initial values to 0
rpart_train_RMSE <- 0
rpart_test_RMSE <- 0
rpart_validation_RMSE <- 0
rpart_holdout_RMSE <- 0
rpart_test_predict_value <- 0
rpart_validation_predict_value <- 0

# Make the function
rpart1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

rpart_train_fit <- rpart::rpart(train$y ~ ., data = train)
rpart_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = rpart_train_fit, newdata = train))
rpart_train_RMSE_mean <- mean(rpart_train_RMSE)
rpart_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = rpart_train_fit, newdata = test))
rpart_test_RMSE_mean <- mean(rpart_test_RMSE)

} # Closing loop for numresamples
return(rpart_test_RMSE_mean)
  
} # Closing brace for rpart1 function

rpart1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 4.919528
warnings() # no warnings for individual rpart function
```

### 21. Support Vector Machines


```r

library(e1071)

# Set initial values to 0
svm_train_RMSE <- 0
svm_test_RMSE <- 0
svm_validation_RMSE <- 0
svm_holdout_RMSE <- 0
svm_test_predict_value <- 0
svm_validation_predict_value <- 0

# Make the function
svm1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

svm_train_fit <- e1071::tune.svm(x = train, y = train$y, data = train)
svm_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = svm_train_fit$best.model, newdata = train))
svm_train_RMSE_mean <- mean(svm_train_RMSE)
svm_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = svm_train_fit$best.model, newdata = test))
svm_test_RMSE_mean <- mean(svm_test_RMSE)

} # Closing brace for numresamples loop
  return(svm_test_RMSE_mean)

} # Closing brace for svm1 function

svm1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 2.288686
warnings() # no warnings for individual Support Vector Machines function
```

### 22. Trees


```r

library(tree)

# Set initial values to 0

tree_train_RMSE <- 0
tree_test_RMSE <- 0
tree_validation_RMSE <- 0
tree_holdout_RMSE <- 0
tree_test_predict_value <- 0
tree_validation_predict_value <- 0

# Make the function
tree1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

tree_train_fit <- tree::tree(train$y ~ ., data = train)
tree_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = tree_train_fit, newdata = train))
tree_train_RMSE_mean <- mean(tree_train_RMSE)
tree_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = tree_train_fit, newdata = test))
tree_test_RMSE_mean <- mean(tree_test_RMSE)

} # Closing brace for numresamples loop
  return(tree_test_RMSE_mean)
  
} # Closing brace for tree1 function

tree1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 5.019375
warnings() # no warnings for individual tree function
```

### 23. XGBoost


```r
library(xgboost)
#> 
#> Attaching package: 'xgboost'
#> The following object is masked from 'package:dplyr':
#> 
#>     slice

# Set initial values to 0
xgb_train_RMSE <- 0
xgb_test_RMSE <- 0
xgb_validation_RMSE <- 0
xgb_holdout_RMSE <- 0
xgb_test_predict_value <- 0
xgb_validation_predict_value <- 0

# Create the function
xgb1 <- function(data, colnum, train_amount, test_amount, validation_amount, numresamples){

# Set up random resampling
for (i in 1:numresamples) {

# Changes the name of the target column to y
y <- 0
colnames(data)[colnum] <- "y"

# Moves the target column to the last column on the right
df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right df <- df[sample(nrow(df)), ] # randomizes the rows

# Breaks the data into train and test sets
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
train <- df[idx == 1,]
test <- df[idx == 2, ]

train_x <- data.matrix(train[, -ncol(train)])
train_y <- train[, ncol(train)]

# define predictor and response variables in test set
test_x <- data.matrix(test[, -ncol(test)])
test_y <- test[, ncol(test)]

# define final train, test and validation sets

xgb_train <- xgboost::xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgboost::xgb.DMatrix(data = test_x, label = test_y)

# define watchlist
watchlist <- list(train = xgb_train)
watchlist_test <- list(train = xgb_train, test = xgb_test)

# fit XGBoost model and display training and validation data at each round
xgb_model <- xgboost::xgb.train(data = xgb_train, max.depth = 3, watchlist = watchlist_test, nrounds = 70)

xgboost_min <- which.min(xgb_model$evaluation_log$validation_rmse)

xgb_train_RMSE[i] <- Metrics::rmse(actual = train$y, predicted = predict(object = xgb_model, newdata = train_x))
xgb_train_RMSE_mean <- mean(xgb_train_RMSE)
xgb_test_RMSE[i] <- Metrics::rmse(actual = test$y, predicted = predict(object = xgb_model, newdata = test_x))
xgb_test_RMSE_mean <- mean(xgb_test_RMSE)

xgb_holdout_RMSE[i] <- mean(xgb_test_RMSE_mean)
xgb_holdout_RMSE_mean <- mean(xgb_holdout_RMSE)

} # Closing brace for numresamples loop
  return(xgb_holdout_RMSE_mean)
  
} # Closing brace for xgb1 function

xgb1(data = MASS::Boston, colnum = 14, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1]	train-rmse:17.486497	test-rmse:16.481208 
#> [2]	train-rmse:12.680142	test-rmse:11.986695 
#> [3]	train-rmse:9.347974	test-rmse:8.749361 
#> [4]	train-rmse:7.008353	test-rmse:6.675755 
#> [5]	train-rmse:5.419391	test-rmse:5.367051 
#> [6]	train-rmse:4.361966	test-rmse:4.561522 
#> [7]	train-rmse:3.624947	test-rmse:4.032414 
#> [8]	train-rmse:3.160847	test-rmse:3.820689 
#> [9]	train-rmse:2.824743	test-rmse:3.610561 
#> [10]	train-rmse:2.619698	test-rmse:3.507650 
#> [11]	train-rmse:2.446987	test-rmse:3.484689 
#> [12]	train-rmse:2.329737	test-rmse:3.439376 
#> [13]	train-rmse:2.240210	test-rmse:3.430371 
#> [14]	train-rmse:2.145244	test-rmse:3.391075 
#> [15]	train-rmse:2.084449	test-rmse:3.360280 
#> [16]	train-rmse:2.037245	test-rmse:3.379184 
#> [17]	train-rmse:2.002897	test-rmse:3.366566 
#> [18]	train-rmse:1.945952	test-rmse:3.376486 
#> [19]	train-rmse:1.897742	test-rmse:3.367665 
#> [20]	train-rmse:1.854231	test-rmse:3.348448 
#> [21]	train-rmse:1.822154	test-rmse:3.349097 
#> [22]	train-rmse:1.787204	test-rmse:3.304920 
#> [23]	train-rmse:1.754270	test-rmse:3.293085 
#> [24]	train-rmse:1.709816	test-rmse:3.272922 
#> [25]	train-rmse:1.674145	test-rmse:3.269630 
#> [26]	train-rmse:1.658251	test-rmse:3.258794 
#> [27]	train-rmse:1.641139	test-rmse:3.242577 
#> [28]	train-rmse:1.614929	test-rmse:3.240223 
#> [29]	train-rmse:1.567291	test-rmse:3.240452 
#> [30]	train-rmse:1.505544	test-rmse:3.201182 
#> [31]	train-rmse:1.476929	test-rmse:3.205659 
#> [32]	train-rmse:1.449647	test-rmse:3.194901 
#> [33]	train-rmse:1.416530	test-rmse:3.203879 
#> [34]	train-rmse:1.374986	test-rmse:3.187995 
#> [35]	train-rmse:1.351330	test-rmse:3.190713 
#> [36]	train-rmse:1.316853	test-rmse:3.191331 
#> [37]	train-rmse:1.309591	test-rmse:3.182175 
#> [38]	train-rmse:1.294090	test-rmse:3.175570 
#> [39]	train-rmse:1.277757	test-rmse:3.180537 
#> [40]	train-rmse:1.242837	test-rmse:3.186998 
#> [41]	train-rmse:1.216383	test-rmse:3.187467 
#> [42]	train-rmse:1.189996	test-rmse:3.184045 
#> [43]	train-rmse:1.154658	test-rmse:3.159547 
#> [44]	train-rmse:1.127478	test-rmse:3.139461 
#> [45]	train-rmse:1.115798	test-rmse:3.134894 
#> [46]	train-rmse:1.104211	test-rmse:3.128842 
#> [47]	train-rmse:1.071158	test-rmse:3.123660 
#> [48]	train-rmse:1.054669	test-rmse:3.138948 
#> [49]	train-rmse:1.046175	test-rmse:3.131209 
#> [50]	train-rmse:1.027679	test-rmse:3.124061 
#> [51]	train-rmse:1.014130	test-rmse:3.101524 
#> [52]	train-rmse:0.993697	test-rmse:3.111012 
#> [53]	train-rmse:0.971893	test-rmse:3.107055 
#> [54]	train-rmse:0.950926	test-rmse:3.110510 
#> [55]	train-rmse:0.941455	test-rmse:3.112877 
#> [56]	train-rmse:0.933776	test-rmse:3.106882 
#> [57]	train-rmse:0.929408	test-rmse:3.108469 
#> [58]	train-rmse:0.923074	test-rmse:3.107276 
#> [59]	train-rmse:0.905856	test-rmse:3.098320 
#> [60]	train-rmse:0.899052	test-rmse:3.094325 
#> [61]	train-rmse:0.885400	test-rmse:3.100936 
#> [62]	train-rmse:0.880479	test-rmse:3.099746 
#> [63]	train-rmse:0.865969	test-rmse:3.094835 
#> [64]	train-rmse:0.844476	test-rmse:3.100751 
#> [65]	train-rmse:0.820401	test-rmse:3.091629 
#> [66]	train-rmse:0.805832	test-rmse:3.097608 
#> [67]	train-rmse:0.789846	test-rmse:3.092611 
#> [68]	train-rmse:0.777879	test-rmse:3.098650 
#> [69]	train-rmse:0.765226	test-rmse:3.102803 
#> [70]	train-rmse:0.760578	test-rmse:3.103667 
#> [1]	train-rmse:17.271000	test-rmse:16.919832 
#> [2]	train-rmse:12.515867	test-rmse:12.332933 
#> [3]	train-rmse:9.176305	test-rmse:9.186107 
#> [4]	train-rmse:6.859001	test-rmse:7.174320 
#> [5]	train-rmse:5.312209	test-rmse:5.824901 
#> [6]	train-rmse:4.221931	test-rmse:5.080645 
#> [7]	train-rmse:3.523406	test-rmse:4.570028 
#> [8]	train-rmse:3.052697	test-rmse:4.377020 
#> [9]	train-rmse:2.707503	test-rmse:4.246629 
#> [10]	train-rmse:2.520544	test-rmse:4.196648 
#> [11]	train-rmse:2.365464	test-rmse:4.141710 
#> [12]	train-rmse:2.269290	test-rmse:4.131828 
#> [13]	train-rmse:2.185941	test-rmse:4.165351 
#> [14]	train-rmse:2.101735	test-rmse:4.138577 
#> [15]	train-rmse:2.024846	test-rmse:4.137209 
#> [16]	train-rmse:1.993709	test-rmse:4.152307 
#> [17]	train-rmse:1.945708	test-rmse:4.175986 
#> [18]	train-rmse:1.847535	test-rmse:4.171342 
#> [19]	train-rmse:1.807317	test-rmse:4.181224 
#> [20]	train-rmse:1.769414	test-rmse:4.178264 
#> [21]	train-rmse:1.731118	test-rmse:4.161658 
#> [22]	train-rmse:1.700002	test-rmse:4.176502 
#> [23]	train-rmse:1.676733	test-rmse:4.192347 
#> [24]	train-rmse:1.640807	test-rmse:4.199664 
#> [25]	train-rmse:1.619645	test-rmse:4.182642 
#> [26]	train-rmse:1.585201	test-rmse:4.188219 
#> [27]	train-rmse:1.538859	test-rmse:4.193522 
#> [28]	train-rmse:1.516105	test-rmse:4.200009 
#> [29]	train-rmse:1.471954	test-rmse:4.171491 
#> [30]	train-rmse:1.447593	test-rmse:4.184000 
#> [31]	train-rmse:1.410964	test-rmse:4.163903 
#> [32]	train-rmse:1.385057	test-rmse:4.159180 
#> [33]	train-rmse:1.370980	test-rmse:4.157527 
#> [34]	train-rmse:1.342666	test-rmse:4.163018 
#> [35]	train-rmse:1.305707	test-rmse:4.167307 
#> [36]	train-rmse:1.276964	test-rmse:4.157421 
#> [37]	train-rmse:1.260409	test-rmse:4.156494 
#> [38]	train-rmse:1.247981	test-rmse:4.153613 
#> [39]	train-rmse:1.206421	test-rmse:4.146320 
#> [40]	train-rmse:1.192975	test-rmse:4.150320 
#> [41]	train-rmse:1.181422	test-rmse:4.152663 
#> [42]	train-rmse:1.163184	test-rmse:4.153904 
#> [43]	train-rmse:1.129128	test-rmse:4.171885 
#> [44]	train-rmse:1.117972	test-rmse:4.162864 
#> [45]	train-rmse:1.089349	test-rmse:4.159964 
#> [46]	train-rmse:1.071597	test-rmse:4.175901 
#> [47]	train-rmse:1.049902	test-rmse:4.172292 
#> [48]	train-rmse:1.033387	test-rmse:4.176787 
#> [49]	train-rmse:1.025796	test-rmse:4.179763 
#> [50]	train-rmse:1.014940	test-rmse:4.177002 
#> [51]	train-rmse:0.988000	test-rmse:4.167849 
#> [52]	train-rmse:0.968624	test-rmse:4.168490 
#> [53]	train-rmse:0.954143	test-rmse:4.184151 
#> [54]	train-rmse:0.938822	test-rmse:4.176074 
#> [55]	train-rmse:0.924480	test-rmse:4.195964 
#> [56]	train-rmse:0.915237	test-rmse:4.191507 
#> [57]	train-rmse:0.896332	test-rmse:4.198041 
#> [58]	train-rmse:0.888031	test-rmse:4.190549 
#> [59]	train-rmse:0.878456	test-rmse:4.190976 
#> [60]	train-rmse:0.866349	test-rmse:4.188612 
#> [61]	train-rmse:0.850429	test-rmse:4.171399 
#> [62]	train-rmse:0.838005	test-rmse:4.171440 
#> [63]	train-rmse:0.818213	test-rmse:4.164876 
#> [64]	train-rmse:0.812399	test-rmse:4.160797 
#> [65]	train-rmse:0.805299	test-rmse:4.160084 
#> [66]	train-rmse:0.785106	test-rmse:4.159871 
#> [67]	train-rmse:0.765562	test-rmse:4.169879 
#> [68]	train-rmse:0.752423	test-rmse:4.161290 
#> [69]	train-rmse:0.744580	test-rmse:4.158569 
#> [70]	train-rmse:0.737120	test-rmse:4.149397 
#> [1]	train-rmse:17.395489	test-rmse:16.510984 
#> [2]	train-rmse:12.561541	test-rmse:12.113637 
#> [3]	train-rmse:9.194729	test-rmse:9.024100 
#> [4]	train-rmse:6.871181	test-rmse:6.819280 
#> [5]	train-rmse:5.283668	test-rmse:5.467459 
#> [6]	train-rmse:4.159390	test-rmse:4.582748 
#> [7]	train-rmse:3.437048	test-rmse:4.007805 
#> [8]	train-rmse:2.938627	test-rmse:3.730191 
#> [9]	train-rmse:2.607829	test-rmse:3.559813 
#> [10]	train-rmse:2.387627	test-rmse:3.441103 
#> [11]	train-rmse:2.259817	test-rmse:3.354965 
#> [12]	train-rmse:2.163043	test-rmse:3.318793 
#> [13]	train-rmse:2.090711	test-rmse:3.260732 
#> [14]	train-rmse:2.001951	test-rmse:3.229978 
#> [15]	train-rmse:1.947110	test-rmse:3.248306 
#> [16]	train-rmse:1.877404	test-rmse:3.216982 
#> [17]	train-rmse:1.825344	test-rmse:3.195938 
#> [18]	train-rmse:1.776860	test-rmse:3.173275 
#> [19]	train-rmse:1.749610	test-rmse:3.168493 
#> [20]	train-rmse:1.696357	test-rmse:3.144102 
#> [21]	train-rmse:1.670021	test-rmse:3.126842 
#> [22]	train-rmse:1.643466	test-rmse:3.105513 
#> [23]	train-rmse:1.596573	test-rmse:3.110415 
#> [24]	train-rmse:1.576580	test-rmse:3.104983 
#> [25]	train-rmse:1.563637	test-rmse:3.099317 
#> [26]	train-rmse:1.540325	test-rmse:3.092570 
#> [27]	train-rmse:1.530043	test-rmse:3.093045 
#> [28]	train-rmse:1.509687	test-rmse:3.106363 
#> [29]	train-rmse:1.481246	test-rmse:3.082575 
#> [30]	train-rmse:1.449740	test-rmse:3.083743 
#> [31]	train-rmse:1.425405	test-rmse:3.098682 
#> [32]	train-rmse:1.382464	test-rmse:3.101694 
#> [33]	train-rmse:1.363775	test-rmse:3.083235 
#> [34]	train-rmse:1.349192	test-rmse:3.075221 
#> [35]	train-rmse:1.323078	test-rmse:3.076637 
#> [36]	train-rmse:1.314223	test-rmse:3.073725 
#> [37]	train-rmse:1.303232	test-rmse:3.069449 
#> [38]	train-rmse:1.284872	test-rmse:3.079453 
#> [39]	train-rmse:1.263972	test-rmse:3.068180 
#> [40]	train-rmse:1.239298	test-rmse:3.084689 
#> [41]	train-rmse:1.213229	test-rmse:3.096412 
#> [42]	train-rmse:1.202738	test-rmse:3.099342 
#> [43]	train-rmse:1.181719	test-rmse:3.097213 
#> [44]	train-rmse:1.175878	test-rmse:3.091491 
#> [45]	train-rmse:1.156718	test-rmse:3.087315 
#> [46]	train-rmse:1.144089	test-rmse:3.095235 
#> [47]	train-rmse:1.130485	test-rmse:3.094214 
#> [48]	train-rmse:1.100811	test-rmse:3.071416 
#> [49]	train-rmse:1.085433	test-rmse:3.068388 
#> [50]	train-rmse:1.053607	test-rmse:3.061209 
#> [51]	train-rmse:1.038903	test-rmse:3.052623 
#> [52]	train-rmse:1.014005	test-rmse:3.035290 
#> [53]	train-rmse:0.996117	test-rmse:3.040920 
#> [54]	train-rmse:0.968683	test-rmse:3.043354 
#> [55]	train-rmse:0.947833	test-rmse:3.020012 
#> [56]	train-rmse:0.935359	test-rmse:3.012993 
#> [57]	train-rmse:0.917079	test-rmse:3.002739 
#> [58]	train-rmse:0.901128	test-rmse:3.009477 
#> [59]	train-rmse:0.887161	test-rmse:2.998941 
#> [60]	train-rmse:0.878894	test-rmse:2.993458 
#> [61]	train-rmse:0.870364	test-rmse:2.988974 
#> [62]	train-rmse:0.854939	test-rmse:2.986013 
#> [63]	train-rmse:0.847620	test-rmse:2.983280 
#> [64]	train-rmse:0.844183	test-rmse:2.980246 
#> [65]	train-rmse:0.834523	test-rmse:2.972377 
#> [66]	train-rmse:0.820483	test-rmse:2.973813 
#> [67]	train-rmse:0.811202	test-rmse:2.962869 
#> [68]	train-rmse:0.795636	test-rmse:2.959582 
#> [69]	train-rmse:0.778703	test-rmse:2.950396 
#> [70]	train-rmse:0.768674	test-rmse:2.959586 
#> [1]	train-rmse:17.380206	test-rmse:17.031772 
#> [2]	train-rmse:12.588357	test-rmse:12.494238 
#> [3]	train-rmse:9.244460	test-rmse:9.391844 
#> [4]	train-rmse:6.930907	test-rmse:7.257440 
#> [5]	train-rmse:5.306428	test-rmse:5.935620 
#> [6]	train-rmse:4.212014	test-rmse:5.070906 
#> [7]	train-rmse:3.480988	test-rmse:4.427449 
#> [8]	train-rmse:2.971855	test-rmse:4.134873 
#> [9]	train-rmse:2.632878	test-rmse:3.940482 
#> [10]	train-rmse:2.428489	test-rmse:3.843724 
#> [11]	train-rmse:2.242322	test-rmse:3.705470 
#> [12]	train-rmse:2.134822	test-rmse:3.654557 
#> [13]	train-rmse:2.039841	test-rmse:3.597136 
#> [14]	train-rmse:1.959057	test-rmse:3.540029 
#> [15]	train-rmse:1.875066	test-rmse:3.546649 
#> [16]	train-rmse:1.829356	test-rmse:3.556482 
#> [17]	train-rmse:1.778048	test-rmse:3.541097 
#> [18]	train-rmse:1.748775	test-rmse:3.534900 
#> [19]	train-rmse:1.681980	test-rmse:3.511449 
#> [20]	train-rmse:1.653488	test-rmse:3.488497 
#> [21]	train-rmse:1.634753	test-rmse:3.488181 
#> [22]	train-rmse:1.588483	test-rmse:3.465585 
#> [23]	train-rmse:1.549751	test-rmse:3.468146 
#> [24]	train-rmse:1.521693	test-rmse:3.459599 
#> [25]	train-rmse:1.484244	test-rmse:3.458086 
#> [26]	train-rmse:1.461672	test-rmse:3.469976 
#> [27]	train-rmse:1.428051	test-rmse:3.464792 
#> [28]	train-rmse:1.397826	test-rmse:3.483781 
#> [29]	train-rmse:1.370554	test-rmse:3.475009 
#> [30]	train-rmse:1.344898	test-rmse:3.475376 
#> [31]	train-rmse:1.330812	test-rmse:3.474912 
#> [32]	train-rmse:1.318602	test-rmse:3.481810 
#> [33]	train-rmse:1.289089	test-rmse:3.468368 
#> [34]	train-rmse:1.267479	test-rmse:3.470361 
#> [35]	train-rmse:1.250811	test-rmse:3.467983 
#> [36]	train-rmse:1.230678	test-rmse:3.485372 
#> [37]	train-rmse:1.209182	test-rmse:3.490498 
#> [38]	train-rmse:1.198441	test-rmse:3.480437 
#> [39]	train-rmse:1.182828	test-rmse:3.484867 
#> [40]	train-rmse:1.172483	test-rmse:3.487583 
#> [41]	train-rmse:1.144719	test-rmse:3.484114 
#> [42]	train-rmse:1.120774	test-rmse:3.478865 
#> [43]	train-rmse:1.094923	test-rmse:3.470110 
#> [44]	train-rmse:1.082106	test-rmse:3.477357 
#> [45]	train-rmse:1.049023	test-rmse:3.470272 
#> [46]	train-rmse:1.023616	test-rmse:3.466944 
#> [47]	train-rmse:1.016039	test-rmse:3.466201 
#> [48]	train-rmse:0.992702	test-rmse:3.473398 
#> [49]	train-rmse:0.969921	test-rmse:3.472677 
#> [50]	train-rmse:0.959166	test-rmse:3.483063 
#> [51]	train-rmse:0.937026	test-rmse:3.475435 
#> [52]	train-rmse:0.919188	test-rmse:3.464488 
#> [53]	train-rmse:0.907835	test-rmse:3.460747 
#> [54]	train-rmse:0.888932	test-rmse:3.459577 
#> [55]	train-rmse:0.883432	test-rmse:3.455239 
#> [56]	train-rmse:0.869079	test-rmse:3.452456 
#> [57]	train-rmse:0.857464	test-rmse:3.453599 
#> [58]	train-rmse:0.843735	test-rmse:3.448456 
#> [59]	train-rmse:0.830671	test-rmse:3.445266 
#> [60]	train-rmse:0.827228	test-rmse:3.443131 
#> [61]	train-rmse:0.811642	test-rmse:3.438849 
#> [62]	train-rmse:0.785364	test-rmse:3.432073 
#> [63]	train-rmse:0.776901	test-rmse:3.436418 
#> [64]	train-rmse:0.769347	test-rmse:3.437193 
#> [65]	train-rmse:0.751765	test-rmse:3.433901 
#> [66]	train-rmse:0.738440	test-rmse:3.432187 
#> [67]	train-rmse:0.726259	test-rmse:3.429399 
#> [68]	train-rmse:0.719352	test-rmse:3.424938 
#> [69]	train-rmse:0.714389	test-rmse:3.432535 
#> [70]	train-rmse:0.709031	test-rmse:3.433105 
#> [1]	train-rmse:16.984875	test-rmse:17.697009 
#> [2]	train-rmse:12.324817	test-rmse:12.953677 
#> [3]	train-rmse:9.045915	test-rmse:9.644390 
#> [4]	train-rmse:6.784900	test-rmse:7.500049 
#> [5]	train-rmse:5.253843	test-rmse:6.065439 
#> [6]	train-rmse:4.178446	test-rmse:5.162901 
#> [7]	train-rmse:3.480812	test-rmse:4.724464 
#> [8]	train-rmse:3.021814	test-rmse:4.466702 
#> [9]	train-rmse:2.700977	test-rmse:4.238377 
#> [10]	train-rmse:2.466828	test-rmse:4.115037 
#> [11]	train-rmse:2.308248	test-rmse:4.046978 
#> [12]	train-rmse:2.200536	test-rmse:4.003357 
#> [13]	train-rmse:2.077451	test-rmse:3.937996 
#> [14]	train-rmse:1.993869	test-rmse:3.915808 
#> [15]	train-rmse:1.935727	test-rmse:3.875300 
#> [16]	train-rmse:1.880466	test-rmse:3.874977 
#> [17]	train-rmse:1.844265	test-rmse:3.858995 
#> [18]	train-rmse:1.806236	test-rmse:3.880547 
#> [19]	train-rmse:1.746038	test-rmse:3.860371 
#> [20]	train-rmse:1.698725	test-rmse:3.859303 
#> [21]	train-rmse:1.677481	test-rmse:3.847714 
#> [22]	train-rmse:1.644719	test-rmse:3.846820 
#> [23]	train-rmse:1.619180	test-rmse:3.864139 
#> [24]	train-rmse:1.586591	test-rmse:3.864736 
#> [25]	train-rmse:1.535714	test-rmse:3.855715 
#> [26]	train-rmse:1.523442	test-rmse:3.848466 
#> [27]	train-rmse:1.493008	test-rmse:3.820469 
#> [28]	train-rmse:1.468519	test-rmse:3.814175 
#> [29]	train-rmse:1.455317	test-rmse:3.819515 
#> [30]	train-rmse:1.422747	test-rmse:3.834053 
#> [31]	train-rmse:1.366023	test-rmse:3.834437 
#> [32]	train-rmse:1.331921	test-rmse:3.818818 
#> [33]	train-rmse:1.302149	test-rmse:3.814104 
#> [34]	train-rmse:1.288872	test-rmse:3.815516 
#> [35]	train-rmse:1.276577	test-rmse:3.801725 
#> [36]	train-rmse:1.253637	test-rmse:3.793647 
#> [37]	train-rmse:1.237153	test-rmse:3.813853 
#> [38]	train-rmse:1.230567	test-rmse:3.813858 
#> [39]	train-rmse:1.219340	test-rmse:3.813696 
#> [40]	train-rmse:1.206634	test-rmse:3.811301 
#> [41]	train-rmse:1.178934	test-rmse:3.817181 
#> [42]	train-rmse:1.153515	test-rmse:3.819049 
#> [43]	train-rmse:1.137846	test-rmse:3.813922 
#> [44]	train-rmse:1.120400	test-rmse:3.797407 
#> [45]	train-rmse:1.110760	test-rmse:3.797720 
#> [46]	train-rmse:1.102493	test-rmse:3.794102 
#> [47]	train-rmse:1.082844	test-rmse:3.787301 
#> [48]	train-rmse:1.066238	test-rmse:3.797389 
#> [49]	train-rmse:1.041964	test-rmse:3.800018 
#> [50]	train-rmse:1.029571	test-rmse:3.801595 
#> [51]	train-rmse:1.014837	test-rmse:3.802143 
#> [52]	train-rmse:0.986400	test-rmse:3.804814 
#> [53]	train-rmse:0.982557	test-rmse:3.799498 
#> [54]	train-rmse:0.972485	test-rmse:3.806163 
#> [55]	train-rmse:0.956978	test-rmse:3.808189 
#> [56]	train-rmse:0.933708	test-rmse:3.801178 
#> [57]	train-rmse:0.918174	test-rmse:3.798300 
#> [58]	train-rmse:0.906931	test-rmse:3.798485 
#> [59]	train-rmse:0.902619	test-rmse:3.801572 
#> [60]	train-rmse:0.898097	test-rmse:3.797797 
#> [61]	train-rmse:0.889969	test-rmse:3.790598 
#> [62]	train-rmse:0.874802	test-rmse:3.789076 
#> [63]	train-rmse:0.858111	test-rmse:3.781398 
#> [64]	train-rmse:0.847368	test-rmse:3.782311 
#> [65]	train-rmse:0.840788	test-rmse:3.784421 
#> [66]	train-rmse:0.824566	test-rmse:3.786287 
#> [67]	train-rmse:0.820674	test-rmse:3.785419 
#> [68]	train-rmse:0.806288	test-rmse:3.782655 
#> [69]	train-rmse:0.794688	test-rmse:3.777987 
#> [70]	train-rmse:0.787247	test-rmse:3.781246 
#> [1]	train-rmse:17.713810	test-rmse:16.241058 
#> [2]	train-rmse:12.828331	test-rmse:11.856221 
#> [3]	train-rmse:9.419469	test-rmse:8.836258 
#> [4]	train-rmse:7.026210	test-rmse:6.923035 
#> [5]	train-rmse:5.389105	test-rmse:5.719839 
#> [6]	train-rmse:4.278109	test-rmse:5.031750 
#> [7]	train-rmse:3.544784	test-rmse:4.677846 
#> [8]	train-rmse:3.083702	test-rmse:4.488374 
#> [9]	train-rmse:2.784028	test-rmse:4.375848 
#> [10]	train-rmse:2.607710	test-rmse:4.356227 
#> [11]	train-rmse:2.457488	test-rmse:4.225116 
#> [12]	train-rmse:2.308632	test-rmse:4.120038 
#> [13]	train-rmse:2.213545	test-rmse:4.118951 
#> [14]	train-rmse:2.101512	test-rmse:4.099988 
#> [15]	train-rmse:2.064077	test-rmse:4.084281 
#> [16]	train-rmse:1.992316	test-rmse:4.112185 
#> [17]	train-rmse:1.941101	test-rmse:4.068031 
#> [18]	train-rmse:1.889634	test-rmse:4.047856 
#> [19]	train-rmse:1.837156	test-rmse:4.047730 
#> [20]	train-rmse:1.796701	test-rmse:4.030062 
#> [21]	train-rmse:1.753100	test-rmse:4.037393 
#> [22]	train-rmse:1.716282	test-rmse:4.035883 
#> [23]	train-rmse:1.688023	test-rmse:4.044036 
#> [24]	train-rmse:1.637178	test-rmse:4.025172 
#> [25]	train-rmse:1.590153	test-rmse:4.001287 
#> [26]	train-rmse:1.546720	test-rmse:3.999570 
#> [27]	train-rmse:1.518661	test-rmse:3.996621 
#> [28]	train-rmse:1.504567	test-rmse:3.997688 
#> [29]	train-rmse:1.494228	test-rmse:4.000123 
#> [30]	train-rmse:1.452043	test-rmse:3.998804 
#> [31]	train-rmse:1.432224	test-rmse:3.995112 
#> [32]	train-rmse:1.414814	test-rmse:3.993324 
#> [33]	train-rmse:1.385945	test-rmse:3.988753 
#> [34]	train-rmse:1.358210	test-rmse:3.974604 
#> [35]	train-rmse:1.327928	test-rmse:3.959248 
#> [36]	train-rmse:1.292086	test-rmse:3.921402 
#> [37]	train-rmse:1.270701	test-rmse:3.925025 
#> [38]	train-rmse:1.250430	test-rmse:3.928783 
#> [39]	train-rmse:1.216950	test-rmse:3.926792 
#> [40]	train-rmse:1.194814	test-rmse:3.923330 
#> [41]	train-rmse:1.179701	test-rmse:3.927600 
#> [42]	train-rmse:1.165947	test-rmse:3.926244 
#> [43]	train-rmse:1.139723	test-rmse:3.925082 
#> [44]	train-rmse:1.126418	test-rmse:3.931113 
#> [45]	train-rmse:1.108641	test-rmse:3.927126 
#> [46]	train-rmse:1.079074	test-rmse:3.912517 
#> [47]	train-rmse:1.067105	test-rmse:3.914097 
#> [48]	train-rmse:1.057996	test-rmse:3.912245 
#> [49]	train-rmse:1.038300	test-rmse:3.911922 
#> [50]	train-rmse:1.026288	test-rmse:3.906951 
#> [51]	train-rmse:1.017630	test-rmse:3.903757 
#> [52]	train-rmse:1.009007	test-rmse:3.906669 
#> [53]	train-rmse:0.986859	test-rmse:3.904847 
#> [54]	train-rmse:0.962731	test-rmse:3.899167 
#> [55]	train-rmse:0.952384	test-rmse:3.887059 
#> [56]	train-rmse:0.939516	test-rmse:3.885651 
#> [57]	train-rmse:0.932103	test-rmse:3.884431 
#> [58]	train-rmse:0.919283	test-rmse:3.878682 
#> [59]	train-rmse:0.905617	test-rmse:3.876369 
#> [60]	train-rmse:0.881147	test-rmse:3.883946 
#> [61]	train-rmse:0.858171	test-rmse:3.884682 
#> [62]	train-rmse:0.844415	test-rmse:3.882521 
#> [63]	train-rmse:0.837669	test-rmse:3.882943 
#> [64]	train-rmse:0.832174	test-rmse:3.880904 
#> [65]	train-rmse:0.810570	test-rmse:3.889605 
#> [66]	train-rmse:0.804834	test-rmse:3.894057 
#> [67]	train-rmse:0.791626	test-rmse:3.893028 
#> [68]	train-rmse:0.768779	test-rmse:3.879874 
#> [69]	train-rmse:0.758766	test-rmse:3.879197 
#> [70]	train-rmse:0.749967	test-rmse:3.879459 
#> [1]	train-rmse:16.997379	test-rmse:17.451532 
#> [2]	train-rmse:12.327516	test-rmse:12.704180 
#> [3]	train-rmse:9.040641	test-rmse:9.445564 
#> [4]	train-rmse:6.807076	test-rmse:7.205961 
#> [5]	train-rmse:5.244356	test-rmse:5.800310 
#> [6]	train-rmse:4.159316	test-rmse:4.861555 
#> [7]	train-rmse:3.462708	test-rmse:4.311892 
#> [8]	train-rmse:3.014366	test-rmse:4.028696 
#> [9]	train-rmse:2.720645	test-rmse:3.843606 
#> [10]	train-rmse:2.520422	test-rmse:3.689043 
#> [11]	train-rmse:2.373553	test-rmse:3.583901 
#> [12]	train-rmse:2.277313	test-rmse:3.518069 
#> [13]	train-rmse:2.201194	test-rmse:3.488645 
#> [14]	train-rmse:2.100364	test-rmse:3.458100 
#> [15]	train-rmse:2.038767	test-rmse:3.436961 
#> [16]	train-rmse:1.981704	test-rmse:3.423819 
#> [17]	train-rmse:1.939475	test-rmse:3.405802 
#> [18]	train-rmse:1.878451	test-rmse:3.391935 
#> [19]	train-rmse:1.822122	test-rmse:3.356745 
#> [20]	train-rmse:1.760522	test-rmse:3.348070 
#> [21]	train-rmse:1.728997	test-rmse:3.338443 
#> [22]	train-rmse:1.678775	test-rmse:3.331897 
#> [23]	train-rmse:1.619983	test-rmse:3.302209 
#> [24]	train-rmse:1.587433	test-rmse:3.294017 
#> [25]	train-rmse:1.558678	test-rmse:3.283673 
#> [26]	train-rmse:1.526820	test-rmse:3.272382 
#> [27]	train-rmse:1.484628	test-rmse:3.279604 
#> [28]	train-rmse:1.453687	test-rmse:3.294263 
#> [29]	train-rmse:1.420792	test-rmse:3.286631 
#> [30]	train-rmse:1.393862	test-rmse:3.271825 
#> [31]	train-rmse:1.375700	test-rmse:3.273577 
#> [32]	train-rmse:1.349823	test-rmse:3.277158 
#> [33]	train-rmse:1.330934	test-rmse:3.276069 
#> [34]	train-rmse:1.306806	test-rmse:3.261469 
#> [35]	train-rmse:1.290785	test-rmse:3.248870 
#> [36]	train-rmse:1.276678	test-rmse:3.256351 
#> [37]	train-rmse:1.250650	test-rmse:3.249693 
#> [38]	train-rmse:1.239255	test-rmse:3.236318 
#> [39]	train-rmse:1.220674	test-rmse:3.239413 
#> [40]	train-rmse:1.183392	test-rmse:3.232085 
#> [41]	train-rmse:1.164879	test-rmse:3.237248 
#> [42]	train-rmse:1.150750	test-rmse:3.229254 
#> [43]	train-rmse:1.145172	test-rmse:3.225219 
#> [44]	train-rmse:1.127229	test-rmse:3.224636 
#> [45]	train-rmse:1.101395	test-rmse:3.223810 
#> [46]	train-rmse:1.091903	test-rmse:3.213180 
#> [47]	train-rmse:1.076728	test-rmse:3.220686 
#> [48]	train-rmse:1.061354	test-rmse:3.215525 
#> [49]	train-rmse:1.051190	test-rmse:3.225935 
#> [50]	train-rmse:1.021991	test-rmse:3.224229 
#> [51]	train-rmse:1.010068	test-rmse:3.224404 
#> [52]	train-rmse:0.999470	test-rmse:3.223292 
#> [53]	train-rmse:0.970067	test-rmse:3.206137 
#> [54]	train-rmse:0.964163	test-rmse:3.209393 
#> [55]	train-rmse:0.942628	test-rmse:3.200447 
#> [56]	train-rmse:0.934682	test-rmse:3.211937 
#> [57]	train-rmse:0.924950	test-rmse:3.206770 
#> [58]	train-rmse:0.904265	test-rmse:3.206695 
#> [59]	train-rmse:0.881713	test-rmse:3.209321 
#> [60]	train-rmse:0.867272	test-rmse:3.208244 
#> [61]	train-rmse:0.854953	test-rmse:3.211635 
#> [62]	train-rmse:0.842953	test-rmse:3.216044 
#> [63]	train-rmse:0.836405	test-rmse:3.213471 
#> [64]	train-rmse:0.830658	test-rmse:3.210704 
#> [65]	train-rmse:0.807639	test-rmse:3.200923 
#> [66]	train-rmse:0.799781	test-rmse:3.199911 
#> [67]	train-rmse:0.788423	test-rmse:3.192200 
#> [68]	train-rmse:0.772786	test-rmse:3.195580 
#> [69]	train-rmse:0.757038	test-rmse:3.199317 
#> [70]	train-rmse:0.752378	test-rmse:3.192875 
#> [1]	train-rmse:17.057881	test-rmse:17.368816 
#> [2]	train-rmse:12.397999	test-rmse:12.721176 
#> [3]	train-rmse:9.179658	test-rmse:9.397576 
#> [4]	train-rmse:6.899103	test-rmse:7.217575 
#> [5]	train-rmse:5.339070	test-rmse:5.748317 
#> [6]	train-rmse:4.268698	test-rmse:4.793658 
#> [7]	train-rmse:3.590526	test-rmse:4.206053 
#> [8]	train-rmse:3.118371	test-rmse:3.869369 
#> [9]	train-rmse:2.790172	test-rmse:3.652297 
#> [10]	train-rmse:2.544289	test-rmse:3.532488 
#> [11]	train-rmse:2.385160	test-rmse:3.444538 
#> [12]	train-rmse:2.267360	test-rmse:3.375863 
#> [13]	train-rmse:2.164550	test-rmse:3.327617 
#> [14]	train-rmse:2.077892	test-rmse:3.311205 
#> [15]	train-rmse:2.007370	test-rmse:3.300080 
#> [16]	train-rmse:1.963144	test-rmse:3.297826 
#> [17]	train-rmse:1.890445	test-rmse:3.244709 
#> [18]	train-rmse:1.840363	test-rmse:3.226010 
#> [19]	train-rmse:1.807124	test-rmse:3.226659 
#> [20]	train-rmse:1.731958	test-rmse:3.221266 
#> [21]	train-rmse:1.712411	test-rmse:3.213196 
#> [22]	train-rmse:1.671912	test-rmse:3.209998 
#> [23]	train-rmse:1.650644	test-rmse:3.207031 
#> [24]	train-rmse:1.597803	test-rmse:3.192777 
#> [25]	train-rmse:1.544983	test-rmse:3.176058 
#> [26]	train-rmse:1.512009	test-rmse:3.164827 
#> [27]	train-rmse:1.490265	test-rmse:3.165031 
#> [28]	train-rmse:1.463455	test-rmse:3.145430 
#> [29]	train-rmse:1.413624	test-rmse:3.118182 
#> [30]	train-rmse:1.381894	test-rmse:3.114592 
#> [31]	train-rmse:1.373177	test-rmse:3.115505 
#> [32]	train-rmse:1.333080	test-rmse:3.099397 
#> [33]	train-rmse:1.296291	test-rmse:3.102611 
#> [34]	train-rmse:1.284492	test-rmse:3.100417 
#> [35]	train-rmse:1.270912	test-rmse:3.096351 
#> [36]	train-rmse:1.255799	test-rmse:3.103873 
#> [37]	train-rmse:1.212618	test-rmse:3.110615 
#> [38]	train-rmse:1.178545	test-rmse:3.099686 
#> [39]	train-rmse:1.165140	test-rmse:3.092176 
#> [40]	train-rmse:1.148148	test-rmse:3.088673 
#> [41]	train-rmse:1.133738	test-rmse:3.073517 
#> [42]	train-rmse:1.118084	test-rmse:3.060142 
#> [43]	train-rmse:1.109201	test-rmse:3.060820 
#> [44]	train-rmse:1.101690	test-rmse:3.059667 
#> [45]	train-rmse:1.090387	test-rmse:3.056292 
#> [46]	train-rmse:1.067051	test-rmse:3.052778 
#> [47]	train-rmse:1.054564	test-rmse:3.050693 
#> [48]	train-rmse:1.032769	test-rmse:3.049460 
#> [49]	train-rmse:1.024019	test-rmse:3.047590 
#> [50]	train-rmse:1.002596	test-rmse:3.046943 
#> [51]	train-rmse:0.977975	test-rmse:3.050363 
#> [52]	train-rmse:0.967894	test-rmse:3.053568 
#> [53]	train-rmse:0.957275	test-rmse:3.051814 
#> [54]	train-rmse:0.940385	test-rmse:3.055941 
#> [55]	train-rmse:0.928878	test-rmse:3.055873 
#> [56]	train-rmse:0.905830	test-rmse:3.052251 
#> [57]	train-rmse:0.896455	test-rmse:3.052866 
#> [58]	train-rmse:0.887365	test-rmse:3.057423 
#> [59]	train-rmse:0.874122	test-rmse:3.056521 
#> [60]	train-rmse:0.865980	test-rmse:3.057188 
#> [61]	train-rmse:0.855847	test-rmse:3.051740 
#> [62]	train-rmse:0.842819	test-rmse:3.047978 
#> [63]	train-rmse:0.832470	test-rmse:3.051294 
#> [64]	train-rmse:0.811250	test-rmse:3.051492 
#> [65]	train-rmse:0.808786	test-rmse:3.051843 
#> [66]	train-rmse:0.798990	test-rmse:3.057929 
#> [67]	train-rmse:0.783010	test-rmse:3.064119 
#> [68]	train-rmse:0.777822	test-rmse:3.062087 
#> [69]	train-rmse:0.770918	test-rmse:3.059189 
#> [70]	train-rmse:0.754899	test-rmse:3.059076 
#> [1]	train-rmse:17.412960	test-rmse:16.681394 
#> [2]	train-rmse:12.652660	test-rmse:12.189434 
#> [3]	train-rmse:9.296199	test-rmse:8.954743 
#> [4]	train-rmse:6.973352	test-rmse:6.788588 
#> [5]	train-rmse:5.367166	test-rmse:5.339162 
#> [6]	train-rmse:4.311594	test-rmse:4.608044 
#> [7]	train-rmse:3.607140	test-rmse:4.078680 
#> [8]	train-rmse:3.163652	test-rmse:3.843206 
#> [9]	train-rmse:2.849114	test-rmse:3.649577 
#> [10]	train-rmse:2.590645	test-rmse:3.550432 
#> [11]	train-rmse:2.440900	test-rmse:3.519477 
#> [12]	train-rmse:2.340064	test-rmse:3.474820 
#> [13]	train-rmse:2.265105	test-rmse:3.441133 
#> [14]	train-rmse:2.189529	test-rmse:3.415320 
#> [15]	train-rmse:2.141899	test-rmse:3.384855 
#> [16]	train-rmse:2.069348	test-rmse:3.358941 
#> [17]	train-rmse:2.034522	test-rmse:3.340774 
#> [18]	train-rmse:1.999661	test-rmse:3.344826 
#> [19]	train-rmse:1.938931	test-rmse:3.319317 
#> [20]	train-rmse:1.912546	test-rmse:3.293988 
#> [21]	train-rmse:1.886576	test-rmse:3.258163 
#> [22]	train-rmse:1.847894	test-rmse:3.253090 
#> [23]	train-rmse:1.831469	test-rmse:3.234605 
#> [24]	train-rmse:1.751538	test-rmse:3.214050 
#> [25]	train-rmse:1.740301	test-rmse:3.206351 
#> [26]	train-rmse:1.688788	test-rmse:3.189390 
#> [27]	train-rmse:1.649900	test-rmse:3.176641 
#> [28]	train-rmse:1.613690	test-rmse:3.180401 
#> [29]	train-rmse:1.569443	test-rmse:3.179498 
#> [30]	train-rmse:1.549176	test-rmse:3.178520 
#> [31]	train-rmse:1.523100	test-rmse:3.163688 
#> [32]	train-rmse:1.508207	test-rmse:3.164436 
#> [33]	train-rmse:1.458552	test-rmse:3.158960 
#> [34]	train-rmse:1.446232	test-rmse:3.143078 
#> [35]	train-rmse:1.431799	test-rmse:3.140469 
#> [36]	train-rmse:1.400321	test-rmse:3.133625 
#> [37]	train-rmse:1.369973	test-rmse:3.115593 
#> [38]	train-rmse:1.352752	test-rmse:3.129119 
#> [39]	train-rmse:1.301946	test-rmse:3.116484 
#> [40]	train-rmse:1.281978	test-rmse:3.113416 
#> [41]	train-rmse:1.259055	test-rmse:3.109692 
#> [42]	train-rmse:1.236656	test-rmse:3.115181 
#> [43]	train-rmse:1.224697	test-rmse:3.121522 
#> [44]	train-rmse:1.215824	test-rmse:3.118802 
#> [45]	train-rmse:1.198657	test-rmse:3.124249 
#> [46]	train-rmse:1.186001	test-rmse:3.116307 
#> [47]	train-rmse:1.171970	test-rmse:3.117426 
#> [48]	train-rmse:1.154399	test-rmse:3.107584 
#> [49]	train-rmse:1.148894	test-rmse:3.108027 
#> [50]	train-rmse:1.137316	test-rmse:3.111089 
#> [51]	train-rmse:1.107201	test-rmse:3.109275 
#> [52]	train-rmse:1.085121	test-rmse:3.120356 
#> [53]	train-rmse:1.061365	test-rmse:3.113679 
#> [54]	train-rmse:1.052869	test-rmse:3.113354 
#> [55]	train-rmse:1.027213	test-rmse:3.112807 
#> [56]	train-rmse:1.018945	test-rmse:3.116606 
#> [57]	train-rmse:1.009530	test-rmse:3.117851 
#> [58]	train-rmse:1.002134	test-rmse:3.105817 
#> [59]	train-rmse:0.990317	test-rmse:3.100990 
#> [60]	train-rmse:0.965903	test-rmse:3.090352 
#> [61]	train-rmse:0.954821	test-rmse:3.085283 
#> [62]	train-rmse:0.944863	test-rmse:3.089749 
#> [63]	train-rmse:0.927950	test-rmse:3.091168 
#> [64]	train-rmse:0.907285	test-rmse:3.087667 
#> [65]	train-rmse:0.889235	test-rmse:3.076890 
#> [66]	train-rmse:0.875737	test-rmse:3.087059 
#> [67]	train-rmse:0.862707	test-rmse:3.090322 
#> [68]	train-rmse:0.853752	test-rmse:3.085296 
#> [69]	train-rmse:0.837307	test-rmse:3.084133 
#> [70]	train-rmse:0.822196	test-rmse:3.084090 
#> [1]	train-rmse:17.169506	test-rmse:17.225022 
#> [2]	train-rmse:12.455913	test-rmse:12.752763 
#> [3]	train-rmse:9.151926	test-rmse:9.553192 
#> [4]	train-rmse:6.884537	test-rmse:7.505901 
#> [5]	train-rmse:5.325362	test-rmse:6.039607 
#> [6]	train-rmse:4.302497	test-rmse:5.109753 
#> [7]	train-rmse:3.606837	test-rmse:4.558336 
#> [8]	train-rmse:3.093282	test-rmse:4.084655 
#> [9]	train-rmse:2.800401	test-rmse:3.924412 
#> [10]	train-rmse:2.540384	test-rmse:3.814947 
#> [11]	train-rmse:2.354165	test-rmse:3.688904 
#> [12]	train-rmse:2.256497	test-rmse:3.611516 
#> [13]	train-rmse:2.132042	test-rmse:3.511207 
#> [14]	train-rmse:2.069947	test-rmse:3.471555 
#> [15]	train-rmse:1.997296	test-rmse:3.390646 
#> [16]	train-rmse:1.940055	test-rmse:3.368159 
#> [17]	train-rmse:1.885953	test-rmse:3.361642 
#> [18]	train-rmse:1.838922	test-rmse:3.390506 
#> [19]	train-rmse:1.798524	test-rmse:3.387458 
#> [20]	train-rmse:1.756683	test-rmse:3.367634 
#> [21]	train-rmse:1.724023	test-rmse:3.356533 
#> [22]	train-rmse:1.685320	test-rmse:3.336750 
#> [23]	train-rmse:1.661951	test-rmse:3.335087 
#> [24]	train-rmse:1.603331	test-rmse:3.323405 
#> [25]	train-rmse:1.555986	test-rmse:3.305589 
#> [26]	train-rmse:1.529450	test-rmse:3.297357 
#> [27]	train-rmse:1.504871	test-rmse:3.280927 
#> [28]	train-rmse:1.485139	test-rmse:3.281735 
#> [29]	train-rmse:1.469784	test-rmse:3.283056 
#> [30]	train-rmse:1.420708	test-rmse:3.286066 
#> [31]	train-rmse:1.392765	test-rmse:3.280234 
#> [32]	train-rmse:1.383273	test-rmse:3.281691 
#> [33]	train-rmse:1.362129	test-rmse:3.280715 
#> [34]	train-rmse:1.331734	test-rmse:3.284393 
#> [35]	train-rmse:1.310938	test-rmse:3.280485 
#> [36]	train-rmse:1.302003	test-rmse:3.268143 
#> [37]	train-rmse:1.285602	test-rmse:3.267923 
#> [38]	train-rmse:1.270166	test-rmse:3.263989 
#> [39]	train-rmse:1.259880	test-rmse:3.269257 
#> [40]	train-rmse:1.234744	test-rmse:3.278989 
#> [41]	train-rmse:1.217414	test-rmse:3.276972 
#> [42]	train-rmse:1.211225	test-rmse:3.271526 
#> [43]	train-rmse:1.199504	test-rmse:3.277273 
#> [44]	train-rmse:1.179804	test-rmse:3.278123 
#> [45]	train-rmse:1.147487	test-rmse:3.258568 
#> [46]	train-rmse:1.134460	test-rmse:3.259104 
#> [47]	train-rmse:1.106672	test-rmse:3.251910 
#> [48]	train-rmse:1.089201	test-rmse:3.247609 
#> [49]	train-rmse:1.079810	test-rmse:3.244909 
#> [50]	train-rmse:1.047328	test-rmse:3.236303 
#> [51]	train-rmse:1.035362	test-rmse:3.237717 
#> [52]	train-rmse:1.004347	test-rmse:3.237464 
#> [53]	train-rmse:0.990274	test-rmse:3.240111 
#> [54]	train-rmse:0.979509	test-rmse:3.245756 
#> [55]	train-rmse:0.975044	test-rmse:3.245766 
#> [56]	train-rmse:0.955212	test-rmse:3.227891 
#> [57]	train-rmse:0.934651	test-rmse:3.233161 
#> [58]	train-rmse:0.912748	test-rmse:3.221678 
#> [59]	train-rmse:0.897412	test-rmse:3.223187 
#> [60]	train-rmse:0.881171	test-rmse:3.222607 
#> [61]	train-rmse:0.871897	test-rmse:3.222442 
#> [62]	train-rmse:0.851408	test-rmse:3.219850 
#> [63]	train-rmse:0.833504	test-rmse:3.221976 
#> [64]	train-rmse:0.826423	test-rmse:3.220464 
#> [65]	train-rmse:0.819444	test-rmse:3.218707 
#> [66]	train-rmse:0.803940	test-rmse:3.222166 
#> [67]	train-rmse:0.798206	test-rmse:3.222465 
#> [68]	train-rmse:0.785378	test-rmse:3.218759 
#> [69]	train-rmse:0.780135	test-rmse:3.219273 
#> [70]	train-rmse:0.773542	test-rmse:3.219333 
#> [1]	train-rmse:17.164552	test-rmse:17.242814 
#> [2]	train-rmse:12.471723	test-rmse:12.579258 
#> [3]	train-rmse:9.207161	test-rmse:9.448715 
#> [4]	train-rmse:6.867933	test-rmse:7.289990 
#> [5]	train-rmse:5.303911	test-rmse:5.857567 
#> [6]	train-rmse:4.206712	test-rmse:4.998401 
#> [7]	train-rmse:3.465868	test-rmse:4.376139 
#> [8]	train-rmse:2.991414	test-rmse:3.999812 
#> [9]	train-rmse:2.668558	test-rmse:3.791950 
#> [10]	train-rmse:2.428843	test-rmse:3.626125 
#> [11]	train-rmse:2.295141	test-rmse:3.532538 
#> [12]	train-rmse:2.209122	test-rmse:3.487994 
#> [13]	train-rmse:2.135309	test-rmse:3.409080 
#> [14]	train-rmse:2.062738	test-rmse:3.342879 
#> [15]	train-rmse:1.972212	test-rmse:3.341761 
#> [16]	train-rmse:1.905831	test-rmse:3.312981 
#> [17]	train-rmse:1.864826	test-rmse:3.294408 
#> [18]	train-rmse:1.821740	test-rmse:3.285478 
#> [19]	train-rmse:1.776895	test-rmse:3.276286 
#> [20]	train-rmse:1.739859	test-rmse:3.256479 
#> [21]	train-rmse:1.679596	test-rmse:3.271925 
#> [22]	train-rmse:1.623119	test-rmse:3.269289 
#> [23]	train-rmse:1.604576	test-rmse:3.269806 
#> [24]	train-rmse:1.577794	test-rmse:3.256249 
#> [25]	train-rmse:1.544814	test-rmse:3.256503 
#> [26]	train-rmse:1.519333	test-rmse:3.246662 
#> [27]	train-rmse:1.498719	test-rmse:3.244969 
#> [28]	train-rmse:1.458306	test-rmse:3.245996 
#> [29]	train-rmse:1.437625	test-rmse:3.240521 
#> [30]	train-rmse:1.410925	test-rmse:3.234643 
#> [31]	train-rmse:1.387669	test-rmse:3.218191 
#> [32]	train-rmse:1.346276	test-rmse:3.200522 
#> [33]	train-rmse:1.322706	test-rmse:3.205671 
#> [34]	train-rmse:1.292469	test-rmse:3.201344 
#> [35]	train-rmse:1.270995	test-rmse:3.193838 
#> [36]	train-rmse:1.243724	test-rmse:3.191893 
#> [37]	train-rmse:1.212973	test-rmse:3.184971 
#> [38]	train-rmse:1.202365	test-rmse:3.172221 
#> [39]	train-rmse:1.163736	test-rmse:3.169551 
#> [40]	train-rmse:1.157949	test-rmse:3.173690 
#> [41]	train-rmse:1.130852	test-rmse:3.183101 
#> [42]	train-rmse:1.118583	test-rmse:3.181193 
#> [43]	train-rmse:1.096160	test-rmse:3.180477 
#> [44]	train-rmse:1.086879	test-rmse:3.167624 
#> [45]	train-rmse:1.077244	test-rmse:3.174473 
#> [46]	train-rmse:1.064686	test-rmse:3.181578 
#> [47]	train-rmse:1.046804	test-rmse:3.178921 
#> [48]	train-rmse:1.036998	test-rmse:3.179391 
#> [49]	train-rmse:1.022280	test-rmse:3.176065 
#> [50]	train-rmse:1.010822	test-rmse:3.174446 
#> [51]	train-rmse:1.001944	test-rmse:3.166613 
#> [52]	train-rmse:0.983659	test-rmse:3.177306 
#> [53]	train-rmse:0.963870	test-rmse:3.166899 
#> [54]	train-rmse:0.945945	test-rmse:3.164375 
#> [55]	train-rmse:0.926894	test-rmse:3.165051 
#> [56]	train-rmse:0.907626	test-rmse:3.156174 
#> [57]	train-rmse:0.892841	test-rmse:3.158669 
#> [58]	train-rmse:0.884488	test-rmse:3.153873 
#> [59]	train-rmse:0.878829	test-rmse:3.155999 
#> [60]	train-rmse:0.874050	test-rmse:3.156696 
#> [61]	train-rmse:0.854229	test-rmse:3.148281 
#> [62]	train-rmse:0.834348	test-rmse:3.143383 
#> [63]	train-rmse:0.821748	test-rmse:3.139207 
#> [64]	train-rmse:0.815034	test-rmse:3.132746 
#> [65]	train-rmse:0.808634	test-rmse:3.126192 
#> [66]	train-rmse:0.805577	test-rmse:3.128236 
#> [67]	train-rmse:0.791926	test-rmse:3.133038 
#> [68]	train-rmse:0.786542	test-rmse:3.127416 
#> [69]	train-rmse:0.782589	test-rmse:3.126790 
#> [70]	train-rmse:0.759756	test-rmse:3.117040 
#> [1]	train-rmse:17.365677	test-rmse:16.981133 
#> [2]	train-rmse:12.627085	test-rmse:12.371665 
#> [3]	train-rmse:9.283167	test-rmse:9.081143 
#> [4]	train-rmse:7.007401	test-rmse:6.895097 
#> [5]	train-rmse:5.414292	test-rmse:5.378851 
#> [6]	train-rmse:4.327584	test-rmse:4.426966 
#> [7]	train-rmse:3.650806	test-rmse:3.858720 
#> [8]	train-rmse:3.187126	test-rmse:3.570247 
#> [9]	train-rmse:2.841159	test-rmse:3.415336 
#> [10]	train-rmse:2.613889	test-rmse:3.311968 
#> [11]	train-rmse:2.442909	test-rmse:3.265147 
#> [12]	train-rmse:2.289956	test-rmse:3.127611 
#> [13]	train-rmse:2.185121	test-rmse:3.084449 
#> [14]	train-rmse:2.089459	test-rmse:3.048229 
#> [15]	train-rmse:2.019436	test-rmse:3.046697 
#> [16]	train-rmse:1.927465	test-rmse:3.041559 
#> [17]	train-rmse:1.874340	test-rmse:3.059228 
#> [18]	train-rmse:1.827809	test-rmse:3.033781 
#> [19]	train-rmse:1.759303	test-rmse:3.054434 
#> [20]	train-rmse:1.736359	test-rmse:3.050429 
#> [21]	train-rmse:1.698841	test-rmse:3.060269 
#> [22]	train-rmse:1.668556	test-rmse:3.057619 
#> [23]	train-rmse:1.645172	test-rmse:3.067866 
#> [24]	train-rmse:1.591316	test-rmse:3.093922 
#> [25]	train-rmse:1.564144	test-rmse:3.092687 
#> [26]	train-rmse:1.525636	test-rmse:3.078053 
#> [27]	train-rmse:1.498496	test-rmse:3.073215 
#> [28]	train-rmse:1.444955	test-rmse:3.091102 
#> [29]	train-rmse:1.412006	test-rmse:3.088701 
#> [30]	train-rmse:1.391050	test-rmse:3.080158 
#> [31]	train-rmse:1.367150	test-rmse:3.089364 
#> [32]	train-rmse:1.328217	test-rmse:3.094719 
#> [33]	train-rmse:1.313198	test-rmse:3.094374 
#> [34]	train-rmse:1.276102	test-rmse:3.094325 
#> [35]	train-rmse:1.246812	test-rmse:3.087894 
#> [36]	train-rmse:1.226088	test-rmse:3.090196 
#> [37]	train-rmse:1.210656	test-rmse:3.096260 
#> [38]	train-rmse:1.193339	test-rmse:3.106484 
#> [39]	train-rmse:1.182922	test-rmse:3.114498 
#> [40]	train-rmse:1.173687	test-rmse:3.112313 
#> [41]	train-rmse:1.160263	test-rmse:3.105608 
#> [42]	train-rmse:1.144793	test-rmse:3.123336 
#> [43]	train-rmse:1.131790	test-rmse:3.123308 
#> [44]	train-rmse:1.098635	test-rmse:3.122902 
#> [45]	train-rmse:1.089013	test-rmse:3.116918 
#> [46]	train-rmse:1.069510	test-rmse:3.120333 
#> [47]	train-rmse:1.055863	test-rmse:3.128643 
#> [48]	train-rmse:1.039188	test-rmse:3.131400 
#> [49]	train-rmse:1.006483	test-rmse:3.125944 
#> [50]	train-rmse:0.984161	test-rmse:3.137047 
#> [51]	train-rmse:0.968635	test-rmse:3.144871 
#> [52]	train-rmse:0.952427	test-rmse:3.139466 
#> [53]	train-rmse:0.930935	test-rmse:3.131501 
#> [54]	train-rmse:0.918740	test-rmse:3.138763 
#> [55]	train-rmse:0.899297	test-rmse:3.138424 
#> [56]	train-rmse:0.891931	test-rmse:3.147669 
#> [57]	train-rmse:0.886033	test-rmse:3.149456 
#> [58]	train-rmse:0.874594	test-rmse:3.151274 
#> [59]	train-rmse:0.860872	test-rmse:3.142111 
#> [60]	train-rmse:0.847046	test-rmse:3.137227 
#> [61]	train-rmse:0.840928	test-rmse:3.141958 
#> [62]	train-rmse:0.819154	test-rmse:3.147682 
#> [63]	train-rmse:0.807491	test-rmse:3.147035 
#> [64]	train-rmse:0.799199	test-rmse:3.149395 
#> [65]	train-rmse:0.785173	test-rmse:3.151627 
#> [66]	train-rmse:0.779256	test-rmse:3.155283 
#> [67]	train-rmse:0.767509	test-rmse:3.151191 
#> [68]	train-rmse:0.752762	test-rmse:3.140265 
#> [69]	train-rmse:0.747249	test-rmse:3.138050 
#> [70]	train-rmse:0.740448	test-rmse:3.142820 
#> [1]	train-rmse:17.106814	test-rmse:17.250180 
#> [2]	train-rmse:12.377038	test-rmse:12.772505 
#> [3]	train-rmse:9.067686	test-rmse:9.741168 
#> [4]	train-rmse:6.773091	test-rmse:7.754187 
#> [5]	train-rmse:5.224824	test-rmse:6.525330 
#> [6]	train-rmse:4.144283	test-rmse:5.684826 
#> [7]	train-rmse:3.430951	test-rmse:5.265582 
#> [8]	train-rmse:2.964625	test-rmse:4.986421 
#> [9]	train-rmse:2.648607	test-rmse:4.886857 
#> [10]	train-rmse:2.416541	test-rmse:4.725083 
#> [11]	train-rmse:2.242877	test-rmse:4.528417 
#> [12]	train-rmse:2.122946	test-rmse:4.397233 
#> [13]	train-rmse:2.045196	test-rmse:4.360970 
#> [14]	train-rmse:1.977138	test-rmse:4.332111 
#> [15]	train-rmse:1.889960	test-rmse:4.321811 
#> [16]	train-rmse:1.814296	test-rmse:4.311061 
#> [17]	train-rmse:1.765740	test-rmse:4.285000 
#> [18]	train-rmse:1.720769	test-rmse:4.266297 
#> [19]	train-rmse:1.694682	test-rmse:4.260227 
#> [20]	train-rmse:1.673406	test-rmse:4.248314 
#> [21]	train-rmse:1.640517	test-rmse:4.242088 
#> [22]	train-rmse:1.613464	test-rmse:4.247419 
#> [23]	train-rmse:1.583350	test-rmse:4.238865 
#> [24]	train-rmse:1.540880	test-rmse:4.231536 
#> [25]	train-rmse:1.495060	test-rmse:4.252198 
#> [26]	train-rmse:1.464764	test-rmse:4.244365 
#> [27]	train-rmse:1.448202	test-rmse:4.198582 
#> [28]	train-rmse:1.425685	test-rmse:4.188188 
#> [29]	train-rmse:1.406633	test-rmse:4.192507 
#> [30]	train-rmse:1.395349	test-rmse:4.193589 
#> [31]	train-rmse:1.376466	test-rmse:4.190414 
#> [32]	train-rmse:1.336125	test-rmse:4.189229 
#> [33]	train-rmse:1.321900	test-rmse:4.154224 
#> [34]	train-rmse:1.305214	test-rmse:4.146886 
#> [35]	train-rmse:1.280075	test-rmse:4.143943 
#> [36]	train-rmse:1.259627	test-rmse:4.139292 
#> [37]	train-rmse:1.229639	test-rmse:4.132941 
#> [38]	train-rmse:1.220821	test-rmse:4.138425 
#> [39]	train-rmse:1.200879	test-rmse:4.129853 
#> [40]	train-rmse:1.186947	test-rmse:4.104963 
#> [41]	train-rmse:1.171959	test-rmse:4.102531 
#> [42]	train-rmse:1.141721	test-rmse:4.086933 
#> [43]	train-rmse:1.110592	test-rmse:4.099258 
#> [44]	train-rmse:1.095057	test-rmse:4.098190 
#> [45]	train-rmse:1.068194	test-rmse:4.097428 
#> [46]	train-rmse:1.046173	test-rmse:4.089784 
#> [47]	train-rmse:1.033784	test-rmse:4.076871 
#> [48]	train-rmse:1.010406	test-rmse:4.068876 
#> [49]	train-rmse:0.987087	test-rmse:4.080664 
#> [50]	train-rmse:0.973166	test-rmse:4.082035 
#> [51]	train-rmse:0.969764	test-rmse:4.082538 
#> [52]	train-rmse:0.963108	test-rmse:4.085631 
#> [53]	train-rmse:0.946885	test-rmse:4.088078 
#> [54]	train-rmse:0.928519	test-rmse:4.095592 
#> [55]	train-rmse:0.915581	test-rmse:4.091596 
#> [56]	train-rmse:0.905270	test-rmse:4.087808 
#> [57]	train-rmse:0.890827	test-rmse:4.082792 
#> [58]	train-rmse:0.879801	test-rmse:4.077462 
#> [59]	train-rmse:0.872119	test-rmse:4.084161 
#> [60]	train-rmse:0.853952	test-rmse:4.089163 
#> [61]	train-rmse:0.842883	test-rmse:4.088458 
#> [62]	train-rmse:0.823886	test-rmse:4.080652 
#> [63]	train-rmse:0.812544	test-rmse:4.080648 
#> [64]	train-rmse:0.800273	test-rmse:4.082317 
#> [65]	train-rmse:0.785629	test-rmse:4.078284 
#> [66]	train-rmse:0.764952	test-rmse:4.077993 
#> [67]	train-rmse:0.748780	test-rmse:4.068827 
#> [68]	train-rmse:0.740748	test-rmse:4.063622 
#> [69]	train-rmse:0.728420	test-rmse:4.071308 
#> [70]	train-rmse:0.713275	test-rmse:4.068478 
#> [1]	train-rmse:16.843327	test-rmse:17.630448 
#> [2]	train-rmse:12.230086	test-rmse:13.025456 
#> [3]	train-rmse:9.033050	test-rmse:9.805279 
#> [4]	train-rmse:6.802771	test-rmse:7.573263 
#> [5]	train-rmse:5.277781	test-rmse:6.218051 
#> [6]	train-rmse:4.181441	test-rmse:5.410309 
#> [7]	train-rmse:3.508052	test-rmse:4.869971 
#> [8]	train-rmse:3.041484	test-rmse:4.473363 
#> [9]	train-rmse:2.702130	test-rmse:4.224139 
#> [10]	train-rmse:2.477018	test-rmse:4.049994 
#> [11]	train-rmse:2.295460	test-rmse:3.941098 
#> [12]	train-rmse:2.158347	test-rmse:3.840117 
#> [13]	train-rmse:2.087225	test-rmse:3.776254 
#> [14]	train-rmse:2.011524	test-rmse:3.755294 
#> [15]	train-rmse:1.923977	test-rmse:3.706982 
#> [16]	train-rmse:1.853436	test-rmse:3.697812 
#> [17]	train-rmse:1.789510	test-rmse:3.637131 
#> [18]	train-rmse:1.735547	test-rmse:3.615964 
#> [19]	train-rmse:1.706160	test-rmse:3.613229 
#> [20]	train-rmse:1.655062	test-rmse:3.593897 
#> [21]	train-rmse:1.627220	test-rmse:3.571851 
#> [22]	train-rmse:1.597271	test-rmse:3.553981 
#> [23]	train-rmse:1.560725	test-rmse:3.536863 
#> [24]	train-rmse:1.541744	test-rmse:3.518866 
#> [25]	train-rmse:1.476404	test-rmse:3.508670 
#> [26]	train-rmse:1.457586	test-rmse:3.496073 
#> [27]	train-rmse:1.401413	test-rmse:3.477101 
#> [28]	train-rmse:1.357625	test-rmse:3.455486 
#> [29]	train-rmse:1.343528	test-rmse:3.465116 
#> [30]	train-rmse:1.331339	test-rmse:3.450320 
#> [31]	train-rmse:1.304855	test-rmse:3.438186 
#> [32]	train-rmse:1.277205	test-rmse:3.429056 
#> [33]	train-rmse:1.267476	test-rmse:3.431337 
#> [34]	train-rmse:1.260252	test-rmse:3.429773 
#> [35]	train-rmse:1.239489	test-rmse:3.418485 
#> [36]	train-rmse:1.211899	test-rmse:3.418756 
#> [37]	train-rmse:1.175095	test-rmse:3.421404 
#> [38]	train-rmse:1.168278	test-rmse:3.431754 
#> [39]	train-rmse:1.145681	test-rmse:3.428072 
#> [40]	train-rmse:1.110845	test-rmse:3.425727 
#> [41]	train-rmse:1.099152	test-rmse:3.415371 
#> [42]	train-rmse:1.072962	test-rmse:3.406520 
#> [43]	train-rmse:1.063818	test-rmse:3.409243 
#> [44]	train-rmse:1.055725	test-rmse:3.402111 
#> [45]	train-rmse:1.039420	test-rmse:3.396152 
#> [46]	train-rmse:1.018120	test-rmse:3.388016 
#> [47]	train-rmse:1.012710	test-rmse:3.396671 
#> [48]	train-rmse:1.007173	test-rmse:3.396966 
#> [49]	train-rmse:0.988877	test-rmse:3.403745 
#> [50]	train-rmse:0.964777	test-rmse:3.414491 
#> [51]	train-rmse:0.950579	test-rmse:3.420110 
#> [52]	train-rmse:0.942251	test-rmse:3.408809 
#> [53]	train-rmse:0.934431	test-rmse:3.404427 
#> [54]	train-rmse:0.910658	test-rmse:3.406597 
#> [55]	train-rmse:0.897827	test-rmse:3.399108 
#> [56]	train-rmse:0.881912	test-rmse:3.396343 
#> [57]	train-rmse:0.864401	test-rmse:3.394349 
#> [58]	train-rmse:0.845182	test-rmse:3.403125 
#> [59]	train-rmse:0.836448	test-rmse:3.400948 
#> [60]	train-rmse:0.820609	test-rmse:3.404693 
#> [61]	train-rmse:0.800922	test-rmse:3.401149 
#> [62]	train-rmse:0.782807	test-rmse:3.398375 
#> [63]	train-rmse:0.763460	test-rmse:3.392056 
#> [64]	train-rmse:0.752839	test-rmse:3.403859 
#> [65]	train-rmse:0.745040	test-rmse:3.398278 
#> [66]	train-rmse:0.738352	test-rmse:3.407817 
#> [67]	train-rmse:0.728654	test-rmse:3.412509 
#> [68]	train-rmse:0.718183	test-rmse:3.410390 
#> [69]	train-rmse:0.708034	test-rmse:3.409943 
#> [70]	train-rmse:0.691025	test-rmse:3.409508 
#> [1]	train-rmse:16.949845	test-rmse:17.824195 
#> [2]	train-rmse:12.341275	test-rmse:13.165814 
#> [3]	train-rmse:9.089034	test-rmse:10.017984 
#> [4]	train-rmse:6.861677	test-rmse:7.900584 
#> [5]	train-rmse:5.303419	test-rmse:6.384306 
#> [6]	train-rmse:4.259769	test-rmse:5.392201 
#> [7]	train-rmse:3.576851	test-rmse:4.763239 
#> [8]	train-rmse:3.067836	test-rmse:4.400579 
#> [9]	train-rmse:2.758527	test-rmse:4.167037 
#> [10]	train-rmse:2.537287	test-rmse:4.075754 
#> [11]	train-rmse:2.385793	test-rmse:4.035144 
#> [12]	train-rmse:2.275513	test-rmse:3.953069 
#> [13]	train-rmse:2.200654	test-rmse:3.912271 
#> [14]	train-rmse:2.080634	test-rmse:3.865646 
#> [15]	train-rmse:2.031579	test-rmse:3.864741 
#> [16]	train-rmse:1.952998	test-rmse:3.852482 
#> [17]	train-rmse:1.925618	test-rmse:3.874928 
#> [18]	train-rmse:1.880780	test-rmse:3.859902 
#> [19]	train-rmse:1.842236	test-rmse:3.840464 
#> [20]	train-rmse:1.801217	test-rmse:3.860147 
#> [21]	train-rmse:1.734936	test-rmse:3.875298 
#> [22]	train-rmse:1.693691	test-rmse:3.847974 
#> [23]	train-rmse:1.664794	test-rmse:3.830507 
#> [24]	train-rmse:1.632001	test-rmse:3.840560 
#> [25]	train-rmse:1.585405	test-rmse:3.813598 
#> [26]	train-rmse:1.562326	test-rmse:3.813233 
#> [27]	train-rmse:1.513265	test-rmse:3.817980 
#> [28]	train-rmse:1.496845	test-rmse:3.816506 
#> [29]	train-rmse:1.482189	test-rmse:3.831948 
#> [30]	train-rmse:1.463350	test-rmse:3.821553 
#> [31]	train-rmse:1.447672	test-rmse:3.823210 
#> [32]	train-rmse:1.436055	test-rmse:3.821823 
#> [33]	train-rmse:1.409757	test-rmse:3.811259 
#> [34]	train-rmse:1.369539	test-rmse:3.807964 
#> [35]	train-rmse:1.343920	test-rmse:3.805067 
#> [36]	train-rmse:1.314949	test-rmse:3.817181 
#> [37]	train-rmse:1.293520	test-rmse:3.817634 
#> [38]	train-rmse:1.270368	test-rmse:3.808399 
#> [39]	train-rmse:1.249667	test-rmse:3.813187 
#> [40]	train-rmse:1.236692	test-rmse:3.811879 
#> [41]	train-rmse:1.213191	test-rmse:3.798800 
#> [42]	train-rmse:1.176563	test-rmse:3.798468 
#> [43]	train-rmse:1.165220	test-rmse:3.800355 
#> [44]	train-rmse:1.135637	test-rmse:3.807197 
#> [45]	train-rmse:1.115599	test-rmse:3.788496 
#> [46]	train-rmse:1.103269	test-rmse:3.792169 
#> [47]	train-rmse:1.091085	test-rmse:3.784152 
#> [48]	train-rmse:1.062291	test-rmse:3.779102 
#> [49]	train-rmse:1.037378	test-rmse:3.759805 
#> [50]	train-rmse:1.010207	test-rmse:3.756861 
#> [51]	train-rmse:0.994277	test-rmse:3.751901 
#> [52]	train-rmse:0.976165	test-rmse:3.748819 
#> [53]	train-rmse:0.953688	test-rmse:3.742070 
#> [54]	train-rmse:0.934867	test-rmse:3.747525 
#> [55]	train-rmse:0.918004	test-rmse:3.745348 
#> [56]	train-rmse:0.906836	test-rmse:3.749111 
#> [57]	train-rmse:0.890291	test-rmse:3.746518 
#> [58]	train-rmse:0.879625	test-rmse:3.747108 
#> [59]	train-rmse:0.864893	test-rmse:3.742562 
#> [60]	train-rmse:0.844640	test-rmse:3.739284 
#> [61]	train-rmse:0.840344	test-rmse:3.736408 
#> [62]	train-rmse:0.828862	test-rmse:3.729288 
#> [63]	train-rmse:0.814817	test-rmse:3.730951 
#> [64]	train-rmse:0.808629	test-rmse:3.725393 
#> [65]	train-rmse:0.802357	test-rmse:3.725300 
#> [66]	train-rmse:0.785755	test-rmse:3.723347 
#> [67]	train-rmse:0.778525	test-rmse:3.717697 
#> [68]	train-rmse:0.767660	test-rmse:3.720729 
#> [69]	train-rmse:0.758042	test-rmse:3.722188 
#> [70]	train-rmse:0.751187	test-rmse:3.714882 
#> [1]	train-rmse:17.164234	test-rmse:17.290856 
#> [2]	train-rmse:12.385317	test-rmse:12.873364 
#> [3]	train-rmse:9.038469	test-rmse:9.814048 
#> [4]	train-rmse:6.712088	test-rmse:7.904425 
#> [5]	train-rmse:5.123116	test-rmse:6.569116 
#> [6]	train-rmse:4.057769	test-rmse:5.819921 
#> [7]	train-rmse:3.311019	test-rmse:5.398898 
#> [8]	train-rmse:2.833277	test-rmse:5.063300 
#> [9]	train-rmse:2.526343	test-rmse:4.910978 
#> [10]	train-rmse:2.330799	test-rmse:4.824775 
#> [11]	train-rmse:2.190479	test-rmse:4.778139 
#> [12]	train-rmse:2.039613	test-rmse:4.722453 
#> [13]	train-rmse:1.937872	test-rmse:4.606422 
#> [14]	train-rmse:1.879678	test-rmse:4.581573 
#> [15]	train-rmse:1.825661	test-rmse:4.561776 
#> [16]	train-rmse:1.759723	test-rmse:4.534257 
#> [17]	train-rmse:1.710874	test-rmse:4.524862 
#> [18]	train-rmse:1.664751	test-rmse:4.508217 
#> [19]	train-rmse:1.626281	test-rmse:4.503706 
#> [20]	train-rmse:1.597528	test-rmse:4.489805 
#> [21]	train-rmse:1.566941	test-rmse:4.483998 
#> [22]	train-rmse:1.531071	test-rmse:4.464565 
#> [23]	train-rmse:1.487685	test-rmse:4.456293 
#> [24]	train-rmse:1.449060	test-rmse:4.454694 
#> [25]	train-rmse:1.413143	test-rmse:4.465825 
#> [26]	train-rmse:1.382532	test-rmse:4.457578 
#> [27]	train-rmse:1.340500	test-rmse:4.452180 
#> [28]	train-rmse:1.314598	test-rmse:4.453048 
#> [29]	train-rmse:1.299256	test-rmse:4.453786 
#> [30]	train-rmse:1.271586	test-rmse:4.434573 
#> [31]	train-rmse:1.229094	test-rmse:4.418154 
#> [32]	train-rmse:1.197729	test-rmse:4.394206 
#> [33]	train-rmse:1.184765	test-rmse:4.397141 
#> [34]	train-rmse:1.166993	test-rmse:4.383526 
#> [35]	train-rmse:1.152211	test-rmse:4.381627 
#> [36]	train-rmse:1.141174	test-rmse:4.381599 
#> [37]	train-rmse:1.112470	test-rmse:4.372176 
#> [38]	train-rmse:1.097799	test-rmse:4.376624 
#> [39]	train-rmse:1.067566	test-rmse:4.378681 
#> [40]	train-rmse:1.053726	test-rmse:4.372034 
#> [41]	train-rmse:1.034273	test-rmse:4.369001 
#> [42]	train-rmse:1.011066	test-rmse:4.363326 
#> [43]	train-rmse:0.985568	test-rmse:4.371523 
#> [44]	train-rmse:0.967149	test-rmse:4.371314 
#> [45]	train-rmse:0.951092	test-rmse:4.372927 
#> [46]	train-rmse:0.945378	test-rmse:4.373406 
#> [47]	train-rmse:0.931181	test-rmse:4.370639 
#> [48]	train-rmse:0.919773	test-rmse:4.368897 
#> [49]	train-rmse:0.898064	test-rmse:4.364811 
#> [50]	train-rmse:0.886152	test-rmse:4.364254 
#> [51]	train-rmse:0.867474	test-rmse:4.369778 
#> [52]	train-rmse:0.860538	test-rmse:4.361710 
#> [53]	train-rmse:0.855384	test-rmse:4.357094 
#> [54]	train-rmse:0.844501	test-rmse:4.357192 
#> [55]	train-rmse:0.828474	test-rmse:4.347019 
#> [56]	train-rmse:0.805482	test-rmse:4.339889 
#> [57]	train-rmse:0.796325	test-rmse:4.339752 
#> [58]	train-rmse:0.783212	test-rmse:4.338486 
#> [59]	train-rmse:0.774764	test-rmse:4.337017 
#> [60]	train-rmse:0.754201	test-rmse:4.343769 
#> [61]	train-rmse:0.748837	test-rmse:4.343565 
#> [62]	train-rmse:0.734607	test-rmse:4.341011 
#> [63]	train-rmse:0.719883	test-rmse:4.342027 
#> [64]	train-rmse:0.711308	test-rmse:4.341379 
#> [65]	train-rmse:0.697173	test-rmse:4.335417 
#> [66]	train-rmse:0.681067	test-rmse:4.336891 
#> [67]	train-rmse:0.675150	test-rmse:4.334851 
#> [68]	train-rmse:0.664202	test-rmse:4.336332 
#> [69]	train-rmse:0.653607	test-rmse:4.336920 
#> [70]	train-rmse:0.646397	test-rmse:4.327037 
#> [1]	train-rmse:17.339362	test-rmse:16.886611 
#> [2]	train-rmse:12.586488	test-rmse:12.335260 
#> [3]	train-rmse:9.224821	test-rmse:9.145505 
#> [4]	train-rmse:6.890770	test-rmse:7.107080 
#> [5]	train-rmse:5.279757	test-rmse:5.951127 
#> [6]	train-rmse:4.172018	test-rmse:5.249710 
#> [7]	train-rmse:3.427074	test-rmse:4.850307 
#> [8]	train-rmse:2.963032	test-rmse:4.625386 
#> [9]	train-rmse:2.662795	test-rmse:4.498435 
#> [10]	train-rmse:2.442344	test-rmse:4.363990 
#> [11]	train-rmse:2.308797	test-rmse:4.349434 
#> [12]	train-rmse:2.185629	test-rmse:4.349068 
#> [13]	train-rmse:2.084013	test-rmse:4.316079 
#> [14]	train-rmse:2.019330	test-rmse:4.297881 
#> [15]	train-rmse:1.966460	test-rmse:4.297906 
#> [16]	train-rmse:1.929337	test-rmse:4.293474 
#> [17]	train-rmse:1.893997	test-rmse:4.273471 
#> [18]	train-rmse:1.852781	test-rmse:4.273586 
#> [19]	train-rmse:1.800863	test-rmse:4.273792 
#> [20]	train-rmse:1.749531	test-rmse:4.262150 
#> [21]	train-rmse:1.722285	test-rmse:4.259934 
#> [22]	train-rmse:1.648740	test-rmse:4.265721 
#> [23]	train-rmse:1.614853	test-rmse:4.250797 
#> [24]	train-rmse:1.587401	test-rmse:4.249821 
#> [25]	train-rmse:1.549045	test-rmse:4.246630 
#> [26]	train-rmse:1.525117	test-rmse:4.224750 
#> [27]	train-rmse:1.489872	test-rmse:4.221634 
#> [28]	train-rmse:1.472498	test-rmse:4.218214 
#> [29]	train-rmse:1.428341	test-rmse:4.228701 
#> [30]	train-rmse:1.409449	test-rmse:4.228808 
#> [31]	train-rmse:1.348274	test-rmse:4.225690 
#> [32]	train-rmse:1.326855	test-rmse:4.216593 
#> [33]	train-rmse:1.305855	test-rmse:4.216675 
#> [34]	train-rmse:1.290848	test-rmse:4.212561 
#> [35]	train-rmse:1.279023	test-rmse:4.215831 
#> [36]	train-rmse:1.266642	test-rmse:4.217184 
#> [37]	train-rmse:1.250990	test-rmse:4.218370 
#> [38]	train-rmse:1.234323	test-rmse:4.219821 
#> [39]	train-rmse:1.221149	test-rmse:4.224593 
#> [40]	train-rmse:1.212641	test-rmse:4.227159 
#> [41]	train-rmse:1.188258	test-rmse:4.228749 
#> [42]	train-rmse:1.152960	test-rmse:4.223627 
#> [43]	train-rmse:1.132255	test-rmse:4.227086 
#> [44]	train-rmse:1.106895	test-rmse:4.225276 
#> [45]	train-rmse:1.096884	test-rmse:4.214814 
#> [46]	train-rmse:1.084345	test-rmse:4.218354 
#> [47]	train-rmse:1.058253	test-rmse:4.217997 
#> [48]	train-rmse:1.029234	test-rmse:4.218360 
#> [49]	train-rmse:1.002912	test-rmse:4.218650 
#> [50]	train-rmse:0.988225	test-rmse:4.207876 
#> [51]	train-rmse:0.974329	test-rmse:4.206019 
#> [52]	train-rmse:0.961530	test-rmse:4.213996 
#> [53]	train-rmse:0.953320	test-rmse:4.212837 
#> [54]	train-rmse:0.943970	test-rmse:4.218621 
#> [55]	train-rmse:0.936817	test-rmse:4.221017 
#> [56]	train-rmse:0.927453	test-rmse:4.219435 
#> [57]	train-rmse:0.921098	test-rmse:4.225236 
#> [58]	train-rmse:0.904326	test-rmse:4.219251 
#> [59]	train-rmse:0.890045	test-rmse:4.223911 
#> [60]	train-rmse:0.881177	test-rmse:4.227973 
#> [61]	train-rmse:0.870725	test-rmse:4.235689 
#> [62]	train-rmse:0.847676	test-rmse:4.243785 
#> [63]	train-rmse:0.839300	test-rmse:4.245725 
#> [64]	train-rmse:0.820093	test-rmse:4.249943 
#> [65]	train-rmse:0.812403	test-rmse:4.243110 
#> [66]	train-rmse:0.804941	test-rmse:4.240816 
#> [67]	train-rmse:0.789119	test-rmse:4.244490 
#> [68]	train-rmse:0.778546	test-rmse:4.248261 
#> [69]	train-rmse:0.768634	test-rmse:4.253473 
#> [70]	train-rmse:0.753906	test-rmse:4.255495 
#> [1]	train-rmse:17.043464	test-rmse:17.270172 
#> [2]	train-rmse:12.361571	test-rmse:12.583885 
#> [3]	train-rmse:9.066605	test-rmse:9.319331 
#> [4]	train-rmse:6.774027	test-rmse:7.262226 
#> [5]	train-rmse:5.211588	test-rmse:5.923800 
#> [6]	train-rmse:4.099974	test-rmse:5.122749 
#> [7]	train-rmse:3.398566	test-rmse:4.580629 
#> [8]	train-rmse:2.931773	test-rmse:4.301766 
#> [9]	train-rmse:2.582530	test-rmse:4.112700 
#> [10]	train-rmse:2.355689	test-rmse:3.955240 
#> [11]	train-rmse:2.214995	test-rmse:3.847516 
#> [12]	train-rmse:2.099497	test-rmse:3.745548 
#> [13]	train-rmse:2.035166	test-rmse:3.738093 
#> [14]	train-rmse:1.990925	test-rmse:3.708766 
#> [15]	train-rmse:1.946326	test-rmse:3.659753 
#> [16]	train-rmse:1.920013	test-rmse:3.660893 
#> [17]	train-rmse:1.866716	test-rmse:3.627870 
#> [18]	train-rmse:1.803806	test-rmse:3.578629 
#> [19]	train-rmse:1.736163	test-rmse:3.574666 
#> [20]	train-rmse:1.714589	test-rmse:3.559812 
#> [21]	train-rmse:1.681649	test-rmse:3.542513 
#> [22]	train-rmse:1.656912	test-rmse:3.531143 
#> [23]	train-rmse:1.622786	test-rmse:3.529608 
#> [24]	train-rmse:1.600940	test-rmse:3.519877 
#> [25]	train-rmse:1.566961	test-rmse:3.510495 
#> [26]	train-rmse:1.547053	test-rmse:3.501320 
#> [27]	train-rmse:1.531756	test-rmse:3.500481 
#> [28]	train-rmse:1.485721	test-rmse:3.460774 
#> [29]	train-rmse:1.462208	test-rmse:3.464586 
#> [30]	train-rmse:1.435784	test-rmse:3.436003 
#> [31]	train-rmse:1.422993	test-rmse:3.423357 
#> [32]	train-rmse:1.396143	test-rmse:3.423470 
#> [33]	train-rmse:1.382470	test-rmse:3.419188 
#> [34]	train-rmse:1.364982	test-rmse:3.422158 
#> [35]	train-rmse:1.354400	test-rmse:3.425399 
#> [36]	train-rmse:1.349015	test-rmse:3.425106 
#> [37]	train-rmse:1.331291	test-rmse:3.417051 
#> [38]	train-rmse:1.322026	test-rmse:3.414100 
#> [39]	train-rmse:1.301484	test-rmse:3.407506 
#> [40]	train-rmse:1.287670	test-rmse:3.409145 
#> [41]	train-rmse:1.273148	test-rmse:3.408364 
#> [42]	train-rmse:1.243073	test-rmse:3.414236 
#> [43]	train-rmse:1.217938	test-rmse:3.411057 
#> [44]	train-rmse:1.204210	test-rmse:3.418863 
#> [45]	train-rmse:1.199341	test-rmse:3.411529 
#> [46]	train-rmse:1.174599	test-rmse:3.378846 
#> [47]	train-rmse:1.163850	test-rmse:3.382828 
#> [48]	train-rmse:1.119012	test-rmse:3.382510 
#> [49]	train-rmse:1.088558	test-rmse:3.381638 
#> [50]	train-rmse:1.081114	test-rmse:3.382802 
#> [51]	train-rmse:1.064135	test-rmse:3.381489 
#> [52]	train-rmse:1.043606	test-rmse:3.398191 
#> [53]	train-rmse:1.029898	test-rmse:3.387862 
#> [54]	train-rmse:1.014860	test-rmse:3.381437 
#> [55]	train-rmse:0.986339	test-rmse:3.373278 
#> [56]	train-rmse:0.981199	test-rmse:3.367442 
#> [57]	train-rmse:0.968102	test-rmse:3.368859 
#> [58]	train-rmse:0.959230	test-rmse:3.364833 
#> [59]	train-rmse:0.951625	test-rmse:3.361161 
#> [60]	train-rmse:0.942096	test-rmse:3.359646 
#> [61]	train-rmse:0.929283	test-rmse:3.348098 
#> [62]	train-rmse:0.921466	test-rmse:3.347140 
#> [63]	train-rmse:0.902562	test-rmse:3.349948 
#> [64]	train-rmse:0.895032	test-rmse:3.351796 
#> [65]	train-rmse:0.873948	test-rmse:3.367141 
#> [66]	train-rmse:0.867975	test-rmse:3.363937 
#> [67]	train-rmse:0.861314	test-rmse:3.372426 
#> [68]	train-rmse:0.843317	test-rmse:3.368465 
#> [69]	train-rmse:0.837294	test-rmse:3.365844 
#> [70]	train-rmse:0.831373	test-rmse:3.364705 
#> [1]	train-rmse:17.456190	test-rmse:16.653549 
#> [2]	train-rmse:12.648194	test-rmse:12.092145 
#> [3]	train-rmse:9.313788	test-rmse:9.082147 
#> [4]	train-rmse:6.970995	test-rmse:7.073219 
#> [5]	train-rmse:5.380071	test-rmse:5.699679 
#> [6]	train-rmse:4.270889	test-rmse:4.754919 
#> [7]	train-rmse:3.515352	test-rmse:4.246503 
#> [8]	train-rmse:3.029775	test-rmse:3.914680 
#> [9]	train-rmse:2.697642	test-rmse:3.754900 
#> [10]	train-rmse:2.504079	test-rmse:3.652176 
#> [11]	train-rmse:2.343365	test-rmse:3.606815 
#> [12]	train-rmse:2.220072	test-rmse:3.558213 
#> [13]	train-rmse:2.124556	test-rmse:3.502181 
#> [14]	train-rmse:2.072974	test-rmse:3.465366 
#> [15]	train-rmse:2.000467	test-rmse:3.433915 
#> [16]	train-rmse:1.924200	test-rmse:3.384735 
#> [17]	train-rmse:1.856506	test-rmse:3.363236 
#> [18]	train-rmse:1.816794	test-rmse:3.357258 
#> [19]	train-rmse:1.750301	test-rmse:3.319879 
#> [20]	train-rmse:1.723350	test-rmse:3.289218 
#> [21]	train-rmse:1.700047	test-rmse:3.265925 
#> [22]	train-rmse:1.653667	test-rmse:3.256395 
#> [23]	train-rmse:1.628537	test-rmse:3.256971 
#> [24]	train-rmse:1.602954	test-rmse:3.249408 
#> [25]	train-rmse:1.570623	test-rmse:3.248375 
#> [26]	train-rmse:1.556067	test-rmse:3.246089 
#> [27]	train-rmse:1.530178	test-rmse:3.247561 
#> [28]	train-rmse:1.504508	test-rmse:3.226086 
#> [29]	train-rmse:1.451478	test-rmse:3.204355 
#> [30]	train-rmse:1.442616	test-rmse:3.207756 
#> [31]	train-rmse:1.379040	test-rmse:3.207266 
#> [32]	train-rmse:1.361567	test-rmse:3.212331 
#> [33]	train-rmse:1.330051	test-rmse:3.209703 
#> [34]	train-rmse:1.309706	test-rmse:3.208598 
#> [35]	train-rmse:1.300922	test-rmse:3.210971 
#> [36]	train-rmse:1.291950	test-rmse:3.213551 
#> [37]	train-rmse:1.265668	test-rmse:3.217192 
#> [38]	train-rmse:1.242499	test-rmse:3.210560 
#> [39]	train-rmse:1.234691	test-rmse:3.209650 
#> [40]	train-rmse:1.203055	test-rmse:3.220415 
#> [41]	train-rmse:1.196558	test-rmse:3.214507 
#> [42]	train-rmse:1.183184	test-rmse:3.203664 
#> [43]	train-rmse:1.162531	test-rmse:3.211341 
#> [44]	train-rmse:1.144301	test-rmse:3.200769 
#> [45]	train-rmse:1.127309	test-rmse:3.210206 
#> [46]	train-rmse:1.104384	test-rmse:3.196201 
#> [47]	train-rmse:1.093444	test-rmse:3.200203 
#> [48]	train-rmse:1.084574	test-rmse:3.194495 
#> [49]	train-rmse:1.065494	test-rmse:3.199787 
#> [50]	train-rmse:1.047072	test-rmse:3.183444 
#> [51]	train-rmse:1.034626	test-rmse:3.183327 
#> [52]	train-rmse:1.012602	test-rmse:3.176554 
#> [53]	train-rmse:0.990191	test-rmse:3.176441 
#> [54]	train-rmse:0.981658	test-rmse:3.171199 
#> [55]	train-rmse:0.971126	test-rmse:3.163631 
#> [56]	train-rmse:0.956953	test-rmse:3.168607 
#> [57]	train-rmse:0.946457	test-rmse:3.174608 
#> [58]	train-rmse:0.938780	test-rmse:3.175984 
#> [59]	train-rmse:0.931616	test-rmse:3.169231 
#> [60]	train-rmse:0.922506	test-rmse:3.166960 
#> [61]	train-rmse:0.913599	test-rmse:3.160929 
#> [62]	train-rmse:0.902772	test-rmse:3.155219 
#> [63]	train-rmse:0.890311	test-rmse:3.152159 
#> [64]	train-rmse:0.880465	test-rmse:3.152945 
#> [65]	train-rmse:0.871798	test-rmse:3.151441 
#> [66]	train-rmse:0.859197	test-rmse:3.153839 
#> [67]	train-rmse:0.846615	test-rmse:3.156859 
#> [68]	train-rmse:0.831658	test-rmse:3.157734 
#> [69]	train-rmse:0.824605	test-rmse:3.156717 
#> [70]	train-rmse:0.804884	test-rmse:3.162866 
#> [1]	train-rmse:17.485217	test-rmse:16.537413 
#> [2]	train-rmse:12.660004	test-rmse:11.973589 
#> [3]	train-rmse:9.300846	test-rmse:8.862548 
#> [4]	train-rmse:6.987644	test-rmse:6.763105 
#> [5]	train-rmse:5.339208	test-rmse:5.351333 
#> [6]	train-rmse:4.226457	test-rmse:4.455682 
#> [7]	train-rmse:3.502347	test-rmse:3.927905 
#> [8]	train-rmse:3.019808	test-rmse:3.673145 
#> [9]	train-rmse:2.685198	test-rmse:3.550330 
#> [10]	train-rmse:2.471760	test-rmse:3.448850 
#> [11]	train-rmse:2.301642	test-rmse:3.351629 
#> [12]	train-rmse:2.180760	test-rmse:3.323814 
#> [13]	train-rmse:2.106010	test-rmse:3.271801 
#> [14]	train-rmse:2.035941	test-rmse:3.256801 
#> [15]	train-rmse:1.951739	test-rmse:3.261532 
#> [16]	train-rmse:1.884756	test-rmse:3.243790 
#> [17]	train-rmse:1.851105	test-rmse:3.235069 
#> [18]	train-rmse:1.807456	test-rmse:3.204093 
#> [19]	train-rmse:1.770250	test-rmse:3.183853 
#> [20]	train-rmse:1.708237	test-rmse:3.175278 
#> [21]	train-rmse:1.657705	test-rmse:3.158464 
#> [22]	train-rmse:1.626835	test-rmse:3.152306 
#> [23]	train-rmse:1.590751	test-rmse:3.154191 
#> [24]	train-rmse:1.563488	test-rmse:3.151662 
#> [25]	train-rmse:1.528616	test-rmse:3.149880 
#> [26]	train-rmse:1.491629	test-rmse:3.143521 
#> [27]	train-rmse:1.476580	test-rmse:3.132445 
#> [28]	train-rmse:1.432317	test-rmse:3.117654 
#> [29]	train-rmse:1.412865	test-rmse:3.097791 
#> [30]	train-rmse:1.378458	test-rmse:3.103508 
#> [31]	train-rmse:1.332700	test-rmse:3.105340 
#> [32]	train-rmse:1.291118	test-rmse:3.082222 
#> [33]	train-rmse:1.263577	test-rmse:3.082855 
#> [34]	train-rmse:1.237787	test-rmse:3.076044 
#> [35]	train-rmse:1.228565	test-rmse:3.071678 
#> [36]	train-rmse:1.221586	test-rmse:3.067981 
#> [37]	train-rmse:1.202036	test-rmse:3.076470 
#> [38]	train-rmse:1.182295	test-rmse:3.077216 
#> [39]	train-rmse:1.174065	test-rmse:3.069785 
#> [40]	train-rmse:1.143644	test-rmse:3.075048 
#> [41]	train-rmse:1.118070	test-rmse:3.086533 
#> [42]	train-rmse:1.083684	test-rmse:3.076990 
#> [43]	train-rmse:1.067657	test-rmse:3.085092 
#> [44]	train-rmse:1.056737	test-rmse:3.075374 
#> [45]	train-rmse:1.042070	test-rmse:3.074781 
#> [46]	train-rmse:1.034100	test-rmse:3.069572 
#> [47]	train-rmse:1.018872	test-rmse:3.060232 
#> [48]	train-rmse:1.009841	test-rmse:3.057160 
#> [49]	train-rmse:0.988343	test-rmse:3.040026 
#> [50]	train-rmse:0.969062	test-rmse:3.043817 
#> [51]	train-rmse:0.948150	test-rmse:3.044994 
#> [52]	train-rmse:0.931555	test-rmse:3.055855 
#> [53]	train-rmse:0.927358	test-rmse:3.052031 
#> [54]	train-rmse:0.919497	test-rmse:3.048521 
#> [55]	train-rmse:0.903379	test-rmse:3.048413 
#> [56]	train-rmse:0.891003	test-rmse:3.043118 
#> [57]	train-rmse:0.878901	test-rmse:3.054984 
#> [58]	train-rmse:0.868691	test-rmse:3.054022 
#> [59]	train-rmse:0.855001	test-rmse:3.048966 
#> [60]	train-rmse:0.834860	test-rmse:3.035608 
#> [61]	train-rmse:0.816049	test-rmse:3.037272 
#> [62]	train-rmse:0.798031	test-rmse:3.025575 
#> [63]	train-rmse:0.783671	test-rmse:3.025780 
#> [64]	train-rmse:0.772990	test-rmse:3.024913 
#> [65]	train-rmse:0.760798	test-rmse:3.019602 
#> [66]	train-rmse:0.743240	test-rmse:3.011554 
#> [67]	train-rmse:0.733533	test-rmse:3.008570 
#> [68]	train-rmse:0.721323	test-rmse:3.005192 
#> [69]	train-rmse:0.704381	test-rmse:3.005989 
#> [70]	train-rmse:0.701263	test-rmse:3.002740 
#> [1]	train-rmse:17.111532	test-rmse:17.435272 
#> [2]	train-rmse:12.387211	test-rmse:13.025047 
#> [3]	train-rmse:9.085726	test-rmse:9.873093 
#> [4]	train-rmse:6.789982	test-rmse:7.752298 
#> [5]	train-rmse:5.207629	test-rmse:6.384945 
#> [6]	train-rmse:4.115838	test-rmse:5.447950 
#> [7]	train-rmse:3.366020	test-rmse:4.894496 
#> [8]	train-rmse:2.879771	test-rmse:4.580277 
#> [9]	train-rmse:2.569030	test-rmse:4.453151 
#> [10]	train-rmse:2.325202	test-rmse:4.457078 
#> [11]	train-rmse:2.191495	test-rmse:4.411079 
#> [12]	train-rmse:2.052944	test-rmse:4.384738 
#> [13]	train-rmse:1.977015	test-rmse:4.345249 
#> [14]	train-rmse:1.924209	test-rmse:4.342288 
#> [15]	train-rmse:1.831577	test-rmse:4.261086 
#> [16]	train-rmse:1.796415	test-rmse:4.231911 
#> [17]	train-rmse:1.754201	test-rmse:4.226254 
#> [18]	train-rmse:1.701770	test-rmse:4.220641 
#> [19]	train-rmse:1.634846	test-rmse:4.220300 
#> [20]	train-rmse:1.616856	test-rmse:4.206086 
#> [21]	train-rmse:1.572590	test-rmse:4.177235 
#> [22]	train-rmse:1.546168	test-rmse:4.169682 
#> [23]	train-rmse:1.504574	test-rmse:4.170857 
#> [24]	train-rmse:1.482746	test-rmse:4.172313 
#> [25]	train-rmse:1.430754	test-rmse:4.178851 
#> [26]	train-rmse:1.404922	test-rmse:4.172277 
#> [27]	train-rmse:1.374729	test-rmse:4.174905 
#> [28]	train-rmse:1.323425	test-rmse:4.161761 
#> [29]	train-rmse:1.307501	test-rmse:4.154302 
#> [30]	train-rmse:1.282739	test-rmse:4.150683 
#> [31]	train-rmse:1.251377	test-rmse:4.131608 
#> [32]	train-rmse:1.221794	test-rmse:4.143298 
#> [33]	train-rmse:1.185037	test-rmse:4.148625 
#> [34]	train-rmse:1.150993	test-rmse:4.156402 
#> [35]	train-rmse:1.119772	test-rmse:4.160789 
#> [36]	train-rmse:1.090457	test-rmse:4.160090 
#> [37]	train-rmse:1.060700	test-rmse:4.162218 
#> [38]	train-rmse:1.044813	test-rmse:4.165447 
#> [39]	train-rmse:1.032574	test-rmse:4.157378 
#> [40]	train-rmse:1.013405	test-rmse:4.153388 
#> [41]	train-rmse:1.006148	test-rmse:4.142999 
#> [42]	train-rmse:1.000142	test-rmse:4.140634 
#> [43]	train-rmse:0.976399	test-rmse:4.131345 
#> [44]	train-rmse:0.963216	test-rmse:4.132121 
#> [45]	train-rmse:0.943239	test-rmse:4.130291 
#> [46]	train-rmse:0.933755	test-rmse:4.129046 
#> [47]	train-rmse:0.923550	test-rmse:4.116899 
#> [48]	train-rmse:0.901284	test-rmse:4.114111 
#> [49]	train-rmse:0.884423	test-rmse:4.106702 
#> [50]	train-rmse:0.870152	test-rmse:4.107586 
#> [51]	train-rmse:0.859978	test-rmse:4.108229 
#> [52]	train-rmse:0.840108	test-rmse:4.108760 
#> [53]	train-rmse:0.816931	test-rmse:4.108813 
#> [54]	train-rmse:0.811576	test-rmse:4.109076 
#> [55]	train-rmse:0.800264	test-rmse:4.109824 
#> [56]	train-rmse:0.781994	test-rmse:4.102288 
#> [57]	train-rmse:0.775396	test-rmse:4.101467 
#> [58]	train-rmse:0.767829	test-rmse:4.104483 
#> [59]	train-rmse:0.760129	test-rmse:4.106617 
#> [60]	train-rmse:0.754196	test-rmse:4.106342 
#> [61]	train-rmse:0.749437	test-rmse:4.105491 
#> [62]	train-rmse:0.731089	test-rmse:4.104831 
#> [63]	train-rmse:0.715924	test-rmse:4.100628 
#> [64]	train-rmse:0.704701	test-rmse:4.105410 
#> [65]	train-rmse:0.693998	test-rmse:4.100563 
#> [66]	train-rmse:0.683421	test-rmse:4.099929 
#> [67]	train-rmse:0.674604	test-rmse:4.101201 
#> [68]	train-rmse:0.667805	test-rmse:4.101812 
#> [69]	train-rmse:0.654981	test-rmse:4.102751 
#> [70]	train-rmse:0.647825	test-rmse:4.100728 
#> [1]	train-rmse:16.923413	test-rmse:17.858287 
#> [2]	train-rmse:12.250960	test-rmse:13.245702 
#> [3]	train-rmse:9.008995	test-rmse:10.269762 
#> [4]	train-rmse:6.790111	test-rmse:8.118332 
#> [5]	train-rmse:5.263897	test-rmse:6.737373 
#> [6]	train-rmse:4.226298	test-rmse:5.910818 
#> [7]	train-rmse:3.513853	test-rmse:5.386048 
#> [8]	train-rmse:3.060588	test-rmse:4.971588 
#> [9]	train-rmse:2.704180	test-rmse:4.667367 
#> [10]	train-rmse:2.500957	test-rmse:4.520007 
#> [11]	train-rmse:2.360276	test-rmse:4.424577 
#> [12]	train-rmse:2.255008	test-rmse:4.268448 
#> [13]	train-rmse:2.150293	test-rmse:4.239951 
#> [14]	train-rmse:2.050543	test-rmse:4.181015 
#> [15]	train-rmse:2.003452	test-rmse:4.137171 
#> [16]	train-rmse:1.924608	test-rmse:4.095233 
#> [17]	train-rmse:1.861110	test-rmse:4.081775 
#> [18]	train-rmse:1.794015	test-rmse:4.070671 
#> [19]	train-rmse:1.750976	test-rmse:3.980266 
#> [20]	train-rmse:1.704884	test-rmse:3.943720 
#> [21]	train-rmse:1.633313	test-rmse:3.872674 
#> [22]	train-rmse:1.600509	test-rmse:3.859818 
#> [23]	train-rmse:1.575060	test-rmse:3.850316 
#> [24]	train-rmse:1.549222	test-rmse:3.848650 
#> [25]	train-rmse:1.512176	test-rmse:3.804888 
#> [26]	train-rmse:1.490101	test-rmse:3.806708 
#> [27]	train-rmse:1.437855	test-rmse:3.796689 
#> [28]	train-rmse:1.413658	test-rmse:3.800822 
#> [29]	train-rmse:1.391086	test-rmse:3.784102 
#> [30]	train-rmse:1.368214	test-rmse:3.751162 
#> [31]	train-rmse:1.341678	test-rmse:3.765602 
#> [32]	train-rmse:1.324352	test-rmse:3.766968 
#> [33]	train-rmse:1.267682	test-rmse:3.733680 
#> [34]	train-rmse:1.231068	test-rmse:3.735550 
#> [35]	train-rmse:1.211650	test-rmse:3.737077 
#> [36]	train-rmse:1.179801	test-rmse:3.718929 
#> [37]	train-rmse:1.145159	test-rmse:3.727204 
#> [38]	train-rmse:1.128239	test-rmse:3.711813 
#> [39]	train-rmse:1.103570	test-rmse:3.699716 
#> [40]	train-rmse:1.085133	test-rmse:3.701658 
#> [41]	train-rmse:1.073124	test-rmse:3.702788 
#> [42]	train-rmse:1.062533	test-rmse:3.699433 
#> [43]	train-rmse:1.045937	test-rmse:3.693495 
#> [44]	train-rmse:1.022968	test-rmse:3.688230 
#> [45]	train-rmse:1.012610	test-rmse:3.700308 
#> [46]	train-rmse:1.002660	test-rmse:3.698268 
#> [47]	train-rmse:0.995678	test-rmse:3.708165 
#> [48]	train-rmse:0.987307	test-rmse:3.690949 
#> [49]	train-rmse:0.976949	test-rmse:3.690472 
#> [50]	train-rmse:0.969292	test-rmse:3.701235 
#> [51]	train-rmse:0.951285	test-rmse:3.692314 
#> [52]	train-rmse:0.929873	test-rmse:3.662860 
#> [53]	train-rmse:0.918629	test-rmse:3.656849 
#> [54]	train-rmse:0.895356	test-rmse:3.640781 
#> [55]	train-rmse:0.881782	test-rmse:3.643711 
#> [56]	train-rmse:0.866274	test-rmse:3.652540 
#> [57]	train-rmse:0.856896	test-rmse:3.650975 
#> [58]	train-rmse:0.842923	test-rmse:3.640053 
#> [59]	train-rmse:0.834285	test-rmse:3.638969 
#> [60]	train-rmse:0.829124	test-rmse:3.643577 
#> [61]	train-rmse:0.818701	test-rmse:3.631377 
#> [62]	train-rmse:0.793543	test-rmse:3.628659 
#> [63]	train-rmse:0.780059	test-rmse:3.628344 
#> [64]	train-rmse:0.766899	test-rmse:3.633353 
#> [65]	train-rmse:0.756453	test-rmse:3.630871 
#> [66]	train-rmse:0.734865	test-rmse:3.627766 
#> [67]	train-rmse:0.730521	test-rmse:3.624072 
#> [68]	train-rmse:0.712303	test-rmse:3.613017 
#> [69]	train-rmse:0.706751	test-rmse:3.618362 
#> [70]	train-rmse:0.691661	test-rmse:3.615868 
#> [1]	train-rmse:17.094472	test-rmse:17.443979 
#> [2]	train-rmse:12.370325	test-rmse:12.648890 
#> [3]	train-rmse:9.136447	test-rmse:9.424303 
#> [4]	train-rmse:6.815337	test-rmse:7.274856 
#> [5]	train-rmse:5.243661	test-rmse:5.833762 
#> [6]	train-rmse:4.178753	test-rmse:4.908622 
#> [7]	train-rmse:3.467034	test-rmse:4.275468 
#> [8]	train-rmse:3.008385	test-rmse:3.939571 
#> [9]	train-rmse:2.675041	test-rmse:3.773246 
#> [10]	train-rmse:2.474950	test-rmse:3.632611 
#> [11]	train-rmse:2.307720	test-rmse:3.533710 
#> [12]	train-rmse:2.177433	test-rmse:3.477157 
#> [13]	train-rmse:2.079583	test-rmse:3.439424 
#> [14]	train-rmse:1.998522	test-rmse:3.388455 
#> [15]	train-rmse:1.900939	test-rmse:3.346838 
#> [16]	train-rmse:1.821591	test-rmse:3.344501 
#> [17]	train-rmse:1.778242	test-rmse:3.331574 
#> [18]	train-rmse:1.716799	test-rmse:3.347789 
#> [19]	train-rmse:1.676291	test-rmse:3.338689 
#> [20]	train-rmse:1.632078	test-rmse:3.342672 
#> [21]	train-rmse:1.610130	test-rmse:3.323012 
#> [22]	train-rmse:1.591450	test-rmse:3.319949 
#> [23]	train-rmse:1.551755	test-rmse:3.299841 
#> [24]	train-rmse:1.536861	test-rmse:3.301777 
#> [25]	train-rmse:1.481012	test-rmse:3.276894 
#> [26]	train-rmse:1.454397	test-rmse:3.269283 
#> [27]	train-rmse:1.415888	test-rmse:3.254819 
#> [28]	train-rmse:1.384320	test-rmse:3.249840 
#> [29]	train-rmse:1.368003	test-rmse:3.243054 
#> [30]	train-rmse:1.344136	test-rmse:3.254499 
#> [31]	train-rmse:1.329390	test-rmse:3.257611 
#> [32]	train-rmse:1.301899	test-rmse:3.254902 
#> [33]	train-rmse:1.285813	test-rmse:3.253238 
#> [34]	train-rmse:1.275745	test-rmse:3.243495 
#> [35]	train-rmse:1.267246	test-rmse:3.239862 
#> [36]	train-rmse:1.244695	test-rmse:3.241509 
#> [37]	train-rmse:1.233288	test-rmse:3.246018 
#> [38]	train-rmse:1.223915	test-rmse:3.244156 
#> [39]	train-rmse:1.211425	test-rmse:3.244399 
#> [40]	train-rmse:1.180329	test-rmse:3.242740 
#> [41]	train-rmse:1.171058	test-rmse:3.241273 
#> [42]	train-rmse:1.161724	test-rmse:3.237195 
#> [43]	train-rmse:1.156259	test-rmse:3.238887 
#> [44]	train-rmse:1.134589	test-rmse:3.234876 
#> [45]	train-rmse:1.111284	test-rmse:3.224325 
#> [46]	train-rmse:1.084095	test-rmse:3.223480 
#> [47]	train-rmse:1.067216	test-rmse:3.226552 
#> [48]	train-rmse:1.059544	test-rmse:3.227267 
#> [49]	train-rmse:1.052601	test-rmse:3.232803 
#> [50]	train-rmse:1.029946	test-rmse:3.229616 
#> [51]	train-rmse:1.007692	test-rmse:3.233152 
#> [52]	train-rmse:0.989256	test-rmse:3.236416 
#> [53]	train-rmse:0.964192	test-rmse:3.234155 
#> [54]	train-rmse:0.947695	test-rmse:3.242226 
#> [55]	train-rmse:0.926670	test-rmse:3.237352 
#> [56]	train-rmse:0.905204	test-rmse:3.235260 
#> [57]	train-rmse:0.890484	test-rmse:3.230032 
#> [58]	train-rmse:0.881130	test-rmse:3.232472 
#> [59]	train-rmse:0.865016	test-rmse:3.228469 
#> [60]	train-rmse:0.842762	test-rmse:3.229937 
#> [61]	train-rmse:0.828306	test-rmse:3.236761 
#> [62]	train-rmse:0.810322	test-rmse:3.237922 
#> [63]	train-rmse:0.793606	test-rmse:3.239330 
#> [64]	train-rmse:0.784651	test-rmse:3.244624 
#> [65]	train-rmse:0.765355	test-rmse:3.251084 
#> [66]	train-rmse:0.754189	test-rmse:3.247930 
#> [67]	train-rmse:0.738733	test-rmse:3.249863 
#> [68]	train-rmse:0.729666	test-rmse:3.248329 
#> [69]	train-rmse:0.714781	test-rmse:3.248858 
#> [70]	train-rmse:0.705361	test-rmse:3.250069 
#> [1]	train-rmse:17.170011	test-rmse:17.171257 
#> [2]	train-rmse:12.426524	test-rmse:12.525679 
#> [3]	train-rmse:9.117093	test-rmse:9.459849 
#> [4]	train-rmse:6.793125	test-rmse:7.351778 
#> [5]	train-rmse:5.236309	test-rmse:5.996796 
#> [6]	train-rmse:4.151535	test-rmse:5.057847 
#> [7]	train-rmse:3.447785	test-rmse:4.453109 
#> [8]	train-rmse:2.964770	test-rmse:4.025576 
#> [9]	train-rmse:2.648792	test-rmse:3.819552 
#> [10]	train-rmse:2.440856	test-rmse:3.740400 
#> [11]	train-rmse:2.284757	test-rmse:3.673686 
#> [12]	train-rmse:2.147940	test-rmse:3.634061 
#> [13]	train-rmse:2.056752	test-rmse:3.567147 
#> [14]	train-rmse:1.988819	test-rmse:3.519069 
#> [15]	train-rmse:1.910203	test-rmse:3.509014 
#> [16]	train-rmse:1.855831	test-rmse:3.488891 
#> [17]	train-rmse:1.825984	test-rmse:3.455567 
#> [18]	train-rmse:1.784249	test-rmse:3.450392 
#> [19]	train-rmse:1.742281	test-rmse:3.429652 
#> [20]	train-rmse:1.683531	test-rmse:3.424123 
#> [21]	train-rmse:1.644295	test-rmse:3.415885 
#> [22]	train-rmse:1.620427	test-rmse:3.397409 
#> [23]	train-rmse:1.591121	test-rmse:3.396034 
#> [24]	train-rmse:1.557370	test-rmse:3.359075 
#> [25]	train-rmse:1.517061	test-rmse:3.363934 
#> [26]	train-rmse:1.476960	test-rmse:3.358385 
#> [27]	train-rmse:1.459281	test-rmse:3.353978 
#> [28]	train-rmse:1.434989	test-rmse:3.373883 
#> [29]	train-rmse:1.417344	test-rmse:3.367324 
#> [30]	train-rmse:1.391160	test-rmse:3.361120 
#> [31]	train-rmse:1.371772	test-rmse:3.367616 
#> [32]	train-rmse:1.337820	test-rmse:3.368026 
#> [33]	train-rmse:1.319915	test-rmse:3.360235 
#> [34]	train-rmse:1.296343	test-rmse:3.350689 
#> [35]	train-rmse:1.263589	test-rmse:3.361618 
#> [36]	train-rmse:1.244863	test-rmse:3.366146 
#> [37]	train-rmse:1.194421	test-rmse:3.335086 
#> [38]	train-rmse:1.184094	test-rmse:3.334097 
#> [39]	train-rmse:1.156357	test-rmse:3.328963 
#> [40]	train-rmse:1.144884	test-rmse:3.326506 
#> [41]	train-rmse:1.117081	test-rmse:3.340203 
#> [42]	train-rmse:1.106197	test-rmse:3.341041 
#> [43]	train-rmse:1.100150	test-rmse:3.342239 
#> [44]	train-rmse:1.077481	test-rmse:3.347784 
#> [45]	train-rmse:1.046810	test-rmse:3.338964 
#> [46]	train-rmse:1.034026	test-rmse:3.351019 
#> [47]	train-rmse:1.023361	test-rmse:3.342007 
#> [48]	train-rmse:0.994702	test-rmse:3.341222 
#> [49]	train-rmse:0.989948	test-rmse:3.337105 
#> [50]	train-rmse:0.975026	test-rmse:3.336172 
#> [51]	train-rmse:0.952664	test-rmse:3.331605 
#> [52]	train-rmse:0.945468	test-rmse:3.334575 
#> [53]	train-rmse:0.922853	test-rmse:3.341329 
#> [54]	train-rmse:0.909634	test-rmse:3.337151 
#> [55]	train-rmse:0.903042	test-rmse:3.333116 
#> [56]	train-rmse:0.884904	test-rmse:3.326880 
#> [57]	train-rmse:0.874922	test-rmse:3.320552 
#> [58]	train-rmse:0.868196	test-rmse:3.319108 
#> [59]	train-rmse:0.848900	test-rmse:3.319444 
#> [60]	train-rmse:0.841934	test-rmse:3.326078 
#> [61]	train-rmse:0.838304	test-rmse:3.326217 
#> [62]	train-rmse:0.822154	test-rmse:3.318379 
#> [63]	train-rmse:0.807249	test-rmse:3.313129 
#> [64]	train-rmse:0.797583	test-rmse:3.309362 
#> [65]	train-rmse:0.782169	test-rmse:3.302684 
#> [66]	train-rmse:0.770329	test-rmse:3.298346 
#> [67]	train-rmse:0.765190	test-rmse:3.305168 
#> [68]	train-rmse:0.758894	test-rmse:3.307550 
#> [69]	train-rmse:0.745356	test-rmse:3.311324 
#> [70]	train-rmse:0.739772	test-rmse:3.313405 
#> [1]	train-rmse:16.582907	test-rmse:18.207723 
#> [2]	train-rmse:12.001622	test-rmse:13.722996 
#> [3]	train-rmse:8.787949	test-rmse:10.474234 
#> [4]	train-rmse:6.600602	test-rmse:8.351502 
#> [5]	train-rmse:5.044425	test-rmse:6.781755 
#> [6]	train-rmse:3.998929	test-rmse:5.842432 
#> [7]	train-rmse:3.313856	test-rmse:5.326463 
#> [8]	train-rmse:2.880562	test-rmse:5.021548 
#> [9]	train-rmse:2.574312	test-rmse:4.744890 
#> [10]	train-rmse:2.372165	test-rmse:4.600522 
#> [11]	train-rmse:2.222040	test-rmse:4.535886 
#> [12]	train-rmse:2.115739	test-rmse:4.437634 
#> [13]	train-rmse:2.042316	test-rmse:4.430661 
#> [14]	train-rmse:1.938443	test-rmse:4.418263 
#> [15]	train-rmse:1.876714	test-rmse:4.404265 
#> [16]	train-rmse:1.823690	test-rmse:4.384746 
#> [17]	train-rmse:1.757006	test-rmse:4.377393 
#> [18]	train-rmse:1.708880	test-rmse:4.372659 
#> [19]	train-rmse:1.670362	test-rmse:4.356115 
#> [20]	train-rmse:1.624955	test-rmse:4.323082 
#> [21]	train-rmse:1.586701	test-rmse:4.314782 
#> [22]	train-rmse:1.563975	test-rmse:4.308469 
#> [23]	train-rmse:1.528573	test-rmse:4.300380 
#> [24]	train-rmse:1.513181	test-rmse:4.301769 
#> [25]	train-rmse:1.456300	test-rmse:4.268720 
#> [26]	train-rmse:1.399742	test-rmse:4.257959 
#> [27]	train-rmse:1.361863	test-rmse:4.220169 
#> [28]	train-rmse:1.345355	test-rmse:4.214260 
#> [29]	train-rmse:1.308177	test-rmse:4.189072 
#> [30]	train-rmse:1.283864	test-rmse:4.176219 
#> [31]	train-rmse:1.259614	test-rmse:4.195716 
#> [32]	train-rmse:1.222292	test-rmse:4.186206 
#> [33]	train-rmse:1.204266	test-rmse:4.187286 
#> [34]	train-rmse:1.173770	test-rmse:4.192971 
#> [35]	train-rmse:1.146007	test-rmse:4.190610 
#> [36]	train-rmse:1.135612	test-rmse:4.191723 
#> [37]	train-rmse:1.116079	test-rmse:4.191155 
#> [38]	train-rmse:1.101917	test-rmse:4.190144 
#> [39]	train-rmse:1.066302	test-rmse:4.181161 
#> [40]	train-rmse:1.045433	test-rmse:4.181066 
#> [41]	train-rmse:1.021536	test-rmse:4.182448 
#> [42]	train-rmse:1.012055	test-rmse:4.172824 
#> [43]	train-rmse:0.998246	test-rmse:4.164631 
#> [44]	train-rmse:0.981645	test-rmse:4.157344 
#> [45]	train-rmse:0.969975	test-rmse:4.158212 
#> [46]	train-rmse:0.962992	test-rmse:4.159398 
#> [47]	train-rmse:0.938172	test-rmse:4.162487 
#> [48]	train-rmse:0.909808	test-rmse:4.149913 
#> [49]	train-rmse:0.888373	test-rmse:4.146672 
#> [50]	train-rmse:0.883702	test-rmse:4.145806 
#> [51]	train-rmse:0.878323	test-rmse:4.147840 
#> [52]	train-rmse:0.865686	test-rmse:4.147482 
#> [53]	train-rmse:0.856467	test-rmse:4.141376 
#> [54]	train-rmse:0.851313	test-rmse:4.140475 
#> [55]	train-rmse:0.824926	test-rmse:4.125626 
#> [56]	train-rmse:0.807213	test-rmse:4.130817 
#> [57]	train-rmse:0.795934	test-rmse:4.133379 
#> [58]	train-rmse:0.782519	test-rmse:4.129859 
#> [59]	train-rmse:0.767906	test-rmse:4.130241 
#> [60]	train-rmse:0.756362	test-rmse:4.126689 
#> [61]	train-rmse:0.750580	test-rmse:4.125184 
#> [62]	train-rmse:0.739081	test-rmse:4.125837 
#> [63]	train-rmse:0.734751	test-rmse:4.124346 
#> [64]	train-rmse:0.727189	test-rmse:4.116006 
#> [65]	train-rmse:0.719912	test-rmse:4.119719 
#> [66]	train-rmse:0.711155	test-rmse:4.123465 
#> [67]	train-rmse:0.692546	test-rmse:4.119317 
#> [68]	train-rmse:0.681082	test-rmse:4.113232 
#> [69]	train-rmse:0.677283	test-rmse:4.110573 
#> [70]	train-rmse:0.661992	test-rmse:4.113574
#> [1] 3.449206
warnings() # no warnings for individual XGBoost function
```
