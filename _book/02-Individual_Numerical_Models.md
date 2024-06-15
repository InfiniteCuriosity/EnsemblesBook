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


``` r
library(e1071) # will allow us to use a tuned random forest model
library(Metrics) # Will allow us to calculate the root mean squared error
library(randomForest) # To use the random forest function
#> randomForest 4.7-1.1
#> Type rfNews() to see new features/changes/bug fixes.
```

``` r
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


``` r
# Set initial values to 0. The function will return an error if any of these are left out.

bag_rf_holdout_RMSE <- 0
bag_rf_holdout_RMSE_mean <- 0
bag_rf_train_RMSE <- 0
bag_rf_test_RMSE <- 0
bag_rf_validation_RMSE <- 0
```


``` r

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
#> [1] 0.3043472
```

``` r
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


``` r
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
#> [1] 4.012994
```

``` r
warnings() # no warnings
```

### 3. BayesGLM


``` r
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
#> Working directory is /Users/russellconte/Library/Mobile Documents/com~apple~CloudDocs/Documents/Machine Learning templates in R/EnsemblesBook
```

``` r

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
#> [1] 4.766784
```

``` r
warnings() # no warnings
```

### 4. BayesRNN


``` r
library(brnn) # so we can use the BayesRNN function
#> Loading required package: Formula
#> Loading required package: truncnorm
```

``` r

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
#> Scaling factor= 0.7016579 
#> gamma= 30.9146 	 alpha= 4.5072 	 beta= 19287.31 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7017412 
#> gamma= 31.0361 	 alpha= 3.1851 	 beta= 14704.06 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015619 
#> gamma= 31.2686 	 alpha= 5.5043 	 beta= 14369.86 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016138 
#> gamma= 31.5051 	 alpha= 3.2518 	 beta= 14287.47 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016246 
#> gamma= 31.4638 	 alpha= 5.7484 	 beta= 14045.49 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016751 
#> gamma= 31.6047 	 alpha= 5.2739 	 beta= 13583.82 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015323 
#> gamma= 30.2174 	 alpha= 3.217 	 beta= 19069.97 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016694 
#> gamma= 30.7135 	 alpha= 5.0462 	 beta= 15276.49 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.701572 
#> gamma= 30.4745 	 alpha= 4.9501 	 beta= 18702.41 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016356 
#> gamma= 30.4193 	 alpha= 4.9632 	 beta= 18163.57 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015132 
#> gamma= 31.4395 	 alpha= 5.3713 	 beta= 14690.09 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015874 
#> gamma= 30.9879 	 alpha= 4.5776 	 beta= 33212.92 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016694 
#> gamma= 30.3109 	 alpha= 5.1169 	 beta= 22330 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7014674 
#> gamma= 31.0504 	 alpha= 5.3045 	 beta= 16203.19 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016579 
#> gamma= 31.3034 	 alpha= 5.5572 	 beta= 14265.58 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015323 
#> gamma= 31.5195 	 alpha= 5.4152 	 beta= 16280.12 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016246 
#> gamma= 29.1298 	 alpha= 4.4154 	 beta= 19890.05 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016523 
#> gamma= 31.4777 	 alpha= 5.7411 	 beta= 14662.8 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7014899 
#> gamma= 31.6415 	 alpha= 4.6533 	 beta= 17256.76 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015469 
#> gamma= 31.3987 	 alpha= 5.3242 	 beta= 14257.68 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016246 
#> gamma= 31.5294 	 alpha= 5.1328 	 beta= 14909.11 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016356 
#> gamma= 31.3673 	 alpha= 5.0348 	 beta= 16423.81 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015771 
#> gamma= 31.6411 	 alpha= 5.7092 	 beta= 14018.46 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015519 
#> gamma= 31.3911 	 alpha= 5.5519 	 beta= 15234.52 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7017288 
#> gamma= 30.9045 	 alpha= 4.9779 	 beta= 14752.83
#> [1] 0.1430853
```

``` r

warnings() # no warnings for BayesRNN function
```

### 5. Boosted Random Forest


``` r
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
#> [1] 0.3278237
```

``` r
warnings() # no warnings for Boosted Random Forest function
```

### 6. Cubist


``` r
library(Cubist)
#> Loading required package: lattice
```

``` r
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
#> [1] 4.400205
```

``` r
warnings() # no warnings for individual cubist function
```

### 7. Elastic


``` r

library(glmnet) # So we can run the elastic model
#> Loaded glmnet 4.1-8
```

``` r
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
#> [1] 5.060079
```

``` r
warnings() # no warnings for individual elastic function
```

### 8. Generalized Additive Models with smoothing splines


``` r
library(gam) # for fitting generalized additive models
#> Loading required package: splines
#> Loading required package: foreach
#> 
#> Attaching package: 'foreach'
#> The following objects are masked from 'package:purrr':
#> 
#>     accumulate, when
#> Loaded gam 1.22-3
```

``` r

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
#> [1] 5.051043
```

``` r
warnings() # no warnings for individual gam function
```

### 9. Gradient Boosted


``` r
library(gbm) # to allow use of gradient boosted models
#> Loaded gbm 2.1.9
#> This version of gbm is no longer under development. Consider transitioning to gbm3, https://github.com/gbm-developers/gbm3
```

``` r

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
#> [1] 3.579291
```

``` r
warnings() # no warnings for individual gradient boosted function
```

### 10. K-Nearest Neighbors (tuned)


``` r

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
#> [1] 7.052156
```

``` r
warnings() # no warnings for individual knn function
```

### 11. Lasso


``` r
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
#> [1] 5.010819
```

``` r
warnings() # no warnings for individual lasso function
```

### 12. Linear (tuned)


``` r

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
#> [1] 4.805244
```

``` r
warnings() # no warnings for individual lasso function
```

### 13. LQS


``` r

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
#> [1] 7.741189
```

``` r
warnings() # no warnings for individual lqs function
```

### 14. Neuralnet


``` r
library(neuralnet)
#> 
#> Attaching package: 'neuralnet'
#> The following object is masked from 'package:dplyr':
#> 
#>     compute
```

``` r

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
#> [1] 3.934191
```

``` r
warnings() # no warnings for individual neuralnet function
```

### 15. Partial Least Squares


``` r

library(pls)
#> 
#> Attaching package: 'pls'
#> The following objects are masked from 'package:arm':
#> 
#>     coefplot, corrplot
#> The following object is masked from 'package:stats':
#> 
#>     loadings
```

``` r

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
#> [1] 6.160096
```

``` r
warnings() # no warnings for individual pls function
```

### 16. Principal Components Regression


``` r

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
#> [1] 6.444914
```

``` r
warnings() # no warnings for individual pls function
```

### 17. Random Forest


``` r
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
#> [1] 1.767656
```

``` r
warnings() # no warnings for individual random forest function
```

### 18. Ridge Regression


``` r

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
#> [1] 4.993241
```

``` r
warnings() # no warnings for individual ridge function
```

### 19. Robust Regression


``` r

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
#> [1] 5.054642
```

``` r
warnings() # no warnings for individual robust function
```

### 20. Rpart


``` r

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
#> [1] 4.793699
```

``` r
warnings() # no warnings for individual rpart function
```

### 21. Support Vector Machines


``` r

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
#> [1] 2.426661
```

``` r
warnings() # no warnings for individual Support Vector Machines function
```

### 22. Trees


``` r

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
#> [1] 4.802842
```

``` r
warnings() # no warnings for individual tree function
```

### 23. XGBoost


``` r
library(xgboost)
#> 
#> Attaching package: 'xgboost'
#> The following object is masked from 'package:dplyr':
#> 
#>     slice
```

``` r

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
#> [1]	train-rmse:16.566847	test-rmse:18.295836 
#> [2]	train-rmse:12.001877	test-rmse:13.600278 
#> [3]	train-rmse:8.857524	test-rmse:10.447422 
#> [4]	train-rmse:6.648565	test-rmse:8.334287 
#> [5]	train-rmse:5.137718	test-rmse:7.013896 
#> [6]	train-rmse:4.116261	test-rmse:6.110523 
#> [7]	train-rmse:3.420488	test-rmse:5.520650 
#> [8]	train-rmse:3.002864	test-rmse:5.227048 
#> [9]	train-rmse:2.707789	test-rmse:5.022301 
#> [10]	train-rmse:2.521882	test-rmse:4.852310 
#> [11]	train-rmse:2.405066	test-rmse:4.791663 
#> [12]	train-rmse:2.320092	test-rmse:4.691732 
#> [13]	train-rmse:2.240695	test-rmse:4.632454 
#> [14]	train-rmse:2.156235	test-rmse:4.562374 
#> [15]	train-rmse:2.119126	test-rmse:4.524297 
#> [16]	train-rmse:2.079369	test-rmse:4.478217 
#> [17]	train-rmse:2.020774	test-rmse:4.452389 
#> [18]	train-rmse:1.946407	test-rmse:4.365773 
#> [19]	train-rmse:1.857629	test-rmse:4.303493 
#> [20]	train-rmse:1.835047	test-rmse:4.283662 
#> [21]	train-rmse:1.796012	test-rmse:4.277377 
#> [22]	train-rmse:1.763180	test-rmse:4.273819 
#> [23]	train-rmse:1.748976	test-rmse:4.272261 
#> [24]	train-rmse:1.689464	test-rmse:4.274587 
#> [25]	train-rmse:1.660090	test-rmse:4.261288 
#> [26]	train-rmse:1.648596	test-rmse:4.273753 
#> [27]	train-rmse:1.587294	test-rmse:4.261752 
#> [28]	train-rmse:1.557944	test-rmse:4.243167 
#> [29]	train-rmse:1.541092	test-rmse:4.250105 
#> [30]	train-rmse:1.512719	test-rmse:4.254290 
#> [31]	train-rmse:1.495594	test-rmse:4.235982 
#> [32]	train-rmse:1.471548	test-rmse:4.237962 
#> [33]	train-rmse:1.461947	test-rmse:4.230278 
#> [34]	train-rmse:1.440187	test-rmse:4.217669 
#> [35]	train-rmse:1.412271	test-rmse:4.217333 
#> [36]	train-rmse:1.394335	test-rmse:4.218771 
#> [37]	train-rmse:1.354841	test-rmse:4.211338 
#> [38]	train-rmse:1.344822	test-rmse:4.211056 
#> [39]	train-rmse:1.312267	test-rmse:4.203068 
#> [40]	train-rmse:1.280977	test-rmse:4.206494 
#> [41]	train-rmse:1.261975	test-rmse:4.198913 
#> [42]	train-rmse:1.248094	test-rmse:4.185089 
#> [43]	train-rmse:1.212051	test-rmse:4.189859 
#> [44]	train-rmse:1.199969	test-rmse:4.181182 
#> [45]	train-rmse:1.193709	test-rmse:4.180919 
#> [46]	train-rmse:1.163349	test-rmse:4.182952 
#> [47]	train-rmse:1.136975	test-rmse:4.158753 
#> [48]	train-rmse:1.112211	test-rmse:4.148918 
#> [49]	train-rmse:1.105006	test-rmse:4.154884 
#> [50]	train-rmse:1.088859	test-rmse:4.143711 
#> [51]	train-rmse:1.076607	test-rmse:4.150342 
#> [52]	train-rmse:1.072393	test-rmse:4.150674 
#> [53]	train-rmse:1.064686	test-rmse:4.150741 
#> [54]	train-rmse:1.053028	test-rmse:4.154434 
#> [55]	train-rmse:1.020980	test-rmse:4.166082 
#> [56]	train-rmse:0.998032	test-rmse:4.181283 
#> [57]	train-rmse:0.987385	test-rmse:4.176308 
#> [58]	train-rmse:0.962621	test-rmse:4.166486 
#> [59]	train-rmse:0.958672	test-rmse:4.172673 
#> [60]	train-rmse:0.935287	test-rmse:4.162383 
#> [61]	train-rmse:0.925546	test-rmse:4.157751 
#> [62]	train-rmse:0.920647	test-rmse:4.161055 
#> [63]	train-rmse:0.914087	test-rmse:4.164519 
#> [64]	train-rmse:0.891765	test-rmse:4.149640 
#> [65]	train-rmse:0.876797	test-rmse:4.151246 
#> [66]	train-rmse:0.866961	test-rmse:4.153165 
#> [67]	train-rmse:0.862838	test-rmse:4.153364 
#> [68]	train-rmse:0.846186	test-rmse:4.153644 
#> [69]	train-rmse:0.833005	test-rmse:4.153704 
#> [70]	train-rmse:0.818758	test-rmse:4.151786 
#> [1]	train-rmse:17.476615	test-rmse:16.712516 
#> [2]	train-rmse:12.706851	test-rmse:12.104079 
#> [3]	train-rmse:9.325248	test-rmse:8.891638 
#> [4]	train-rmse:6.967080	test-rmse:6.638769 
#> [5]	train-rmse:5.361310	test-rmse:5.243285 
#> [6]	train-rmse:4.273568	test-rmse:4.382146 
#> [7]	train-rmse:3.571172	test-rmse:3.854521 
#> [8]	train-rmse:3.096809	test-rmse:3.553356 
#> [9]	train-rmse:2.785035	test-rmse:3.411058 
#> [10]	train-rmse:2.542470	test-rmse:3.289487 
#> [11]	train-rmse:2.396477	test-rmse:3.266160 
#> [12]	train-rmse:2.249223	test-rmse:3.219625 
#> [13]	train-rmse:2.165404	test-rmse:3.248116 
#> [14]	train-rmse:2.082891	test-rmse:3.199139 
#> [15]	train-rmse:2.024894	test-rmse:3.190043 
#> [16]	train-rmse:1.965081	test-rmse:3.156556 
#> [17]	train-rmse:1.882508	test-rmse:3.154637 
#> [18]	train-rmse:1.828034	test-rmse:3.162224 
#> [19]	train-rmse:1.778963	test-rmse:3.188254 
#> [20]	train-rmse:1.726786	test-rmse:3.180868 
#> [21]	train-rmse:1.702797	test-rmse:3.170957 
#> [22]	train-rmse:1.667876	test-rmse:3.193273 
#> [23]	train-rmse:1.636254	test-rmse:3.197338 
#> [24]	train-rmse:1.588035	test-rmse:3.199324 
#> [25]	train-rmse:1.561565	test-rmse:3.203509 
#> [26]	train-rmse:1.523799	test-rmse:3.181351 
#> [27]	train-rmse:1.490133	test-rmse:3.183189 
#> [28]	train-rmse:1.474789	test-rmse:3.183359 
#> [29]	train-rmse:1.432077	test-rmse:3.168793 
#> [30]	train-rmse:1.399415	test-rmse:3.179798 
#> [31]	train-rmse:1.368031	test-rmse:3.164018 
#> [32]	train-rmse:1.339634	test-rmse:3.155773 
#> [33]	train-rmse:1.322923	test-rmse:3.167707 
#> [34]	train-rmse:1.312939	test-rmse:3.162325 
#> [35]	train-rmse:1.291858	test-rmse:3.162030 
#> [36]	train-rmse:1.266042	test-rmse:3.157177 
#> [37]	train-rmse:1.249867	test-rmse:3.155634 
#> [38]	train-rmse:1.233007	test-rmse:3.162457 
#> [39]	train-rmse:1.202567	test-rmse:3.158863 
#> [40]	train-rmse:1.192454	test-rmse:3.166728 
#> [41]	train-rmse:1.185654	test-rmse:3.161986 
#> [42]	train-rmse:1.156329	test-rmse:3.155749 
#> [43]	train-rmse:1.132966	test-rmse:3.156878 
#> [44]	train-rmse:1.099172	test-rmse:3.160830 
#> [45]	train-rmse:1.084264	test-rmse:3.166902 
#> [46]	train-rmse:1.077450	test-rmse:3.176762 
#> [47]	train-rmse:1.069471	test-rmse:3.172082 
#> [48]	train-rmse:1.047696	test-rmse:3.161650 
#> [49]	train-rmse:1.029634	test-rmse:3.164930 
#> [50]	train-rmse:1.011847	test-rmse:3.161962 
#> [51]	train-rmse:0.988660	test-rmse:3.161577 
#> [52]	train-rmse:0.966914	test-rmse:3.157507 
#> [53]	train-rmse:0.958587	test-rmse:3.158530 
#> [54]	train-rmse:0.946078	test-rmse:3.158107 
#> [55]	train-rmse:0.936126	test-rmse:3.159700 
#> [56]	train-rmse:0.920098	test-rmse:3.164522 
#> [57]	train-rmse:0.903369	test-rmse:3.158688 
#> [58]	train-rmse:0.888368	test-rmse:3.159415 
#> [59]	train-rmse:0.880155	test-rmse:3.161047 
#> [60]	train-rmse:0.869179	test-rmse:3.160468 
#> [61]	train-rmse:0.862545	test-rmse:3.161371 
#> [62]	train-rmse:0.858606	test-rmse:3.162070 
#> [63]	train-rmse:0.843402	test-rmse:3.155911 
#> [64]	train-rmse:0.839111	test-rmse:3.155235 
#> [65]	train-rmse:0.820921	test-rmse:3.167764 
#> [66]	train-rmse:0.813685	test-rmse:3.167962 
#> [67]	train-rmse:0.799078	test-rmse:3.165993 
#> [68]	train-rmse:0.792911	test-rmse:3.161664 
#> [69]	train-rmse:0.783525	test-rmse:3.157303 
#> [70]	train-rmse:0.763116	test-rmse:3.158526 
#> [1]	train-rmse:16.940826	test-rmse:17.620587 
#> [2]	train-rmse:12.287029	test-rmse:12.895806 
#> [3]	train-rmse:9.027421	test-rmse:9.716785 
#> [4]	train-rmse:6.807202	test-rmse:7.638635 
#> [5]	train-rmse:5.285410	test-rmse:6.357400 
#> [6]	train-rmse:4.227195	test-rmse:5.469514 
#> [7]	train-rmse:3.530453	test-rmse:4.865071 
#> [8]	train-rmse:3.043198	test-rmse:4.480110 
#> [9]	train-rmse:2.743274	test-rmse:4.283322 
#> [10]	train-rmse:2.503662	test-rmse:4.143151 
#> [11]	train-rmse:2.341390	test-rmse:4.081705 
#> [12]	train-rmse:2.223119	test-rmse:3.949371 
#> [13]	train-rmse:2.145158	test-rmse:3.869653 
#> [14]	train-rmse:2.070649	test-rmse:3.838803 
#> [15]	train-rmse:2.018632	test-rmse:3.823521 
#> [16]	train-rmse:1.932049	test-rmse:3.813291 
#> [17]	train-rmse:1.869901	test-rmse:3.774552 
#> [18]	train-rmse:1.834118	test-rmse:3.751064 
#> [19]	train-rmse:1.793967	test-rmse:3.744829 
#> [20]	train-rmse:1.752734	test-rmse:3.733358 
#> [21]	train-rmse:1.702178	test-rmse:3.752904 
#> [22]	train-rmse:1.654633	test-rmse:3.752624 
#> [23]	train-rmse:1.628374	test-rmse:3.750038 
#> [24]	train-rmse:1.595693	test-rmse:3.753647 
#> [25]	train-rmse:1.571161	test-rmse:3.746137 
#> [26]	train-rmse:1.544879	test-rmse:3.745566 
#> [27]	train-rmse:1.517060	test-rmse:3.732738 
#> [28]	train-rmse:1.485936	test-rmse:3.703911 
#> [29]	train-rmse:1.447648	test-rmse:3.696419 
#> [30]	train-rmse:1.413967	test-rmse:3.705468 
#> [31]	train-rmse:1.381059	test-rmse:3.706452 
#> [32]	train-rmse:1.344287	test-rmse:3.709454 
#> [33]	train-rmse:1.305949	test-rmse:3.701470 
#> [34]	train-rmse:1.267424	test-rmse:3.698444 
#> [35]	train-rmse:1.249651	test-rmse:3.686810 
#> [36]	train-rmse:1.242664	test-rmse:3.682350 
#> [37]	train-rmse:1.222246	test-rmse:3.676340 
#> [38]	train-rmse:1.200541	test-rmse:3.672680 
#> [39]	train-rmse:1.183422	test-rmse:3.674608 
#> [40]	train-rmse:1.162316	test-rmse:3.663248 
#> [41]	train-rmse:1.141830	test-rmse:3.655922 
#> [42]	train-rmse:1.128091	test-rmse:3.655636 
#> [43]	train-rmse:1.096402	test-rmse:3.649646 
#> [44]	train-rmse:1.085053	test-rmse:3.645754 
#> [45]	train-rmse:1.065475	test-rmse:3.635741 
#> [46]	train-rmse:1.058745	test-rmse:3.639901 
#> [47]	train-rmse:1.045324	test-rmse:3.631098 
#> [48]	train-rmse:1.020385	test-rmse:3.629003 
#> [49]	train-rmse:1.009853	test-rmse:3.629030 
#> [50]	train-rmse:0.989406	test-rmse:3.628485 
#> [51]	train-rmse:0.979389	test-rmse:3.630748 
#> [52]	train-rmse:0.960420	test-rmse:3.629449 
#> [53]	train-rmse:0.944153	test-rmse:3.632180 
#> [54]	train-rmse:0.932651	test-rmse:3.632644 
#> [55]	train-rmse:0.922669	test-rmse:3.633330 
#> [56]	train-rmse:0.903807	test-rmse:3.629754 
#> [57]	train-rmse:0.875761	test-rmse:3.624106 
#> [58]	train-rmse:0.862721	test-rmse:3.626499 
#> [59]	train-rmse:0.843045	test-rmse:3.622818 
#> [60]	train-rmse:0.829702	test-rmse:3.618196 
#> [61]	train-rmse:0.813128	test-rmse:3.612194 
#> [62]	train-rmse:0.806017	test-rmse:3.609211 
#> [63]	train-rmse:0.792542	test-rmse:3.600465 
#> [64]	train-rmse:0.781655	test-rmse:3.594607 
#> [65]	train-rmse:0.768599	test-rmse:3.583260 
#> [66]	train-rmse:0.759774	test-rmse:3.584345 
#> [67]	train-rmse:0.748496	test-rmse:3.585526 
#> [68]	train-rmse:0.733068	test-rmse:3.584602 
#> [69]	train-rmse:0.727257	test-rmse:3.583001 
#> [70]	train-rmse:0.714099	test-rmse:3.581297 
#> [1]	train-rmse:17.385765	test-rmse:16.794756 
#> [2]	train-rmse:12.670146	test-rmse:12.264586 
#> [3]	train-rmse:9.314821	test-rmse:9.065502 
#> [4]	train-rmse:7.000134	test-rmse:6.953405 
#> [5]	train-rmse:5.405117	test-rmse:5.523697 
#> [6]	train-rmse:4.326401	test-rmse:4.681973 
#> [7]	train-rmse:3.623907	test-rmse:4.130183 
#> [8]	train-rmse:3.178648	test-rmse:3.823766 
#> [9]	train-rmse:2.855376	test-rmse:3.663213 
#> [10]	train-rmse:2.593623	test-rmse:3.573890 
#> [11]	train-rmse:2.439111	test-rmse:3.489961 
#> [12]	train-rmse:2.312032	test-rmse:3.455668 
#> [13]	train-rmse:2.221071	test-rmse:3.439968 
#> [14]	train-rmse:2.108566	test-rmse:3.380220 
#> [15]	train-rmse:2.004805	test-rmse:3.372307 
#> [16]	train-rmse:1.951018	test-rmse:3.361490 
#> [17]	train-rmse:1.891175	test-rmse:3.336966 
#> [18]	train-rmse:1.860596	test-rmse:3.342489 
#> [19]	train-rmse:1.820702	test-rmse:3.337561 
#> [20]	train-rmse:1.770482	test-rmse:3.301025 
#> [21]	train-rmse:1.727679	test-rmse:3.271536 
#> [22]	train-rmse:1.692517	test-rmse:3.275990 
#> [23]	train-rmse:1.643101	test-rmse:3.288711 
#> [24]	train-rmse:1.616868	test-rmse:3.286152 
#> [25]	train-rmse:1.578256	test-rmse:3.285970 
#> [26]	train-rmse:1.515484	test-rmse:3.289535 
#> [27]	train-rmse:1.478759	test-rmse:3.303692 
#> [28]	train-rmse:1.449194	test-rmse:3.311711 
#> [29]	train-rmse:1.408619	test-rmse:3.345546 
#> [30]	train-rmse:1.387780	test-rmse:3.338399 
#> [31]	train-rmse:1.354574	test-rmse:3.330419 
#> [32]	train-rmse:1.322153	test-rmse:3.345416 
#> [33]	train-rmse:1.310442	test-rmse:3.356520 
#> [34]	train-rmse:1.279219	test-rmse:3.359315 
#> [35]	train-rmse:1.262447	test-rmse:3.369878 
#> [36]	train-rmse:1.225963	test-rmse:3.364055 
#> [37]	train-rmse:1.196766	test-rmse:3.366739 
#> [38]	train-rmse:1.177675	test-rmse:3.365982 
#> [39]	train-rmse:1.170192	test-rmse:3.372613 
#> [40]	train-rmse:1.155881	test-rmse:3.374185 
#> [41]	train-rmse:1.127849	test-rmse:3.394838 
#> [42]	train-rmse:1.088085	test-rmse:3.387102 
#> [43]	train-rmse:1.069883	test-rmse:3.384299 
#> [44]	train-rmse:1.060051	test-rmse:3.390040 
#> [45]	train-rmse:1.033736	test-rmse:3.409566 
#> [46]	train-rmse:1.007798	test-rmse:3.414272 
#> [47]	train-rmse:0.997527	test-rmse:3.414263 
#> [48]	train-rmse:0.979780	test-rmse:3.409749 
#> [49]	train-rmse:0.949130	test-rmse:3.415667 
#> [50]	train-rmse:0.939405	test-rmse:3.414446 
#> [51]	train-rmse:0.921432	test-rmse:3.414354 
#> [52]	train-rmse:0.902725	test-rmse:3.418486 
#> [53]	train-rmse:0.889964	test-rmse:3.413368 
#> [54]	train-rmse:0.870040	test-rmse:3.412262 
#> [55]	train-rmse:0.853867	test-rmse:3.410928 
#> [56]	train-rmse:0.834157	test-rmse:3.407093 
#> [57]	train-rmse:0.824457	test-rmse:3.409571 
#> [58]	train-rmse:0.809492	test-rmse:3.410608 
#> [59]	train-rmse:0.799750	test-rmse:3.410038 
#> [60]	train-rmse:0.785987	test-rmse:3.407532 
#> [61]	train-rmse:0.774586	test-rmse:3.406803 
#> [62]	train-rmse:0.765715	test-rmse:3.406814 
#> [63]	train-rmse:0.746053	test-rmse:3.403769 
#> [64]	train-rmse:0.732835	test-rmse:3.408745 
#> [65]	train-rmse:0.721152	test-rmse:3.407695 
#> [66]	train-rmse:0.714833	test-rmse:3.404789 
#> [67]	train-rmse:0.704015	test-rmse:3.406468 
#> [68]	train-rmse:0.687991	test-rmse:3.407474 
#> [69]	train-rmse:0.682328	test-rmse:3.406557 
#> [70]	train-rmse:0.666911	test-rmse:3.406488 
#> [1]	train-rmse:16.988515	test-rmse:17.627070 
#> [2]	train-rmse:12.328131	test-rmse:12.957543 
#> [3]	train-rmse:9.031109	test-rmse:9.525598 
#> [4]	train-rmse:6.729942	test-rmse:7.223944 
#> [5]	train-rmse:5.143532	test-rmse:5.704133 
#> [6]	train-rmse:4.056384	test-rmse:4.679890 
#> [7]	train-rmse:3.351660	test-rmse:4.103296 
#> [8]	train-rmse:2.886013	test-rmse:3.758627 
#> [9]	train-rmse:2.575575	test-rmse:3.524426 
#> [10]	train-rmse:2.374128	test-rmse:3.419636 
#> [11]	train-rmse:2.195364	test-rmse:3.359844 
#> [12]	train-rmse:2.074972	test-rmse:3.272500 
#> [13]	train-rmse:1.961812	test-rmse:3.194225 
#> [14]	train-rmse:1.901956	test-rmse:3.201507 
#> [15]	train-rmse:1.870940	test-rmse:3.186217 
#> [16]	train-rmse:1.802847	test-rmse:3.184240 
#> [17]	train-rmse:1.753162	test-rmse:3.158260 
#> [18]	train-rmse:1.704044	test-rmse:3.149618 
#> [19]	train-rmse:1.678835	test-rmse:3.137091 
#> [20]	train-rmse:1.612179	test-rmse:3.140177 
#> [21]	train-rmse:1.585593	test-rmse:3.130803 
#> [22]	train-rmse:1.572557	test-rmse:3.118183 
#> [23]	train-rmse:1.531643	test-rmse:3.140256 
#> [24]	train-rmse:1.509104	test-rmse:3.133420 
#> [25]	train-rmse:1.476578	test-rmse:3.138309 
#> [26]	train-rmse:1.461619	test-rmse:3.122291 
#> [27]	train-rmse:1.453615	test-rmse:3.123351 
#> [28]	train-rmse:1.420323	test-rmse:3.104772 
#> [29]	train-rmse:1.405515	test-rmse:3.108274 
#> [30]	train-rmse:1.367252	test-rmse:3.110382 
#> [31]	train-rmse:1.355996	test-rmse:3.104787 
#> [32]	train-rmse:1.330491	test-rmse:3.119113 
#> [33]	train-rmse:1.316774	test-rmse:3.121103 
#> [34]	train-rmse:1.303470	test-rmse:3.111014 
#> [35]	train-rmse:1.280759	test-rmse:3.119781 
#> [36]	train-rmse:1.269170	test-rmse:3.115679 
#> [37]	train-rmse:1.234440	test-rmse:3.117343 
#> [38]	train-rmse:1.203275	test-rmse:3.116117 
#> [39]	train-rmse:1.170265	test-rmse:3.119718 
#> [40]	train-rmse:1.161107	test-rmse:3.120461 
#> [41]	train-rmse:1.156858	test-rmse:3.122691 
#> [42]	train-rmse:1.122966	test-rmse:3.107496 
#> [43]	train-rmse:1.099592	test-rmse:3.105319 
#> [44]	train-rmse:1.083191	test-rmse:3.096163 
#> [45]	train-rmse:1.062763	test-rmse:3.106104 
#> [46]	train-rmse:1.049289	test-rmse:3.099127 
#> [47]	train-rmse:1.029801	test-rmse:3.096994 
#> [48]	train-rmse:1.012888	test-rmse:3.100607 
#> [49]	train-rmse:0.996024	test-rmse:3.107249 
#> [50]	train-rmse:0.980183	test-rmse:3.106304 
#> [51]	train-rmse:0.972154	test-rmse:3.102175 
#> [52]	train-rmse:0.966438	test-rmse:3.101945 
#> [53]	train-rmse:0.954552	test-rmse:3.104130 
#> [54]	train-rmse:0.948947	test-rmse:3.098030 
#> [55]	train-rmse:0.929715	test-rmse:3.096164 
#> [56]	train-rmse:0.918693	test-rmse:3.087866 
#> [57]	train-rmse:0.911486	test-rmse:3.085831 
#> [58]	train-rmse:0.896170	test-rmse:3.083192 
#> [59]	train-rmse:0.877669	test-rmse:3.084674 
#> [60]	train-rmse:0.867397	test-rmse:3.091211 
#> [61]	train-rmse:0.861446	test-rmse:3.089973 
#> [62]	train-rmse:0.845836	test-rmse:3.086549 
#> [63]	train-rmse:0.828992	test-rmse:3.081455 
#> [64]	train-rmse:0.812027	test-rmse:3.074534 
#> [65]	train-rmse:0.793364	test-rmse:3.075464 
#> [66]	train-rmse:0.783802	test-rmse:3.078706 
#> [67]	train-rmse:0.779578	test-rmse:3.077914 
#> [68]	train-rmse:0.771363	test-rmse:3.078589 
#> [69]	train-rmse:0.756333	test-rmse:3.074773 
#> [70]	train-rmse:0.742274	test-rmse:3.069944 
#> [1]	train-rmse:17.321959	test-rmse:16.774258 
#> [2]	train-rmse:12.561934	test-rmse:12.298570 
#> [3]	train-rmse:9.286895	test-rmse:9.245498 
#> [4]	train-rmse:6.933006	test-rmse:7.179272 
#> [5]	train-rmse:5.328855	test-rmse:5.874374 
#> [6]	train-rmse:4.253427	test-rmse:5.083352 
#> [7]	train-rmse:3.548404	test-rmse:4.565694 
#> [8]	train-rmse:3.072836	test-rmse:4.339920 
#> [9]	train-rmse:2.714252	test-rmse:4.210879 
#> [10]	train-rmse:2.500005	test-rmse:4.114217 
#> [11]	train-rmse:2.324530	test-rmse:4.050798 
#> [12]	train-rmse:2.200729	test-rmse:4.035300 
#> [13]	train-rmse:2.112021	test-rmse:3.970368 
#> [14]	train-rmse:2.012419	test-rmse:3.940639 
#> [15]	train-rmse:1.951779	test-rmse:3.910923 
#> [16]	train-rmse:1.911819	test-rmse:3.911717 
#> [17]	train-rmse:1.852198	test-rmse:3.871663 
#> [18]	train-rmse:1.795860	test-rmse:3.886116 
#> [19]	train-rmse:1.769939	test-rmse:3.878583 
#> [20]	train-rmse:1.735634	test-rmse:3.863904 
#> [21]	train-rmse:1.693778	test-rmse:3.875972 
#> [22]	train-rmse:1.678357	test-rmse:3.872397 
#> [23]	train-rmse:1.635299	test-rmse:3.838241 
#> [24]	train-rmse:1.607056	test-rmse:3.839355 
#> [25]	train-rmse:1.582201	test-rmse:3.835966 
#> [26]	train-rmse:1.533953	test-rmse:3.832666 
#> [27]	train-rmse:1.523273	test-rmse:3.845692 
#> [28]	train-rmse:1.509610	test-rmse:3.842331 
#> [29]	train-rmse:1.494145	test-rmse:3.842357 
#> [30]	train-rmse:1.432359	test-rmse:3.837947 
#> [31]	train-rmse:1.401277	test-rmse:3.830856 
#> [32]	train-rmse:1.376588	test-rmse:3.831958 
#> [33]	train-rmse:1.349368	test-rmse:3.826925 
#> [34]	train-rmse:1.304012	test-rmse:3.807976 
#> [35]	train-rmse:1.286216	test-rmse:3.810429 
#> [36]	train-rmse:1.253230	test-rmse:3.812646 
#> [37]	train-rmse:1.237035	test-rmse:3.810238 
#> [38]	train-rmse:1.207467	test-rmse:3.793580 
#> [39]	train-rmse:1.180626	test-rmse:3.786512 
#> [40]	train-rmse:1.162318	test-rmse:3.779690 
#> [41]	train-rmse:1.150061	test-rmse:3.782207 
#> [42]	train-rmse:1.132723	test-rmse:3.780529 
#> [43]	train-rmse:1.103424	test-rmse:3.768010 
#> [44]	train-rmse:1.091152	test-rmse:3.744968 
#> [45]	train-rmse:1.075710	test-rmse:3.743391 
#> [46]	train-rmse:1.060600	test-rmse:3.740504 
#> [47]	train-rmse:1.045323	test-rmse:3.745132 
#> [48]	train-rmse:1.032176	test-rmse:3.742003 
#> [49]	train-rmse:1.019129	test-rmse:3.743353 
#> [50]	train-rmse:1.009357	test-rmse:3.739067 
#> [51]	train-rmse:0.998351	test-rmse:3.739515 
#> [52]	train-rmse:0.968744	test-rmse:3.737050 
#> [53]	train-rmse:0.948280	test-rmse:3.733943 
#> [54]	train-rmse:0.922214	test-rmse:3.735062 
#> [55]	train-rmse:0.894645	test-rmse:3.738614 
#> [56]	train-rmse:0.881368	test-rmse:3.735006 
#> [57]	train-rmse:0.862454	test-rmse:3.735164 
#> [58]	train-rmse:0.845040	test-rmse:3.736717 
#> [59]	train-rmse:0.824536	test-rmse:3.742466 
#> [60]	train-rmse:0.817390	test-rmse:3.741655 
#> [61]	train-rmse:0.799640	test-rmse:3.724825 
#> [62]	train-rmse:0.778461	test-rmse:3.727749 
#> [63]	train-rmse:0.768820	test-rmse:3.726508 
#> [64]	train-rmse:0.757031	test-rmse:3.717994 
#> [65]	train-rmse:0.750929	test-rmse:3.719656 
#> [66]	train-rmse:0.732423	test-rmse:3.722865 
#> [67]	train-rmse:0.715445	test-rmse:3.722268 
#> [68]	train-rmse:0.710682	test-rmse:3.718600 
#> [69]	train-rmse:0.702250	test-rmse:3.718964 
#> [70]	train-rmse:0.680833	test-rmse:3.715453 
#> [1]	train-rmse:17.323344	test-rmse:16.887683 
#> [2]	train-rmse:12.512398	test-rmse:12.409795 
#> [3]	train-rmse:9.171131	test-rmse:9.399116 
#> [4]	train-rmse:6.865599	test-rmse:7.366663 
#> [5]	train-rmse:5.290222	test-rmse:6.117032 
#> [6]	train-rmse:4.207435	test-rmse:5.199151 
#> [7]	train-rmse:3.479597	test-rmse:4.719487 
#> [8]	train-rmse:2.990186	test-rmse:4.396071 
#> [9]	train-rmse:2.668102	test-rmse:4.194580 
#> [10]	train-rmse:2.455356	test-rmse:4.098731 
#> [11]	train-rmse:2.323447	test-rmse:3.989118 
#> [12]	train-rmse:2.215922	test-rmse:3.992581 
#> [13]	train-rmse:2.109730	test-rmse:3.948598 
#> [14]	train-rmse:2.054346	test-rmse:3.909207 
#> [15]	train-rmse:1.982701	test-rmse:3.879111 
#> [16]	train-rmse:1.932368	test-rmse:3.877432 
#> [17]	train-rmse:1.818206	test-rmse:3.829744 
#> [18]	train-rmse:1.757775	test-rmse:3.800581 
#> [19]	train-rmse:1.675877	test-rmse:3.767973 
#> [20]	train-rmse:1.654063	test-rmse:3.766804 
#> [21]	train-rmse:1.587751	test-rmse:3.739617 
#> [22]	train-rmse:1.552599	test-rmse:3.726380 
#> [23]	train-rmse:1.520064	test-rmse:3.711067 
#> [24]	train-rmse:1.508405	test-rmse:3.720312 
#> [25]	train-rmse:1.470924	test-rmse:3.735472 
#> [26]	train-rmse:1.442288	test-rmse:3.730755 
#> [27]	train-rmse:1.412341	test-rmse:3.733803 
#> [28]	train-rmse:1.401643	test-rmse:3.732055 
#> [29]	train-rmse:1.382986	test-rmse:3.732116 
#> [30]	train-rmse:1.366508	test-rmse:3.753377 
#> [31]	train-rmse:1.330571	test-rmse:3.753575 
#> [32]	train-rmse:1.311493	test-rmse:3.755554 
#> [33]	train-rmse:1.260804	test-rmse:3.765435 
#> [34]	train-rmse:1.252092	test-rmse:3.766213 
#> [35]	train-rmse:1.225012	test-rmse:3.764774 
#> [36]	train-rmse:1.210006	test-rmse:3.755831 
#> [37]	train-rmse:1.202187	test-rmse:3.760723 
#> [38]	train-rmse:1.188864	test-rmse:3.759230 
#> [39]	train-rmse:1.181029	test-rmse:3.761895 
#> [40]	train-rmse:1.168316	test-rmse:3.766661 
#> [41]	train-rmse:1.138756	test-rmse:3.758480 
#> [42]	train-rmse:1.110167	test-rmse:3.758335 
#> [43]	train-rmse:1.091759	test-rmse:3.761693 
#> [44]	train-rmse:1.061706	test-rmse:3.755408 
#> [45]	train-rmse:1.044165	test-rmse:3.750081 
#> [46]	train-rmse:1.033604	test-rmse:3.741435 
#> [47]	train-rmse:1.011237	test-rmse:3.747237 
#> [48]	train-rmse:0.983079	test-rmse:3.747406 
#> [49]	train-rmse:0.977700	test-rmse:3.750498 
#> [50]	train-rmse:0.970572	test-rmse:3.750838 
#> [51]	train-rmse:0.948303	test-rmse:3.754534 
#> [52]	train-rmse:0.933015	test-rmse:3.748746 
#> [53]	train-rmse:0.913438	test-rmse:3.754442 
#> [54]	train-rmse:0.909884	test-rmse:3.758545 
#> [55]	train-rmse:0.901655	test-rmse:3.770935 
#> [56]	train-rmse:0.894783	test-rmse:3.771958 
#> [57]	train-rmse:0.887991	test-rmse:3.763164 
#> [58]	train-rmse:0.881060	test-rmse:3.762108 
#> [59]	train-rmse:0.867273	test-rmse:3.751898 
#> [60]	train-rmse:0.858725	test-rmse:3.753705 
#> [61]	train-rmse:0.838373	test-rmse:3.763830 
#> [62]	train-rmse:0.830943	test-rmse:3.759233 
#> [63]	train-rmse:0.826514	test-rmse:3.766267 
#> [64]	train-rmse:0.823175	test-rmse:3.766559 
#> [65]	train-rmse:0.807660	test-rmse:3.767533 
#> [66]	train-rmse:0.794098	test-rmse:3.765165 
#> [67]	train-rmse:0.778397	test-rmse:3.770035 
#> [68]	train-rmse:0.772503	test-rmse:3.771494 
#> [69]	train-rmse:0.747584	test-rmse:3.767552 
#> [70]	train-rmse:0.740816	test-rmse:3.764540 
#> [1]	train-rmse:17.044051	test-rmse:17.598582 
#> [2]	train-rmse:12.361020	test-rmse:13.070777 
#> [3]	train-rmse:9.094743	test-rmse:9.941374 
#> [4]	train-rmse:6.823861	test-rmse:7.838050 
#> [5]	train-rmse:5.263417	test-rmse:6.425471 
#> [6]	train-rmse:4.215828	test-rmse:5.438853 
#> [7]	train-rmse:3.502785	test-rmse:4.984603 
#> [8]	train-rmse:3.026065	test-rmse:4.662811 
#> [9]	train-rmse:2.721890	test-rmse:4.363687 
#> [10]	train-rmse:2.495576	test-rmse:4.231543 
#> [11]	train-rmse:2.338947	test-rmse:4.141785 
#> [12]	train-rmse:2.234066	test-rmse:4.081315 
#> [13]	train-rmse:2.158419	test-rmse:4.021757 
#> [14]	train-rmse:2.085335	test-rmse:4.008236 
#> [15]	train-rmse:2.012131	test-rmse:4.001397 
#> [16]	train-rmse:1.933621	test-rmse:3.986264 
#> [17]	train-rmse:1.896756	test-rmse:3.925873 
#> [18]	train-rmse:1.867964	test-rmse:3.925774 
#> [19]	train-rmse:1.829713	test-rmse:3.879238 
#> [20]	train-rmse:1.790533	test-rmse:3.855955 
#> [21]	train-rmse:1.746201	test-rmse:3.861145 
#> [22]	train-rmse:1.690864	test-rmse:3.817658 
#> [23]	train-rmse:1.658930	test-rmse:3.775611 
#> [24]	train-rmse:1.637712	test-rmse:3.777781 
#> [25]	train-rmse:1.595910	test-rmse:3.757286 
#> [26]	train-rmse:1.579251	test-rmse:3.748755 
#> [27]	train-rmse:1.564803	test-rmse:3.744023 
#> [28]	train-rmse:1.543134	test-rmse:3.721144 
#> [29]	train-rmse:1.510350	test-rmse:3.709961 
#> [30]	train-rmse:1.488938	test-rmse:3.700683 
#> [31]	train-rmse:1.472471	test-rmse:3.706287 
#> [32]	train-rmse:1.449769	test-rmse:3.709890 
#> [33]	train-rmse:1.430000	test-rmse:3.714988 
#> [34]	train-rmse:1.397398	test-rmse:3.707030 
#> [35]	train-rmse:1.363955	test-rmse:3.699587 
#> [36]	train-rmse:1.333442	test-rmse:3.698103 
#> [37]	train-rmse:1.299325	test-rmse:3.698002 
#> [38]	train-rmse:1.280210	test-rmse:3.698826 
#> [39]	train-rmse:1.262763	test-rmse:3.698071 
#> [40]	train-rmse:1.241038	test-rmse:3.699788 
#> [41]	train-rmse:1.224637	test-rmse:3.696921 
#> [42]	train-rmse:1.210522	test-rmse:3.684259 
#> [43]	train-rmse:1.202150	test-rmse:3.690122 
#> [44]	train-rmse:1.190017	test-rmse:3.688138 
#> [45]	train-rmse:1.166656	test-rmse:3.684822 
#> [46]	train-rmse:1.155027	test-rmse:3.684041 
#> [47]	train-rmse:1.135007	test-rmse:3.687548 
#> [48]	train-rmse:1.114548	test-rmse:3.682242 
#> [49]	train-rmse:1.092828	test-rmse:3.683085 
#> [50]	train-rmse:1.075588	test-rmse:3.680993 
#> [51]	train-rmse:1.059238	test-rmse:3.664334 
#> [52]	train-rmse:1.037551	test-rmse:3.660962 
#> [53]	train-rmse:1.022884	test-rmse:3.643521 
#> [54]	train-rmse:1.006510	test-rmse:3.640152 
#> [55]	train-rmse:0.972259	test-rmse:3.641487 
#> [56]	train-rmse:0.964384	test-rmse:3.643671 
#> [57]	train-rmse:0.933300	test-rmse:3.642574 
#> [58]	train-rmse:0.924543	test-rmse:3.646467 
#> [59]	train-rmse:0.912426	test-rmse:3.647027 
#> [60]	train-rmse:0.901072	test-rmse:3.644761 
#> [61]	train-rmse:0.894141	test-rmse:3.641944 
#> [62]	train-rmse:0.874174	test-rmse:3.650901 
#> [63]	train-rmse:0.861625	test-rmse:3.654305 
#> [64]	train-rmse:0.847851	test-rmse:3.636964 
#> [65]	train-rmse:0.831352	test-rmse:3.634862 
#> [66]	train-rmse:0.823780	test-rmse:3.629843 
#> [67]	train-rmse:0.812325	test-rmse:3.625395 
#> [68]	train-rmse:0.796279	test-rmse:3.614405 
#> [69]	train-rmse:0.776594	test-rmse:3.617129 
#> [70]	train-rmse:0.767391	test-rmse:3.613929 
#> [1]	train-rmse:17.170895	test-rmse:17.540865 
#> [2]	train-rmse:12.518578	test-rmse:13.151335 
#> [3]	train-rmse:9.206967	test-rmse:9.794633 
#> [4]	train-rmse:6.914678	test-rmse:7.547047 
#> [5]	train-rmse:5.320936	test-rmse:5.979701 
#> [6]	train-rmse:4.300025	test-rmse:5.106135 
#> [7]	train-rmse:3.628148	test-rmse:4.486839 
#> [8]	train-rmse:3.167276	test-rmse:4.080448 
#> [9]	train-rmse:2.856795	test-rmse:3.851353 
#> [10]	train-rmse:2.638298	test-rmse:3.608062 
#> [11]	train-rmse:2.487609	test-rmse:3.467947 
#> [12]	train-rmse:2.366785	test-rmse:3.409284 
#> [13]	train-rmse:2.232873	test-rmse:3.368363 
#> [14]	train-rmse:2.129441	test-rmse:3.277761 
#> [15]	train-rmse:2.052034	test-rmse:3.226649 
#> [16]	train-rmse:2.009219	test-rmse:3.206701 
#> [17]	train-rmse:1.940478	test-rmse:3.175203 
#> [18]	train-rmse:1.893435	test-rmse:3.158259 
#> [19]	train-rmse:1.862652	test-rmse:3.172486 
#> [20]	train-rmse:1.801562	test-rmse:3.186940 
#> [21]	train-rmse:1.770672	test-rmse:3.188426 
#> [22]	train-rmse:1.732201	test-rmse:3.187165 
#> [23]	train-rmse:1.673226	test-rmse:3.164867 
#> [24]	train-rmse:1.656786	test-rmse:3.153857 
#> [25]	train-rmse:1.637156	test-rmse:3.159684 
#> [26]	train-rmse:1.578651	test-rmse:3.160574 
#> [27]	train-rmse:1.542109	test-rmse:3.153191 
#> [28]	train-rmse:1.521919	test-rmse:3.158687 
#> [29]	train-rmse:1.497534	test-rmse:3.150210 
#> [30]	train-rmse:1.480342	test-rmse:3.131334 
#> [31]	train-rmse:1.441255	test-rmse:3.116201 
#> [32]	train-rmse:1.430575	test-rmse:3.124205 
#> [33]	train-rmse:1.400415	test-rmse:3.127266 
#> [34]	train-rmse:1.364291	test-rmse:3.120414 
#> [35]	train-rmse:1.334748	test-rmse:3.113060 
#> [36]	train-rmse:1.302551	test-rmse:3.113071 
#> [37]	train-rmse:1.281874	test-rmse:3.115077 
#> [38]	train-rmse:1.244729	test-rmse:3.108771 
#> [39]	train-rmse:1.225519	test-rmse:3.106393 
#> [40]	train-rmse:1.209392	test-rmse:3.103222 
#> [41]	train-rmse:1.182575	test-rmse:3.103951 
#> [42]	train-rmse:1.171319	test-rmse:3.106968 
#> [43]	train-rmse:1.153272	test-rmse:3.093656 
#> [44]	train-rmse:1.125798	test-rmse:3.090910 
#> [45]	train-rmse:1.103927	test-rmse:3.097984 
#> [46]	train-rmse:1.097017	test-rmse:3.093829 
#> [47]	train-rmse:1.078219	test-rmse:3.091931 
#> [48]	train-rmse:1.065487	test-rmse:3.088405 
#> [49]	train-rmse:1.042989	test-rmse:3.094644 
#> [50]	train-rmse:1.029686	test-rmse:3.099256 
#> [51]	train-rmse:1.012739	test-rmse:3.089073 
#> [52]	train-rmse:1.002601	test-rmse:3.088385 
#> [53]	train-rmse:0.983105	test-rmse:3.093128 
#> [54]	train-rmse:0.969584	test-rmse:3.086581 
#> [55]	train-rmse:0.951424	test-rmse:3.081786 
#> [56]	train-rmse:0.938979	test-rmse:3.083971 
#> [57]	train-rmse:0.920503	test-rmse:3.076928 
#> [58]	train-rmse:0.908392	test-rmse:3.076756 
#> [59]	train-rmse:0.904085	test-rmse:3.078821 
#> [60]	train-rmse:0.880048	test-rmse:3.080675 
#> [61]	train-rmse:0.873038	test-rmse:3.082606 
#> [62]	train-rmse:0.853135	test-rmse:3.073058 
#> [63]	train-rmse:0.844088	test-rmse:3.069322 
#> [64]	train-rmse:0.829665	test-rmse:3.073606 
#> [65]	train-rmse:0.816083	test-rmse:3.071443 
#> [66]	train-rmse:0.802582	test-rmse:3.067585 
#> [67]	train-rmse:0.789541	test-rmse:3.060139 
#> [68]	train-rmse:0.777767	test-rmse:3.068019 
#> [69]	train-rmse:0.771780	test-rmse:3.067337 
#> [70]	train-rmse:0.758804	test-rmse:3.067511 
#> [1]	train-rmse:17.120072	test-rmse:17.205633 
#> [2]	train-rmse:12.441414	test-rmse:12.381690 
#> [3]	train-rmse:9.157665	test-rmse:9.012166 
#> [4]	train-rmse:6.856267	test-rmse:6.942229 
#> [5]	train-rmse:5.292154	test-rmse:5.541123 
#> [6]	train-rmse:4.218657	test-rmse:4.641700 
#> [7]	train-rmse:3.491688	test-rmse:4.170801 
#> [8]	train-rmse:3.020847	test-rmse:3.884707 
#> [9]	train-rmse:2.713909	test-rmse:3.667915 
#> [10]	train-rmse:2.513685	test-rmse:3.561887 
#> [11]	train-rmse:2.342838	test-rmse:3.430305 
#> [12]	train-rmse:2.201219	test-rmse:3.381002 
#> [13]	train-rmse:2.124123	test-rmse:3.381588 
#> [14]	train-rmse:2.060950	test-rmse:3.357941 
#> [15]	train-rmse:2.006157	test-rmse:3.334989 
#> [16]	train-rmse:1.905435	test-rmse:3.314851 
#> [17]	train-rmse:1.860574	test-rmse:3.302205 
#> [18]	train-rmse:1.825841	test-rmse:3.286552 
#> [19]	train-rmse:1.764498	test-rmse:3.255861 
#> [20]	train-rmse:1.734601	test-rmse:3.260446 
#> [21]	train-rmse:1.691928	test-rmse:3.236840 
#> [22]	train-rmse:1.648116	test-rmse:3.235027 
#> [23]	train-rmse:1.631069	test-rmse:3.230025 
#> [24]	train-rmse:1.605170	test-rmse:3.225316 
#> [25]	train-rmse:1.579480	test-rmse:3.213046 
#> [26]	train-rmse:1.540571	test-rmse:3.221617 
#> [27]	train-rmse:1.503467	test-rmse:3.214518 
#> [28]	train-rmse:1.480958	test-rmse:3.210128 
#> [29]	train-rmse:1.463905	test-rmse:3.206344 
#> [30]	train-rmse:1.447949	test-rmse:3.197237 
#> [31]	train-rmse:1.416408	test-rmse:3.202577 
#> [32]	train-rmse:1.400635	test-rmse:3.208664 
#> [33]	train-rmse:1.380655	test-rmse:3.198639 
#> [34]	train-rmse:1.355856	test-rmse:3.181233 
#> [35]	train-rmse:1.335305	test-rmse:3.179564 
#> [36]	train-rmse:1.298753	test-rmse:3.182537 
#> [37]	train-rmse:1.268010	test-rmse:3.173512 
#> [38]	train-rmse:1.241321	test-rmse:3.177074 
#> [39]	train-rmse:1.213382	test-rmse:3.184895 
#> [40]	train-rmse:1.175449	test-rmse:3.180922 
#> [41]	train-rmse:1.141018	test-rmse:3.177054 
#> [42]	train-rmse:1.130889	test-rmse:3.178959 
#> [43]	train-rmse:1.120073	test-rmse:3.183034 
#> [44]	train-rmse:1.113177	test-rmse:3.178551 
#> [45]	train-rmse:1.084548	test-rmse:3.187671 
#> [46]	train-rmse:1.079891	test-rmse:3.188788 
#> [47]	train-rmse:1.047196	test-rmse:3.178450 
#> [48]	train-rmse:1.027513	test-rmse:3.174070 
#> [49]	train-rmse:1.001774	test-rmse:3.154042 
#> [50]	train-rmse:0.977110	test-rmse:3.153992 
#> [51]	train-rmse:0.965694	test-rmse:3.157393 
#> [52]	train-rmse:0.953294	test-rmse:3.152014 
#> [53]	train-rmse:0.930325	test-rmse:3.150492 
#> [54]	train-rmse:0.917532	test-rmse:3.152998 
#> [55]	train-rmse:0.897069	test-rmse:3.150624 
#> [56]	train-rmse:0.888930	test-rmse:3.149678 
#> [57]	train-rmse:0.878738	test-rmse:3.148072 
#> [58]	train-rmse:0.862263	test-rmse:3.144069 
#> [59]	train-rmse:0.846056	test-rmse:3.143937 
#> [60]	train-rmse:0.840649	test-rmse:3.143465 
#> [61]	train-rmse:0.825235	test-rmse:3.146202 
#> [62]	train-rmse:0.818611	test-rmse:3.145530 
#> [63]	train-rmse:0.801086	test-rmse:3.150825 
#> [64]	train-rmse:0.789049	test-rmse:3.158372 
#> [65]	train-rmse:0.784521	test-rmse:3.156429 
#> [66]	train-rmse:0.761073	test-rmse:3.152421 
#> [67]	train-rmse:0.746497	test-rmse:3.149011 
#> [68]	train-rmse:0.738125	test-rmse:3.151721 
#> [69]	train-rmse:0.720135	test-rmse:3.153331 
#> [70]	train-rmse:0.707547	test-rmse:3.163377 
#> [1]	train-rmse:17.023882	test-rmse:17.563864 
#> [2]	train-rmse:12.269273	test-rmse:13.191065 
#> [3]	train-rmse:8.945738	test-rmse:10.222077 
#> [4]	train-rmse:6.633549	test-rmse:8.228290 
#> [5]	train-rmse:5.040055	test-rmse:7.011920 
#> [6]	train-rmse:3.953076	test-rmse:6.163945 
#> [7]	train-rmse:3.232299	test-rmse:5.642217 
#> [8]	train-rmse:2.737867	test-rmse:5.318758 
#> [9]	train-rmse:2.438273	test-rmse:5.118100 
#> [10]	train-rmse:2.224739	test-rmse:5.047133 
#> [11]	train-rmse:2.074463	test-rmse:4.937526 
#> [12]	train-rmse:1.976551	test-rmse:4.878560 
#> [13]	train-rmse:1.897620	test-rmse:4.853702 
#> [14]	train-rmse:1.816070	test-rmse:4.796739 
#> [15]	train-rmse:1.778009	test-rmse:4.791865 
#> [16]	train-rmse:1.739783	test-rmse:4.765143 
#> [17]	train-rmse:1.683148	test-rmse:4.747970 
#> [18]	train-rmse:1.660325	test-rmse:4.747637 
#> [19]	train-rmse:1.605003	test-rmse:4.734799 
#> [20]	train-rmse:1.581467	test-rmse:4.735818 
#> [21]	train-rmse:1.543347	test-rmse:4.720051 
#> [22]	train-rmse:1.516976	test-rmse:4.716742 
#> [23]	train-rmse:1.493828	test-rmse:4.708630 
#> [24]	train-rmse:1.475674	test-rmse:4.714964 
#> [25]	train-rmse:1.444951	test-rmse:4.717055 
#> [26]	train-rmse:1.397475	test-rmse:4.712563 
#> [27]	train-rmse:1.387247	test-rmse:4.701317 
#> [28]	train-rmse:1.358862	test-rmse:4.709803 
#> [29]	train-rmse:1.330462	test-rmse:4.694522 
#> [30]	train-rmse:1.307189	test-rmse:4.688540 
#> [31]	train-rmse:1.295407	test-rmse:4.675326 
#> [32]	train-rmse:1.281960	test-rmse:4.676934 
#> [33]	train-rmse:1.256713	test-rmse:4.671896 
#> [34]	train-rmse:1.229986	test-rmse:4.669061 
#> [35]	train-rmse:1.194477	test-rmse:4.670156 
#> [36]	train-rmse:1.160844	test-rmse:4.685237 
#> [37]	train-rmse:1.143638	test-rmse:4.681791 
#> [38]	train-rmse:1.111245	test-rmse:4.684169 
#> [39]	train-rmse:1.099496	test-rmse:4.673255 
#> [40]	train-rmse:1.064249	test-rmse:4.654325 
#> [41]	train-rmse:1.049932	test-rmse:4.650486 
#> [42]	train-rmse:1.037882	test-rmse:4.656002 
#> [43]	train-rmse:1.009303	test-rmse:4.644184 
#> [44]	train-rmse:0.989942	test-rmse:4.636316 
#> [45]	train-rmse:0.973487	test-rmse:4.632211 
#> [46]	train-rmse:0.963268	test-rmse:4.639360 
#> [47]	train-rmse:0.946897	test-rmse:4.632759 
#> [48]	train-rmse:0.929681	test-rmse:4.628163 
#> [49]	train-rmse:0.925963	test-rmse:4.626612 
#> [50]	train-rmse:0.909648	test-rmse:4.617172 
#> [51]	train-rmse:0.895271	test-rmse:4.610602 
#> [52]	train-rmse:0.867565	test-rmse:4.605412 
#> [53]	train-rmse:0.860844	test-rmse:4.610389 
#> [54]	train-rmse:0.848021	test-rmse:4.609577 
#> [55]	train-rmse:0.837087	test-rmse:4.603181 
#> [56]	train-rmse:0.823987	test-rmse:4.603746 
#> [57]	train-rmse:0.798760	test-rmse:4.593155 
#> [58]	train-rmse:0.789874	test-rmse:4.589935 
#> [59]	train-rmse:0.781605	test-rmse:4.593616 
#> [60]	train-rmse:0.777567	test-rmse:4.590960 
#> [61]	train-rmse:0.760254	test-rmse:4.593655 
#> [62]	train-rmse:0.741331	test-rmse:4.592714 
#> [63]	train-rmse:0.732111	test-rmse:4.592398 
#> [64]	train-rmse:0.729240	test-rmse:4.595556 
#> [65]	train-rmse:0.723922	test-rmse:4.589123 
#> [66]	train-rmse:0.710034	test-rmse:4.589352 
#> [67]	train-rmse:0.700937	test-rmse:4.590984 
#> [68]	train-rmse:0.680924	test-rmse:4.590127 
#> [69]	train-rmse:0.668874	test-rmse:4.582640 
#> [70]	train-rmse:0.660979	test-rmse:4.583414 
#> [1]	train-rmse:16.632975	test-rmse:18.402482 
#> [2]	train-rmse:12.084679	test-rmse:14.024370 
#> [3]	train-rmse:8.872609	test-rmse:10.594402 
#> [4]	train-rmse:6.666727	test-rmse:8.359604 
#> [5]	train-rmse:5.135195	test-rmse:6.910258 
#> [6]	train-rmse:4.118610	test-rmse:6.021983 
#> [7]	train-rmse:3.421797	test-rmse:5.277892 
#> [8]	train-rmse:2.984047	test-rmse:4.875201 
#> [9]	train-rmse:2.666508	test-rmse:4.669088 
#> [10]	train-rmse:2.442604	test-rmse:4.400238 
#> [11]	train-rmse:2.263024	test-rmse:4.230482 
#> [12]	train-rmse:2.121456	test-rmse:4.079175 
#> [13]	train-rmse:2.009415	test-rmse:3.969395 
#> [14]	train-rmse:1.925757	test-rmse:3.935991 
#> [15]	train-rmse:1.877151	test-rmse:3.906905 
#> [16]	train-rmse:1.815046	test-rmse:3.890979 
#> [17]	train-rmse:1.722805	test-rmse:3.843355 
#> [18]	train-rmse:1.678251	test-rmse:3.822660 
#> [19]	train-rmse:1.638035	test-rmse:3.805111 
#> [20]	train-rmse:1.586498	test-rmse:3.772204 
#> [21]	train-rmse:1.555798	test-rmse:3.754926 
#> [22]	train-rmse:1.521831	test-rmse:3.739119 
#> [23]	train-rmse:1.485348	test-rmse:3.720224 
#> [24]	train-rmse:1.452430	test-rmse:3.720954 
#> [25]	train-rmse:1.401942	test-rmse:3.678907 
#> [26]	train-rmse:1.347862	test-rmse:3.626140 
#> [27]	train-rmse:1.333065	test-rmse:3.620101 
#> [28]	train-rmse:1.304386	test-rmse:3.602639 
#> [29]	train-rmse:1.276703	test-rmse:3.607050 
#> [30]	train-rmse:1.264345	test-rmse:3.608845 
#> [31]	train-rmse:1.250547	test-rmse:3.614718 
#> [32]	train-rmse:1.218531	test-rmse:3.599399 
#> [33]	train-rmse:1.177848	test-rmse:3.594882 
#> [34]	train-rmse:1.144331	test-rmse:3.618441 
#> [35]	train-rmse:1.132708	test-rmse:3.619260 
#> [36]	train-rmse:1.121853	test-rmse:3.618322 
#> [37]	train-rmse:1.107662	test-rmse:3.623112 
#> [38]	train-rmse:1.097314	test-rmse:3.622504 
#> [39]	train-rmse:1.091235	test-rmse:3.613645 
#> [40]	train-rmse:1.080512	test-rmse:3.610760 
#> [41]	train-rmse:1.053192	test-rmse:3.602607 
#> [42]	train-rmse:1.040840	test-rmse:3.602440 
#> [43]	train-rmse:1.023147	test-rmse:3.599051 
#> [44]	train-rmse:1.002725	test-rmse:3.593594 
#> [45]	train-rmse:0.975447	test-rmse:3.591580 
#> [46]	train-rmse:0.955695	test-rmse:3.587632 
#> [47]	train-rmse:0.945853	test-rmse:3.589384 
#> [48]	train-rmse:0.940748	test-rmse:3.586710 
#> [49]	train-rmse:0.924930	test-rmse:3.581747 
#> [50]	train-rmse:0.918651	test-rmse:3.579613 
#> [51]	train-rmse:0.906455	test-rmse:3.580222 
#> [52]	train-rmse:0.890968	test-rmse:3.579130 
#> [53]	train-rmse:0.872706	test-rmse:3.568474 
#> [54]	train-rmse:0.858351	test-rmse:3.564070 
#> [55]	train-rmse:0.849300	test-rmse:3.560762 
#> [56]	train-rmse:0.839321	test-rmse:3.562311 
#> [57]	train-rmse:0.822098	test-rmse:3.571710 
#> [58]	train-rmse:0.810739	test-rmse:3.568650 
#> [59]	train-rmse:0.797141	test-rmse:3.564591 
#> [60]	train-rmse:0.784108	test-rmse:3.568265 
#> [61]	train-rmse:0.775899	test-rmse:3.566012 
#> [62]	train-rmse:0.771441	test-rmse:3.564280 
#> [63]	train-rmse:0.760396	test-rmse:3.565345 
#> [64]	train-rmse:0.750564	test-rmse:3.566748 
#> [65]	train-rmse:0.742772	test-rmse:3.570976 
#> [66]	train-rmse:0.726596	test-rmse:3.574597 
#> [67]	train-rmse:0.719990	test-rmse:3.577241 
#> [68]	train-rmse:0.703763	test-rmse:3.587239 
#> [69]	train-rmse:0.688217	test-rmse:3.581395 
#> [70]	train-rmse:0.678489	test-rmse:3.584277 
#> [1]	train-rmse:17.420586	test-rmse:16.803931 
#> [2]	train-rmse:12.632942	test-rmse:12.057325 
#> [3]	train-rmse:9.321768	test-rmse:9.048377 
#> [4]	train-rmse:7.009684	test-rmse:6.770864 
#> [5]	train-rmse:5.419812	test-rmse:5.420105 
#> [6]	train-rmse:4.380042	test-rmse:4.697201 
#> [7]	train-rmse:3.626264	test-rmse:4.264077 
#> [8]	train-rmse:3.152862	test-rmse:3.967050 
#> [9]	train-rmse:2.828154	test-rmse:3.805159 
#> [10]	train-rmse:2.581239	test-rmse:3.609835 
#> [11]	train-rmse:2.400240	test-rmse:3.508118 
#> [12]	train-rmse:2.247041	test-rmse:3.492614 
#> [13]	train-rmse:2.139544	test-rmse:3.480955 
#> [14]	train-rmse:2.047876	test-rmse:3.441028 
#> [15]	train-rmse:1.969279	test-rmse:3.418413 
#> [16]	train-rmse:1.928213	test-rmse:3.406006 
#> [17]	train-rmse:1.859272	test-rmse:3.376072 
#> [18]	train-rmse:1.793148	test-rmse:3.375488 
#> [19]	train-rmse:1.755568	test-rmse:3.356838 
#> [20]	train-rmse:1.716810	test-rmse:3.349448 
#> [21]	train-rmse:1.683483	test-rmse:3.332201 
#> [22]	train-rmse:1.657629	test-rmse:3.339370 
#> [23]	train-rmse:1.610634	test-rmse:3.316619 
#> [24]	train-rmse:1.571209	test-rmse:3.299268 
#> [25]	train-rmse:1.549698	test-rmse:3.306010 
#> [26]	train-rmse:1.538426	test-rmse:3.302109 
#> [27]	train-rmse:1.513778	test-rmse:3.302920 
#> [28]	train-rmse:1.469105	test-rmse:3.318813 
#> [29]	train-rmse:1.424691	test-rmse:3.299384 
#> [30]	train-rmse:1.407023	test-rmse:3.309983 
#> [31]	train-rmse:1.386734	test-rmse:3.318788 
#> [32]	train-rmse:1.350868	test-rmse:3.311731 
#> [33]	train-rmse:1.342330	test-rmse:3.314040 
#> [34]	train-rmse:1.307345	test-rmse:3.307154 
#> [35]	train-rmse:1.291813	test-rmse:3.306095 
#> [36]	train-rmse:1.280663	test-rmse:3.313352 
#> [37]	train-rmse:1.245434	test-rmse:3.308364 
#> [38]	train-rmse:1.230180	test-rmse:3.305575 
#> [39]	train-rmse:1.215316	test-rmse:3.307525 
#> [40]	train-rmse:1.183399	test-rmse:3.301347 
#> [41]	train-rmse:1.160984	test-rmse:3.299754 
#> [42]	train-rmse:1.147713	test-rmse:3.295631 
#> [43]	train-rmse:1.139061	test-rmse:3.289814 
#> [44]	train-rmse:1.112078	test-rmse:3.293741 
#> [45]	train-rmse:1.104979	test-rmse:3.293868 
#> [46]	train-rmse:1.078384	test-rmse:3.290930 
#> [47]	train-rmse:1.058113	test-rmse:3.284939 
#> [48]	train-rmse:1.035028	test-rmse:3.280781 
#> [49]	train-rmse:1.025479	test-rmse:3.277803 
#> [50]	train-rmse:0.995829	test-rmse:3.293224 
#> [51]	train-rmse:0.976585	test-rmse:3.295553 
#> [52]	train-rmse:0.967754	test-rmse:3.294619 
#> [53]	train-rmse:0.946435	test-rmse:3.292584 
#> [54]	train-rmse:0.934532	test-rmse:3.283227 
#> [55]	train-rmse:0.914604	test-rmse:3.281806 
#> [56]	train-rmse:0.909798	test-rmse:3.283366 
#> [57]	train-rmse:0.901885	test-rmse:3.287257 
#> [58]	train-rmse:0.883715	test-rmse:3.281733 
#> [59]	train-rmse:0.869019	test-rmse:3.278942 
#> [60]	train-rmse:0.851390	test-rmse:3.273285 
#> [61]	train-rmse:0.834642	test-rmse:3.279204 
#> [62]	train-rmse:0.826403	test-rmse:3.278465 
#> [63]	train-rmse:0.815742	test-rmse:3.275544 
#> [64]	train-rmse:0.799335	test-rmse:3.275844 
#> [65]	train-rmse:0.785954	test-rmse:3.271935 
#> [66]	train-rmse:0.775118	test-rmse:3.274313 
#> [67]	train-rmse:0.770264	test-rmse:3.265371 
#> [68]	train-rmse:0.763633	test-rmse:3.266875 
#> [69]	train-rmse:0.752399	test-rmse:3.266356 
#> [70]	train-rmse:0.733955	test-rmse:3.254622 
#> [1]	train-rmse:17.206143	test-rmse:17.310606 
#> [2]	train-rmse:12.475230	test-rmse:12.761037 
#> [3]	train-rmse:9.211373	test-rmse:9.676766 
#> [4]	train-rmse:6.937853	test-rmse:7.458836 
#> [5]	train-rmse:5.354684	test-rmse:5.988964 
#> [6]	train-rmse:4.312569	test-rmse:5.071245 
#> [7]	train-rmse:3.603969	test-rmse:4.505517 
#> [8]	train-rmse:3.130441	test-rmse:4.239380 
#> [9]	train-rmse:2.824796	test-rmse:4.091811 
#> [10]	train-rmse:2.606360	test-rmse:3.995039 
#> [11]	train-rmse:2.468222	test-rmse:3.900546 
#> [12]	train-rmse:2.356540	test-rmse:3.883096 
#> [13]	train-rmse:2.228585	test-rmse:3.844089 
#> [14]	train-rmse:2.156309	test-rmse:3.795136 
#> [15]	train-rmse:2.066807	test-rmse:3.756831 
#> [16]	train-rmse:2.020746	test-rmse:3.755588 
#> [17]	train-rmse:1.973536	test-rmse:3.714579 
#> [18]	train-rmse:1.939959	test-rmse:3.742088 
#> [19]	train-rmse:1.909877	test-rmse:3.720112 
#> [20]	train-rmse:1.879773	test-rmse:3.717705 
#> [21]	train-rmse:1.845108	test-rmse:3.694559 
#> [22]	train-rmse:1.821461	test-rmse:3.700721 
#> [23]	train-rmse:1.740543	test-rmse:3.727705 
#> [24]	train-rmse:1.704768	test-rmse:3.713805 
#> [25]	train-rmse:1.622321	test-rmse:3.699541 
#> [26]	train-rmse:1.586920	test-rmse:3.665485 
#> [27]	train-rmse:1.561497	test-rmse:3.654759 
#> [28]	train-rmse:1.535832	test-rmse:3.634603 
#> [29]	train-rmse:1.505243	test-rmse:3.621200 
#> [30]	train-rmse:1.484575	test-rmse:3.616501 
#> [31]	train-rmse:1.459651	test-rmse:3.646291 
#> [32]	train-rmse:1.433545	test-rmse:3.635112 
#> [33]	train-rmse:1.394871	test-rmse:3.627458 
#> [34]	train-rmse:1.339169	test-rmse:3.621203 
#> [35]	train-rmse:1.301722	test-rmse:3.609021 
#> [36]	train-rmse:1.288915	test-rmse:3.603531 
#> [37]	train-rmse:1.273098	test-rmse:3.599050 
#> [38]	train-rmse:1.261219	test-rmse:3.596048 
#> [39]	train-rmse:1.239203	test-rmse:3.594390 
#> [40]	train-rmse:1.222750	test-rmse:3.600512 
#> [41]	train-rmse:1.188663	test-rmse:3.597116 
#> [42]	train-rmse:1.166662	test-rmse:3.589139 
#> [43]	train-rmse:1.156084	test-rmse:3.602857 
#> [44]	train-rmse:1.132052	test-rmse:3.592791 
#> [45]	train-rmse:1.122293	test-rmse:3.587474 
#> [46]	train-rmse:1.104472	test-rmse:3.581073 
#> [47]	train-rmse:1.097093	test-rmse:3.574353 
#> [48]	train-rmse:1.072563	test-rmse:3.567235 
#> [49]	train-rmse:1.042838	test-rmse:3.562499 
#> [50]	train-rmse:1.023958	test-rmse:3.565569 
#> [51]	train-rmse:1.012712	test-rmse:3.562736 
#> [52]	train-rmse:1.004095	test-rmse:3.558615 
#> [53]	train-rmse:0.981545	test-rmse:3.549809 
#> [54]	train-rmse:0.975209	test-rmse:3.548871 
#> [55]	train-rmse:0.967470	test-rmse:3.541378 
#> [56]	train-rmse:0.935473	test-rmse:3.534337 
#> [57]	train-rmse:0.927019	test-rmse:3.542098 
#> [58]	train-rmse:0.904968	test-rmse:3.538517 
#> [59]	train-rmse:0.898375	test-rmse:3.535053 
#> [60]	train-rmse:0.881136	test-rmse:3.534564 
#> [61]	train-rmse:0.869588	test-rmse:3.529943 
#> [62]	train-rmse:0.858381	test-rmse:3.529081 
#> [63]	train-rmse:0.843445	test-rmse:3.528588 
#> [64]	train-rmse:0.825035	test-rmse:3.539658 
#> [65]	train-rmse:0.801397	test-rmse:3.538800 
#> [66]	train-rmse:0.785303	test-rmse:3.536702 
#> [67]	train-rmse:0.764640	test-rmse:3.527970 
#> [68]	train-rmse:0.752227	test-rmse:3.522709 
#> [69]	train-rmse:0.725835	test-rmse:3.522032 
#> [70]	train-rmse:0.717273	test-rmse:3.528527 
#> [1]	train-rmse:17.497372	test-rmse:16.705213 
#> [2]	train-rmse:12.675198	test-rmse:12.169965 
#> [3]	train-rmse:9.319557	test-rmse:8.980046 
#> [4]	train-rmse:6.985055	test-rmse:6.970945 
#> [5]	train-rmse:5.390176	test-rmse:5.667425 
#> [6]	train-rmse:4.319376	test-rmse:4.787396 
#> [7]	train-rmse:3.576026	test-rmse:4.186170 
#> [8]	train-rmse:3.091486	test-rmse:3.923885 
#> [9]	train-rmse:2.737457	test-rmse:3.746149 
#> [10]	train-rmse:2.494671	test-rmse:3.634128 
#> [11]	train-rmse:2.347501	test-rmse:3.570130 
#> [12]	train-rmse:2.189063	test-rmse:3.474114 
#> [13]	train-rmse:2.084397	test-rmse:3.450230 
#> [14]	train-rmse:2.012493	test-rmse:3.413229 
#> [15]	train-rmse:1.960176	test-rmse:3.396044 
#> [16]	train-rmse:1.907340	test-rmse:3.388413 
#> [17]	train-rmse:1.866869	test-rmse:3.386020 
#> [18]	train-rmse:1.820518	test-rmse:3.359068 
#> [19]	train-rmse:1.757013	test-rmse:3.358409 
#> [20]	train-rmse:1.713591	test-rmse:3.347679 
#> [21]	train-rmse:1.658347	test-rmse:3.324007 
#> [22]	train-rmse:1.628121	test-rmse:3.302077 
#> [23]	train-rmse:1.595465	test-rmse:3.295175 
#> [24]	train-rmse:1.573173	test-rmse:3.301710 
#> [25]	train-rmse:1.557127	test-rmse:3.296946 
#> [26]	train-rmse:1.512672	test-rmse:3.267271 
#> [27]	train-rmse:1.479931	test-rmse:3.266640 
#> [28]	train-rmse:1.444716	test-rmse:3.271621 
#> [29]	train-rmse:1.425789	test-rmse:3.270048 
#> [30]	train-rmse:1.410675	test-rmse:3.276684 
#> [31]	train-rmse:1.394562	test-rmse:3.268245 
#> [32]	train-rmse:1.367539	test-rmse:3.283266 
#> [33]	train-rmse:1.339492	test-rmse:3.284132 
#> [34]	train-rmse:1.320548	test-rmse:3.283232 
#> [35]	train-rmse:1.292458	test-rmse:3.304806 
#> [36]	train-rmse:1.250490	test-rmse:3.304315 
#> [37]	train-rmse:1.215407	test-rmse:3.306732 
#> [38]	train-rmse:1.196963	test-rmse:3.315427 
#> [39]	train-rmse:1.159897	test-rmse:3.310838 
#> [40]	train-rmse:1.127439	test-rmse:3.297007 
#> [41]	train-rmse:1.114539	test-rmse:3.290294 
#> [42]	train-rmse:1.092330	test-rmse:3.295700 
#> [43]	train-rmse:1.083105	test-rmse:3.298736 
#> [44]	train-rmse:1.067643	test-rmse:3.295123 
#> [45]	train-rmse:1.033607	test-rmse:3.298065 
#> [46]	train-rmse:1.007633	test-rmse:3.315789 
#> [47]	train-rmse:0.983573	test-rmse:3.312440 
#> [48]	train-rmse:0.964810	test-rmse:3.316091 
#> [49]	train-rmse:0.956272	test-rmse:3.316081 
#> [50]	train-rmse:0.950150	test-rmse:3.319850 
#> [51]	train-rmse:0.943727	test-rmse:3.330000 
#> [52]	train-rmse:0.933783	test-rmse:3.330933 
#> [53]	train-rmse:0.924993	test-rmse:3.329577 
#> [54]	train-rmse:0.910063	test-rmse:3.330154 
#> [55]	train-rmse:0.900702	test-rmse:3.327405 
#> [56]	train-rmse:0.885572	test-rmse:3.316851 
#> [57]	train-rmse:0.866154	test-rmse:3.324717 
#> [58]	train-rmse:0.850172	test-rmse:3.324540 
#> [59]	train-rmse:0.836721	test-rmse:3.334744 
#> [60]	train-rmse:0.824576	test-rmse:3.329172 
#> [61]	train-rmse:0.819755	test-rmse:3.331554 
#> [62]	train-rmse:0.809218	test-rmse:3.330982 
#> [63]	train-rmse:0.801793	test-rmse:3.338507 
#> [64]	train-rmse:0.789447	test-rmse:3.336580 
#> [65]	train-rmse:0.775831	test-rmse:3.334755 
#> [66]	train-rmse:0.763123	test-rmse:3.332028 
#> [67]	train-rmse:0.750921	test-rmse:3.327178 
#> [68]	train-rmse:0.727153	test-rmse:3.324443 
#> [69]	train-rmse:0.716003	test-rmse:3.324771 
#> [70]	train-rmse:0.707029	test-rmse:3.323727 
#> [1]	train-rmse:17.746953	test-rmse:16.088614 
#> [2]	train-rmse:12.848058	test-rmse:11.738874 
#> [3]	train-rmse:9.445858	test-rmse:8.724812 
#> [4]	train-rmse:7.082045	test-rmse:6.689311 
#> [5]	train-rmse:5.473613	test-rmse:5.404417 
#> [6]	train-rmse:4.382359	test-rmse:4.585756 
#> [7]	train-rmse:3.663277	test-rmse:4.135243 
#> [8]	train-rmse:3.151459	test-rmse:3.810141 
#> [9]	train-rmse:2.818541	test-rmse:3.623605 
#> [10]	train-rmse:2.535063	test-rmse:3.503854 
#> [11]	train-rmse:2.347760	test-rmse:3.427004 
#> [12]	train-rmse:2.220339	test-rmse:3.383669 
#> [13]	train-rmse:2.120341	test-rmse:3.333476 
#> [14]	train-rmse:2.032737	test-rmse:3.324670 
#> [15]	train-rmse:1.965348	test-rmse:3.298366 
#> [16]	train-rmse:1.908704	test-rmse:3.267163 
#> [17]	train-rmse:1.854787	test-rmse:3.264975 
#> [18]	train-rmse:1.791170	test-rmse:3.243805 
#> [19]	train-rmse:1.740559	test-rmse:3.214075 
#> [20]	train-rmse:1.704997	test-rmse:3.181556 
#> [21]	train-rmse:1.643535	test-rmse:3.166293 
#> [22]	train-rmse:1.592413	test-rmse:3.165949 
#> [23]	train-rmse:1.565962	test-rmse:3.166594 
#> [24]	train-rmse:1.511071	test-rmse:3.183016 
#> [25]	train-rmse:1.462349	test-rmse:3.165835 
#> [26]	train-rmse:1.444158	test-rmse:3.168277 
#> [27]	train-rmse:1.419945	test-rmse:3.138727 
#> [28]	train-rmse:1.381454	test-rmse:3.132851 
#> [29]	train-rmse:1.342097	test-rmse:3.124041 
#> [30]	train-rmse:1.318796	test-rmse:3.107525 
#> [31]	train-rmse:1.293426	test-rmse:3.107286 
#> [32]	train-rmse:1.281636	test-rmse:3.080113 
#> [33]	train-rmse:1.247643	test-rmse:3.073300 
#> [34]	train-rmse:1.229100	test-rmse:3.081445 
#> [35]	train-rmse:1.207040	test-rmse:3.070193 
#> [36]	train-rmse:1.186971	test-rmse:3.064941 
#> [37]	train-rmse:1.174614	test-rmse:3.060739 
#> [38]	train-rmse:1.152327	test-rmse:3.052294 
#> [39]	train-rmse:1.131991	test-rmse:3.055758 
#> [40]	train-rmse:1.116384	test-rmse:3.052850 
#> [41]	train-rmse:1.110065	test-rmse:3.046765 
#> [42]	train-rmse:1.098135	test-rmse:3.046463 
#> [43]	train-rmse:1.077706	test-rmse:3.038278 
#> [44]	train-rmse:1.051197	test-rmse:3.043176 
#> [45]	train-rmse:1.038141	test-rmse:3.050361 
#> [46]	train-rmse:1.031067	test-rmse:3.046429 
#> [47]	train-rmse:1.010061	test-rmse:3.042669 
#> [48]	train-rmse:0.991994	test-rmse:3.041003 
#> [49]	train-rmse:0.974017	test-rmse:3.043991 
#> [50]	train-rmse:0.954146	test-rmse:3.043860 
#> [51]	train-rmse:0.940643	test-rmse:3.043554 
#> [52]	train-rmse:0.925621	test-rmse:3.048857 
#> [53]	train-rmse:0.907251	test-rmse:3.063045 
#> [54]	train-rmse:0.895753	test-rmse:3.064538 
#> [55]	train-rmse:0.881166	test-rmse:3.064388 
#> [56]	train-rmse:0.866438	test-rmse:3.067355 
#> [57]	train-rmse:0.860904	test-rmse:3.064627 
#> [58]	train-rmse:0.849280	test-rmse:3.071963 
#> [59]	train-rmse:0.834151	test-rmse:3.077752 
#> [60]	train-rmse:0.823731	test-rmse:3.074505 
#> [61]	train-rmse:0.806629	test-rmse:3.072093 
#> [62]	train-rmse:0.795235	test-rmse:3.063652 
#> [63]	train-rmse:0.782851	test-rmse:3.057359 
#> [64]	train-rmse:0.771702	test-rmse:3.057988 
#> [65]	train-rmse:0.761396	test-rmse:3.063430 
#> [66]	train-rmse:0.750726	test-rmse:3.064544 
#> [67]	train-rmse:0.737999	test-rmse:3.063073 
#> [68]	train-rmse:0.726631	test-rmse:3.069981 
#> [69]	train-rmse:0.709756	test-rmse:3.068084 
#> [70]	train-rmse:0.702123	test-rmse:3.072209 
#> [1]	train-rmse:17.757345	test-rmse:16.205676 
#> [2]	train-rmse:12.875820	test-rmse:11.608635 
#> [3]	train-rmse:9.462338	test-rmse:8.523874 
#> [4]	train-rmse:7.056919	test-rmse:6.321476 
#> [5]	train-rmse:5.456711	test-rmse:5.045230 
#> [6]	train-rmse:4.360768	test-rmse:4.358151 
#> [7]	train-rmse:3.590120	test-rmse:3.811730 
#> [8]	train-rmse:3.122081	test-rmse:3.503827 
#> [9]	train-rmse:2.821735	test-rmse:3.434037 
#> [10]	train-rmse:2.600017	test-rmse:3.309007 
#> [11]	train-rmse:2.433657	test-rmse:3.292402 
#> [12]	train-rmse:2.328481	test-rmse:3.270210 
#> [13]	train-rmse:2.235187	test-rmse:3.215807 
#> [14]	train-rmse:2.159686	test-rmse:3.195468 
#> [15]	train-rmse:2.070844	test-rmse:3.175675 
#> [16]	train-rmse:2.011279	test-rmse:3.197592 
#> [17]	train-rmse:1.975715	test-rmse:3.198668 
#> [18]	train-rmse:1.928552	test-rmse:3.204729 
#> [19]	train-rmse:1.870943	test-rmse:3.208802 
#> [20]	train-rmse:1.821949	test-rmse:3.203317 
#> [21]	train-rmse:1.797158	test-rmse:3.200699 
#> [22]	train-rmse:1.752589	test-rmse:3.163103 
#> [23]	train-rmse:1.720680	test-rmse:3.155974 
#> [24]	train-rmse:1.660530	test-rmse:3.160287 
#> [25]	train-rmse:1.625455	test-rmse:3.158182 
#> [26]	train-rmse:1.601794	test-rmse:3.164349 
#> [27]	train-rmse:1.588628	test-rmse:3.175391 
#> [28]	train-rmse:1.537036	test-rmse:3.163229 
#> [29]	train-rmse:1.500139	test-rmse:3.181958 
#> [30]	train-rmse:1.470186	test-rmse:3.189175 
#> [31]	train-rmse:1.445744	test-rmse:3.181643 
#> [32]	train-rmse:1.422065	test-rmse:3.176888 
#> [33]	train-rmse:1.381573	test-rmse:3.190551 
#> [34]	train-rmse:1.350240	test-rmse:3.181641 
#> [35]	train-rmse:1.334542	test-rmse:3.180492 
#> [36]	train-rmse:1.317453	test-rmse:3.186737 
#> [37]	train-rmse:1.296085	test-rmse:3.202003 
#> [38]	train-rmse:1.269124	test-rmse:3.203128 
#> [39]	train-rmse:1.250360	test-rmse:3.193755 
#> [40]	train-rmse:1.223990	test-rmse:3.186283 
#> [41]	train-rmse:1.191271	test-rmse:3.195541 
#> [42]	train-rmse:1.180929	test-rmse:3.198217 
#> [43]	train-rmse:1.154820	test-rmse:3.204580 
#> [44]	train-rmse:1.144795	test-rmse:3.201658 
#> [45]	train-rmse:1.136989	test-rmse:3.204554 
#> [46]	train-rmse:1.123091	test-rmse:3.206007 
#> [47]	train-rmse:1.101891	test-rmse:3.189574 
#> [48]	train-rmse:1.085828	test-rmse:3.179985 
#> [49]	train-rmse:1.055641	test-rmse:3.189480 
#> [50]	train-rmse:1.042670	test-rmse:3.200033 
#> [51]	train-rmse:1.030204	test-rmse:3.198881 
#> [52]	train-rmse:1.009598	test-rmse:3.198746 
#> [53]	train-rmse:0.994990	test-rmse:3.206915 
#> [54]	train-rmse:0.989716	test-rmse:3.208727 
#> [55]	train-rmse:0.976126	test-rmse:3.205484 
#> [56]	train-rmse:0.964928	test-rmse:3.209258 
#> [57]	train-rmse:0.946495	test-rmse:3.203329 
#> [58]	train-rmse:0.921372	test-rmse:3.201259 
#> [59]	train-rmse:0.907611	test-rmse:3.204599 
#> [60]	train-rmse:0.889665	test-rmse:3.206765 
#> [61]	train-rmse:0.864126	test-rmse:3.207128 
#> [62]	train-rmse:0.853117	test-rmse:3.206127 
#> [63]	train-rmse:0.846385	test-rmse:3.204270 
#> [64]	train-rmse:0.826482	test-rmse:3.208616 
#> [65]	train-rmse:0.813973	test-rmse:3.204869 
#> [66]	train-rmse:0.807103	test-rmse:3.203939 
#> [67]	train-rmse:0.789126	test-rmse:3.203539 
#> [68]	train-rmse:0.781736	test-rmse:3.204418 
#> [69]	train-rmse:0.768352	test-rmse:3.194666 
#> [70]	train-rmse:0.748397	test-rmse:3.191436 
#> [1]	train-rmse:17.153358	test-rmse:17.247926 
#> [2]	train-rmse:12.441687	test-rmse:12.652032 
#> [3]	train-rmse:9.187360	test-rmse:9.469573 
#> [4]	train-rmse:6.922300	test-rmse:7.287823 
#> [5]	train-rmse:5.318238	test-rmse:5.978485 
#> [6]	train-rmse:4.268214	test-rmse:5.063714 
#> [7]	train-rmse:3.581960	test-rmse:4.560218 
#> [8]	train-rmse:3.042221	test-rmse:4.064226 
#> [9]	train-rmse:2.724658	test-rmse:3.933850 
#> [10]	train-rmse:2.495427	test-rmse:3.809304 
#> [11]	train-rmse:2.316863	test-rmse:3.620897 
#> [12]	train-rmse:2.204720	test-rmse:3.567965 
#> [13]	train-rmse:2.092047	test-rmse:3.518681 
#> [14]	train-rmse:2.028509	test-rmse:3.495556 
#> [15]	train-rmse:1.955107	test-rmse:3.423853 
#> [16]	train-rmse:1.909453	test-rmse:3.412761 
#> [17]	train-rmse:1.827261	test-rmse:3.333483 
#> [18]	train-rmse:1.793152	test-rmse:3.316521 
#> [19]	train-rmse:1.756339	test-rmse:3.302823 
#> [20]	train-rmse:1.698962	test-rmse:3.318053 
#> [21]	train-rmse:1.673975	test-rmse:3.318148 
#> [22]	train-rmse:1.652197	test-rmse:3.300669 
#> [23]	train-rmse:1.627174	test-rmse:3.287246 
#> [24]	train-rmse:1.580979	test-rmse:3.260976 
#> [25]	train-rmse:1.547157	test-rmse:3.254116 
#> [26]	train-rmse:1.515574	test-rmse:3.249121 
#> [27]	train-rmse:1.471449	test-rmse:3.265800 
#> [28]	train-rmse:1.443393	test-rmse:3.275661 
#> [29]	train-rmse:1.417479	test-rmse:3.269999 
#> [30]	train-rmse:1.402362	test-rmse:3.274996 
#> [31]	train-rmse:1.394953	test-rmse:3.269014 
#> [32]	train-rmse:1.375341	test-rmse:3.261593 
#> [33]	train-rmse:1.353052	test-rmse:3.260496 
#> [34]	train-rmse:1.332705	test-rmse:3.262382 
#> [35]	train-rmse:1.324803	test-rmse:3.256628 
#> [36]	train-rmse:1.285419	test-rmse:3.225293 
#> [37]	train-rmse:1.258540	test-rmse:3.228784 
#> [38]	train-rmse:1.246189	test-rmse:3.224981 
#> [39]	train-rmse:1.229884	test-rmse:3.197306 
#> [40]	train-rmse:1.207481	test-rmse:3.191800 
#> [41]	train-rmse:1.194144	test-rmse:3.191253 
#> [42]	train-rmse:1.178394	test-rmse:3.189043 
#> [43]	train-rmse:1.159140	test-rmse:3.198097 
#> [44]	train-rmse:1.132824	test-rmse:3.184262 
#> [45]	train-rmse:1.115965	test-rmse:3.187304 
#> [46]	train-rmse:1.081659	test-rmse:3.183274 
#> [47]	train-rmse:1.049598	test-rmse:3.185394 
#> [48]	train-rmse:1.035387	test-rmse:3.185552 
#> [49]	train-rmse:1.023945	test-rmse:3.185979 
#> [50]	train-rmse:0.998014	test-rmse:3.184844 
#> [51]	train-rmse:0.981964	test-rmse:3.182987 
#> [52]	train-rmse:0.974405	test-rmse:3.184497 
#> [53]	train-rmse:0.961793	test-rmse:3.188064 
#> [54]	train-rmse:0.951966	test-rmse:3.185168 
#> [55]	train-rmse:0.945683	test-rmse:3.183075 
#> [56]	train-rmse:0.935008	test-rmse:3.181035 
#> [57]	train-rmse:0.932093	test-rmse:3.184060 
#> [58]	train-rmse:0.917324	test-rmse:3.189954 
#> [59]	train-rmse:0.906029	test-rmse:3.187350 
#> [60]	train-rmse:0.889095	test-rmse:3.189489 
#> [61]	train-rmse:0.878756	test-rmse:3.189283 
#> [62]	train-rmse:0.872081	test-rmse:3.190268 
#> [63]	train-rmse:0.861711	test-rmse:3.186995 
#> [64]	train-rmse:0.849970	test-rmse:3.188500 
#> [65]	train-rmse:0.832303	test-rmse:3.176611 
#> [66]	train-rmse:0.818736	test-rmse:3.185935 
#> [67]	train-rmse:0.805083	test-rmse:3.183172 
#> [68]	train-rmse:0.798027	test-rmse:3.176018 
#> [69]	train-rmse:0.788944	test-rmse:3.181227 
#> [70]	train-rmse:0.778040	test-rmse:3.174212 
#> [1]	train-rmse:17.079528	test-rmse:17.377109 
#> [2]	train-rmse:12.384860	test-rmse:12.625349 
#> [3]	train-rmse:9.165314	test-rmse:9.311551 
#> [4]	train-rmse:6.893540	test-rmse:7.021701 
#> [5]	train-rmse:5.309133	test-rmse:5.393646 
#> [6]	train-rmse:4.271522	test-rmse:4.430744 
#> [7]	train-rmse:3.561669	test-rmse:3.837535 
#> [8]	train-rmse:3.064646	test-rmse:3.522244 
#> [9]	train-rmse:2.769855	test-rmse:3.251925 
#> [10]	train-rmse:2.561877	test-rmse:3.167872 
#> [11]	train-rmse:2.420761	test-rmse:3.088785 
#> [12]	train-rmse:2.296889	test-rmse:3.020767 
#> [13]	train-rmse:2.209150	test-rmse:2.976007 
#> [14]	train-rmse:2.157451	test-rmse:2.961612 
#> [15]	train-rmse:2.054361	test-rmse:2.959658 
#> [16]	train-rmse:2.005199	test-rmse:2.964704 
#> [17]	train-rmse:1.941198	test-rmse:2.960652 
#> [18]	train-rmse:1.881997	test-rmse:2.935266 
#> [19]	train-rmse:1.822826	test-rmse:2.920812 
#> [20]	train-rmse:1.790203	test-rmse:2.904936 
#> [21]	train-rmse:1.760105	test-rmse:2.891240 
#> [22]	train-rmse:1.707131	test-rmse:2.896128 
#> [23]	train-rmse:1.662741	test-rmse:2.901868 
#> [24]	train-rmse:1.633643	test-rmse:2.898332 
#> [25]	train-rmse:1.608720	test-rmse:2.891816 
#> [26]	train-rmse:1.566568	test-rmse:2.928903 
#> [27]	train-rmse:1.542651	test-rmse:2.920520 
#> [28]	train-rmse:1.514065	test-rmse:2.928540 
#> [29]	train-rmse:1.478750	test-rmse:2.930061 
#> [30]	train-rmse:1.462296	test-rmse:2.917730 
#> [31]	train-rmse:1.421458	test-rmse:2.910186 
#> [32]	train-rmse:1.392717	test-rmse:2.909048 
#> [33]	train-rmse:1.360096	test-rmse:2.901942 
#> [34]	train-rmse:1.348711	test-rmse:2.895804 
#> [35]	train-rmse:1.331264	test-rmse:2.881988 
#> [36]	train-rmse:1.301336	test-rmse:2.895174 
#> [37]	train-rmse:1.270702	test-rmse:2.877713 
#> [38]	train-rmse:1.246264	test-rmse:2.900819 
#> [39]	train-rmse:1.237540	test-rmse:2.903270 
#> [40]	train-rmse:1.204781	test-rmse:2.899446 
#> [41]	train-rmse:1.186700	test-rmse:2.888834 
#> [42]	train-rmse:1.167358	test-rmse:2.898562 
#> [43]	train-rmse:1.145142	test-rmse:2.901068 
#> [44]	train-rmse:1.137023	test-rmse:2.896725 
#> [45]	train-rmse:1.110583	test-rmse:2.896236 
#> [46]	train-rmse:1.088840	test-rmse:2.889204 
#> [47]	train-rmse:1.079157	test-rmse:2.888155 
#> [48]	train-rmse:1.070868	test-rmse:2.884162 
#> [49]	train-rmse:1.059954	test-rmse:2.880864 
#> [50]	train-rmse:1.042157	test-rmse:2.888545 
#> [51]	train-rmse:1.017776	test-rmse:2.878213 
#> [52]	train-rmse:0.988539	test-rmse:2.874488 
#> [53]	train-rmse:0.971234	test-rmse:2.868676 
#> [54]	train-rmse:0.956523	test-rmse:2.863017 
#> [55]	train-rmse:0.941375	test-rmse:2.862686 
#> [56]	train-rmse:0.927121	test-rmse:2.861085 
#> [57]	train-rmse:0.919926	test-rmse:2.855945 
#> [58]	train-rmse:0.912473	test-rmse:2.865741 
#> [59]	train-rmse:0.899319	test-rmse:2.865994 
#> [60]	train-rmse:0.886971	test-rmse:2.860850 
#> [61]	train-rmse:0.875922	test-rmse:2.856875 
#> [62]	train-rmse:0.870501	test-rmse:2.855029 
#> [63]	train-rmse:0.853772	test-rmse:2.853909 
#> [64]	train-rmse:0.847366	test-rmse:2.860050 
#> [65]	train-rmse:0.831146	test-rmse:2.859409 
#> [66]	train-rmse:0.818748	test-rmse:2.848222 
#> [67]	train-rmse:0.806568	test-rmse:2.842176 
#> [68]	train-rmse:0.799445	test-rmse:2.833018 
#> [69]	train-rmse:0.784608	test-rmse:2.814668 
#> [70]	train-rmse:0.774472	test-rmse:2.817666 
#> [1]	train-rmse:17.174497	test-rmse:17.264833 
#> [2]	train-rmse:12.466474	test-rmse:12.619905 
#> [3]	train-rmse:9.148978	test-rmse:9.390790 
#> [4]	train-rmse:6.885397	test-rmse:7.204675 
#> [5]	train-rmse:5.279459	test-rmse:5.841540 
#> [6]	train-rmse:4.242885	test-rmse:5.017919 
#> [7]	train-rmse:3.486216	test-rmse:4.416454 
#> [8]	train-rmse:3.015869	test-rmse:4.114004 
#> [9]	train-rmse:2.708992	test-rmse:3.924625 
#> [10]	train-rmse:2.490774	test-rmse:3.806508 
#> [11]	train-rmse:2.333321	test-rmse:3.718458 
#> [12]	train-rmse:2.227711	test-rmse:3.679917 
#> [13]	train-rmse:2.141888	test-rmse:3.660030 
#> [14]	train-rmse:2.082735	test-rmse:3.674052 
#> [15]	train-rmse:1.996223	test-rmse:3.611874 
#> [16]	train-rmse:1.960814	test-rmse:3.588602 
#> [17]	train-rmse:1.903594	test-rmse:3.572304 
#> [18]	train-rmse:1.854117	test-rmse:3.560568 
#> [19]	train-rmse:1.821896	test-rmse:3.559986 
#> [20]	train-rmse:1.744679	test-rmse:3.563565 
#> [21]	train-rmse:1.720437	test-rmse:3.557131 
#> [22]	train-rmse:1.687881	test-rmse:3.559080 
#> [23]	train-rmse:1.647137	test-rmse:3.538544 
#> [24]	train-rmse:1.618445	test-rmse:3.541620 
#> [25]	train-rmse:1.557838	test-rmse:3.483289 
#> [26]	train-rmse:1.535155	test-rmse:3.493737 
#> [27]	train-rmse:1.495709	test-rmse:3.496462 
#> [28]	train-rmse:1.475953	test-rmse:3.497386 
#> [29]	train-rmse:1.451411	test-rmse:3.506798 
#> [30]	train-rmse:1.427954	test-rmse:3.507897 
#> [31]	train-rmse:1.402805	test-rmse:3.493114 
#> [32]	train-rmse:1.367876	test-rmse:3.466924 
#> [33]	train-rmse:1.340332	test-rmse:3.460579 
#> [34]	train-rmse:1.316756	test-rmse:3.466749 
#> [35]	train-rmse:1.281867	test-rmse:3.467192 
#> [36]	train-rmse:1.272643	test-rmse:3.455104 
#> [37]	train-rmse:1.254233	test-rmse:3.457215 
#> [38]	train-rmse:1.225753	test-rmse:3.449404 
#> [39]	train-rmse:1.209607	test-rmse:3.452840 
#> [40]	train-rmse:1.190480	test-rmse:3.455185 
#> [41]	train-rmse:1.179451	test-rmse:3.454639 
#> [42]	train-rmse:1.161244	test-rmse:3.452408 
#> [43]	train-rmse:1.149862	test-rmse:3.457055 
#> [44]	train-rmse:1.130977	test-rmse:3.462610 
#> [45]	train-rmse:1.111614	test-rmse:3.462926 
#> [46]	train-rmse:1.085415	test-rmse:3.457275 
#> [47]	train-rmse:1.070715	test-rmse:3.472729 
#> [48]	train-rmse:1.060586	test-rmse:3.472500 
#> [49]	train-rmse:1.053762	test-rmse:3.461818 
#> [50]	train-rmse:1.032720	test-rmse:3.463770 
#> [51]	train-rmse:1.007656	test-rmse:3.464170 
#> [52]	train-rmse:0.997822	test-rmse:3.464671 
#> [53]	train-rmse:0.985003	test-rmse:3.463193 
#> [54]	train-rmse:0.973618	test-rmse:3.464843 
#> [55]	train-rmse:0.953144	test-rmse:3.461074 
#> [56]	train-rmse:0.936272	test-rmse:3.460912 
#> [57]	train-rmse:0.921454	test-rmse:3.468553 
#> [58]	train-rmse:0.914720	test-rmse:3.476876 
#> [59]	train-rmse:0.905542	test-rmse:3.472793 
#> [60]	train-rmse:0.884523	test-rmse:3.474688 
#> [61]	train-rmse:0.862461	test-rmse:3.472315 
#> [62]	train-rmse:0.848348	test-rmse:3.478386 
#> [63]	train-rmse:0.835287	test-rmse:3.478017 
#> [64]	train-rmse:0.826955	test-rmse:3.480483 
#> [65]	train-rmse:0.816417	test-rmse:3.479547 
#> [66]	train-rmse:0.811387	test-rmse:3.490310 
#> [67]	train-rmse:0.806071	test-rmse:3.492417 
#> [68]	train-rmse:0.795524	test-rmse:3.500655 
#> [69]	train-rmse:0.782489	test-rmse:3.503188 
#> [70]	train-rmse:0.774461	test-rmse:3.500021 
#> [1]	train-rmse:16.779350	test-rmse:18.207919 
#> [2]	train-rmse:12.171179	test-rmse:13.445699 
#> [3]	train-rmse:8.927165	test-rmse:10.283390 
#> [4]	train-rmse:6.696618	test-rmse:8.345708 
#> [5]	train-rmse:5.136253	test-rmse:6.780196 
#> [6]	train-rmse:4.105116	test-rmse:5.797880 
#> [7]	train-rmse:3.415105	test-rmse:5.308461 
#> [8]	train-rmse:2.943451	test-rmse:4.918321 
#> [9]	train-rmse:2.658312	test-rmse:4.680732 
#> [10]	train-rmse:2.454329	test-rmse:4.567930 
#> [11]	train-rmse:2.281712	test-rmse:4.379558 
#> [12]	train-rmse:2.198491	test-rmse:4.335912 
#> [13]	train-rmse:2.071788	test-rmse:4.304987 
#> [14]	train-rmse:1.977886	test-rmse:4.205243 
#> [15]	train-rmse:1.894389	test-rmse:4.182313 
#> [16]	train-rmse:1.854751	test-rmse:4.159849 
#> [17]	train-rmse:1.808117	test-rmse:4.152690 
#> [18]	train-rmse:1.756165	test-rmse:4.155184 
#> [19]	train-rmse:1.717520	test-rmse:4.161827 
#> [20]	train-rmse:1.684331	test-rmse:4.163554 
#> [21]	train-rmse:1.646353	test-rmse:4.162023 
#> [22]	train-rmse:1.623560	test-rmse:4.156331 
#> [23]	train-rmse:1.575575	test-rmse:4.121608 
#> [24]	train-rmse:1.552324	test-rmse:4.118739 
#> [25]	train-rmse:1.506545	test-rmse:4.111395 
#> [26]	train-rmse:1.482322	test-rmse:4.104987 
#> [27]	train-rmse:1.435784	test-rmse:4.090974 
#> [28]	train-rmse:1.413571	test-rmse:4.090047 
#> [29]	train-rmse:1.376245	test-rmse:4.078865 
#> [30]	train-rmse:1.343570	test-rmse:4.054152 
#> [31]	train-rmse:1.329362	test-rmse:4.052484 
#> [32]	train-rmse:1.287098	test-rmse:4.053874 
#> [33]	train-rmse:1.266878	test-rmse:4.055464 
#> [34]	train-rmse:1.230480	test-rmse:4.036984 
#> [35]	train-rmse:1.207075	test-rmse:4.024112 
#> [36]	train-rmse:1.175968	test-rmse:4.021759 
#> [37]	train-rmse:1.140574	test-rmse:3.998033 
#> [38]	train-rmse:1.127684	test-rmse:3.998154 
#> [39]	train-rmse:1.104108	test-rmse:3.994652 
#> [40]	train-rmse:1.082294	test-rmse:4.015869 
#> [41]	train-rmse:1.049164	test-rmse:4.023144 
#> [42]	train-rmse:1.021966	test-rmse:4.025873 
#> [43]	train-rmse:1.000787	test-rmse:4.022535 
#> [44]	train-rmse:0.985148	test-rmse:4.018350 
#> [45]	train-rmse:0.975522	test-rmse:4.013462 
#> [46]	train-rmse:0.955586	test-rmse:4.018598 
#> [47]	train-rmse:0.936245	test-rmse:4.013409 
#> [48]	train-rmse:0.926763	test-rmse:4.014300 
#> [49]	train-rmse:0.916715	test-rmse:4.005393 
#> [50]	train-rmse:0.902231	test-rmse:4.002152 
#> [51]	train-rmse:0.887179	test-rmse:4.001605 
#> [52]	train-rmse:0.873935	test-rmse:4.002588 
#> [53]	train-rmse:0.865345	test-rmse:4.001689 
#> [54]	train-rmse:0.857539	test-rmse:4.001055 
#> [55]	train-rmse:0.843897	test-rmse:4.009306 
#> [56]	train-rmse:0.829447	test-rmse:4.002423 
#> [57]	train-rmse:0.817708	test-rmse:4.001942 
#> [58]	train-rmse:0.804459	test-rmse:4.002660 
#> [59]	train-rmse:0.791290	test-rmse:3.990508 
#> [60]	train-rmse:0.784867	test-rmse:3.989261 
#> [61]	train-rmse:0.767499	test-rmse:3.980973 
#> [62]	train-rmse:0.749397	test-rmse:3.985475 
#> [63]	train-rmse:0.740022	test-rmse:3.986033 
#> [64]	train-rmse:0.731210	test-rmse:3.987539 
#> [65]	train-rmse:0.725246	test-rmse:3.995125 
#> [66]	train-rmse:0.712231	test-rmse:3.996640 
#> [67]	train-rmse:0.701003	test-rmse:3.995766 
#> [68]	train-rmse:0.687988	test-rmse:3.997611 
#> [69]	train-rmse:0.672621	test-rmse:3.992692 
#> [70]	train-rmse:0.660912	test-rmse:3.992390 
#> [1]	train-rmse:17.644143	test-rmse:16.369328 
#> [2]	train-rmse:12.794632	test-rmse:11.929539 
#> [3]	train-rmse:9.413120	test-rmse:8.985036 
#> [4]	train-rmse:7.030341	test-rmse:7.014112 
#> [5]	train-rmse:5.395858	test-rmse:5.668539 
#> [6]	train-rmse:4.284418	test-rmse:4.805353 
#> [7]	train-rmse:3.532857	test-rmse:4.321955 
#> [8]	train-rmse:3.062053	test-rmse:4.041404 
#> [9]	train-rmse:2.727744	test-rmse:3.880189 
#> [10]	train-rmse:2.437574	test-rmse:3.759992 
#> [11]	train-rmse:2.271091	test-rmse:3.729054 
#> [12]	train-rmse:2.158544	test-rmse:3.660379 
#> [13]	train-rmse:2.056166	test-rmse:3.614323 
#> [14]	train-rmse:2.008425	test-rmse:3.581949 
#> [15]	train-rmse:1.963335	test-rmse:3.562357 
#> [16]	train-rmse:1.892000	test-rmse:3.550607 
#> [17]	train-rmse:1.838552	test-rmse:3.593554 
#> [18]	train-rmse:1.806830	test-rmse:3.585898 
#> [19]	train-rmse:1.736500	test-rmse:3.585468 
#> [20]	train-rmse:1.680566	test-rmse:3.569657 
#> [21]	train-rmse:1.641793	test-rmse:3.554016 
#> [22]	train-rmse:1.621415	test-rmse:3.546730 
#> [23]	train-rmse:1.602346	test-rmse:3.569991 
#> [24]	train-rmse:1.576039	test-rmse:3.560067 
#> [25]	train-rmse:1.541867	test-rmse:3.543751 
#> [26]	train-rmse:1.521339	test-rmse:3.536509 
#> [27]	train-rmse:1.495745	test-rmse:3.541574 
#> [28]	train-rmse:1.479090	test-rmse:3.548629 
#> [29]	train-rmse:1.461375	test-rmse:3.553321 
#> [30]	train-rmse:1.431670	test-rmse:3.549392 
#> [31]	train-rmse:1.404947	test-rmse:3.552035 
#> [32]	train-rmse:1.363493	test-rmse:3.549040 
#> [33]	train-rmse:1.341505	test-rmse:3.548294 
#> [34]	train-rmse:1.334772	test-rmse:3.545150 
#> [35]	train-rmse:1.304933	test-rmse:3.532957 
#> [36]	train-rmse:1.267487	test-rmse:3.538322 
#> [37]	train-rmse:1.256616	test-rmse:3.535697 
#> [38]	train-rmse:1.240113	test-rmse:3.522899 
#> [39]	train-rmse:1.221487	test-rmse:3.520452 
#> [40]	train-rmse:1.211621	test-rmse:3.527968 
#> [41]	train-rmse:1.187889	test-rmse:3.522384 
#> [42]	train-rmse:1.176825	test-rmse:3.516388 
#> [43]	train-rmse:1.151752	test-rmse:3.521675 
#> [44]	train-rmse:1.143956	test-rmse:3.520928 
#> [45]	train-rmse:1.125564	test-rmse:3.533644 
#> [46]	train-rmse:1.091267	test-rmse:3.526531 
#> [47]	train-rmse:1.080519	test-rmse:3.526824 
#> [48]	train-rmse:1.068470	test-rmse:3.523016 
#> [49]	train-rmse:1.027419	test-rmse:3.513712 
#> [50]	train-rmse:0.999058	test-rmse:3.510759 
#> [51]	train-rmse:0.980671	test-rmse:3.507549 
#> [52]	train-rmse:0.958796	test-rmse:3.499991 
#> [53]	train-rmse:0.952550	test-rmse:3.504307 
#> [54]	train-rmse:0.932789	test-rmse:3.490623 
#> [55]	train-rmse:0.924410	test-rmse:3.496458 
#> [56]	train-rmse:0.919869	test-rmse:3.498303 
#> [57]	train-rmse:0.902118	test-rmse:3.499676 
#> [58]	train-rmse:0.885793	test-rmse:3.499246 
#> [59]	train-rmse:0.863765	test-rmse:3.484370 
#> [60]	train-rmse:0.847728	test-rmse:3.477669 
#> [61]	train-rmse:0.836233	test-rmse:3.475560 
#> [62]	train-rmse:0.825694	test-rmse:3.471706 
#> [63]	train-rmse:0.817225	test-rmse:3.472442 
#> [64]	train-rmse:0.803572	test-rmse:3.472471 
#> [65]	train-rmse:0.796228	test-rmse:3.472661 
#> [66]	train-rmse:0.779853	test-rmse:3.468595 
#> [67]	train-rmse:0.773933	test-rmse:3.467307 
#> [68]	train-rmse:0.764835	test-rmse:3.459772 
#> [69]	train-rmse:0.761240	test-rmse:3.459294 
#> [70]	train-rmse:0.755777	test-rmse:3.459214 
#> [1]	train-rmse:17.684751	test-rmse:16.500476 
#> [2]	train-rmse:12.793272	test-rmse:11.937067 
#> [3]	train-rmse:9.401096	test-rmse:8.867871 
#> [4]	train-rmse:7.007000	test-rmse:6.741049 
#> [5]	train-rmse:5.403779	test-rmse:5.400790 
#> [6]	train-rmse:4.300519	test-rmse:4.586066 
#> [7]	train-rmse:3.573068	test-rmse:4.168921 
#> [8]	train-rmse:3.040646	test-rmse:3.871437 
#> [9]	train-rmse:2.696866	test-rmse:3.704518 
#> [10]	train-rmse:2.477213	test-rmse:3.618700 
#> [11]	train-rmse:2.334902	test-rmse:3.575200 
#> [12]	train-rmse:2.231843	test-rmse:3.554308 
#> [13]	train-rmse:2.147690	test-rmse:3.488702 
#> [14]	train-rmse:2.063000	test-rmse:3.411661 
#> [15]	train-rmse:1.997862	test-rmse:3.390124 
#> [16]	train-rmse:1.950103	test-rmse:3.368649 
#> [17]	train-rmse:1.903784	test-rmse:3.362779 
#> [18]	train-rmse:1.873729	test-rmse:3.361938 
#> [19]	train-rmse:1.830452	test-rmse:3.356057 
#> [20]	train-rmse:1.797668	test-rmse:3.327144 
#> [21]	train-rmse:1.715509	test-rmse:3.297517 
#> [22]	train-rmse:1.651769	test-rmse:3.295284 
#> [23]	train-rmse:1.614815	test-rmse:3.294483 
#> [24]	train-rmse:1.589045	test-rmse:3.279542 
#> [25]	train-rmse:1.557201	test-rmse:3.268105 
#> [26]	train-rmse:1.509195	test-rmse:3.242756 
#> [27]	train-rmse:1.493377	test-rmse:3.237224 
#> [28]	train-rmse:1.454785	test-rmse:3.246490 
#> [29]	train-rmse:1.415584	test-rmse:3.247198 
#> [30]	train-rmse:1.400648	test-rmse:3.249404 
#> [31]	train-rmse:1.381394	test-rmse:3.253635 
#> [32]	train-rmse:1.350793	test-rmse:3.255294 
#> [33]	train-rmse:1.313821	test-rmse:3.237575 
#> [34]	train-rmse:1.284443	test-rmse:3.244345 
#> [35]	train-rmse:1.248565	test-rmse:3.249575 
#> [36]	train-rmse:1.230584	test-rmse:3.242797 
#> [37]	train-rmse:1.205088	test-rmse:3.241367 
#> [38]	train-rmse:1.189449	test-rmse:3.225965 
#> [39]	train-rmse:1.161576	test-rmse:3.236384 
#> [40]	train-rmse:1.132401	test-rmse:3.232241 
#> [41]	train-rmse:1.117285	test-rmse:3.238234 
#> [42]	train-rmse:1.108723	test-rmse:3.241522 
#> [43]	train-rmse:1.077248	test-rmse:3.257564 
#> [44]	train-rmse:1.069290	test-rmse:3.265557 
#> [45]	train-rmse:1.046985	test-rmse:3.257532 
#> [46]	train-rmse:1.041975	test-rmse:3.266573 
#> [47]	train-rmse:1.025338	test-rmse:3.261892 
#> [48]	train-rmse:1.019173	test-rmse:3.263357 
#> [49]	train-rmse:1.006527	test-rmse:3.264308 
#> [50]	train-rmse:0.999569	test-rmse:3.263293 
#> [51]	train-rmse:0.990109	test-rmse:3.265685 
#> [52]	train-rmse:0.976736	test-rmse:3.267067 
#> [53]	train-rmse:0.950653	test-rmse:3.272872 
#> [54]	train-rmse:0.942109	test-rmse:3.264780 
#> [55]	train-rmse:0.937606	test-rmse:3.274221 
#> [56]	train-rmse:0.922337	test-rmse:3.272372 
#> [57]	train-rmse:0.902798	test-rmse:3.268210 
#> [58]	train-rmse:0.889290	test-rmse:3.267985 
#> [59]	train-rmse:0.883539	test-rmse:3.270015 
#> [60]	train-rmse:0.862743	test-rmse:3.261608 
#> [61]	train-rmse:0.846788	test-rmse:3.266571 
#> [62]	train-rmse:0.835584	test-rmse:3.261188 
#> [63]	train-rmse:0.816480	test-rmse:3.257493 
#> [64]	train-rmse:0.805986	test-rmse:3.247675 
#> [65]	train-rmse:0.791130	test-rmse:3.246013 
#> [66]	train-rmse:0.779006	test-rmse:3.244313 
#> [67]	train-rmse:0.762253	test-rmse:3.239574 
#> [68]	train-rmse:0.747020	test-rmse:3.240649 
#> [69]	train-rmse:0.741704	test-rmse:3.240951 
#> [70]	train-rmse:0.728181	test-rmse:3.239402 
#> [1]	train-rmse:17.500320	test-rmse:16.577935 
#> [2]	train-rmse:12.700782	test-rmse:12.026178 
#> [3]	train-rmse:9.374607	test-rmse:8.883171 
#> [4]	train-rmse:7.019407	test-rmse:6.866400 
#> [5]	train-rmse:5.400458	test-rmse:5.585809 
#> [6]	train-rmse:4.289105	test-rmse:4.803949 
#> [7]	train-rmse:3.574879	test-rmse:4.343962 
#> [8]	train-rmse:3.081702	test-rmse:4.088238 
#> [9]	train-rmse:2.760779	test-rmse:3.924238 
#> [10]	train-rmse:2.528432	test-rmse:3.842127 
#> [11]	train-rmse:2.371529	test-rmse:3.794781 
#> [12]	train-rmse:2.258979	test-rmse:3.748469 
#> [13]	train-rmse:2.167540	test-rmse:3.738536 
#> [14]	train-rmse:2.045130	test-rmse:3.704941 
#> [15]	train-rmse:1.995062	test-rmse:3.693663 
#> [16]	train-rmse:1.931012	test-rmse:3.685969 
#> [17]	train-rmse:1.894994	test-rmse:3.695818 
#> [18]	train-rmse:1.844343	test-rmse:3.687795 
#> [19]	train-rmse:1.798131	test-rmse:3.695430 
#> [20]	train-rmse:1.739249	test-rmse:3.654595 
#> [21]	train-rmse:1.700298	test-rmse:3.601839 
#> [22]	train-rmse:1.649804	test-rmse:3.576904 
#> [23]	train-rmse:1.630151	test-rmse:3.563748 
#> [24]	train-rmse:1.581735	test-rmse:3.553712 
#> [25]	train-rmse:1.567455	test-rmse:3.537648 
#> [26]	train-rmse:1.539873	test-rmse:3.529298 
#> [27]	train-rmse:1.525838	test-rmse:3.522211 
#> [28]	train-rmse:1.490469	test-rmse:3.526956 
#> [29]	train-rmse:1.443220	test-rmse:3.521233 
#> [30]	train-rmse:1.408247	test-rmse:3.518112 
#> [31]	train-rmse:1.391281	test-rmse:3.508094 
#> [32]	train-rmse:1.361160	test-rmse:3.513120 
#> [33]	train-rmse:1.336057	test-rmse:3.514023 
#> [34]	train-rmse:1.315465	test-rmse:3.508017 
#> [35]	train-rmse:1.290935	test-rmse:3.511331 
#> [36]	train-rmse:1.281778	test-rmse:3.508712 
#> [37]	train-rmse:1.263491	test-rmse:3.503052 
#> [38]	train-rmse:1.251507	test-rmse:3.493851 
#> [39]	train-rmse:1.241208	test-rmse:3.493632 
#> [40]	train-rmse:1.221975	test-rmse:3.485807 
#> [41]	train-rmse:1.196739	test-rmse:3.490003 
#> [42]	train-rmse:1.172633	test-rmse:3.490853 
#> [43]	train-rmse:1.152701	test-rmse:3.485805 
#> [44]	train-rmse:1.138961	test-rmse:3.487476 
#> [45]	train-rmse:1.109313	test-rmse:3.475582 
#> [46]	train-rmse:1.100207	test-rmse:3.469868 
#> [47]	train-rmse:1.090441	test-rmse:3.470839 
#> [48]	train-rmse:1.084244	test-rmse:3.469064 
#> [49]	train-rmse:1.077133	test-rmse:3.469469 
#> [50]	train-rmse:1.055232	test-rmse:3.461626 
#> [51]	train-rmse:1.037104	test-rmse:3.462026 
#> [52]	train-rmse:1.019209	test-rmse:3.470342 
#> [53]	train-rmse:1.012538	test-rmse:3.471178 
#> [54]	train-rmse:0.994863	test-rmse:3.461374 
#> [55]	train-rmse:0.965409	test-rmse:3.453446 
#> [56]	train-rmse:0.953566	test-rmse:3.450462 
#> [57]	train-rmse:0.940082	test-rmse:3.452440 
#> [58]	train-rmse:0.919081	test-rmse:3.452695 
#> [59]	train-rmse:0.903928	test-rmse:3.460760 
#> [60]	train-rmse:0.883478	test-rmse:3.462497 
#> [61]	train-rmse:0.877497	test-rmse:3.464765 
#> [62]	train-rmse:0.872967	test-rmse:3.463513 
#> [63]	train-rmse:0.858929	test-rmse:3.466758 
#> [64]	train-rmse:0.849939	test-rmse:3.462989 
#> [65]	train-rmse:0.832749	test-rmse:3.465008 
#> [66]	train-rmse:0.820749	test-rmse:3.467366 
#> [67]	train-rmse:0.799739	test-rmse:3.470420 
#> [68]	train-rmse:0.781536	test-rmse:3.467527 
#> [69]	train-rmse:0.776637	test-rmse:3.467608 
#> [70]	train-rmse:0.756403	test-rmse:3.466820 
#> [1]	train-rmse:17.402750	test-rmse:16.672237 
#> [2]	train-rmse:12.613865	test-rmse:12.068149 
#> [3]	train-rmse:9.287956	test-rmse:8.946529 
#> [4]	train-rmse:6.933049	test-rmse:6.872692 
#> [5]	train-rmse:5.333376	test-rmse:5.543267 
#> [6]	train-rmse:4.232659	test-rmse:4.710821 
#> [7]	train-rmse:3.518554	test-rmse:4.225892 
#> [8]	train-rmse:3.065487	test-rmse:3.984023 
#> [9]	train-rmse:2.744047	test-rmse:3.822014 
#> [10]	train-rmse:2.538783	test-rmse:3.735906 
#> [11]	train-rmse:2.391563	test-rmse:3.711962 
#> [12]	train-rmse:2.249077	test-rmse:3.637363 
#> [13]	train-rmse:2.145327	test-rmse:3.639505 
#> [14]	train-rmse:2.071817	test-rmse:3.617094 
#> [15]	train-rmse:2.021594	test-rmse:3.594986 
#> [16]	train-rmse:1.948477	test-rmse:3.571381 
#> [17]	train-rmse:1.896602	test-rmse:3.555057 
#> [18]	train-rmse:1.823711	test-rmse:3.546802 
#> [19]	train-rmse:1.803024	test-rmse:3.535221 
#> [20]	train-rmse:1.757521	test-rmse:3.523680 
#> [21]	train-rmse:1.719815	test-rmse:3.505092 
#> [22]	train-rmse:1.673410	test-rmse:3.516841 
#> [23]	train-rmse:1.644435	test-rmse:3.522851 
#> [24]	train-rmse:1.620934	test-rmse:3.504999 
#> [25]	train-rmse:1.567626	test-rmse:3.513036 
#> [26]	train-rmse:1.537336	test-rmse:3.501669 
#> [27]	train-rmse:1.521802	test-rmse:3.507671 
#> [28]	train-rmse:1.477356	test-rmse:3.515000 
#> [29]	train-rmse:1.463061	test-rmse:3.511038 
#> [30]	train-rmse:1.443271	test-rmse:3.516023 
#> [31]	train-rmse:1.429034	test-rmse:3.500366 
#> [32]	train-rmse:1.409486	test-rmse:3.513899 
#> [33]	train-rmse:1.367020	test-rmse:3.530032 
#> [34]	train-rmse:1.351872	test-rmse:3.529120 
#> [35]	train-rmse:1.340334	test-rmse:3.534267 
#> [36]	train-rmse:1.306939	test-rmse:3.527196 
#> [37]	train-rmse:1.285643	test-rmse:3.530349 
#> [38]	train-rmse:1.265536	test-rmse:3.536605 
#> [39]	train-rmse:1.254320	test-rmse:3.527217 
#> [40]	train-rmse:1.240069	test-rmse:3.540603 
#> [41]	train-rmse:1.219325	test-rmse:3.545401 
#> [42]	train-rmse:1.204344	test-rmse:3.538206 
#> [43]	train-rmse:1.173548	test-rmse:3.537709 
#> [44]	train-rmse:1.166427	test-rmse:3.544395 
#> [45]	train-rmse:1.140547	test-rmse:3.563474 
#> [46]	train-rmse:1.128052	test-rmse:3.568552 
#> [47]	train-rmse:1.120925	test-rmse:3.571155 
#> [48]	train-rmse:1.092155	test-rmse:3.586326 
#> [49]	train-rmse:1.062622	test-rmse:3.584373 
#> [50]	train-rmse:1.044057	test-rmse:3.581142 
#> [51]	train-rmse:1.013747	test-rmse:3.573442 
#> [52]	train-rmse:1.004978	test-rmse:3.573550 
#> [53]	train-rmse:0.979008	test-rmse:3.571830 
#> [54]	train-rmse:0.973746	test-rmse:3.575230 
#> [55]	train-rmse:0.963112	test-rmse:3.577156 
#> [56]	train-rmse:0.948297	test-rmse:3.554859 
#> [57]	train-rmse:0.927097	test-rmse:3.557713 
#> [58]	train-rmse:0.903443	test-rmse:3.564630 
#> [59]	train-rmse:0.886617	test-rmse:3.571356 
#> [60]	train-rmse:0.870010	test-rmse:3.568536 
#> [61]	train-rmse:0.855544	test-rmse:3.565833 
#> [62]	train-rmse:0.848755	test-rmse:3.568257 
#> [63]	train-rmse:0.838522	test-rmse:3.567631 
#> [64]	train-rmse:0.817649	test-rmse:3.561768 
#> [65]	train-rmse:0.803753	test-rmse:3.558734 
#> [66]	train-rmse:0.799405	test-rmse:3.560965 
#> [67]	train-rmse:0.794757	test-rmse:3.556835 
#> [68]	train-rmse:0.779397	test-rmse:3.554966 
#> [69]	train-rmse:0.768486	test-rmse:3.557377 
#> [70]	train-rmse:0.762586	test-rmse:3.558965
#> [1] 3.538308
```

``` r
warnings() # no warnings for individual XGBoost function
```

### Exercise: Pick four of the functions and apply them with other numerical data sets.

### Post the results of your work.

### Test four of your functions on a neutral data set (mean = 0, sd = 1). Post your results, compare most accurate to least accurate results.


``` r

bland <- data.frame(
  X1 = rnorm(n = 1000, mean = 0, sd = 1),
  X2 = rnorm(n = 1000, mean = 0, sd = 1),
  X3 = rnorm(n = 1000, mean = 0, sd = 1),
  X4 = rnorm(n = 1000, mean = 0, sd = 1),
  y = rnorm(n = 1000, mean = 0, sd = 1)
)
# test with cubist function:
cubist_1(data = bland, colnum = 5, train_amount = 0.60, test_amount = 0.40, numresamples = 25)
#> [1] 1.004373
```

``` r

# apply with lqs function:
lqs1(data = bland, colnum = 5, train_amount = 0.60, test_amount = 0.40, validation_amount = 0.00, numresamples = 25)
#> [1] 1.052853
```
