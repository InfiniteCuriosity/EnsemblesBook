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
#> [1] 0.3032176
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
#> [1] 3.951048
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
#> [1] 4.826467
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
#> Scaling factor= 0.7015979 
#> gamma= 30.9135 	 alpha= 4.8006 	 beta= 20224.62 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015323 
#> gamma= 31.1033 	 alpha= 5.5126 	 beta= 15023.08 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7014854 
#> gamma= 31.076 	 alpha= 3.9929 	 beta= 23622.04 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015874 
#> gamma= 31.6271 	 alpha= 5.7396 	 beta= 13894.38 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016138 
#> gamma= 31.398 	 alpha= 4.8438 	 beta= 17147.03 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015669 
#> gamma= 30.6412 	 alpha= 2.7592 	 beta= 18760.23 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016868 
#> gamma= 31.0864 	 alpha= 4.0625 	 beta= 27297.18 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016809 
#> gamma= 31.1076 	 alpha= 4.011 	 beta= 16655.02 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015619 
#> gamma= 30.6972 	 alpha= 3.5254 	 beta= 60325.72 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015227 
#> gamma= 30.1146 	 alpha= 2.9039 	 beta= 17425.96 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7014854 
#> gamma= 31.4493 	 alpha= 5.3886 	 beta= 15882.44 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015179 
#> gamma= 30.9071 	 alpha= 2.7797 	 beta= 15990.24 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7017045 
#> gamma= 31.2656 	 alpha= 4.6721 	 beta= 14287.39 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015469 
#> gamma= 31.4396 	 alpha= 3.8918 	 beta= 32878.72 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016138 
#> gamma= 31.6352 	 alpha= 4.1275 	 beta= 13990.78 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.701542 
#> gamma= 31.5509 	 alpha= 5.3686 	 beta= 17132.37 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015823 
#> gamma= 30.621 	 alpha= 4.5094 	 beta= 38520.84 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015519 
#> gamma= 31.125 	 alpha= 4.393 	 beta= 14133.96 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016636 
#> gamma= 30.6163 	 alpha= 5.3253 	 beta= 13300.03 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015979 
#> gamma= 30.5402 	 alpha= 5.0743 	 beta= 20441.23 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015569 
#> gamma= 30.5387 	 alpha= 5.1334 	 beta= 16952.94 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016192 
#> gamma= 31.4644 	 alpha= 5.4873 	 beta= 15997.12 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7016085 
#> gamma= 31.4464 	 alpha= 5.1745 	 beta= 15916.15 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015619 
#> gamma= 31.1062 	 alpha= 5.1507 	 beta= 16418.65 
#> Number of parameters (weights and biases) to estimate: 32 
#> Nguyen-Widrow method
#> Scaling factor= 0.7015085 
#> gamma= 30.9124 	 alpha= 3.9584 	 beta= 48498.02
#> [1] 0.1362577
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
#> [1] 0.3119975
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
#> [1] 4.40312
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
#> [1] 4.97869
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
#> [1] 4.755033
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
#> [1] 3.370866
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
#> [1] 6.898357
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
#> [1] 5.001568
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
#> [1] 4.607703
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
#> [1] 6.838138
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
#> [1] 3.943471
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
#> [1] 6.068721
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
#> [1] 6.527915
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
#> [1] 1.818037
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
#> [1] 5.09164
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
#> [1] 4.950721
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
#> [1] 4.899757
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
#> [1] 2.3304
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
#> [1] 4.803725
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
#> [1]	train-rmse:17.106138	test-rmse:17.366893 
#> [2]	train-rmse:12.381786	test-rmse:12.791063 
#> [3]	train-rmse:9.054376	test-rmse:9.622087 
#> [4]	train-rmse:6.732554	test-rmse:7.565010 
#> [5]	train-rmse:5.149285	test-rmse:6.266702 
#> [6]	train-rmse:4.076553	test-rmse:5.494029 
#> [7]	train-rmse:3.324973	test-rmse:4.903619 
#> [8]	train-rmse:2.847707	test-rmse:4.542260 
#> [9]	train-rmse:2.530861	test-rmse:4.413045 
#> [10]	train-rmse:2.307846	test-rmse:4.231765 
#> [11]	train-rmse:2.152082	test-rmse:4.184429 
#> [12]	train-rmse:2.055640	test-rmse:4.161266 
#> [13]	train-rmse:1.946959	test-rmse:4.136018 
#> [14]	train-rmse:1.869913	test-rmse:4.143107 
#> [15]	train-rmse:1.805064	test-rmse:4.058231 
#> [16]	train-rmse:1.758146	test-rmse:4.040621 
#> [17]	train-rmse:1.713256	test-rmse:3.986970 
#> [18]	train-rmse:1.681229	test-rmse:3.985656 
#> [19]	train-rmse:1.620191	test-rmse:3.996229 
#> [20]	train-rmse:1.564413	test-rmse:3.989849 
#> [21]	train-rmse:1.533410	test-rmse:3.992445 
#> [22]	train-rmse:1.512751	test-rmse:3.997344 
#> [23]	train-rmse:1.473980	test-rmse:3.991080 
#> [24]	train-rmse:1.450180	test-rmse:3.959146 
#> [25]	train-rmse:1.419758	test-rmse:3.971367 
#> [26]	train-rmse:1.381709	test-rmse:3.972488 
#> [27]	train-rmse:1.341355	test-rmse:3.967456 
#> [28]	train-rmse:1.327296	test-rmse:3.967204 
#> [29]	train-rmse:1.315809	test-rmse:3.963369 
#> [30]	train-rmse:1.276372	test-rmse:3.939496 
#> [31]	train-rmse:1.245258	test-rmse:3.941929 
#> [32]	train-rmse:1.227539	test-rmse:3.943494 
#> [33]	train-rmse:1.205570	test-rmse:3.944429 
#> [34]	train-rmse:1.178578	test-rmse:3.908206 
#> [35]	train-rmse:1.147782	test-rmse:3.890774 
#> [36]	train-rmse:1.130770	test-rmse:3.895550 
#> [37]	train-rmse:1.110949	test-rmse:3.892502 
#> [38]	train-rmse:1.086514	test-rmse:3.894933 
#> [39]	train-rmse:1.059807	test-rmse:3.895144 
#> [40]	train-rmse:1.037346	test-rmse:3.899691 
#> [41]	train-rmse:1.027192	test-rmse:3.906265 
#> [42]	train-rmse:1.008947	test-rmse:3.906913 
#> [43]	train-rmse:0.982172	test-rmse:3.895650 
#> [44]	train-rmse:0.955719	test-rmse:3.886293 
#> [45]	train-rmse:0.934445	test-rmse:3.885709 
#> [46]	train-rmse:0.922770	test-rmse:3.883157 
#> [47]	train-rmse:0.904506	test-rmse:3.880058 
#> [48]	train-rmse:0.888704	test-rmse:3.877216 
#> [49]	train-rmse:0.873289	test-rmse:3.881064 
#> [50]	train-rmse:0.863270	test-rmse:3.870378 
#> [51]	train-rmse:0.856263	test-rmse:3.870180 
#> [52]	train-rmse:0.849087	test-rmse:3.871422 
#> [53]	train-rmse:0.831005	test-rmse:3.864546 
#> [54]	train-rmse:0.812538	test-rmse:3.863428 
#> [55]	train-rmse:0.803447	test-rmse:3.860963 
#> [56]	train-rmse:0.792266	test-rmse:3.852648 
#> [57]	train-rmse:0.783392	test-rmse:3.848370 
#> [58]	train-rmse:0.778657	test-rmse:3.850442 
#> [59]	train-rmse:0.773384	test-rmse:3.853208 
#> [60]	train-rmse:0.766044	test-rmse:3.852157 
#> [61]	train-rmse:0.761048	test-rmse:3.849829 
#> [62]	train-rmse:0.752227	test-rmse:3.840653 
#> [63]	train-rmse:0.736047	test-rmse:3.852588 
#> [64]	train-rmse:0.724781	test-rmse:3.846104 
#> [65]	train-rmse:0.713466	test-rmse:3.837874 
#> [66]	train-rmse:0.703244	test-rmse:3.836673 
#> [67]	train-rmse:0.689611	test-rmse:3.836109 
#> [68]	train-rmse:0.684932	test-rmse:3.826088 
#> [69]	train-rmse:0.664265	test-rmse:3.818385 
#> [70]	train-rmse:0.652603	test-rmse:3.810074 
#> [1]	train-rmse:16.924814	test-rmse:17.491308 
#> [2]	train-rmse:12.224077	test-rmse:12.977083 
#> [3]	train-rmse:8.935319	test-rmse:9.811723 
#> [4]	train-rmse:6.643097	test-rmse:7.739571 
#> [5]	train-rmse:5.062052	test-rmse:6.470779 
#> [6]	train-rmse:3.994706	test-rmse:5.659490 
#> [7]	train-rmse:3.290321	test-rmse:5.208584 
#> [8]	train-rmse:2.811032	test-rmse:4.926356 
#> [9]	train-rmse:2.511393	test-rmse:4.616665 
#> [10]	train-rmse:2.302334	test-rmse:4.464360 
#> [11]	train-rmse:2.133457	test-rmse:4.325686 
#> [12]	train-rmse:2.020290	test-rmse:4.268084 
#> [13]	train-rmse:1.952641	test-rmse:4.176074 
#> [14]	train-rmse:1.909818	test-rmse:4.157850 
#> [15]	train-rmse:1.828420	test-rmse:4.114806 
#> [16]	train-rmse:1.762910	test-rmse:4.090413 
#> [17]	train-rmse:1.716270	test-rmse:4.040192 
#> [18]	train-rmse:1.684788	test-rmse:3.999977 
#> [19]	train-rmse:1.629414	test-rmse:3.970050 
#> [20]	train-rmse:1.570881	test-rmse:3.975561 
#> [21]	train-rmse:1.539230	test-rmse:3.971791 
#> [22]	train-rmse:1.483834	test-rmse:3.966041 
#> [23]	train-rmse:1.455368	test-rmse:3.981049 
#> [24]	train-rmse:1.440889	test-rmse:3.923917 
#> [25]	train-rmse:1.417689	test-rmse:3.904749 
#> [26]	train-rmse:1.377943	test-rmse:3.927352 
#> [27]	train-rmse:1.351410	test-rmse:3.924684 
#> [28]	train-rmse:1.307542	test-rmse:3.927266 
#> [29]	train-rmse:1.289389	test-rmse:3.920995 
#> [30]	train-rmse:1.281154	test-rmse:3.913963 
#> [31]	train-rmse:1.262924	test-rmse:3.913638 
#> [32]	train-rmse:1.240791	test-rmse:3.907200 
#> [33]	train-rmse:1.225516	test-rmse:3.905394 
#> [34]	train-rmse:1.212163	test-rmse:3.912072 
#> [35]	train-rmse:1.195203	test-rmse:3.916352 
#> [36]	train-rmse:1.187114	test-rmse:3.919584 
#> [37]	train-rmse:1.169093	test-rmse:3.912839 
#> [38]	train-rmse:1.139805	test-rmse:3.898301 
#> [39]	train-rmse:1.128596	test-rmse:3.896211 
#> [40]	train-rmse:1.117456	test-rmse:3.875217 
#> [41]	train-rmse:1.082428	test-rmse:3.880888 
#> [42]	train-rmse:1.073824	test-rmse:3.878762 
#> [43]	train-rmse:1.057046	test-rmse:3.881334 
#> [44]	train-rmse:1.044397	test-rmse:3.879126 
#> [45]	train-rmse:1.039668	test-rmse:3.878866 
#> [46]	train-rmse:1.023359	test-rmse:3.881689 
#> [47]	train-rmse:0.999350	test-rmse:3.887873 
#> [48]	train-rmse:0.978395	test-rmse:3.896517 
#> [49]	train-rmse:0.968111	test-rmse:3.876398 
#> [50]	train-rmse:0.959579	test-rmse:3.874545 
#> [51]	train-rmse:0.940500	test-rmse:3.873539 
#> [52]	train-rmse:0.917210	test-rmse:3.863689 
#> [53]	train-rmse:0.912667	test-rmse:3.861802 
#> [54]	train-rmse:0.906458	test-rmse:3.864835 
#> [55]	train-rmse:0.899376	test-rmse:3.845328 
#> [56]	train-rmse:0.894771	test-rmse:3.848354 
#> [57]	train-rmse:0.888213	test-rmse:3.848686 
#> [58]	train-rmse:0.884812	test-rmse:3.846327 
#> [59]	train-rmse:0.868224	test-rmse:3.841791 
#> [60]	train-rmse:0.850141	test-rmse:3.839369 
#> [61]	train-rmse:0.838281	test-rmse:3.842239 
#> [62]	train-rmse:0.809893	test-rmse:3.845711 
#> [63]	train-rmse:0.801592	test-rmse:3.847079 
#> [64]	train-rmse:0.797066	test-rmse:3.846509 
#> [65]	train-rmse:0.783371	test-rmse:3.836707 
#> [66]	train-rmse:0.760317	test-rmse:3.839897 
#> [67]	train-rmse:0.755642	test-rmse:3.833284 
#> [68]	train-rmse:0.750886	test-rmse:3.834663 
#> [69]	train-rmse:0.737716	test-rmse:3.832866 
#> [70]	train-rmse:0.730757	test-rmse:3.833470 
#> [1]	train-rmse:16.769430	test-rmse:17.894876 
#> [2]	train-rmse:12.179074	test-rmse:13.314768 
#> [3]	train-rmse:8.959736	test-rmse:9.939200 
#> [4]	train-rmse:6.744677	test-rmse:7.862458 
#> [5]	train-rmse:5.215327	test-rmse:6.289669 
#> [6]	train-rmse:4.166223	test-rmse:5.344343 
#> [7]	train-rmse:3.448274	test-rmse:4.808412 
#> [8]	train-rmse:3.000724	test-rmse:4.346214 
#> [9]	train-rmse:2.676890	test-rmse:4.018198 
#> [10]	train-rmse:2.454332	test-rmse:3.864414 
#> [11]	train-rmse:2.320214	test-rmse:3.706553 
#> [12]	train-rmse:2.168896	test-rmse:3.625093 
#> [13]	train-rmse:2.087479	test-rmse:3.566353 
#> [14]	train-rmse:1.988021	test-rmse:3.482312 
#> [15]	train-rmse:1.900084	test-rmse:3.473857 
#> [16]	train-rmse:1.830392	test-rmse:3.442453 
#> [17]	train-rmse:1.798654	test-rmse:3.423610 
#> [18]	train-rmse:1.735404	test-rmse:3.392003 
#> [19]	train-rmse:1.689292	test-rmse:3.344842 
#> [20]	train-rmse:1.658038	test-rmse:3.329957 
#> [21]	train-rmse:1.615931	test-rmse:3.323461 
#> [22]	train-rmse:1.597195	test-rmse:3.320635 
#> [23]	train-rmse:1.547681	test-rmse:3.315211 
#> [24]	train-rmse:1.522034	test-rmse:3.289021 
#> [25]	train-rmse:1.502220	test-rmse:3.273123 
#> [26]	train-rmse:1.452952	test-rmse:3.264563 
#> [27]	train-rmse:1.440006	test-rmse:3.266037 
#> [28]	train-rmse:1.407211	test-rmse:3.267691 
#> [29]	train-rmse:1.380584	test-rmse:3.252523 
#> [30]	train-rmse:1.347878	test-rmse:3.254423 
#> [31]	train-rmse:1.323356	test-rmse:3.247834 
#> [32]	train-rmse:1.307834	test-rmse:3.243761 
#> [33]	train-rmse:1.284736	test-rmse:3.229237 
#> [34]	train-rmse:1.261906	test-rmse:3.218660 
#> [35]	train-rmse:1.230486	test-rmse:3.216589 
#> [36]	train-rmse:1.201938	test-rmse:3.205829 
#> [37]	train-rmse:1.186259	test-rmse:3.204344 
#> [38]	train-rmse:1.179335	test-rmse:3.192806 
#> [39]	train-rmse:1.162139	test-rmse:3.192417 
#> [40]	train-rmse:1.147543	test-rmse:3.182245 
#> [41]	train-rmse:1.130188	test-rmse:3.175907 
#> [42]	train-rmse:1.111416	test-rmse:3.182595 
#> [43]	train-rmse:1.098352	test-rmse:3.190045 
#> [44]	train-rmse:1.093224	test-rmse:3.190668 
#> [45]	train-rmse:1.057084	test-rmse:3.183423 
#> [46]	train-rmse:1.049201	test-rmse:3.184489 
#> [47]	train-rmse:1.030767	test-rmse:3.184654 
#> [48]	train-rmse:1.005605	test-rmse:3.189390 
#> [49]	train-rmse:0.983366	test-rmse:3.196911 
#> [50]	train-rmse:0.964676	test-rmse:3.202144 
#> [51]	train-rmse:0.948305	test-rmse:3.189985 
#> [52]	train-rmse:0.920270	test-rmse:3.199393 
#> [53]	train-rmse:0.915890	test-rmse:3.199914 
#> [54]	train-rmse:0.892648	test-rmse:3.199218 
#> [55]	train-rmse:0.882845	test-rmse:3.196315 
#> [56]	train-rmse:0.872703	test-rmse:3.185576 
#> [57]	train-rmse:0.861601	test-rmse:3.193927 
#> [58]	train-rmse:0.838209	test-rmse:3.184323 
#> [59]	train-rmse:0.828727	test-rmse:3.182709 
#> [60]	train-rmse:0.810560	test-rmse:3.176715 
#> [61]	train-rmse:0.805448	test-rmse:3.176383 
#> [62]	train-rmse:0.798437	test-rmse:3.180692 
#> [63]	train-rmse:0.782868	test-rmse:3.188952 
#> [64]	train-rmse:0.775681	test-rmse:3.180862 
#> [65]	train-rmse:0.759254	test-rmse:3.183831 
#> [66]	train-rmse:0.745204	test-rmse:3.182465 
#> [67]	train-rmse:0.741159	test-rmse:3.183134 
#> [68]	train-rmse:0.736253	test-rmse:3.184094 
#> [69]	train-rmse:0.729998	test-rmse:3.191249 
#> [70]	train-rmse:0.716184	test-rmse:3.184404 
#> [1]	train-rmse:17.087106	test-rmse:17.547975 
#> [2]	train-rmse:12.433314	test-rmse:12.898015 
#> [3]	train-rmse:9.181515	test-rmse:9.806146 
#> [4]	train-rmse:6.907137	test-rmse:7.608362 
#> [5]	train-rmse:5.315277	test-rmse:6.191429 
#> [6]	train-rmse:4.277091	test-rmse:5.247489 
#> [7]	train-rmse:3.555920	test-rmse:4.620634 
#> [8]	train-rmse:3.091994	test-rmse:4.277427 
#> [9]	train-rmse:2.773289	test-rmse:4.039136 
#> [10]	train-rmse:2.542963	test-rmse:3.837833 
#> [11]	train-rmse:2.358420	test-rmse:3.719522 
#> [12]	train-rmse:2.259905	test-rmse:3.657435 
#> [13]	train-rmse:2.143198	test-rmse:3.614771 
#> [14]	train-rmse:2.016831	test-rmse:3.509346 
#> [15]	train-rmse:1.926708	test-rmse:3.452235 
#> [16]	train-rmse:1.831159	test-rmse:3.424980 
#> [17]	train-rmse:1.764247	test-rmse:3.401318 
#> [18]	train-rmse:1.717560	test-rmse:3.386264 
#> [19]	train-rmse:1.686371	test-rmse:3.359821 
#> [20]	train-rmse:1.635582	test-rmse:3.344121 
#> [21]	train-rmse:1.597862	test-rmse:3.321821 
#> [22]	train-rmse:1.551536	test-rmse:3.314438 
#> [23]	train-rmse:1.493920	test-rmse:3.301745 
#> [24]	train-rmse:1.468887	test-rmse:3.307657 
#> [25]	train-rmse:1.440494	test-rmse:3.318599 
#> [26]	train-rmse:1.399639	test-rmse:3.308133 
#> [27]	train-rmse:1.371753	test-rmse:3.321056 
#> [28]	train-rmse:1.356785	test-rmse:3.311057 
#> [29]	train-rmse:1.334746	test-rmse:3.315460 
#> [30]	train-rmse:1.314431	test-rmse:3.313619 
#> [31]	train-rmse:1.286656	test-rmse:3.301253 
#> [32]	train-rmse:1.275029	test-rmse:3.300700 
#> [33]	train-rmse:1.261278	test-rmse:3.313080 
#> [34]	train-rmse:1.228144	test-rmse:3.307229 
#> [35]	train-rmse:1.209504	test-rmse:3.301115 
#> [36]	train-rmse:1.186792	test-rmse:3.290215 
#> [37]	train-rmse:1.168979	test-rmse:3.284940 
#> [38]	train-rmse:1.140622	test-rmse:3.289209 
#> [39]	train-rmse:1.125910	test-rmse:3.283581 
#> [40]	train-rmse:1.114858	test-rmse:3.286099 
#> [41]	train-rmse:1.090374	test-rmse:3.302602 
#> [42]	train-rmse:1.076803	test-rmse:3.301566 
#> [43]	train-rmse:1.068855	test-rmse:3.296318 
#> [44]	train-rmse:1.042066	test-rmse:3.292759 
#> [45]	train-rmse:1.028457	test-rmse:3.294081 
#> [46]	train-rmse:0.995959	test-rmse:3.270298 
#> [47]	train-rmse:0.980871	test-rmse:3.268985 
#> [48]	train-rmse:0.958533	test-rmse:3.274377 
#> [49]	train-rmse:0.938282	test-rmse:3.272335 
#> [50]	train-rmse:0.928156	test-rmse:3.275671 
#> [51]	train-rmse:0.921367	test-rmse:3.278549 
#> [52]	train-rmse:0.911435	test-rmse:3.271091 
#> [53]	train-rmse:0.903200	test-rmse:3.273346 
#> [54]	train-rmse:0.878155	test-rmse:3.275489 
#> [55]	train-rmse:0.856695	test-rmse:3.268017 
#> [56]	train-rmse:0.843457	test-rmse:3.263470 
#> [57]	train-rmse:0.825039	test-rmse:3.262049 
#> [58]	train-rmse:0.801914	test-rmse:3.260554 
#> [59]	train-rmse:0.790788	test-rmse:3.264901 
#> [60]	train-rmse:0.780473	test-rmse:3.265901 
#> [61]	train-rmse:0.764497	test-rmse:3.257577 
#> [62]	train-rmse:0.755715	test-rmse:3.257109 
#> [63]	train-rmse:0.743633	test-rmse:3.251935 
#> [64]	train-rmse:0.735593	test-rmse:3.251824 
#> [65]	train-rmse:0.716100	test-rmse:3.252355 
#> [66]	train-rmse:0.708078	test-rmse:3.256886 
#> [67]	train-rmse:0.701784	test-rmse:3.254948 
#> [68]	train-rmse:0.694658	test-rmse:3.253093 
#> [69]	train-rmse:0.692237	test-rmse:3.252560 
#> [70]	train-rmse:0.682665	test-rmse:3.252795 
#> [1]	train-rmse:17.039314	test-rmse:17.372493 
#> [2]	train-rmse:12.364930	test-rmse:12.737714 
#> [3]	train-rmse:9.152790	test-rmse:9.706442 
#> [4]	train-rmse:6.880103	test-rmse:7.469698 
#> [5]	train-rmse:5.385033	test-rmse:6.221769 
#> [6]	train-rmse:4.334779	test-rmse:5.418746 
#> [7]	train-rmse:3.623685	test-rmse:4.894852 
#> [8]	train-rmse:3.119375	test-rmse:4.530647 
#> [9]	train-rmse:2.808541	test-rmse:4.299940 
#> [10]	train-rmse:2.608400	test-rmse:4.144907 
#> [11]	train-rmse:2.438045	test-rmse:4.021411 
#> [12]	train-rmse:2.284360	test-rmse:3.923246 
#> [13]	train-rmse:2.160649	test-rmse:3.878706 
#> [14]	train-rmse:2.074205	test-rmse:3.879430 
#> [15]	train-rmse:2.001493	test-rmse:3.882180 
#> [16]	train-rmse:1.931897	test-rmse:3.853905 
#> [17]	train-rmse:1.886886	test-rmse:3.839471 
#> [18]	train-rmse:1.842014	test-rmse:3.817174 
#> [19]	train-rmse:1.770499	test-rmse:3.809698 
#> [20]	train-rmse:1.741151	test-rmse:3.788846 
#> [21]	train-rmse:1.706989	test-rmse:3.786862 
#> [22]	train-rmse:1.682979	test-rmse:3.788694 
#> [23]	train-rmse:1.667438	test-rmse:3.801298 
#> [24]	train-rmse:1.644861	test-rmse:3.800593 
#> [25]	train-rmse:1.625596	test-rmse:3.807905 
#> [26]	train-rmse:1.588505	test-rmse:3.830860 
#> [27]	train-rmse:1.556805	test-rmse:3.830283 
#> [28]	train-rmse:1.504327	test-rmse:3.831677 
#> [29]	train-rmse:1.484864	test-rmse:3.827646 
#> [30]	train-rmse:1.436177	test-rmse:3.831639 
#> [31]	train-rmse:1.406546	test-rmse:3.843213 
#> [32]	train-rmse:1.386777	test-rmse:3.850847 
#> [33]	train-rmse:1.362030	test-rmse:3.850510 
#> [34]	train-rmse:1.337454	test-rmse:3.846228 
#> [35]	train-rmse:1.321338	test-rmse:3.831758 
#> [36]	train-rmse:1.296574	test-rmse:3.823850 
#> [37]	train-rmse:1.275097	test-rmse:3.802191 
#> [38]	train-rmse:1.258453	test-rmse:3.806852 
#> [39]	train-rmse:1.247943	test-rmse:3.816175 
#> [40]	train-rmse:1.208008	test-rmse:3.815497 
#> [41]	train-rmse:1.180333	test-rmse:3.810803 
#> [42]	train-rmse:1.166932	test-rmse:3.802153 
#> [43]	train-rmse:1.141532	test-rmse:3.800314 
#> [44]	train-rmse:1.125231	test-rmse:3.801838 
#> [45]	train-rmse:1.114843	test-rmse:3.801716 
#> [46]	train-rmse:1.103012	test-rmse:3.790983 
#> [47]	train-rmse:1.090691	test-rmse:3.789959 
#> [48]	train-rmse:1.059705	test-rmse:3.758672 
#> [49]	train-rmse:1.043280	test-rmse:3.760127 
#> [50]	train-rmse:1.021525	test-rmse:3.757465 
#> [51]	train-rmse:1.009641	test-rmse:3.761157 
#> [52]	train-rmse:0.995215	test-rmse:3.761631 
#> [53]	train-rmse:0.987668	test-rmse:3.767853 
#> [54]	train-rmse:0.976287	test-rmse:3.764637 
#> [55]	train-rmse:0.961583	test-rmse:3.763548 
#> [56]	train-rmse:0.948269	test-rmse:3.765997 
#> [57]	train-rmse:0.939112	test-rmse:3.760699 
#> [58]	train-rmse:0.914079	test-rmse:3.762832 
#> [59]	train-rmse:0.894459	test-rmse:3.750531 
#> [60]	train-rmse:0.871483	test-rmse:3.749053 
#> [61]	train-rmse:0.861584	test-rmse:3.756978 
#> [62]	train-rmse:0.844814	test-rmse:3.756580 
#> [63]	train-rmse:0.829690	test-rmse:3.756497 
#> [64]	train-rmse:0.810550	test-rmse:3.755744 
#> [65]	train-rmse:0.804361	test-rmse:3.762233 
#> [66]	train-rmse:0.800803	test-rmse:3.762128 
#> [67]	train-rmse:0.776651	test-rmse:3.751693 
#> [68]	train-rmse:0.772505	test-rmse:3.748052 
#> [69]	train-rmse:0.760311	test-rmse:3.744858 
#> [70]	train-rmse:0.750680	test-rmse:3.737034 
#> [1]	train-rmse:17.101483	test-rmse:17.221152 
#> [2]	train-rmse:12.444646	test-rmse:12.615811 
#> [3]	train-rmse:9.172284	test-rmse:9.598002 
#> [4]	train-rmse:6.884956	test-rmse:7.412542 
#> [5]	train-rmse:5.322263	test-rmse:5.931355 
#> [6]	train-rmse:4.280939	test-rmse:5.076201 
#> [7]	train-rmse:3.569942	test-rmse:4.549952 
#> [8]	train-rmse:3.081976	test-rmse:4.256225 
#> [9]	train-rmse:2.781709	test-rmse:4.040227 
#> [10]	train-rmse:2.530248	test-rmse:3.816286 
#> [11]	train-rmse:2.366431	test-rmse:3.756852 
#> [12]	train-rmse:2.261999	test-rmse:3.717622 
#> [13]	train-rmse:2.164082	test-rmse:3.678948 
#> [14]	train-rmse:2.076715	test-rmse:3.604042 
#> [15]	train-rmse:2.014051	test-rmse:3.578147 
#> [16]	train-rmse:1.969042	test-rmse:3.556411 
#> [17]	train-rmse:1.883153	test-rmse:3.516556 
#> [18]	train-rmse:1.834229	test-rmse:3.496479 
#> [19]	train-rmse:1.792077	test-rmse:3.487370 
#> [20]	train-rmse:1.724855	test-rmse:3.454408 
#> [21]	train-rmse:1.697145	test-rmse:3.424916 
#> [22]	train-rmse:1.660443	test-rmse:3.425995 
#> [23]	train-rmse:1.615386	test-rmse:3.415980 
#> [24]	train-rmse:1.596917	test-rmse:3.384274 
#> [25]	train-rmse:1.569208	test-rmse:3.387216 
#> [26]	train-rmse:1.522134	test-rmse:3.372645 
#> [27]	train-rmse:1.498584	test-rmse:3.373395 
#> [28]	train-rmse:1.465949	test-rmse:3.366745 
#> [29]	train-rmse:1.457054	test-rmse:3.369736 
#> [30]	train-rmse:1.429412	test-rmse:3.367073 
#> [31]	train-rmse:1.385224	test-rmse:3.361896 
#> [32]	train-rmse:1.365340	test-rmse:3.369315 
#> [33]	train-rmse:1.348392	test-rmse:3.338163 
#> [34]	train-rmse:1.314974	test-rmse:3.347388 
#> [35]	train-rmse:1.279139	test-rmse:3.354428 
#> [36]	train-rmse:1.271114	test-rmse:3.360353 
#> [37]	train-rmse:1.255547	test-rmse:3.370428 
#> [38]	train-rmse:1.244996	test-rmse:3.364618 
#> [39]	train-rmse:1.233660	test-rmse:3.355773 
#> [40]	train-rmse:1.204007	test-rmse:3.354770 
#> [41]	train-rmse:1.176796	test-rmse:3.350602 
#> [42]	train-rmse:1.158870	test-rmse:3.351266 
#> [43]	train-rmse:1.142927	test-rmse:3.354656 
#> [44]	train-rmse:1.127056	test-rmse:3.359469 
#> [45]	train-rmse:1.122091	test-rmse:3.365171 
#> [46]	train-rmse:1.105127	test-rmse:3.363288 
#> [47]	train-rmse:1.092647	test-rmse:3.366496 
#> [48]	train-rmse:1.076022	test-rmse:3.364519 
#> [49]	train-rmse:1.061236	test-rmse:3.363286 
#> [50]	train-rmse:1.049585	test-rmse:3.369344 
#> [51]	train-rmse:1.029361	test-rmse:3.373790 
#> [52]	train-rmse:1.014524	test-rmse:3.370027 
#> [53]	train-rmse:1.004017	test-rmse:3.368937 
#> [54]	train-rmse:0.994354	test-rmse:3.369895 
#> [55]	train-rmse:0.991787	test-rmse:3.370348 
#> [56]	train-rmse:0.968377	test-rmse:3.365827 
#> [57]	train-rmse:0.944894	test-rmse:3.368889 
#> [58]	train-rmse:0.940092	test-rmse:3.365939 
#> [59]	train-rmse:0.930072	test-rmse:3.365950 
#> [60]	train-rmse:0.913780	test-rmse:3.371885 
#> [61]	train-rmse:0.894146	test-rmse:3.370491 
#> [62]	train-rmse:0.882759	test-rmse:3.372006 
#> [63]	train-rmse:0.869799	test-rmse:3.380177 
#> [64]	train-rmse:0.859722	test-rmse:3.369491 
#> [65]	train-rmse:0.844869	test-rmse:3.376588 
#> [66]	train-rmse:0.827649	test-rmse:3.382764 
#> [67]	train-rmse:0.819069	test-rmse:3.384356 
#> [68]	train-rmse:0.800752	test-rmse:3.387998 
#> [69]	train-rmse:0.789417	test-rmse:3.393218 
#> [70]	train-rmse:0.779049	test-rmse:3.397949 
#> [1]	train-rmse:17.526194	test-rmse:16.532628 
#> [2]	train-rmse:12.705034	test-rmse:12.224050 
#> [3]	train-rmse:9.338710	test-rmse:9.098551 
#> [4]	train-rmse:7.012505	test-rmse:7.200110 
#> [5]	train-rmse:5.414610	test-rmse:5.915792 
#> [6]	train-rmse:4.332967	test-rmse:5.032077 
#> [7]	train-rmse:3.629258	test-rmse:4.617346 
#> [8]	train-rmse:3.135208	test-rmse:4.254942 
#> [9]	train-rmse:2.795479	test-rmse:4.046990 
#> [10]	train-rmse:2.560827	test-rmse:3.883091 
#> [11]	train-rmse:2.386630	test-rmse:3.803737 
#> [12]	train-rmse:2.279182	test-rmse:3.746729 
#> [13]	train-rmse:2.172721	test-rmse:3.744097 
#> [14]	train-rmse:2.107357	test-rmse:3.706228 
#> [15]	train-rmse:2.030903	test-rmse:3.674772 
#> [16]	train-rmse:1.968384	test-rmse:3.674313 
#> [17]	train-rmse:1.939995	test-rmse:3.661536 
#> [18]	train-rmse:1.898635	test-rmse:3.636892 
#> [19]	train-rmse:1.870243	test-rmse:3.640828 
#> [20]	train-rmse:1.837476	test-rmse:3.620710 
#> [21]	train-rmse:1.770018	test-rmse:3.621513 
#> [22]	train-rmse:1.705829	test-rmse:3.569995 
#> [23]	train-rmse:1.685498	test-rmse:3.568959 
#> [24]	train-rmse:1.642241	test-rmse:3.541168 
#> [25]	train-rmse:1.618518	test-rmse:3.535743 
#> [26]	train-rmse:1.601418	test-rmse:3.541819 
#> [27]	train-rmse:1.570618	test-rmse:3.513438 
#> [28]	train-rmse:1.535657	test-rmse:3.505832 
#> [29]	train-rmse:1.509630	test-rmse:3.493689 
#> [30]	train-rmse:1.492543	test-rmse:3.497684 
#> [31]	train-rmse:1.452068	test-rmse:3.493939 
#> [32]	train-rmse:1.427910	test-rmse:3.486587 
#> [33]	train-rmse:1.418973	test-rmse:3.491128 
#> [34]	train-rmse:1.398376	test-rmse:3.490988 
#> [35]	train-rmse:1.372867	test-rmse:3.486871 
#> [36]	train-rmse:1.335636	test-rmse:3.477362 
#> [37]	train-rmse:1.313383	test-rmse:3.455708 
#> [38]	train-rmse:1.281301	test-rmse:3.431872 
#> [39]	train-rmse:1.267660	test-rmse:3.424936 
#> [40]	train-rmse:1.245438	test-rmse:3.427774 
#> [41]	train-rmse:1.228806	test-rmse:3.413491 
#> [42]	train-rmse:1.200659	test-rmse:3.411115 
#> [43]	train-rmse:1.183586	test-rmse:3.410945 
#> [44]	train-rmse:1.161724	test-rmse:3.413806 
#> [45]	train-rmse:1.144605	test-rmse:3.407710 
#> [46]	train-rmse:1.129759	test-rmse:3.406934 
#> [47]	train-rmse:1.093113	test-rmse:3.395997 
#> [48]	train-rmse:1.068301	test-rmse:3.397111 
#> [49]	train-rmse:1.044730	test-rmse:3.402970 
#> [50]	train-rmse:1.030032	test-rmse:3.404501 
#> [51]	train-rmse:1.020507	test-rmse:3.401476 
#> [52]	train-rmse:1.002199	test-rmse:3.400382 
#> [53]	train-rmse:0.991375	test-rmse:3.400221 
#> [54]	train-rmse:0.980811	test-rmse:3.394219 
#> [55]	train-rmse:0.969603	test-rmse:3.387702 
#> [56]	train-rmse:0.960491	test-rmse:3.384092 
#> [57]	train-rmse:0.931411	test-rmse:3.376298 
#> [58]	train-rmse:0.906902	test-rmse:3.379070 
#> [59]	train-rmse:0.892301	test-rmse:3.378847 
#> [60]	train-rmse:0.870655	test-rmse:3.379813 
#> [61]	train-rmse:0.862239	test-rmse:3.373054 
#> [62]	train-rmse:0.852125	test-rmse:3.362553 
#> [63]	train-rmse:0.847975	test-rmse:3.363875 
#> [64]	train-rmse:0.833654	test-rmse:3.361672 
#> [65]	train-rmse:0.825251	test-rmse:3.358061 
#> [66]	train-rmse:0.810980	test-rmse:3.361093 
#> [67]	train-rmse:0.796412	test-rmse:3.357863 
#> [68]	train-rmse:0.780503	test-rmse:3.360487 
#> [69]	train-rmse:0.773635	test-rmse:3.354451 
#> [70]	train-rmse:0.754839	test-rmse:3.344973 
#> [1]	train-rmse:17.483723	test-rmse:16.602402 
#> [2]	train-rmse:12.677203	test-rmse:11.971884 
#> [3]	train-rmse:9.334857	test-rmse:8.970090 
#> [4]	train-rmse:6.974731	test-rmse:6.865298 
#> [5]	train-rmse:5.294664	test-rmse:5.452871 
#> [6]	train-rmse:4.147150	test-rmse:4.587184 
#> [7]	train-rmse:3.400253	test-rmse:4.028282 
#> [8]	train-rmse:2.909693	test-rmse:3.673200 
#> [9]	train-rmse:2.595444	test-rmse:3.490276 
#> [10]	train-rmse:2.391398	test-rmse:3.390995 
#> [11]	train-rmse:2.223429	test-rmse:3.317579 
#> [12]	train-rmse:2.114670	test-rmse:3.244576 
#> [13]	train-rmse:2.040906	test-rmse:3.243893 
#> [14]	train-rmse:1.961224	test-rmse:3.237762 
#> [15]	train-rmse:1.882405	test-rmse:3.230505 
#> [16]	train-rmse:1.804731	test-rmse:3.211189 
#> [17]	train-rmse:1.754230	test-rmse:3.187206 
#> [18]	train-rmse:1.705020	test-rmse:3.164158 
#> [19]	train-rmse:1.680249	test-rmse:3.155934 
#> [20]	train-rmse:1.650263	test-rmse:3.148356 
#> [21]	train-rmse:1.592729	test-rmse:3.138569 
#> [22]	train-rmse:1.557427	test-rmse:3.137305 
#> [23]	train-rmse:1.529177	test-rmse:3.140099 
#> [24]	train-rmse:1.493994	test-rmse:3.130350 
#> [25]	train-rmse:1.470884	test-rmse:3.131758 
#> [26]	train-rmse:1.448832	test-rmse:3.132505 
#> [27]	train-rmse:1.415059	test-rmse:3.131703 
#> [28]	train-rmse:1.377509	test-rmse:3.121365 
#> [29]	train-rmse:1.356290	test-rmse:3.122179 
#> [30]	train-rmse:1.322679	test-rmse:3.120752 
#> [31]	train-rmse:1.297073	test-rmse:3.114314 
#> [32]	train-rmse:1.277549	test-rmse:3.120934 
#> [33]	train-rmse:1.251928	test-rmse:3.113675 
#> [34]	train-rmse:1.231383	test-rmse:3.110015 
#> [35]	train-rmse:1.199163	test-rmse:3.099596 
#> [36]	train-rmse:1.178040	test-rmse:3.091530 
#> [37]	train-rmse:1.161323	test-rmse:3.096724 
#> [38]	train-rmse:1.138615	test-rmse:3.096789 
#> [39]	train-rmse:1.125595	test-rmse:3.096618 
#> [40]	train-rmse:1.111898	test-rmse:3.094366 
#> [41]	train-rmse:1.094701	test-rmse:3.097805 
#> [42]	train-rmse:1.085883	test-rmse:3.092502 
#> [43]	train-rmse:1.072696	test-rmse:3.087899 
#> [44]	train-rmse:1.064667	test-rmse:3.088633 
#> [45]	train-rmse:1.047441	test-rmse:3.090840 
#> [46]	train-rmse:1.021197	test-rmse:3.073791 
#> [47]	train-rmse:1.010193	test-rmse:3.078463 
#> [48]	train-rmse:0.985304	test-rmse:3.073091 
#> [49]	train-rmse:0.977418	test-rmse:3.076953 
#> [50]	train-rmse:0.967475	test-rmse:3.077545 
#> [51]	train-rmse:0.942087	test-rmse:3.083860 
#> [52]	train-rmse:0.911916	test-rmse:3.081977 
#> [53]	train-rmse:0.886563	test-rmse:3.078180 
#> [54]	train-rmse:0.855724	test-rmse:3.078831 
#> [55]	train-rmse:0.845150	test-rmse:3.077916 
#> [56]	train-rmse:0.826518	test-rmse:3.080387 
#> [57]	train-rmse:0.821122	test-rmse:3.081350 
#> [58]	train-rmse:0.813287	test-rmse:3.083647 
#> [59]	train-rmse:0.795808	test-rmse:3.085114 
#> [60]	train-rmse:0.790514	test-rmse:3.083795 
#> [61]	train-rmse:0.783901	test-rmse:3.070323 
#> [62]	train-rmse:0.773177	test-rmse:3.068162 
#> [63]	train-rmse:0.762865	test-rmse:3.055478 
#> [64]	train-rmse:0.749724	test-rmse:3.054018 
#> [65]	train-rmse:0.734997	test-rmse:3.053921 
#> [66]	train-rmse:0.725137	test-rmse:3.063874 
#> [67]	train-rmse:0.716506	test-rmse:3.065351 
#> [68]	train-rmse:0.709745	test-rmse:3.067690 
#> [69]	train-rmse:0.701490	test-rmse:3.067873 
#> [70]	train-rmse:0.696366	test-rmse:3.066660 
#> [1]	train-rmse:17.103693	test-rmse:17.252991 
#> [2]	train-rmse:12.348859	test-rmse:12.743414 
#> [3]	train-rmse:9.045874	test-rmse:9.692068 
#> [4]	train-rmse:6.783909	test-rmse:7.692183 
#> [5]	train-rmse:5.199627	test-rmse:6.321737 
#> [6]	train-rmse:4.150670	test-rmse:5.533554 
#> [7]	train-rmse:3.470625	test-rmse:5.048434 
#> [8]	train-rmse:3.040186	test-rmse:4.743094 
#> [9]	train-rmse:2.743806	test-rmse:4.468214 
#> [10]	train-rmse:2.512907	test-rmse:4.274009 
#> [11]	train-rmse:2.335801	test-rmse:4.174132 
#> [12]	train-rmse:2.216330	test-rmse:4.133427 
#> [13]	train-rmse:2.159221	test-rmse:4.105270 
#> [14]	train-rmse:2.083429	test-rmse:4.084567 
#> [15]	train-rmse:2.021866	test-rmse:4.010781 
#> [16]	train-rmse:1.937201	test-rmse:3.993215 
#> [17]	train-rmse:1.874537	test-rmse:3.959648 
#> [18]	train-rmse:1.830324	test-rmse:3.939004 
#> [19]	train-rmse:1.795591	test-rmse:3.903961 
#> [20]	train-rmse:1.762106	test-rmse:3.888590 
#> [21]	train-rmse:1.744185	test-rmse:3.883671 
#> [22]	train-rmse:1.710589	test-rmse:3.892285 
#> [23]	train-rmse:1.690421	test-rmse:3.863746 
#> [24]	train-rmse:1.643268	test-rmse:3.858944 
#> [25]	train-rmse:1.606434	test-rmse:3.845628 
#> [26]	train-rmse:1.557324	test-rmse:3.858439 
#> [27]	train-rmse:1.535222	test-rmse:3.865503 
#> [28]	train-rmse:1.510793	test-rmse:3.855497 
#> [29]	train-rmse:1.455015	test-rmse:3.858603 
#> [30]	train-rmse:1.437971	test-rmse:3.863025 
#> [31]	train-rmse:1.410683	test-rmse:3.859476 
#> [32]	train-rmse:1.378122	test-rmse:3.852221 
#> [33]	train-rmse:1.349468	test-rmse:3.853856 
#> [34]	train-rmse:1.337037	test-rmse:3.854812 
#> [35]	train-rmse:1.315696	test-rmse:3.858581 
#> [36]	train-rmse:1.297927	test-rmse:3.864516 
#> [37]	train-rmse:1.282889	test-rmse:3.852512 
#> [38]	train-rmse:1.268213	test-rmse:3.846431 
#> [39]	train-rmse:1.249264	test-rmse:3.852976 
#> [40]	train-rmse:1.226594	test-rmse:3.850417 
#> [41]	train-rmse:1.193590	test-rmse:3.836281 
#> [42]	train-rmse:1.168915	test-rmse:3.819372 
#> [43]	train-rmse:1.151118	test-rmse:3.815312 
#> [44]	train-rmse:1.127839	test-rmse:3.812875 
#> [45]	train-rmse:1.099423	test-rmse:3.813757 
#> [46]	train-rmse:1.081614	test-rmse:3.816851 
#> [47]	train-rmse:1.071098	test-rmse:3.813645 
#> [48]	train-rmse:1.051975	test-rmse:3.811646 
#> [49]	train-rmse:1.034362	test-rmse:3.808453 
#> [50]	train-rmse:1.016399	test-rmse:3.810196 
#> [51]	train-rmse:1.005012	test-rmse:3.813361 
#> [52]	train-rmse:0.983274	test-rmse:3.823564 
#> [53]	train-rmse:0.957256	test-rmse:3.809035 
#> [54]	train-rmse:0.952174	test-rmse:3.810448 
#> [55]	train-rmse:0.942637	test-rmse:3.807311 
#> [56]	train-rmse:0.934262	test-rmse:3.805659 
#> [57]	train-rmse:0.913344	test-rmse:3.794403 
#> [58]	train-rmse:0.892867	test-rmse:3.775166 
#> [59]	train-rmse:0.881712	test-rmse:3.779293 
#> [60]	train-rmse:0.866893	test-rmse:3.776759 
#> [61]	train-rmse:0.848013	test-rmse:3.767446 
#> [62]	train-rmse:0.838068	test-rmse:3.769425 
#> [63]	train-rmse:0.820571	test-rmse:3.764812 
#> [64]	train-rmse:0.800530	test-rmse:3.769180 
#> [65]	train-rmse:0.788218	test-rmse:3.771906 
#> [66]	train-rmse:0.763266	test-rmse:3.774454 
#> [67]	train-rmse:0.753960	test-rmse:3.771909 
#> [68]	train-rmse:0.742168	test-rmse:3.771918 
#> [69]	train-rmse:0.725973	test-rmse:3.775693 
#> [70]	train-rmse:0.718817	test-rmse:3.774767 
#> [1]	train-rmse:17.086836	test-rmse:17.271259 
#> [2]	train-rmse:12.423159	test-rmse:12.482463 
#> [3]	train-rmse:9.137312	test-rmse:9.356235 
#> [4]	train-rmse:6.844112	test-rmse:7.146619 
#> [5]	train-rmse:5.306246	test-rmse:5.573422 
#> [6]	train-rmse:4.189220	test-rmse:4.663308 
#> [7]	train-rmse:3.462537	test-rmse:4.120639 
#> [8]	train-rmse:2.944524	test-rmse:3.822552 
#> [9]	train-rmse:2.606719	test-rmse:3.610832 
#> [10]	train-rmse:2.366434	test-rmse:3.531172 
#> [11]	train-rmse:2.204029	test-rmse:3.501713 
#> [12]	train-rmse:2.089627	test-rmse:3.483928 
#> [13]	train-rmse:1.995011	test-rmse:3.479363 
#> [14]	train-rmse:1.933802	test-rmse:3.494651 
#> [15]	train-rmse:1.870223	test-rmse:3.485901 
#> [16]	train-rmse:1.830508	test-rmse:3.495230 
#> [17]	train-rmse:1.786605	test-rmse:3.484070 
#> [18]	train-rmse:1.720169	test-rmse:3.456908 
#> [19]	train-rmse:1.670580	test-rmse:3.443953 
#> [20]	train-rmse:1.620001	test-rmse:3.443137 
#> [21]	train-rmse:1.604795	test-rmse:3.445152 
#> [22]	train-rmse:1.578824	test-rmse:3.445412 
#> [23]	train-rmse:1.548743	test-rmse:3.441911 
#> [24]	train-rmse:1.518229	test-rmse:3.438313 
#> [25]	train-rmse:1.504679	test-rmse:3.458323 
#> [26]	train-rmse:1.484650	test-rmse:3.451733 
#> [27]	train-rmse:1.450388	test-rmse:3.432473 
#> [28]	train-rmse:1.427110	test-rmse:3.416934 
#> [29]	train-rmse:1.412530	test-rmse:3.415223 
#> [30]	train-rmse:1.376340	test-rmse:3.411791 
#> [31]	train-rmse:1.363297	test-rmse:3.417721 
#> [32]	train-rmse:1.345455	test-rmse:3.423720 
#> [33]	train-rmse:1.308462	test-rmse:3.436598 
#> [34]	train-rmse:1.289718	test-rmse:3.438217 
#> [35]	train-rmse:1.266830	test-rmse:3.432317 
#> [36]	train-rmse:1.249031	test-rmse:3.443272 
#> [37]	train-rmse:1.212633	test-rmse:3.454168 
#> [38]	train-rmse:1.178920	test-rmse:3.454109 
#> [39]	train-rmse:1.145651	test-rmse:3.452442 
#> [40]	train-rmse:1.131976	test-rmse:3.465035 
#> [41]	train-rmse:1.108406	test-rmse:3.468541 
#> [42]	train-rmse:1.089188	test-rmse:3.463397 
#> [43]	train-rmse:1.083508	test-rmse:3.465469 
#> [44]	train-rmse:1.076285	test-rmse:3.463153 
#> [45]	train-rmse:1.055508	test-rmse:3.469290 
#> [46]	train-rmse:1.046914	test-rmse:3.470375 
#> [47]	train-rmse:1.033266	test-rmse:3.467631 
#> [48]	train-rmse:1.013558	test-rmse:3.475531 
#> [49]	train-rmse:0.996832	test-rmse:3.475539 
#> [50]	train-rmse:0.980940	test-rmse:3.457934 
#> [51]	train-rmse:0.965369	test-rmse:3.466527 
#> [52]	train-rmse:0.953598	test-rmse:3.461404 
#> [53]	train-rmse:0.947059	test-rmse:3.462820 
#> [54]	train-rmse:0.925900	test-rmse:3.454268 
#> [55]	train-rmse:0.919322	test-rmse:3.460876 
#> [56]	train-rmse:0.912147	test-rmse:3.459374 
#> [57]	train-rmse:0.903527	test-rmse:3.456502 
#> [58]	train-rmse:0.894283	test-rmse:3.450772 
#> [59]	train-rmse:0.888327	test-rmse:3.449615 
#> [60]	train-rmse:0.863139	test-rmse:3.459265 
#> [61]	train-rmse:0.841046	test-rmse:3.468838 
#> [62]	train-rmse:0.835520	test-rmse:3.473780 
#> [63]	train-rmse:0.832331	test-rmse:3.475061 
#> [64]	train-rmse:0.829117	test-rmse:3.476901 
#> [65]	train-rmse:0.823743	test-rmse:3.472604 
#> [66]	train-rmse:0.811426	test-rmse:3.480739 
#> [67]	train-rmse:0.795589	test-rmse:3.479755 
#> [68]	train-rmse:0.786904	test-rmse:3.481300 
#> [69]	train-rmse:0.767472	test-rmse:3.485711 
#> [70]	train-rmse:0.760890	test-rmse:3.494196 
#> [1]	train-rmse:17.132009	test-rmse:17.251055 
#> [2]	train-rmse:12.474579	test-rmse:12.813663 
#> [3]	train-rmse:9.196238	test-rmse:9.473038 
#> [4]	train-rmse:6.926752	test-rmse:7.332922 
#> [5]	train-rmse:5.335834	test-rmse:5.797878 
#> [6]	train-rmse:4.291225	test-rmse:4.941513 
#> [7]	train-rmse:3.591254	test-rmse:4.424441 
#> [8]	train-rmse:3.083757	test-rmse:4.016705 
#> [9]	train-rmse:2.779699	test-rmse:3.772171 
#> [10]	train-rmse:2.584003	test-rmse:3.615159 
#> [11]	train-rmse:2.422954	test-rmse:3.545441 
#> [12]	train-rmse:2.304330	test-rmse:3.491544 
#> [13]	train-rmse:2.186087	test-rmse:3.436542 
#> [14]	train-rmse:2.122891	test-rmse:3.391602 
#> [15]	train-rmse:2.068880	test-rmse:3.389993 
#> [16]	train-rmse:2.031223	test-rmse:3.356006 
#> [17]	train-rmse:1.928655	test-rmse:3.314187 
#> [18]	train-rmse:1.890378	test-rmse:3.295842 
#> [19]	train-rmse:1.851579	test-rmse:3.278900 
#> [20]	train-rmse:1.804133	test-rmse:3.283719 
#> [21]	train-rmse:1.752445	test-rmse:3.292249 
#> [22]	train-rmse:1.726916	test-rmse:3.280337 
#> [23]	train-rmse:1.691543	test-rmse:3.255034 
#> [24]	train-rmse:1.660882	test-rmse:3.244141 
#> [25]	train-rmse:1.586876	test-rmse:3.253855 
#> [26]	train-rmse:1.544081	test-rmse:3.226763 
#> [27]	train-rmse:1.516795	test-rmse:3.211534 
#> [28]	train-rmse:1.474672	test-rmse:3.209140 
#> [29]	train-rmse:1.434904	test-rmse:3.204160 
#> [30]	train-rmse:1.415943	test-rmse:3.199675 
#> [31]	train-rmse:1.398198	test-rmse:3.209786 
#> [32]	train-rmse:1.355882	test-rmse:3.195988 
#> [33]	train-rmse:1.333300	test-rmse:3.198368 
#> [34]	train-rmse:1.318887	test-rmse:3.192620 
#> [35]	train-rmse:1.299203	test-rmse:3.186128 
#> [36]	train-rmse:1.259877	test-rmse:3.193987 
#> [37]	train-rmse:1.244086	test-rmse:3.192305 
#> [38]	train-rmse:1.214626	test-rmse:3.184889 
#> [39]	train-rmse:1.176503	test-rmse:3.189001 
#> [40]	train-rmse:1.147220	test-rmse:3.184795 
#> [41]	train-rmse:1.133412	test-rmse:3.178203 
#> [42]	train-rmse:1.107119	test-rmse:3.176926 
#> [43]	train-rmse:1.096034	test-rmse:3.179563 
#> [44]	train-rmse:1.084785	test-rmse:3.180497 
#> [45]	train-rmse:1.073845	test-rmse:3.174839 
#> [46]	train-rmse:1.063945	test-rmse:3.164721 
#> [47]	train-rmse:1.040868	test-rmse:3.161488 
#> [48]	train-rmse:1.008156	test-rmse:3.173606 
#> [49]	train-rmse:0.996106	test-rmse:3.177930 
#> [50]	train-rmse:0.990406	test-rmse:3.181367 
#> [51]	train-rmse:0.963512	test-rmse:3.186441 
#> [52]	train-rmse:0.954483	test-rmse:3.171163 
#> [53]	train-rmse:0.929998	test-rmse:3.170383 
#> [54]	train-rmse:0.912704	test-rmse:3.153999 
#> [55]	train-rmse:0.897730	test-rmse:3.148199 
#> [56]	train-rmse:0.890404	test-rmse:3.148111 
#> [57]	train-rmse:0.877286	test-rmse:3.139555 
#> [58]	train-rmse:0.855869	test-rmse:3.144100 
#> [59]	train-rmse:0.846142	test-rmse:3.148114 
#> [60]	train-rmse:0.838000	test-rmse:3.147561 
#> [61]	train-rmse:0.824123	test-rmse:3.147864 
#> [62]	train-rmse:0.802475	test-rmse:3.158312 
#> [63]	train-rmse:0.790599	test-rmse:3.156652 
#> [64]	train-rmse:0.781960	test-rmse:3.158303 
#> [65]	train-rmse:0.776998	test-rmse:3.158212 
#> [66]	train-rmse:0.771446	test-rmse:3.157373 
#> [67]	train-rmse:0.756462	test-rmse:3.156132 
#> [68]	train-rmse:0.747627	test-rmse:3.155212 
#> [69]	train-rmse:0.731652	test-rmse:3.163408 
#> [70]	train-rmse:0.728644	test-rmse:3.163302 
#> [1]	train-rmse:17.278292	test-rmse:17.131271 
#> [2]	train-rmse:12.556230	test-rmse:12.753764 
#> [3]	train-rmse:9.281087	test-rmse:9.710914 
#> [4]	train-rmse:6.964630	test-rmse:7.441855 
#> [5]	train-rmse:5.380251	test-rmse:6.080617 
#> [6]	train-rmse:4.302356	test-rmse:5.150695 
#> [7]	train-rmse:3.597071	test-rmse:4.589758 
#> [8]	train-rmse:3.156724	test-rmse:4.261594 
#> [9]	train-rmse:2.860281	test-rmse:4.087005 
#> [10]	train-rmse:2.670655	test-rmse:4.006988 
#> [11]	train-rmse:2.486593	test-rmse:3.949593 
#> [12]	train-rmse:2.355701	test-rmse:3.857403 
#> [13]	train-rmse:2.284873	test-rmse:3.850330 
#> [14]	train-rmse:2.147371	test-rmse:3.810094 
#> [15]	train-rmse:2.072834	test-rmse:3.798598 
#> [16]	train-rmse:1.991673	test-rmse:3.761367 
#> [17]	train-rmse:1.946457	test-rmse:3.704476 
#> [18]	train-rmse:1.894600	test-rmse:3.707522 
#> [19]	train-rmse:1.851904	test-rmse:3.692939 
#> [20]	train-rmse:1.779696	test-rmse:3.674443 
#> [21]	train-rmse:1.747371	test-rmse:3.630174 
#> [22]	train-rmse:1.691118	test-rmse:3.579103 
#> [23]	train-rmse:1.653306	test-rmse:3.547489 
#> [24]	train-rmse:1.615739	test-rmse:3.547281 
#> [25]	train-rmse:1.583314	test-rmse:3.524839 
#> [26]	train-rmse:1.535739	test-rmse:3.516026 
#> [27]	train-rmse:1.499918	test-rmse:3.527197 
#> [28]	train-rmse:1.482618	test-rmse:3.543929 
#> [29]	train-rmse:1.445713	test-rmse:3.535451 
#> [30]	train-rmse:1.412574	test-rmse:3.518838 
#> [31]	train-rmse:1.396444	test-rmse:3.524303 
#> [32]	train-rmse:1.376443	test-rmse:3.515206 
#> [33]	train-rmse:1.336527	test-rmse:3.489955 
#> [34]	train-rmse:1.312602	test-rmse:3.490443 
#> [35]	train-rmse:1.288183	test-rmse:3.488138 
#> [36]	train-rmse:1.264677	test-rmse:3.498917 
#> [37]	train-rmse:1.244928	test-rmse:3.489153 
#> [38]	train-rmse:1.214395	test-rmse:3.470848 
#> [39]	train-rmse:1.199964	test-rmse:3.465575 
#> [40]	train-rmse:1.166921	test-rmse:3.465804 
#> [41]	train-rmse:1.133459	test-rmse:3.471084 
#> [42]	train-rmse:1.105368	test-rmse:3.475036 
#> [43]	train-rmse:1.083675	test-rmse:3.476203 
#> [44]	train-rmse:1.069402	test-rmse:3.464823 
#> [45]	train-rmse:1.060441	test-rmse:3.448605 
#> [46]	train-rmse:1.044142	test-rmse:3.440452 
#> [47]	train-rmse:1.024233	test-rmse:3.447371 
#> [48]	train-rmse:1.011220	test-rmse:3.449913 
#> [49]	train-rmse:1.000908	test-rmse:3.446062 
#> [50]	train-rmse:0.982938	test-rmse:3.441698 
#> [51]	train-rmse:0.977451	test-rmse:3.451431 
#> [52]	train-rmse:0.970785	test-rmse:3.447929 
#> [53]	train-rmse:0.939961	test-rmse:3.448962 
#> [54]	train-rmse:0.929331	test-rmse:3.447889 
#> [55]	train-rmse:0.909236	test-rmse:3.458525 
#> [56]	train-rmse:0.895298	test-rmse:3.446665 
#> [57]	train-rmse:0.888991	test-rmse:3.448505 
#> [58]	train-rmse:0.881190	test-rmse:3.449063 
#> [59]	train-rmse:0.870374	test-rmse:3.446085 
#> [60]	train-rmse:0.852684	test-rmse:3.443754 
#> [61]	train-rmse:0.834644	test-rmse:3.435974 
#> [62]	train-rmse:0.815564	test-rmse:3.426854 
#> [63]	train-rmse:0.799357	test-rmse:3.423715 
#> [64]	train-rmse:0.793762	test-rmse:3.428804 
#> [65]	train-rmse:0.782843	test-rmse:3.428723 
#> [66]	train-rmse:0.771491	test-rmse:3.432789 
#> [67]	train-rmse:0.762924	test-rmse:3.433431 
#> [68]	train-rmse:0.746974	test-rmse:3.427707 
#> [69]	train-rmse:0.728521	test-rmse:3.430997 
#> [70]	train-rmse:0.715470	test-rmse:3.432857 
#> [1]	train-rmse:17.395387	test-rmse:16.807481 
#> [2]	train-rmse:12.607171	test-rmse:12.209140 
#> [3]	train-rmse:9.287325	test-rmse:8.937328 
#> [4]	train-rmse:7.013758	test-rmse:6.761207 
#> [5]	train-rmse:5.418121	test-rmse:5.271652 
#> [6]	train-rmse:4.309007	test-rmse:4.362471 
#> [7]	train-rmse:3.593338	test-rmse:3.809283 
#> [8]	train-rmse:3.125462	test-rmse:3.488303 
#> [9]	train-rmse:2.808215	test-rmse:3.257830 
#> [10]	train-rmse:2.613578	test-rmse:3.154585 
#> [11]	train-rmse:2.471688	test-rmse:3.079359 
#> [12]	train-rmse:2.326925	test-rmse:2.990096 
#> [13]	train-rmse:2.243460	test-rmse:2.929664 
#> [14]	train-rmse:2.134441	test-rmse:2.890038 
#> [15]	train-rmse:2.068489	test-rmse:2.834206 
#> [16]	train-rmse:2.009764	test-rmse:2.824391 
#> [17]	train-rmse:1.972183	test-rmse:2.825643 
#> [18]	train-rmse:1.915850	test-rmse:2.827552 
#> [19]	train-rmse:1.860962	test-rmse:2.804798 
#> [20]	train-rmse:1.825913	test-rmse:2.805106 
#> [21]	train-rmse:1.795406	test-rmse:2.794276 
#> [22]	train-rmse:1.755088	test-rmse:2.794035 
#> [23]	train-rmse:1.721797	test-rmse:2.775509 
#> [24]	train-rmse:1.676102	test-rmse:2.762334 
#> [25]	train-rmse:1.645750	test-rmse:2.755726 
#> [26]	train-rmse:1.606884	test-rmse:2.752308 
#> [27]	train-rmse:1.554420	test-rmse:2.725800 
#> [28]	train-rmse:1.533663	test-rmse:2.709923 
#> [29]	train-rmse:1.515497	test-rmse:2.715273 
#> [30]	train-rmse:1.487916	test-rmse:2.688154 
#> [31]	train-rmse:1.438739	test-rmse:2.688176 
#> [32]	train-rmse:1.423625	test-rmse:2.672790 
#> [33]	train-rmse:1.390451	test-rmse:2.663642 
#> [34]	train-rmse:1.375332	test-rmse:2.664944 
#> [35]	train-rmse:1.338564	test-rmse:2.667463 
#> [36]	train-rmse:1.303356	test-rmse:2.654060 
#> [37]	train-rmse:1.275568	test-rmse:2.665540 
#> [38]	train-rmse:1.261996	test-rmse:2.659455 
#> [39]	train-rmse:1.251079	test-rmse:2.665207 
#> [40]	train-rmse:1.227448	test-rmse:2.653628 
#> [41]	train-rmse:1.213734	test-rmse:2.646677 
#> [42]	train-rmse:1.176958	test-rmse:2.649511 
#> [43]	train-rmse:1.158550	test-rmse:2.659746 
#> [44]	train-rmse:1.146393	test-rmse:2.661095 
#> [45]	train-rmse:1.112010	test-rmse:2.650501 
#> [46]	train-rmse:1.092421	test-rmse:2.659812 
#> [47]	train-rmse:1.075534	test-rmse:2.660230 
#> [48]	train-rmse:1.049945	test-rmse:2.655150 
#> [49]	train-rmse:1.035587	test-rmse:2.660354 
#> [50]	train-rmse:1.027057	test-rmse:2.658988 
#> [51]	train-rmse:1.005431	test-rmse:2.649592 
#> [52]	train-rmse:0.987424	test-rmse:2.645430 
#> [53]	train-rmse:0.971436	test-rmse:2.647272 
#> [54]	train-rmse:0.951233	test-rmse:2.637944 
#> [55]	train-rmse:0.934642	test-rmse:2.638550 
#> [56]	train-rmse:0.914529	test-rmse:2.639281 
#> [57]	train-rmse:0.905744	test-rmse:2.634091 
#> [58]	train-rmse:0.897733	test-rmse:2.635307 
#> [59]	train-rmse:0.883033	test-rmse:2.625912 
#> [60]	train-rmse:0.865160	test-rmse:2.627433 
#> [61]	train-rmse:0.853266	test-rmse:2.628132 
#> [62]	train-rmse:0.848135	test-rmse:2.625140 
#> [63]	train-rmse:0.835240	test-rmse:2.620551 
#> [64]	train-rmse:0.819862	test-rmse:2.629525 
#> [65]	train-rmse:0.816406	test-rmse:2.631767 
#> [66]	train-rmse:0.797353	test-rmse:2.636911 
#> [67]	train-rmse:0.789818	test-rmse:2.636512 
#> [68]	train-rmse:0.775842	test-rmse:2.636128 
#> [69]	train-rmse:0.762010	test-rmse:2.640448 
#> [70]	train-rmse:0.755448	test-rmse:2.634867 
#> [1]	train-rmse:17.369700	test-rmse:17.163387 
#> [2]	train-rmse:12.538664	test-rmse:12.632404 
#> [3]	train-rmse:9.180276	test-rmse:9.463454 
#> [4]	train-rmse:6.822798	test-rmse:7.408056 
#> [5]	train-rmse:5.157894	test-rmse:6.144156 
#> [6]	train-rmse:4.053398	test-rmse:5.392924 
#> [7]	train-rmse:3.282756	test-rmse:4.813340 
#> [8]	train-rmse:2.763495	test-rmse:4.541041 
#> [9]	train-rmse:2.432271	test-rmse:4.383377 
#> [10]	train-rmse:2.200080	test-rmse:4.191137 
#> [11]	train-rmse:2.020068	test-rmse:4.085083 
#> [12]	train-rmse:1.924612	test-rmse:4.041232 
#> [13]	train-rmse:1.848469	test-rmse:4.027526 
#> [14]	train-rmse:1.782436	test-rmse:4.080986 
#> [15]	train-rmse:1.730807	test-rmse:4.065269 
#> [16]	train-rmse:1.664313	test-rmse:4.020690 
#> [17]	train-rmse:1.614103	test-rmse:3.966595 
#> [18]	train-rmse:1.575948	test-rmse:3.973121 
#> [19]	train-rmse:1.547222	test-rmse:3.971151 
#> [20]	train-rmse:1.514206	test-rmse:3.951577 
#> [21]	train-rmse:1.462316	test-rmse:3.959171 
#> [22]	train-rmse:1.440970	test-rmse:3.968296 
#> [23]	train-rmse:1.399896	test-rmse:3.967540 
#> [24]	train-rmse:1.368519	test-rmse:3.964019 
#> [25]	train-rmse:1.353368	test-rmse:3.948828 
#> [26]	train-rmse:1.327715	test-rmse:3.951640 
#> [27]	train-rmse:1.293445	test-rmse:3.937234 
#> [28]	train-rmse:1.272693	test-rmse:3.912807 
#> [29]	train-rmse:1.240026	test-rmse:3.901195 
#> [30]	train-rmse:1.206814	test-rmse:3.899558 
#> [31]	train-rmse:1.193308	test-rmse:3.902120 
#> [32]	train-rmse:1.179732	test-rmse:3.887835 
#> [33]	train-rmse:1.130219	test-rmse:3.853357 
#> [34]	train-rmse:1.108433	test-rmse:3.837914 
#> [35]	train-rmse:1.082396	test-rmse:3.830936 
#> [36]	train-rmse:1.072405	test-rmse:3.831802 
#> [37]	train-rmse:1.059950	test-rmse:3.836254 
#> [38]	train-rmse:1.040796	test-rmse:3.836272 
#> [39]	train-rmse:1.034363	test-rmse:3.835807 
#> [40]	train-rmse:1.009050	test-rmse:3.830396 
#> [41]	train-rmse:0.992680	test-rmse:3.829704 
#> [42]	train-rmse:0.979284	test-rmse:3.818401 
#> [43]	train-rmse:0.959056	test-rmse:3.817649 
#> [44]	train-rmse:0.931938	test-rmse:3.802608 
#> [45]	train-rmse:0.915750	test-rmse:3.806075 
#> [46]	train-rmse:0.902265	test-rmse:3.805916 
#> [47]	train-rmse:0.874482	test-rmse:3.802125 
#> [48]	train-rmse:0.864981	test-rmse:3.801639 
#> [49]	train-rmse:0.852830	test-rmse:3.806106 
#> [50]	train-rmse:0.835721	test-rmse:3.797447 
#> [51]	train-rmse:0.824056	test-rmse:3.802764 
#> [52]	train-rmse:0.808376	test-rmse:3.800457 
#> [53]	train-rmse:0.800746	test-rmse:3.802641 
#> [54]	train-rmse:0.783422	test-rmse:3.796594 
#> [55]	train-rmse:0.771401	test-rmse:3.804255 
#> [56]	train-rmse:0.762217	test-rmse:3.798605 
#> [57]	train-rmse:0.749706	test-rmse:3.799337 
#> [58]	train-rmse:0.739627	test-rmse:3.796423 
#> [59]	train-rmse:0.721745	test-rmse:3.796050 
#> [60]	train-rmse:0.709733	test-rmse:3.789439 
#> [61]	train-rmse:0.702526	test-rmse:3.787663 
#> [62]	train-rmse:0.696618	test-rmse:3.785630 
#> [63]	train-rmse:0.690819	test-rmse:3.780841 
#> [64]	train-rmse:0.679432	test-rmse:3.782988 
#> [65]	train-rmse:0.674282	test-rmse:3.784567 
#> [66]	train-rmse:0.664273	test-rmse:3.784069 
#> [67]	train-rmse:0.651228	test-rmse:3.785054 
#> [68]	train-rmse:0.644693	test-rmse:3.786495 
#> [69]	train-rmse:0.638943	test-rmse:3.788826 
#> [70]	train-rmse:0.631541	test-rmse:3.779159 
#> [1]	train-rmse:16.851618	test-rmse:17.689038 
#> [2]	train-rmse:12.247046	test-rmse:13.146214 
#> [3]	train-rmse:9.055132	test-rmse:9.992292 
#> [4]	train-rmse:6.811049	test-rmse:7.597927 
#> [5]	train-rmse:5.287718	test-rmse:6.017995 
#> [6]	train-rmse:4.287591	test-rmse:5.168742 
#> [7]	train-rmse:3.604025	test-rmse:4.545065 
#> [8]	train-rmse:3.144249	test-rmse:4.192109 
#> [9]	train-rmse:2.836724	test-rmse:3.936045 
#> [10]	train-rmse:2.629665	test-rmse:3.844183 
#> [11]	train-rmse:2.446611	test-rmse:3.774374 
#> [12]	train-rmse:2.350734	test-rmse:3.656684 
#> [13]	train-rmse:2.246979	test-rmse:3.608158 
#> [14]	train-rmse:2.199923	test-rmse:3.614383 
#> [15]	train-rmse:2.130916	test-rmse:3.566512 
#> [16]	train-rmse:2.092222	test-rmse:3.569966 
#> [17]	train-rmse:2.056247	test-rmse:3.531677 
#> [18]	train-rmse:1.989196	test-rmse:3.509704 
#> [19]	train-rmse:1.904293	test-rmse:3.526812 
#> [20]	train-rmse:1.874183	test-rmse:3.542455 
#> [21]	train-rmse:1.836572	test-rmse:3.529170 
#> [22]	train-rmse:1.803295	test-rmse:3.528720 
#> [23]	train-rmse:1.751647	test-rmse:3.509608 
#> [24]	train-rmse:1.733535	test-rmse:3.533424 
#> [25]	train-rmse:1.690835	test-rmse:3.526375 
#> [26]	train-rmse:1.675032	test-rmse:3.528541 
#> [27]	train-rmse:1.637364	test-rmse:3.531943 
#> [28]	train-rmse:1.595909	test-rmse:3.532742 
#> [29]	train-rmse:1.530506	test-rmse:3.529922 
#> [30]	train-rmse:1.495419	test-rmse:3.524897 
#> [31]	train-rmse:1.482784	test-rmse:3.530953 
#> [32]	train-rmse:1.459573	test-rmse:3.524708 
#> [33]	train-rmse:1.443076	test-rmse:3.540299 
#> [34]	train-rmse:1.419467	test-rmse:3.559708 
#> [35]	train-rmse:1.393598	test-rmse:3.551996 
#> [36]	train-rmse:1.381127	test-rmse:3.549354 
#> [37]	train-rmse:1.343972	test-rmse:3.546839 
#> [38]	train-rmse:1.293013	test-rmse:3.547004 
#> [39]	train-rmse:1.258523	test-rmse:3.550533 
#> [40]	train-rmse:1.227583	test-rmse:3.552073 
#> [41]	train-rmse:1.218001	test-rmse:3.565416 
#> [42]	train-rmse:1.191352	test-rmse:3.555782 
#> [43]	train-rmse:1.167781	test-rmse:3.555559 
#> [44]	train-rmse:1.154672	test-rmse:3.560110 
#> [45]	train-rmse:1.143491	test-rmse:3.561748 
#> [46]	train-rmse:1.112355	test-rmse:3.573442 
#> [47]	train-rmse:1.104252	test-rmse:3.576608 
#> [48]	train-rmse:1.086042	test-rmse:3.564718 
#> [49]	train-rmse:1.065225	test-rmse:3.573229 
#> [50]	train-rmse:1.055505	test-rmse:3.572608 
#> [51]	train-rmse:1.050356	test-rmse:3.574039 
#> [52]	train-rmse:1.024839	test-rmse:3.582760 
#> [53]	train-rmse:1.007063	test-rmse:3.594317 
#> [54]	train-rmse:0.989760	test-rmse:3.596190 
#> [55]	train-rmse:0.985288	test-rmse:3.589626 
#> [56]	train-rmse:0.968332	test-rmse:3.587761 
#> [57]	train-rmse:0.955451	test-rmse:3.588921 
#> [58]	train-rmse:0.935644	test-rmse:3.581341 
#> [59]	train-rmse:0.915201	test-rmse:3.576290 
#> [60]	train-rmse:0.894053	test-rmse:3.575263 
#> [61]	train-rmse:0.886351	test-rmse:3.578476 
#> [62]	train-rmse:0.868200	test-rmse:3.574945 
#> [63]	train-rmse:0.864606	test-rmse:3.570811 
#> [64]	train-rmse:0.843844	test-rmse:3.571999 
#> [65]	train-rmse:0.825551	test-rmse:3.578927 
#> [66]	train-rmse:0.822725	test-rmse:3.575870 
#> [67]	train-rmse:0.814655	test-rmse:3.573540 
#> [68]	train-rmse:0.803614	test-rmse:3.577983 
#> [69]	train-rmse:0.797038	test-rmse:3.584130 
#> [70]	train-rmse:0.791149	test-rmse:3.583348 
#> [1]	train-rmse:17.477357	test-rmse:16.527321 
#> [2]	train-rmse:12.664641	test-rmse:11.995947 
#> [3]	train-rmse:9.302649	test-rmse:8.958430 
#> [4]	train-rmse:6.986884	test-rmse:6.865748 
#> [5]	train-rmse:5.376294	test-rmse:5.499326 
#> [6]	train-rmse:4.299956	test-rmse:4.772482 
#> [7]	train-rmse:3.560704	test-rmse:4.274495 
#> [8]	train-rmse:3.081282	test-rmse:3.934724 
#> [9]	train-rmse:2.734935	test-rmse:3.754312 
#> [10]	train-rmse:2.515792	test-rmse:3.590660 
#> [11]	train-rmse:2.348332	test-rmse:3.475234 
#> [12]	train-rmse:2.250497	test-rmse:3.442904 
#> [13]	train-rmse:2.157210	test-rmse:3.383382 
#> [14]	train-rmse:2.092833	test-rmse:3.351163 
#> [15]	train-rmse:2.019703	test-rmse:3.355036 
#> [16]	train-rmse:1.964612	test-rmse:3.332666 
#> [17]	train-rmse:1.907572	test-rmse:3.299823 
#> [18]	train-rmse:1.874738	test-rmse:3.292053 
#> [19]	train-rmse:1.822290	test-rmse:3.308171 
#> [20]	train-rmse:1.774891	test-rmse:3.277486 
#> [21]	train-rmse:1.727159	test-rmse:3.252848 
#> [22]	train-rmse:1.705368	test-rmse:3.217827 
#> [23]	train-rmse:1.684295	test-rmse:3.203909 
#> [24]	train-rmse:1.653245	test-rmse:3.201445 
#> [25]	train-rmse:1.617504	test-rmse:3.202111 
#> [26]	train-rmse:1.580065	test-rmse:3.188726 
#> [27]	train-rmse:1.546369	test-rmse:3.184266 
#> [28]	train-rmse:1.516384	test-rmse:3.181718 
#> [29]	train-rmse:1.476499	test-rmse:3.170822 
#> [30]	train-rmse:1.449459	test-rmse:3.162337 
#> [31]	train-rmse:1.416426	test-rmse:3.176795 
#> [32]	train-rmse:1.398788	test-rmse:3.177891 
#> [33]	train-rmse:1.367230	test-rmse:3.175756 
#> [34]	train-rmse:1.337373	test-rmse:3.174424 
#> [35]	train-rmse:1.314864	test-rmse:3.169255 
#> [36]	train-rmse:1.294271	test-rmse:3.176421 
#> [37]	train-rmse:1.282166	test-rmse:3.171310 
#> [38]	train-rmse:1.263780	test-rmse:3.168343 
#> [39]	train-rmse:1.250331	test-rmse:3.168549 
#> [40]	train-rmse:1.239088	test-rmse:3.170896 
#> [41]	train-rmse:1.227630	test-rmse:3.165811 
#> [42]	train-rmse:1.219705	test-rmse:3.167467 
#> [43]	train-rmse:1.197678	test-rmse:3.158448 
#> [44]	train-rmse:1.189214	test-rmse:3.150690 
#> [45]	train-rmse:1.174734	test-rmse:3.144025 
#> [46]	train-rmse:1.167863	test-rmse:3.140134 
#> [47]	train-rmse:1.153933	test-rmse:3.138493 
#> [48]	train-rmse:1.144011	test-rmse:3.136328 
#> [49]	train-rmse:1.117278	test-rmse:3.118836 
#> [50]	train-rmse:1.094564	test-rmse:3.093019 
#> [51]	train-rmse:1.078295	test-rmse:3.092524 
#> [52]	train-rmse:1.059747	test-rmse:3.090600 
#> [53]	train-rmse:1.055997	test-rmse:3.091937 
#> [54]	train-rmse:1.038776	test-rmse:3.088562 
#> [55]	train-rmse:1.024767	test-rmse:3.092815 
#> [56]	train-rmse:1.009588	test-rmse:3.096786 
#> [57]	train-rmse:0.987335	test-rmse:3.099280 
#> [58]	train-rmse:0.977918	test-rmse:3.100591 
#> [59]	train-rmse:0.965779	test-rmse:3.105325 
#> [60]	train-rmse:0.959166	test-rmse:3.106332 
#> [61]	train-rmse:0.941844	test-rmse:3.110665 
#> [62]	train-rmse:0.927847	test-rmse:3.097986 
#> [63]	train-rmse:0.908501	test-rmse:3.103467 
#> [64]	train-rmse:0.897267	test-rmse:3.105768 
#> [65]	train-rmse:0.885763	test-rmse:3.104582 
#> [66]	train-rmse:0.877413	test-rmse:3.109884 
#> [67]	train-rmse:0.858192	test-rmse:3.113015 
#> [68]	train-rmse:0.841577	test-rmse:3.112886 
#> [69]	train-rmse:0.827335	test-rmse:3.111785 
#> [70]	train-rmse:0.810480	test-rmse:3.102859 
#> [1]	train-rmse:17.516712	test-rmse:16.439532 
#> [2]	train-rmse:12.704872	test-rmse:11.972499 
#> [3]	train-rmse:9.309501	test-rmse:8.962723 
#> [4]	train-rmse:6.952581	test-rmse:6.891552 
#> [5]	train-rmse:5.359371	test-rmse:5.631844 
#> [6]	train-rmse:4.216955	test-rmse:4.851716 
#> [7]	train-rmse:3.465746	test-rmse:4.414256 
#> [8]	train-rmse:2.948136	test-rmse:4.126199 
#> [9]	train-rmse:2.632189	test-rmse:3.988165 
#> [10]	train-rmse:2.396595	test-rmse:3.910492 
#> [11]	train-rmse:2.235672	test-rmse:3.861430 
#> [12]	train-rmse:2.128849	test-rmse:3.879963 
#> [13]	train-rmse:2.007587	test-rmse:3.839521 
#> [14]	train-rmse:1.917538	test-rmse:3.819519 
#> [15]	train-rmse:1.834964	test-rmse:3.834588 
#> [16]	train-rmse:1.799470	test-rmse:3.846834 
#> [17]	train-rmse:1.732873	test-rmse:3.842931 
#> [18]	train-rmse:1.666987	test-rmse:3.819122 
#> [19]	train-rmse:1.625099	test-rmse:3.788526 
#> [20]	train-rmse:1.588039	test-rmse:3.763336 
#> [21]	train-rmse:1.558551	test-rmse:3.747225 
#> [22]	train-rmse:1.510424	test-rmse:3.756382 
#> [23]	train-rmse:1.497681	test-rmse:3.734523 
#> [24]	train-rmse:1.482040	test-rmse:3.733861 
#> [25]	train-rmse:1.463634	test-rmse:3.753645 
#> [26]	train-rmse:1.448568	test-rmse:3.741722 
#> [27]	train-rmse:1.419823	test-rmse:3.727728 
#> [28]	train-rmse:1.399649	test-rmse:3.719883 
#> [29]	train-rmse:1.353190	test-rmse:3.690821 
#> [30]	train-rmse:1.319409	test-rmse:3.697237 
#> [31]	train-rmse:1.305254	test-rmse:3.692799 
#> [32]	train-rmse:1.284274	test-rmse:3.683033 
#> [33]	train-rmse:1.260786	test-rmse:3.694678 
#> [34]	train-rmse:1.252847	test-rmse:3.687495 
#> [35]	train-rmse:1.232505	test-rmse:3.690181 
#> [36]	train-rmse:1.216862	test-rmse:3.689011 
#> [37]	train-rmse:1.180909	test-rmse:3.689362 
#> [38]	train-rmse:1.163879	test-rmse:3.686103 
#> [39]	train-rmse:1.146814	test-rmse:3.695191 
#> [40]	train-rmse:1.118845	test-rmse:3.702853 
#> [41]	train-rmse:1.091918	test-rmse:3.707081 
#> [42]	train-rmse:1.076218	test-rmse:3.704513 
#> [43]	train-rmse:1.056559	test-rmse:3.704579 
#> [44]	train-rmse:1.034916	test-rmse:3.696735 
#> [45]	train-rmse:1.017518	test-rmse:3.686453 
#> [46]	train-rmse:1.010944	test-rmse:3.672369 
#> [47]	train-rmse:0.998823	test-rmse:3.671160 
#> [48]	train-rmse:0.985772	test-rmse:3.669297 
#> [49]	train-rmse:0.967363	test-rmse:3.678855 
#> [50]	train-rmse:0.944085	test-rmse:3.677862 
#> [51]	train-rmse:0.925227	test-rmse:3.675951 
#> [52]	train-rmse:0.921101	test-rmse:3.666867 
#> [53]	train-rmse:0.902544	test-rmse:3.658930 
#> [54]	train-rmse:0.881444	test-rmse:3.660465 
#> [55]	train-rmse:0.872768	test-rmse:3.665478 
#> [56]	train-rmse:0.864121	test-rmse:3.662788 
#> [57]	train-rmse:0.848972	test-rmse:3.651639 
#> [58]	train-rmse:0.828335	test-rmse:3.641569 
#> [59]	train-rmse:0.816252	test-rmse:3.636778 
#> [60]	train-rmse:0.806125	test-rmse:3.639459 
#> [61]	train-rmse:0.788547	test-rmse:3.634627 
#> [62]	train-rmse:0.773397	test-rmse:3.629928 
#> [63]	train-rmse:0.757270	test-rmse:3.619143 
#> [64]	train-rmse:0.743927	test-rmse:3.622712 
#> [65]	train-rmse:0.734305	test-rmse:3.624204 
#> [66]	train-rmse:0.723893	test-rmse:3.626602 
#> [67]	train-rmse:0.715963	test-rmse:3.626080 
#> [68]	train-rmse:0.708651	test-rmse:3.625414 
#> [69]	train-rmse:0.692336	test-rmse:3.621810 
#> [70]	train-rmse:0.684827	test-rmse:3.617717 
#> [1]	train-rmse:16.922243	test-rmse:17.651039 
#> [2]	train-rmse:12.250031	test-rmse:12.887329 
#> [3]	train-rmse:9.013010	test-rmse:9.756794 
#> [4]	train-rmse:6.793117	test-rmse:7.768504 
#> [5]	train-rmse:5.235041	test-rmse:6.281144 
#> [6]	train-rmse:4.170303	test-rmse:5.362771 
#> [7]	train-rmse:3.486009	test-rmse:4.830497 
#> [8]	train-rmse:3.051639	test-rmse:4.501959 
#> [9]	train-rmse:2.750394	test-rmse:4.297157 
#> [10]	train-rmse:2.528267	test-rmse:4.143705 
#> [11]	train-rmse:2.385107	test-rmse:4.056694 
#> [12]	train-rmse:2.282677	test-rmse:4.011703 
#> [13]	train-rmse:2.206975	test-rmse:3.983616 
#> [14]	train-rmse:2.131304	test-rmse:3.921142 
#> [15]	train-rmse:2.077097	test-rmse:3.898018 
#> [16]	train-rmse:1.992156	test-rmse:3.874507 
#> [17]	train-rmse:1.946122	test-rmse:3.832192 
#> [18]	train-rmse:1.879951	test-rmse:3.791515 
#> [19]	train-rmse:1.814209	test-rmse:3.778795 
#> [20]	train-rmse:1.777613	test-rmse:3.767908 
#> [21]	train-rmse:1.730406	test-rmse:3.749057 
#> [22]	train-rmse:1.710667	test-rmse:3.742828 
#> [23]	train-rmse:1.672707	test-rmse:3.727354 
#> [24]	train-rmse:1.624492	test-rmse:3.710978 
#> [25]	train-rmse:1.585969	test-rmse:3.717527 
#> [26]	train-rmse:1.539809	test-rmse:3.718164 
#> [27]	train-rmse:1.502219	test-rmse:3.711564 
#> [28]	train-rmse:1.487363	test-rmse:3.716270 
#> [29]	train-rmse:1.472990	test-rmse:3.729529 
#> [30]	train-rmse:1.456847	test-rmse:3.730548 
#> [31]	train-rmse:1.438577	test-rmse:3.720914 
#> [32]	train-rmse:1.414038	test-rmse:3.713137 
#> [33]	train-rmse:1.393750	test-rmse:3.698787 
#> [34]	train-rmse:1.377710	test-rmse:3.685371 
#> [35]	train-rmse:1.366862	test-rmse:3.677508 
#> [36]	train-rmse:1.353890	test-rmse:3.680680 
#> [37]	train-rmse:1.318616	test-rmse:3.679199 
#> [38]	train-rmse:1.292043	test-rmse:3.676492 
#> [39]	train-rmse:1.269161	test-rmse:3.670951 
#> [40]	train-rmse:1.256788	test-rmse:3.679901 
#> [41]	train-rmse:1.237353	test-rmse:3.677076 
#> [42]	train-rmse:1.217982	test-rmse:3.671050 
#> [43]	train-rmse:1.182463	test-rmse:3.674857 
#> [44]	train-rmse:1.157537	test-rmse:3.667582 
#> [45]	train-rmse:1.145929	test-rmse:3.660398 
#> [46]	train-rmse:1.124955	test-rmse:3.661673 
#> [47]	train-rmse:1.108189	test-rmse:3.659959 
#> [48]	train-rmse:1.102524	test-rmse:3.661394 
#> [49]	train-rmse:1.064052	test-rmse:3.663788 
#> [50]	train-rmse:1.034119	test-rmse:3.665225 
#> [51]	train-rmse:1.010867	test-rmse:3.658235 
#> [52]	train-rmse:1.002326	test-rmse:3.648886 
#> [53]	train-rmse:0.982503	test-rmse:3.663341 
#> [54]	train-rmse:0.970714	test-rmse:3.660239 
#> [55]	train-rmse:0.955684	test-rmse:3.655650 
#> [56]	train-rmse:0.933728	test-rmse:3.649022 
#> [57]	train-rmse:0.921975	test-rmse:3.646999 
#> [58]	train-rmse:0.907513	test-rmse:3.651628 
#> [59]	train-rmse:0.886558	test-rmse:3.654107 
#> [60]	train-rmse:0.866756	test-rmse:3.662212 
#> [61]	train-rmse:0.862832	test-rmse:3.658910 
#> [62]	train-rmse:0.847068	test-rmse:3.651673 
#> [63]	train-rmse:0.837279	test-rmse:3.652087 
#> [64]	train-rmse:0.822307	test-rmse:3.651748 
#> [65]	train-rmse:0.807431	test-rmse:3.652063 
#> [66]	train-rmse:0.797902	test-rmse:3.655239 
#> [67]	train-rmse:0.791800	test-rmse:3.658549 
#> [68]	train-rmse:0.776808	test-rmse:3.658271 
#> [69]	train-rmse:0.770397	test-rmse:3.659264 
#> [70]	train-rmse:0.767124	test-rmse:3.662660 
#> [1]	train-rmse:17.152837	test-rmse:17.312627 
#> [2]	train-rmse:12.417863	test-rmse:12.623049 
#> [3]	train-rmse:9.109260	test-rmse:9.430341 
#> [4]	train-rmse:6.806087	test-rmse:7.340749 
#> [5]	train-rmse:5.236341	test-rmse:5.969851 
#> [6]	train-rmse:4.124409	test-rmse:5.180520 
#> [7]	train-rmse:3.409541	test-rmse:4.725769 
#> [8]	train-rmse:2.907675	test-rmse:4.370898 
#> [9]	train-rmse:2.616327	test-rmse:4.193904 
#> [10]	train-rmse:2.403490	test-rmse:4.063455 
#> [11]	train-rmse:2.245480	test-rmse:3.954606 
#> [12]	train-rmse:2.131974	test-rmse:3.909208 
#> [13]	train-rmse:2.056478	test-rmse:3.883845 
#> [14]	train-rmse:1.988316	test-rmse:3.829907 
#> [15]	train-rmse:1.918525	test-rmse:3.846480 
#> [16]	train-rmse:1.853409	test-rmse:3.844287 
#> [17]	train-rmse:1.764102	test-rmse:3.808609 
#> [18]	train-rmse:1.728920	test-rmse:3.798865 
#> [19]	train-rmse:1.686462	test-rmse:3.801505 
#> [20]	train-rmse:1.643720	test-rmse:3.768817 
#> [21]	train-rmse:1.615638	test-rmse:3.772158 
#> [22]	train-rmse:1.581222	test-rmse:3.754727 
#> [23]	train-rmse:1.532340	test-rmse:3.742759 
#> [24]	train-rmse:1.511468	test-rmse:3.754419 
#> [25]	train-rmse:1.464715	test-rmse:3.738069 
#> [26]	train-rmse:1.425049	test-rmse:3.750589 
#> [27]	train-rmse:1.403122	test-rmse:3.755480 
#> [28]	train-rmse:1.358186	test-rmse:3.725604 
#> [29]	train-rmse:1.339987	test-rmse:3.718350 
#> [30]	train-rmse:1.315209	test-rmse:3.706850 
#> [31]	train-rmse:1.283096	test-rmse:3.697370 
#> [32]	train-rmse:1.253204	test-rmse:3.704360 
#> [33]	train-rmse:1.239898	test-rmse:3.701379 
#> [34]	train-rmse:1.218439	test-rmse:3.705304 
#> [35]	train-rmse:1.202165	test-rmse:3.717953 
#> [36]	train-rmse:1.188075	test-rmse:3.712864 
#> [37]	train-rmse:1.172806	test-rmse:3.706612 
#> [38]	train-rmse:1.130514	test-rmse:3.693831 
#> [39]	train-rmse:1.123755	test-rmse:3.687604 
#> [40]	train-rmse:1.110395	test-rmse:3.683130 
#> [41]	train-rmse:1.088629	test-rmse:3.684874 
#> [42]	train-rmse:1.057146	test-rmse:3.673015 
#> [43]	train-rmse:1.037357	test-rmse:3.669375 
#> [44]	train-rmse:1.012744	test-rmse:3.667050 
#> [45]	train-rmse:0.995983	test-rmse:3.663645 
#> [46]	train-rmse:0.987530	test-rmse:3.662034 
#> [47]	train-rmse:0.982015	test-rmse:3.661063 
#> [48]	train-rmse:0.963825	test-rmse:3.663633 
#> [49]	train-rmse:0.949556	test-rmse:3.656571 
#> [50]	train-rmse:0.929863	test-rmse:3.654831 
#> [51]	train-rmse:0.917359	test-rmse:3.656136 
#> [52]	train-rmse:0.906694	test-rmse:3.650071 
#> [53]	train-rmse:0.896932	test-rmse:3.651673 
#> [54]	train-rmse:0.886899	test-rmse:3.646640 
#> [55]	train-rmse:0.881587	test-rmse:3.644913 
#> [56]	train-rmse:0.865127	test-rmse:3.649565 
#> [57]	train-rmse:0.852795	test-rmse:3.653401 
#> [58]	train-rmse:0.832879	test-rmse:3.655664 
#> [59]	train-rmse:0.828237	test-rmse:3.649560 
#> [60]	train-rmse:0.813814	test-rmse:3.649660 
#> [61]	train-rmse:0.801674	test-rmse:3.646815 
#> [62]	train-rmse:0.795732	test-rmse:3.652086 
#> [63]	train-rmse:0.792014	test-rmse:3.650749 
#> [64]	train-rmse:0.780086	test-rmse:3.654672 
#> [65]	train-rmse:0.768030	test-rmse:3.653059 
#> [66]	train-rmse:0.759222	test-rmse:3.653827 
#> [67]	train-rmse:0.751381	test-rmse:3.647739 
#> [68]	train-rmse:0.733078	test-rmse:3.651626 
#> [69]	train-rmse:0.715336	test-rmse:3.653775 
#> [70]	train-rmse:0.703349	test-rmse:3.657565 
#> [1]	train-rmse:17.431329	test-rmse:16.703654 
#> [2]	train-rmse:12.696194	test-rmse:12.218873 
#> [3]	train-rmse:9.337002	test-rmse:9.118414 
#> [4]	train-rmse:7.018587	test-rmse:7.088290 
#> [5]	train-rmse:5.415362	test-rmse:5.711857 
#> [6]	train-rmse:4.329083	test-rmse:4.831852 
#> [7]	train-rmse:3.601945	test-rmse:4.270706 
#> [8]	train-rmse:3.149155	test-rmse:4.009935 
#> [9]	train-rmse:2.825517	test-rmse:3.852722 
#> [10]	train-rmse:2.529904	test-rmse:3.700324 
#> [11]	train-rmse:2.371822	test-rmse:3.619504 
#> [12]	train-rmse:2.248299	test-rmse:3.526853 
#> [13]	train-rmse:2.148486	test-rmse:3.510621 
#> [14]	train-rmse:2.083448	test-rmse:3.522890 
#> [15]	train-rmse:2.014768	test-rmse:3.492488 
#> [16]	train-rmse:1.969852	test-rmse:3.492779 
#> [17]	train-rmse:1.919457	test-rmse:3.484013 
#> [18]	train-rmse:1.836285	test-rmse:3.480110 
#> [19]	train-rmse:1.789915	test-rmse:3.494580 
#> [20]	train-rmse:1.745980	test-rmse:3.485723 
#> [21]	train-rmse:1.688302	test-rmse:3.466251 
#> [22]	train-rmse:1.651237	test-rmse:3.448205 
#> [23]	train-rmse:1.624050	test-rmse:3.451121 
#> [24]	train-rmse:1.544799	test-rmse:3.454887 
#> [25]	train-rmse:1.508494	test-rmse:3.443285 
#> [26]	train-rmse:1.494843	test-rmse:3.448496 
#> [27]	train-rmse:1.470220	test-rmse:3.435564 
#> [28]	train-rmse:1.425408	test-rmse:3.425121 
#> [29]	train-rmse:1.380475	test-rmse:3.413315 
#> [30]	train-rmse:1.340690	test-rmse:3.408225 
#> [31]	train-rmse:1.318358	test-rmse:3.410019 
#> [32]	train-rmse:1.305773	test-rmse:3.408686 
#> [33]	train-rmse:1.281862	test-rmse:3.411578 
#> [34]	train-rmse:1.269213	test-rmse:3.401081 
#> [35]	train-rmse:1.243276	test-rmse:3.401679 
#> [36]	train-rmse:1.230148	test-rmse:3.400291 
#> [37]	train-rmse:1.212566	test-rmse:3.396719 
#> [38]	train-rmse:1.199863	test-rmse:3.398044 
#> [39]	train-rmse:1.171560	test-rmse:3.402795 
#> [40]	train-rmse:1.154470	test-rmse:3.399611 
#> [41]	train-rmse:1.125709	test-rmse:3.389426 
#> [42]	train-rmse:1.117133	test-rmse:3.384284 
#> [43]	train-rmse:1.100629	test-rmse:3.384461 
#> [44]	train-rmse:1.074666	test-rmse:3.384487 
#> [45]	train-rmse:1.048657	test-rmse:3.389365 
#> [46]	train-rmse:1.040674	test-rmse:3.390344 
#> [47]	train-rmse:1.030349	test-rmse:3.391706 
#> [48]	train-rmse:1.004669	test-rmse:3.384443 
#> [49]	train-rmse:0.996423	test-rmse:3.387312 
#> [50]	train-rmse:0.986165	test-rmse:3.379558 
#> [51]	train-rmse:0.970477	test-rmse:3.379549 
#> [52]	train-rmse:0.965665	test-rmse:3.380382 
#> [53]	train-rmse:0.948489	test-rmse:3.382174 
#> [54]	train-rmse:0.910145	test-rmse:3.388917 
#> [55]	train-rmse:0.899244	test-rmse:3.380107 
#> [56]	train-rmse:0.894504	test-rmse:3.377116 
#> [57]	train-rmse:0.888464	test-rmse:3.379834 
#> [58]	train-rmse:0.870660	test-rmse:3.377150 
#> [59]	train-rmse:0.866385	test-rmse:3.382585 
#> [60]	train-rmse:0.856161	test-rmse:3.374949 
#> [61]	train-rmse:0.850367	test-rmse:3.375091 
#> [62]	train-rmse:0.843837	test-rmse:3.378089 
#> [63]	train-rmse:0.835610	test-rmse:3.380332 
#> [64]	train-rmse:0.828715	test-rmse:3.380860 
#> [65]	train-rmse:0.821685	test-rmse:3.381212 
#> [66]	train-rmse:0.805571	test-rmse:3.384954 
#> [67]	train-rmse:0.796083	test-rmse:3.388942 
#> [68]	train-rmse:0.781292	test-rmse:3.395359 
#> [69]	train-rmse:0.768496	test-rmse:3.386272 
#> [70]	train-rmse:0.760739	test-rmse:3.386573 
#> [1]	train-rmse:17.233425	test-rmse:17.250586 
#> [2]	train-rmse:12.485570	test-rmse:12.623902 
#> [3]	train-rmse:9.195193	test-rmse:9.623004 
#> [4]	train-rmse:6.837540	test-rmse:7.513643 
#> [5]	train-rmse:5.241085	test-rmse:6.061415 
#> [6]	train-rmse:4.133397	test-rmse:5.169767 
#> [7]	train-rmse:3.391679	test-rmse:4.621963 
#> [8]	train-rmse:2.904424	test-rmse:4.311491 
#> [9]	train-rmse:2.583741	test-rmse:4.097347 
#> [10]	train-rmse:2.386201	test-rmse:3.944636 
#> [11]	train-rmse:2.250730	test-rmse:3.868993 
#> [12]	train-rmse:2.108683	test-rmse:3.764592 
#> [13]	train-rmse:2.052999	test-rmse:3.729400 
#> [14]	train-rmse:1.976534	test-rmse:3.697717 
#> [15]	train-rmse:1.921351	test-rmse:3.658235 
#> [16]	train-rmse:1.823806	test-rmse:3.602679 
#> [17]	train-rmse:1.791829	test-rmse:3.590524 
#> [18]	train-rmse:1.732673	test-rmse:3.588665 
#> [19]	train-rmse:1.691548	test-rmse:3.541515 
#> [20]	train-rmse:1.663846	test-rmse:3.533724 
#> [21]	train-rmse:1.644180	test-rmse:3.518984 
#> [22]	train-rmse:1.574901	test-rmse:3.499891 
#> [23]	train-rmse:1.559760	test-rmse:3.497970 
#> [24]	train-rmse:1.511802	test-rmse:3.470968 
#> [25]	train-rmse:1.472486	test-rmse:3.463712 
#> [26]	train-rmse:1.453882	test-rmse:3.467987 
#> [27]	train-rmse:1.425199	test-rmse:3.454385 
#> [28]	train-rmse:1.371651	test-rmse:3.477113 
#> [29]	train-rmse:1.356004	test-rmse:3.478002 
#> [30]	train-rmse:1.341601	test-rmse:3.466878 
#> [31]	train-rmse:1.324580	test-rmse:3.460670 
#> [32]	train-rmse:1.295253	test-rmse:3.451498 
#> [33]	train-rmse:1.283414	test-rmse:3.444746 
#> [34]	train-rmse:1.253856	test-rmse:3.437694 
#> [35]	train-rmse:1.236857	test-rmse:3.436742 
#> [36]	train-rmse:1.215436	test-rmse:3.425606 
#> [37]	train-rmse:1.180066	test-rmse:3.407364 
#> [38]	train-rmse:1.157078	test-rmse:3.406668 
#> [39]	train-rmse:1.141348	test-rmse:3.401793 
#> [40]	train-rmse:1.134249	test-rmse:3.398003 
#> [41]	train-rmse:1.100634	test-rmse:3.382796 
#> [42]	train-rmse:1.091744	test-rmse:3.386709 
#> [43]	train-rmse:1.061454	test-rmse:3.377536 
#> [44]	train-rmse:1.039501	test-rmse:3.379761 
#> [45]	train-rmse:1.020237	test-rmse:3.375047 
#> [46]	train-rmse:1.000813	test-rmse:3.376329 
#> [47]	train-rmse:0.977879	test-rmse:3.370643 
#> [48]	train-rmse:0.974470	test-rmse:3.368579 
#> [49]	train-rmse:0.957972	test-rmse:3.370339 
#> [50]	train-rmse:0.951925	test-rmse:3.369920 
#> [51]	train-rmse:0.935729	test-rmse:3.366010 
#> [52]	train-rmse:0.923628	test-rmse:3.372610 
#> [53]	train-rmse:0.912039	test-rmse:3.357990 
#> [54]	train-rmse:0.895802	test-rmse:3.361504 
#> [55]	train-rmse:0.870712	test-rmse:3.362466 
#> [56]	train-rmse:0.852411	test-rmse:3.359917 
#> [57]	train-rmse:0.837318	test-rmse:3.352954 
#> [58]	train-rmse:0.822292	test-rmse:3.342966 
#> [59]	train-rmse:0.812269	test-rmse:3.340620 
#> [60]	train-rmse:0.798920	test-rmse:3.341524 
#> [61]	train-rmse:0.787027	test-rmse:3.336136 
#> [62]	train-rmse:0.772426	test-rmse:3.333496 
#> [63]	train-rmse:0.756167	test-rmse:3.329261 
#> [64]	train-rmse:0.746415	test-rmse:3.332921 
#> [65]	train-rmse:0.733260	test-rmse:3.331723 
#> [66]	train-rmse:0.725389	test-rmse:3.327671 
#> [67]	train-rmse:0.719412	test-rmse:3.326824 
#> [68]	train-rmse:0.706745	test-rmse:3.326814 
#> [69]	train-rmse:0.697934	test-rmse:3.319295 
#> [70]	train-rmse:0.681594	test-rmse:3.318715 
#> [1]	train-rmse:17.128594	test-rmse:17.124916 
#> [2]	train-rmse:12.473696	test-rmse:12.576642 
#> [3]	train-rmse:9.155455	test-rmse:9.317343 
#> [4]	train-rmse:6.869103	test-rmse:7.051151 
#> [5]	train-rmse:5.318934	test-rmse:5.554716 
#> [6]	train-rmse:4.305869	test-rmse:4.620594 
#> [7]	train-rmse:3.600153	test-rmse:3.880415 
#> [8]	train-rmse:3.165971	test-rmse:3.499136 
#> [9]	train-rmse:2.890240	test-rmse:3.277098 
#> [10]	train-rmse:2.648240	test-rmse:3.197614 
#> [11]	train-rmse:2.466549	test-rmse:3.075079 
#> [12]	train-rmse:2.373009	test-rmse:3.025359 
#> [13]	train-rmse:2.295470	test-rmse:2.996508 
#> [14]	train-rmse:2.242929	test-rmse:2.987894 
#> [15]	train-rmse:2.189766	test-rmse:2.936767 
#> [16]	train-rmse:2.115994	test-rmse:2.928127 
#> [17]	train-rmse:2.047733	test-rmse:2.922381 
#> [18]	train-rmse:1.993770	test-rmse:2.893833 
#> [19]	train-rmse:1.925173	test-rmse:2.868403 
#> [20]	train-rmse:1.891927	test-rmse:2.858402 
#> [21]	train-rmse:1.855631	test-rmse:2.846958 
#> [22]	train-rmse:1.817977	test-rmse:2.817564 
#> [23]	train-rmse:1.793867	test-rmse:2.826849 
#> [24]	train-rmse:1.760676	test-rmse:2.799355 
#> [25]	train-rmse:1.733016	test-rmse:2.806548 
#> [26]	train-rmse:1.698300	test-rmse:2.790279 
#> [27]	train-rmse:1.667252	test-rmse:2.818433 
#> [28]	train-rmse:1.632844	test-rmse:2.801740 
#> [29]	train-rmse:1.584815	test-rmse:2.803550 
#> [30]	train-rmse:1.559019	test-rmse:2.791002 
#> [31]	train-rmse:1.523995	test-rmse:2.788743 
#> [32]	train-rmse:1.496033	test-rmse:2.808156 
#> [33]	train-rmse:1.480189	test-rmse:2.802236 
#> [34]	train-rmse:1.453507	test-rmse:2.801068 
#> [35]	train-rmse:1.441042	test-rmse:2.798688 
#> [36]	train-rmse:1.425898	test-rmse:2.795132 
#> [37]	train-rmse:1.380954	test-rmse:2.779995 
#> [38]	train-rmse:1.356294	test-rmse:2.787496 
#> [39]	train-rmse:1.333006	test-rmse:2.790582 
#> [40]	train-rmse:1.316990	test-rmse:2.789516 
#> [41]	train-rmse:1.291145	test-rmse:2.784462 
#> [42]	train-rmse:1.267506	test-rmse:2.772754 
#> [43]	train-rmse:1.236319	test-rmse:2.773344 
#> [44]	train-rmse:1.209988	test-rmse:2.779797 
#> [45]	train-rmse:1.181657	test-rmse:2.770077 
#> [46]	train-rmse:1.156871	test-rmse:2.777735 
#> [47]	train-rmse:1.121685	test-rmse:2.780174 
#> [48]	train-rmse:1.100434	test-rmse:2.755684 
#> [49]	train-rmse:1.082500	test-rmse:2.758530 
#> [50]	train-rmse:1.060479	test-rmse:2.755751 
#> [51]	train-rmse:1.042871	test-rmse:2.749594 
#> [52]	train-rmse:1.026156	test-rmse:2.750159 
#> [53]	train-rmse:1.014550	test-rmse:2.748956 
#> [54]	train-rmse:0.992050	test-rmse:2.740920 
#> [55]	train-rmse:0.979768	test-rmse:2.744747 
#> [56]	train-rmse:0.957779	test-rmse:2.744108 
#> [57]	train-rmse:0.936233	test-rmse:2.743346 
#> [58]	train-rmse:0.930738	test-rmse:2.744985 
#> [59]	train-rmse:0.914788	test-rmse:2.746578 
#> [60]	train-rmse:0.897444	test-rmse:2.742718 
#> [61]	train-rmse:0.881162	test-rmse:2.746288 
#> [62]	train-rmse:0.873362	test-rmse:2.739386 
#> [63]	train-rmse:0.855085	test-rmse:2.746520 
#> [64]	train-rmse:0.835583	test-rmse:2.751073 
#> [65]	train-rmse:0.825417	test-rmse:2.744734 
#> [66]	train-rmse:0.805799	test-rmse:2.757439 
#> [67]	train-rmse:0.793748	test-rmse:2.762758 
#> [68]	train-rmse:0.780451	test-rmse:2.767125 
#> [69]	train-rmse:0.774147	test-rmse:2.769935 
#> [70]	train-rmse:0.762300	test-rmse:2.769790 
#> [1]	train-rmse:17.292655	test-rmse:16.876471 
#> [2]	train-rmse:12.577736	test-rmse:12.195365 
#> [3]	train-rmse:9.241283	test-rmse:8.922933 
#> [4]	train-rmse:6.913518	test-rmse:6.724334 
#> [5]	train-rmse:5.326545	test-rmse:5.265766 
#> [6]	train-rmse:4.262574	test-rmse:4.416342 
#> [7]	train-rmse:3.480536	test-rmse:3.935688 
#> [8]	train-rmse:2.985960	test-rmse:3.632995 
#> [9]	train-rmse:2.660206	test-rmse:3.538935 
#> [10]	train-rmse:2.435057	test-rmse:3.492154 
#> [11]	train-rmse:2.264139	test-rmse:3.480262 
#> [12]	train-rmse:2.115240	test-rmse:3.447267 
#> [13]	train-rmse:1.997048	test-rmse:3.432714 
#> [14]	train-rmse:1.913323	test-rmse:3.423979 
#> [15]	train-rmse:1.836868	test-rmse:3.391161 
#> [16]	train-rmse:1.781762	test-rmse:3.377973 
#> [17]	train-rmse:1.708092	test-rmse:3.359732 
#> [18]	train-rmse:1.681469	test-rmse:3.367809 
#> [19]	train-rmse:1.625341	test-rmse:3.353789 
#> [20]	train-rmse:1.564421	test-rmse:3.327785 
#> [21]	train-rmse:1.531215	test-rmse:3.320419 
#> [22]	train-rmse:1.488833	test-rmse:3.341321 
#> [23]	train-rmse:1.456582	test-rmse:3.319002 
#> [24]	train-rmse:1.425750	test-rmse:3.304326 
#> [25]	train-rmse:1.379122	test-rmse:3.285657 
#> [26]	train-rmse:1.335484	test-rmse:3.281933 
#> [27]	train-rmse:1.307765	test-rmse:3.261889 
#> [28]	train-rmse:1.264471	test-rmse:3.255168 
#> [29]	train-rmse:1.254344	test-rmse:3.249777 
#> [30]	train-rmse:1.217107	test-rmse:3.232775 
#> [31]	train-rmse:1.201509	test-rmse:3.229382 
#> [32]	train-rmse:1.184240	test-rmse:3.222288 
#> [33]	train-rmse:1.156229	test-rmse:3.213939 
#> [34]	train-rmse:1.136092	test-rmse:3.216067 
#> [35]	train-rmse:1.110480	test-rmse:3.216788 
#> [36]	train-rmse:1.094752	test-rmse:3.206662 
#> [37]	train-rmse:1.075361	test-rmse:3.211364 
#> [38]	train-rmse:1.061166	test-rmse:3.207463 
#> [39]	train-rmse:1.044627	test-rmse:3.212024 
#> [40]	train-rmse:1.017754	test-rmse:3.214319 
#> [41]	train-rmse:0.995682	test-rmse:3.211204 
#> [42]	train-rmse:0.975897	test-rmse:3.204311 
#> [43]	train-rmse:0.969302	test-rmse:3.209325 
#> [44]	train-rmse:0.962742	test-rmse:3.213350 
#> [45]	train-rmse:0.950907	test-rmse:3.205705 
#> [46]	train-rmse:0.926037	test-rmse:3.203800 
#> [47]	train-rmse:0.913792	test-rmse:3.188461 
#> [48]	train-rmse:0.895957	test-rmse:3.184823 
#> [49]	train-rmse:0.883827	test-rmse:3.181743 
#> [50]	train-rmse:0.861841	test-rmse:3.181888 
#> [51]	train-rmse:0.850236	test-rmse:3.177201 
#> [52]	train-rmse:0.842394	test-rmse:3.176773 
#> [53]	train-rmse:0.827710	test-rmse:3.171668 
#> [54]	train-rmse:0.817296	test-rmse:3.175845 
#> [55]	train-rmse:0.812589	test-rmse:3.174930 
#> [56]	train-rmse:0.807248	test-rmse:3.178411 
#> [57]	train-rmse:0.799478	test-rmse:3.172622 
#> [58]	train-rmse:0.785276	test-rmse:3.171110 
#> [59]	train-rmse:0.780453	test-rmse:3.172351 
#> [60]	train-rmse:0.767954	test-rmse:3.173666 
#> [61]	train-rmse:0.764007	test-rmse:3.172314 
#> [62]	train-rmse:0.758229	test-rmse:3.173465 
#> [63]	train-rmse:0.747242	test-rmse:3.172619 
#> [64]	train-rmse:0.735999	test-rmse:3.170532 
#> [65]	train-rmse:0.725830	test-rmse:3.174557 
#> [66]	train-rmse:0.708749	test-rmse:3.173571 
#> [67]	train-rmse:0.689192	test-rmse:3.178361 
#> [68]	train-rmse:0.675599	test-rmse:3.176243 
#> [69]	train-rmse:0.657948	test-rmse:3.178651 
#> [70]	train-rmse:0.649430	test-rmse:3.176308 
#> [1]	train-rmse:17.544533	test-rmse:16.355680 
#> [2]	train-rmse:12.719916	test-rmse:11.847680 
#> [3]	train-rmse:9.325918	test-rmse:8.587711 
#> [4]	train-rmse:6.978470	test-rmse:6.520126 
#> [5]	train-rmse:5.330168	test-rmse:5.197999 
#> [6]	train-rmse:4.227713	test-rmse:4.370258 
#> [7]	train-rmse:3.452389	test-rmse:3.899105 
#> [8]	train-rmse:2.953947	test-rmse:3.646518 
#> [9]	train-rmse:2.621221	test-rmse:3.488626 
#> [10]	train-rmse:2.390508	test-rmse:3.434280 
#> [11]	train-rmse:2.248430	test-rmse:3.393405 
#> [12]	train-rmse:2.148335	test-rmse:3.397108 
#> [13]	train-rmse:2.062937	test-rmse:3.364962 
#> [14]	train-rmse:1.965071	test-rmse:3.319566 
#> [15]	train-rmse:1.898697	test-rmse:3.274744 
#> [16]	train-rmse:1.864320	test-rmse:3.256337 
#> [17]	train-rmse:1.837230	test-rmse:3.262007 
#> [18]	train-rmse:1.803225	test-rmse:3.235490 
#> [19]	train-rmse:1.749646	test-rmse:3.211386 
#> [20]	train-rmse:1.711134	test-rmse:3.192852 
#> [21]	train-rmse:1.666622	test-rmse:3.184116 
#> [22]	train-rmse:1.628228	test-rmse:3.183039 
#> [23]	train-rmse:1.593111	test-rmse:3.160884 
#> [24]	train-rmse:1.569067	test-rmse:3.152188 
#> [25]	train-rmse:1.538398	test-rmse:3.129954 
#> [26]	train-rmse:1.487646	test-rmse:3.142378 
#> [27]	train-rmse:1.463082	test-rmse:3.140700 
#> [28]	train-rmse:1.420317	test-rmse:3.114777 
#> [29]	train-rmse:1.403061	test-rmse:3.115688 
#> [30]	train-rmse:1.358206	test-rmse:3.101864 
#> [31]	train-rmse:1.340879	test-rmse:3.100730 
#> [32]	train-rmse:1.311146	test-rmse:3.082216 
#> [33]	train-rmse:1.290238	test-rmse:3.077993 
#> [34]	train-rmse:1.280471	test-rmse:3.072337 
#> [35]	train-rmse:1.237008	test-rmse:3.043744 
#> [36]	train-rmse:1.223663	test-rmse:3.029086 
#> [37]	train-rmse:1.214975	test-rmse:3.022611 
#> [38]	train-rmse:1.185739	test-rmse:3.017997 
#> [39]	train-rmse:1.164385	test-rmse:3.013672 
#> [40]	train-rmse:1.154593	test-rmse:3.013402 
#> [41]	train-rmse:1.139351	test-rmse:3.012511 
#> [42]	train-rmse:1.121388	test-rmse:3.008539 
#> [43]	train-rmse:1.104007	test-rmse:2.995401 
#> [44]	train-rmse:1.097245	test-rmse:2.987109 
#> [45]	train-rmse:1.077283	test-rmse:2.990181 
#> [46]	train-rmse:1.050955	test-rmse:2.983796 
#> [47]	train-rmse:1.036467	test-rmse:2.979257 
#> [48]	train-rmse:1.028401	test-rmse:2.984757 
#> [49]	train-rmse:0.994774	test-rmse:2.976397 
#> [50]	train-rmse:0.985299	test-rmse:2.964851 
#> [51]	train-rmse:0.977850	test-rmse:2.957399 
#> [52]	train-rmse:0.956651	test-rmse:2.944479 
#> [53]	train-rmse:0.949326	test-rmse:2.947867 
#> [54]	train-rmse:0.927821	test-rmse:2.952446 
#> [55]	train-rmse:0.908993	test-rmse:2.944909 
#> [56]	train-rmse:0.884882	test-rmse:2.935588 
#> [57]	train-rmse:0.873803	test-rmse:2.938716 
#> [58]	train-rmse:0.845808	test-rmse:2.936628 
#> [59]	train-rmse:0.840582	test-rmse:2.937588 
#> [60]	train-rmse:0.831460	test-rmse:2.938824 
#> [61]	train-rmse:0.813577	test-rmse:2.944781 
#> [62]	train-rmse:0.800639	test-rmse:2.936827 
#> [63]	train-rmse:0.790201	test-rmse:2.933134 
#> [64]	train-rmse:0.776988	test-rmse:2.928452 
#> [65]	train-rmse:0.757110	test-rmse:2.933903 
#> [66]	train-rmse:0.743001	test-rmse:2.931065 
#> [67]	train-rmse:0.735839	test-rmse:2.929767 
#> [68]	train-rmse:0.723563	test-rmse:2.927346 
#> [69]	train-rmse:0.713754	test-rmse:2.925463 
#> [70]	train-rmse:0.698813	test-rmse:2.927692 
#> [1]	train-rmse:16.674860	test-rmse:18.015271 
#> [2]	train-rmse:12.063769	test-rmse:13.222241 
#> [3]	train-rmse:8.811503	test-rmse:9.994677 
#> [4]	train-rmse:6.582876	test-rmse:7.760594 
#> [5]	train-rmse:5.018353	test-rmse:6.298805 
#> [6]	train-rmse:3.953114	test-rmse:5.304116 
#> [7]	train-rmse:3.254362	test-rmse:4.729481 
#> [8]	train-rmse:2.794439	test-rmse:4.387156 
#> [9]	train-rmse:2.501569	test-rmse:4.179851 
#> [10]	train-rmse:2.274560	test-rmse:4.039873 
#> [11]	train-rmse:2.137563	test-rmse:3.954427 
#> [12]	train-rmse:2.045113	test-rmse:3.914518 
#> [13]	train-rmse:1.954443	test-rmse:3.868235 
#> [14]	train-rmse:1.882999	test-rmse:3.860332 
#> [15]	train-rmse:1.804658	test-rmse:3.836931 
#> [16]	train-rmse:1.754922	test-rmse:3.828111 
#> [17]	train-rmse:1.716032	test-rmse:3.805052 
#> [18]	train-rmse:1.650214	test-rmse:3.778186 
#> [19]	train-rmse:1.600299	test-rmse:3.757008 
#> [20]	train-rmse:1.553732	test-rmse:3.739166 
#> [21]	train-rmse:1.525169	test-rmse:3.731639 
#> [22]	train-rmse:1.489868	test-rmse:3.712193 
#> [23]	train-rmse:1.442279	test-rmse:3.697139 
#> [24]	train-rmse:1.414500	test-rmse:3.692106 
#> [25]	train-rmse:1.365578	test-rmse:3.680468 
#> [26]	train-rmse:1.347896	test-rmse:3.673943 
#> [27]	train-rmse:1.332444	test-rmse:3.672703 
#> [28]	train-rmse:1.305014	test-rmse:3.661026 
#> [29]	train-rmse:1.277232	test-rmse:3.662294 
#> [30]	train-rmse:1.248798	test-rmse:3.650389 
#> [31]	train-rmse:1.241240	test-rmse:3.651887 
#> [32]	train-rmse:1.217933	test-rmse:3.639778 
#> [33]	train-rmse:1.206731	test-rmse:3.638429 
#> [34]	train-rmse:1.187312	test-rmse:3.635264 
#> [35]	train-rmse:1.164374	test-rmse:3.632264 
#> [36]	train-rmse:1.131843	test-rmse:3.640556 
#> [37]	train-rmse:1.124082	test-rmse:3.633127 
#> [38]	train-rmse:1.110077	test-rmse:3.630717 
#> [39]	train-rmse:1.100095	test-rmse:3.631217 
#> [40]	train-rmse:1.081659	test-rmse:3.635696 
#> [41]	train-rmse:1.072853	test-rmse:3.632723 
#> [42]	train-rmse:1.055953	test-rmse:3.633422 
#> [43]	train-rmse:1.031922	test-rmse:3.637305 
#> [44]	train-rmse:1.023225	test-rmse:3.625068 
#> [45]	train-rmse:1.006294	test-rmse:3.620948 
#> [46]	train-rmse:0.994529	test-rmse:3.617117 
#> [47]	train-rmse:0.979627	test-rmse:3.611547 
#> [48]	train-rmse:0.971881	test-rmse:3.608559 
#> [49]	train-rmse:0.965366	test-rmse:3.604040 
#> [50]	train-rmse:0.954073	test-rmse:3.605317 
#> [51]	train-rmse:0.931940	test-rmse:3.602057 
#> [52]	train-rmse:0.920522	test-rmse:3.596360 
#> [53]	train-rmse:0.916464	test-rmse:3.593168 
#> [54]	train-rmse:0.898960	test-rmse:3.587739 
#> [55]	train-rmse:0.873178	test-rmse:3.578689 
#> [56]	train-rmse:0.863999	test-rmse:3.576982 
#> [57]	train-rmse:0.844966	test-rmse:3.583278 
#> [58]	train-rmse:0.833886	test-rmse:3.580366 
#> [59]	train-rmse:0.818516	test-rmse:3.577260 
#> [60]	train-rmse:0.805208	test-rmse:3.577898 
#> [61]	train-rmse:0.797068	test-rmse:3.577023 
#> [62]	train-rmse:0.785239	test-rmse:3.568169 
#> [63]	train-rmse:0.776081	test-rmse:3.572902 
#> [64]	train-rmse:0.756783	test-rmse:3.572786 
#> [65]	train-rmse:0.737046	test-rmse:3.571570 
#> [66]	train-rmse:0.729238	test-rmse:3.572501 
#> [67]	train-rmse:0.723132	test-rmse:3.578412 
#> [68]	train-rmse:0.710213	test-rmse:3.568142 
#> [69]	train-rmse:0.703534	test-rmse:3.568780 
#> [70]	train-rmse:0.688026	test-rmse:3.571313
#> [1] 3.486078
```

``` r
warnings() # no warnings for individual XGBoost function
```
