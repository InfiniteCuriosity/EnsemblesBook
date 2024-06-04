# Classification data: How to make 14 individual classification models

Ensembles of numerical data give results that are often superior to any individual model. We've seen results where ensembles of numerical data beat nearly all of the individual models, as measures by the lowest error rate.

Now we are going to do the same with classification data. We will build 15 individual models of classification data in this chapter.

The basic series of steps is the same as with numerical data. We will follow the same steps, but we will complete the process with models for classification data.

> Load the library
>
> Set initial values to 0
>
> Create the function
>
> Break the data into train and test sets
>
> Set up random resampling
>
> Fit the model on the training data, make predictions and measure error on the test data
>
> Return the results
>
> Check for errors or warnings
>
> Test on a different data set

All our models will be structured in a way that is as close to identical as possible. A very high level of consistency makes it easier to spot errors.

### What is classification data?

[Classification models](https://en.wikipedia.org/wiki/Statistical_classification) are a set of models to identify the class of a specific observation. For example, in the Carseats data set, the Shelve Location is an example of this data type:


```r
library(ISLR)
head(Carseats)
#>   Sales CompPrice Income Advertising Population Price
#> 1  9.50       138     73          11        276   120
#> 2 11.22       111     48          16        260    83
#> 3 10.06       113     35          10        269    80
#> 4  7.40       117    100           4        466    97
#> 5  4.15       141     64           3        340   128
#> 6 10.81       124    113          13        501    72
#>   ShelveLoc Age Education Urban  US
#> 1       Bad  42        17   Yes Yes
#> 2      Good  65        10   Yes Yes
#> 3    Medium  59        12   Yes Yes
#> 4    Medium  55        14   Yes Yes
#> 5       Bad  38        13   Yes  No
#> 6       Bad  78        16    No Yes
```

When we look at the ShelveLoc column, it is not a set of numbers, but one of three locations on the shelf: Bad, Medium or Good. Classification models are statistical models to predict the class of the data.

Classification models are very similar to numerical models. We will follow the same basic steps as the numerical models, but use classification models instead.

One big difference is how accuracy is measured in classification data. For numerical models we used root mean squared error. Our measure for classification will simply be the accuracy of the result. This can be measured directly from a matrix of values. For example:

C50 test results:

![C50 table example](Images/C50_table.png)

The accuracy is determined by calculating the number of correct responses divided by the total number of responses. The correct responses are along the main diagonal.

For example, the model was correct when it predicted 83 bad responses, however it also predicted the response was bad when it was actually Good, and it also predicted Bad when it was actually Medium. The correct responses are down the main diagonal, the other responses are errors. We will use these in our calculation of accuracy of the model results.

In this case, that is (83 + 75 + 272) / (83 + 1 + 105 + 4 + 75 + 56 + 106 + 90 + 272) = 0.5429293. We will create a function to calculate this result automatically for each classification model. Clearly the higher the accuracy, the better the results. The best possible accuracy result is 1.00.

### Build our first classification model from the structure:


```r

# Load libraries

# Set initial values to 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){

# Change target column name to y

# Set up resamples

# Randomize the rows

# Split data into train and test sets

# Fit the model on the training data

# Check accuracy and make predictions from the model, applied to the test data

# Calculate overall model accuracy

# Calculate table
# Print table results

# Return accuracy results

# Closing braces for numresamples loop
# Closing brace for classification1 function

# Test the function

# Check for errors
```

Now that we have the structure, let's build the model:


```r

# Load libraries
library(C50)
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
#> ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

# Set initial values to 0:
C50_train_accuracy <- 0
C50_test_accuracy <- 0
C50_validation_accuracy <- 0
C50_overfitting <- 0
C50_holdout <- 0
C50_table_total <- 0

# Build the function:
C50_1 <- function(data, colnum, numresamples, train_amount, test_amount){

# Changes target column name to y
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

# Calculate table
C50_table <- C50_test_table
C50_table_total <- C50_table_total + C50_table

print(C50_table_total)

# Return accuracy result
return(c(C50_holdout_mean))

} # Closing braces for numresamples loop
} # Closing brace for classification1 function

C50_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>              y_test
#> C50_test_pred Bad Good Medium
#>        Bad     11    2     10
#>        Good     1   12     10
#>        Medium  15    8     39
#> [1] 0.5740741
```

Now that we see how one individual classification model is made, let's make 11 more (total of 12).

### Adabag for classification data


```r

# Load libraries
library(ipred)

# Set initial values to 0
adabag_train_accuracy <- 0
adabag_test_accuracy <- 0
adabag_validation_accuracy <- 0
adabag_holdout <- 0
adabag_table_total <- 0

# Build the function
adabag_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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
adabag_train_fit <- ipred::bagging(formula = y ~ ., data = train01)
adabag_train_pred <- predict(object = adabag_train_fit, newdata = train)
adabag_train_table <- table(adabag_train_pred, y_train)
adabag_train_accuracy[i] <- sum(diag(adabag_train_table)) / sum(adabag_train_table)
adabag_train_accuracy_mean <- mean(adabag_train_accuracy)
  
# Check accuracy and make predictions from the model, applied to the test data
adabag_test_pred <- predict(object = adabag_train_fit, newdata = test01)
adabag_test_table <- table(adabag_test_pred, y_test)
adabag_test_accuracy[i] <- sum(diag(adabag_test_table)) / sum(adabag_test_table)
adabag_test_accuracy_mean <- mean(adabag_test_accuracy)
adabag_test_mean <- mean(diag(adabag_test_table)) / mean(adabag_test_table)

# Calculate accuracy
adabag_holdout[i] <- mean(c(adabag_test_accuracy_mean))
adabag_holdout_mean <- mean(adabag_holdout)

# Calculate table
adabag_table <- adabag_test_table
adabag_table_total <- adabag_table_total + adabag_table

print(adabag_table_total)

# Return accuracy results
return(adabag_holdout_mean)

} # Closing braces for numresamples loop

} # Closing brace for classification1 function

# Test the function
adabag_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>                 y_test
#> adabag_test_pred Bad Good Medium
#>           Bad     12    1      6
#>           Good     0   11      5
#>           Medium  14    9     44
#> [1] 0.6568627

# Check for errors
warnings()
```

### Bagged Random Forest


```r

# Load libraries
library(randomForest)
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

# Set initial values to 0
bag_rf_train_accuracy <- 0
bag_rf_test_accuracy <- 0
bag_rf_validation_accuracy <- 0
bag_rf_overfitting <- 0
bag_rf_holdout <- 0
bag_rf_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
bag_rf_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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

# Print table results
print(bag_rf_table_total)

# Return accuracy results
return(bag_rf_holdout_mean)

}# Closing braces for numresamples loop
}# Closing brace for classification1 function

# Test the function
bag_rf_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>                 y_test
#> bag_rf_test_pred Bad Good Medium
#>           Bad     11    0      2
#>           Good     0   11      5
#>           Medium  19    9     37
#> [1] 0.6276596

# Check for errors
warnings()
```

### Linear model


```r

# Load libraries
library(MachineShop)
#> 
#> Attaching package: 'MachineShop'
#> The following object is masked from 'package:purrr':
#> 
#>     lift
#> The following object is masked from 'package:stats':
#> 
#>     ppr

# Set initial values to 0
linear_train_accuracy <- 0
linear_validation_accuracy <- 0
linear_test_accuracy <- 0
linear_test_accuracy_mean <- 0
linear_holdout <- 0
linear_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
linear1 <- function(data, colnum, numresamples, train_amount, test_amount){

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
linear_train_fit <- MachineShop::fit(y ~ ., data = train01, model = "LMModel")
linear_train_pred <- predict(object = linear_train_fit, newdata = train01)
linear_train_table <- table(linear_train_pred, y_train)
linear_train_accuracy[i] <- sum(diag(linear_train_table)) / sum(linear_train_table)
linear_train_accuracy_mean <- mean(linear_train_accuracy)
linear_train_mean <- mean(diag(linear_train_table)) / mean(linear_train_table)

# Check accuracy and make predictions from the model, applied to the test data
linear_test_pred <- predict(object = linear_train_fit, newdata = test01)
linear_test_table <- table(linear_test_pred, y_test)
linear_test_accuracy[i] <- sum(diag(linear_test_table)) / sum(linear_test_table)
linear_test_accuracy_mean <- mean(linear_test_accuracy)

# Calculate overall model accuracy
linear_holdout[i] <- mean(c(linear_test_accuracy_mean))
linear_holdout_mean <- mean(linear_holdout)

# Calculate table
linear_table <- linear_test_table
linear_table_total <- linear_table_total + linear_table

# Print table results
print(linear_table_total)

# Return accuracy results
return(linear_holdout_mean)

}# Closing braces for numresamples loop
}# Closing brace for classification1 function

# Test the function
linear1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>                 y_test
#> linear_test_pred Bad Good Medium
#>           Bad      5    0      0
#>           Good     0   11      1
#>           Medium  30    9     45
#> [1] 0.6039604

# Check for errors
warnings()
```

### Naive Bayes model


```r

# Load libraries
library(e1071)

# Set initial values to 0
n_bayes_train_accuracy <- 0
n_bayes_test_accuracy <- 0
n_bayes_accuracy <- 0
n_bayes_test_accuracy_mean <- 0
n_bayes_holdout <- 0
n_bayes_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
n_bayes_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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
n_bayes_train_fit <- e1071::naiveBayes(y_train ~ ., data = train)
n_bayes_train_pred <- predict(n_bayes_train_fit, train)
n_bayes_train_table <- table(n_bayes_train_pred, y_train)
n_bayes_train_accuracy[i] <- sum(diag(n_bayes_train_table)) / sum(n_bayes_train_table)
n_bayes_train_accuracy_mean <- mean(n_bayes_train_accuracy)

# Check accuracy and make predictions from the model, applied to the test data
n_bayes_test_pred <- predict(n_bayes_train_fit, test)
n_bayes_test_table <- table(n_bayes_test_pred, y_test)
n_bayes_test_accuracy[i] <- sum(diag(n_bayes_test_table)) / sum(n_bayes_test_table)
n_bayes_test_accuracy_mean <- mean(n_bayes_test_accuracy)

# Calculate overall model accuracy
n_bayes_holdout[i] <- mean(c(n_bayes_test_accuracy_mean))
n_bayes_holdout_mean <- mean(n_bayes_holdout)

# Calculate table
n_bayes_table <- n_bayes_test_table
n_bayes_table_total <- n_bayes_table_total + n_bayes_table

# Print table results
print(n_bayes_table_total)

# Return accuracy results
return(n_bayes_holdout_mean)

} # Closing braces for numresamples loop
} # Closing brace for classification1 function

# Test the function
n_bayes_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>                  y_test
#> n_bayes_test_pred Bad Good Medium
#>            Bad      5    0      3
#>            Good     0    6      5
#>            Medium  21   11     39
#> [1] 0.5555556

# Check for errors
warnings()
```

### Partial Least Squares


```r

# Load libraries
library(MachineShop)

# Set initial values to 0
pls_train_accuracy <- 0
pls_test_accuracy <- 0
pls_accuracy <- 0
pls_test_accuracy_mean <- 0
pls_holdout <- 0
pls_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
pls_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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
pls_train_fit <- MachineShop::fit(y ~ ., data = train01, model = "PLSModel")
pls_train_predict <- predict(object = pls_train_fit, newdata = train01)
pls_train_table <- table(pls_train_predict, y_train)
pls_train_accuracy[i] <- sum(diag(pls_train_table)) / sum(pls_train_table)
pls_train_accuracy_mean <- mean(pls_train_accuracy)

# Check accuracy and make predictions from the model, applied to the test data
pls_test_predict <- predict(object = pls_train_fit, newdata = test01)
pls_test_table <- table(pls_test_predict, y_test)
pls_test_accuracy[i] <- sum(diag(pls_test_table)) / sum(pls_test_table)
pls_test_accuracy_mean <- mean(pls_test_accuracy)
pls_test_pred <- pls_test_predict

# Calculate overall model accuracy
pls_holdout[i] <- mean(c(pls_test_accuracy_mean))
pls_holdout_mean <- mean(pls_holdout)

# Calculate table
pls_table <- pls_test_table
pls_table_total <- pls_table_total + pls_table

# Print table results
print(pls_table_total)

# Return accuracy results
return(pls_holdout_mean)

} # Closing braces for numresamples loop
} # Closing brace for classification1 function

# Test the function
pls_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>                 y_test
#> pls_test_predict Bad Good Medium
#>           Bad      0    0      0
#>           Good     0    0      0
#>           Medium  24   18     47
#> [1] 0.5280899

# Check for errors
warnings()
```

### Penalized Discriminant Analysis Model


```r

# Load libraries
library(MachineShop)

# Set initial values to 0
pda_train_accuracy <- 0
pda_test_accuracy <- 0
pda_accuracy <- 0
pda_test_accuracy_mean <- 0
pda_holdout <- 0
pda_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
pda_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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
pda_train_fit <- MachineShop::fit(y ~ ., data = train01, model = "PDAModel")
pda_train_predict <- predict(object = pda_train_fit, newdata = train01)
pda_train_table <- table(pda_train_predict, y_train)
pda_train_accuracy[i] <- sum(diag(pda_train_table)) / sum(pda_train_table)
pda_train_accuracy_mean <- mean(pda_train_accuracy)

# Check accuracy and make predictions from the model, applied to the test data
pda_test_predict <- predict(object = pda_train_fit, newdata = test01)
pda_test_table <- table(pda_test_predict, y_test)
pda_test_accuracy[i] <- sum(diag(pda_test_table)) / sum(pda_test_table)
pda_test_accuracy_mean <- mean(pda_test_accuracy)
pda_test_pred <- pda_test_predict

# Calculate overall model accuracy
pda_holdout[i] <- mean(c(pda_test_accuracy_mean))
pda_holdout_mean <- mean(pda_holdout)

# Calculate table
pda_table <- pda_test_table
pda_table_total <- pda_table_total + pda_table

# Print table results
print(pda_table_total)

# Return accuracy results
return(pda_holdout_mean)

} # Closing braces for numresamples loop
} # Closing brace for classification1 function

# Test the function
pda_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>                 y_test
#> pda_test_predict Bad Good Medium
#>           Bad     12    0      6
#>           Good     0   17      4
#>           Medium   7    2     42
#> [1] 0.7888889

# Check for errors
warnings()
```

### Random Forest


```r

# Load libraries
library(randomForest)

# Set initial values to 0
rf_train_accuracy <- 0
rf_test_accuracy <- 0
rf_accuracy <- 0
rf_test_accuracy_mean <- 0
rf_holdout <- 0
rf_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
rf_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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
rf_train_fit <- randomForest::randomForest(x = train, y = y_train, data = df)
rf_train_pred <- predict(rf_train_fit, train, type = "class")
rf_train_table <- table(rf_train_pred, y_train)
rf_train_accuracy[i] <- sum(diag(rf_train_table)) / sum(rf_train_table)
rf_train_accuracy_mean <- mean(rf_train_accuracy)

# Check accuracy and make predictions from the model, applied to the test data
rf_test_pred <- predict(rf_train_fit, test, type = "class")
rf_test_table <- table(rf_test_pred, y_test)
rf_test_accuracy[i] <- sum(diag(rf_test_table)) / sum(rf_test_table)
rf_test_accuracy_mean <- mean(rf_test_accuracy)

# Calculate overall model accuracy
rf_holdout[i] <- mean(c(rf_test_accuracy_mean))
rf_holdout_mean <- mean(rf_holdout)

# Calculate table
rf_table <- rf_test_table
rf_table_total <- rf_table_total + rf_table

# Print table results
print(rf_table_total)

# Return accuracy results
return(rf_holdout_mean)

} # Closing braces for numresamples loop
} # Closing brace for classification1 function

# Test the function
rf_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>             y_test
#> rf_test_pred Bad Good Medium
#>       Bad      5    0      2
#>       Good     1   12      0
#>       Medium  24   10     41
#> [1] 0.6105263

# Check for errors
warnings()
```

### Ranger


```r

# Load libraries
library(MachineShop)

# Set initial values to 0
ranger_train_accuracy <- 0
ranger_test_accuracy <- 0
ranger_accuracy <- 0
ranger_test_accuracy_mean <- 0
ranger_holdout <- 0
ranger_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
ranger_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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

# Calculate table
ranger_table <- ranger_test_table
ranger_table_total <- ranger_table_total + ranger_table

# Print table results
print(ranger_table_total)

# Return accuracy results
return(ranger_holdout_mean)

} # Closing braces for numresamples loop
} # Closing brace for classification1 function

# Test the function
ranger_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>                    y_test
#> ranger_test_predict Bad Good Medium
#>              Bad      8    0      2
#>              Good     0    9      4
#>              Medium  21   19     54
#> [1] 0.6068376

# Check for errors
warnings()
```

### Regularized Discriminant Analysis


```r

# Load libraries
library(klaR)
#> Loading required package: MASS
#> 
#> Attaching package: 'MASS'
#> The following object is masked from 'package:dplyr':
#> 
#>     select

# Set initial values to 0
rda_train_accuracy <- 0
rda_test_accuracy <- 0
rda_accuracy <- 0
rda_test_accuracy_mean <- 0
rda_holdout <- 0
rda_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
rda_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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
rda_train_fit <- klaR::rda(y_train ~ ., data = train)
rda_train_pred <- predict(object = rda_train_fit, newdata = train)
rda_train_table <- table(rda_train_pred$class, y_train)
rda_train_accuracy[i] <- sum(diag(rda_train_table)) / sum(rda_train_table)
rda_train_accuracy_mean <- mean(rda_train_accuracy)

# Check accuracy and make predictions from the model, applied to the test data
rda_test_pred <- predict(object = rda_train_fit, newdata = test)
rda_test_table <- table(rda_test_pred$class, y_test)
rda_test_accuracy[i] <- sum(diag(rda_test_table)) / sum(rda_test_table)
rda_test_accuracy_mean <- mean(rda_test_accuracy)

# Calculate overall model accuracy
rda_holdout[i] <- mean(c(rda_test_accuracy_mean))
rda_holdout_mean <- mean(rda_holdout)

# Calculate table
rda_table <- rda_test_table
rda_table_total <- rda_table_total + rda_table

# Print table results
print(rda_table_total)

# Return accuracy results
return(rda_holdout_mean)

} # Closing braces for numresamples loop
} # Closing brace for classification1 function

# Test the function
rda_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>         y_test
#>          Bad Good Medium
#>   Bad      0    0      0
#>   Good     0    0      0
#>   Medium  19   20     48
#> [1] 0.5517241

# Check for errors
warnings()
```

### Rpart


```r

# Load libraries
library(MachineShop)

# Set initial values to 0
rpart_train_accuracy <- 0
rpart_test_accuracy <- 0
rpart_accuracy <- 0
rpart_test_accuracy_mean <- 0
rpart_holdout <- 0
rpart_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
rpart_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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
rpart_train_fit <- MachineShop::fit(y ~ ., data = train01, model = "RPartModel")
rpart_train_predict <- predict(object = rpart_train_fit, newdata = train01)
rpart_train_table <- table(rpart_train_predict, y_train)
rpart_train_accuracy[i] <- sum(diag(rpart_train_table)) / sum(rpart_train_table)
rpart_train_accuracy_mean <- mean(rpart_train_accuracy)

# Check accuracy and make predictions from the model, applied to the test data
rpart_test_predict <- predict(object = rpart_train_fit, newdata = test01)
rpart_test_table <- table(rpart_test_predict, y_test)
rpart_test_accuracy[i] <- sum(diag(rpart_test_table)) / sum(rpart_test_table)
rpart_test_accuracy_mean <- mean(rpart_test_accuracy)
rpart_test_pred <- rpart_test_predict

# Calculate overall model accuracy
rpart_holdout[i] <- mean(c(rpart_test_accuracy_mean))
rpart_holdout_mean <- mean(rpart_holdout)

# Calculate table
rpart_table <- rpart_test_table
rpart_table_total <- rpart_table_total + rpart_table
  
# Print table results
print(rpart_table_total)
  
# Return accuracy results
return(rpart_holdout_mean)

} # Closing braces for numresamples loop
} # Closing brace for classification1 function

# Test the function
rpart_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>                   y_test
#> rpart_test_predict Bad Good Medium
#>             Bad     12    0      9
#>             Good     0   13      6
#>             Medium  12   17     38
#> [1] 0.588785

# Check for errors
warnings()
```

### Support Vector Machines


```r

# Load libraries
library(e1071)

# Set initial values to 0
svm_train_accuracy <- 0
svm_test_accuracy <- 0
svm_accuracy <- 0
svm_test_accuracy_mean <- 0
svm_holdout <- 0
svm_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
svm_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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

# Calculate table
svm_table <- svm_test_table
svm_table_total <- svm_table_total + svm_table

# Print table results
print(svm_table_total)

# Return accuracy results
return(svm_holdout_mean)

} # Closing braces for numresamples loop
} # Closing brace for classification1 function

# Test the function
svm_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>              y_test
#> svm_test_pred Bad Good Medium
#>        Bad      0    0      0
#>        Good     0    0      0
#>        Medium  20   29     59
#> [1] 0.5462963

# Check for errors
warnings()
```

### Trees


```r

# Load libraries
library(tree)

# Set initial values to 0
tree_train_accuracy <- 0
tree_test_accuracy <- 0
tree_accuracy <- 0
tree_test_accuracy_mean <- 0
tree_holdout <- 0
tree_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
tree_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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
tree_train_fit <- tree::tree(y_train ~ ., data = train)
tree_train_pred <- predict(tree_train_fit, train, type = "class")
tree_train_table <- table(tree_train_pred, y_train)
tree_train_accuracy[i] <- sum(diag(tree_train_table)) / sum(tree_train_table)
tree_train_accuracy_mean <- mean(tree_train_accuracy)

# Check accuracy and make predictions from the model, applied to the test data
tree_test_pred <- predict(tree_train_fit, test, type = "class")
tree_test_table <- table(tree_test_pred, y_test)
tree_test_accuracy[i] <- sum(diag(tree_test_table)) / sum(tree_test_table)
tree_test_accuracy_mean <- mean(tree_test_accuracy)

# Calculate overall model accuracy
tree_holdout[i] <- mean(c(tree_test_accuracy_mean))
tree_holdout_mean <- mean(tree_holdout)

# Calculate table
tree_table <- tree_test_table
tree_table_total <- tree_table_total + tree_table  

# Print table results
print(tree_table_total)

# Return accuracy results
return(tree_holdout_mean)

} # Closing braces for numresamples loop
} # Closing brace for classification1 function

# Test the function
tree_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>               y_test
#> tree_test_pred Bad Good Medium
#>         Bad      8    0     10
#>         Good     0   10      6
#>         Medium  15   13     44
#> [1] 0.5849057

# Check for errors
warnings()
```

### XGBoost


```r

# Load libraries
library(xgboost)
#> 
#> Attaching package: 'xgboost'
#> The following object is masked from 'package:dplyr':
#> 
#>     slice

# Set initial values to 0
xgb_train_accuracy <- 0
xgb_test_accuracy <- 0
xgb_accuracy <- 0
xgb_test_accuracy_mean <- 0
xgb_holdout <- 0
xgb_table_total <- 0

# Build the function, functionname <- function(data, colnum, numresamples, train_amount, test_amount){
xgb_1 <- function(data, colnum, numresamples, train_amount, test_amount){

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

# Calculate table
xgb_table <- xgb_test_table
xgb_table_total <- xgb_table_total + xgb_table

# Print table results
print(xgb_table_total)

# Return accuracy results
return(xgb_holdout_mean)

} # Closing braces for numresamples loop
} # Closing brace for classification1 function

# Test the function
xgb_1(data = ISLR::Carseats, colnum = 7, numresamples = 5, train_amount = 0.60, test_amount = 0.20)
#>         
#>          Bad Good Medium
#>   Bad      7    0      3
#>   Good     1   10      5
#>   Medium  20   15     42
#> [1] 0.5728155

# Check for errors
warnings()
```

### Post your results
