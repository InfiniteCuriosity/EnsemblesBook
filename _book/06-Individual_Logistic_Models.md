# Individual logistic models

Logistic data sets are extremely powerful. In this chapter we'll use them to evaluate the risk of type 2 diabetes in Pima Indian women, and in the logistic ensembles chapter we'll use it to make recommendations to improve the performance of Lebron James.

That raises a very good question: How can two fields as far apart as scientific research (Pima Indians) and sports analytics (Lebron) be connected? It's because the structure of the data is the same, and it's the structure of the data that makes it easy to use.

Logistic regression is rooted in the idea of a logical variable (hence the name). A variable is a logical variable if there are a specific number of options, usually two options. They have many possible names which all come down to the same result. Names might include true or false, presence or absence of a condition (such as diabetes), or success or failure of making a basket.

In logistic modeling those values are converted to 1 or 0 (if they are not converted already). Let's start by getting the Pima Indians data set from the Kaggle web site:

<https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database>

Download the data set, and open it in your system. For example,


``` r

diabetes <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/EnsemblesData/main/diabetes.csv')
head(diabetes)
#>   Pregnancies Glucose BloodPressure SkinThickness Insulin
#> 1           6     148            72            35       0
#> 2           1      85            66            29       0
#> 3           8     183            64             0       0
#> 4           1      89            66            23      94
#> 5           0     137            40            35     168
#> 6           5     116            74             0       0
#>    BMI DiabetesPedigreeFunction Age Outcome
#> 1 33.6                    0.627  50       1
#> 2 26.6                    0.351  31       0
#> 3 23.3                    0.672  32       1
#> 4 28.1                    0.167  21       0
#> 5 43.1                    2.288  33       1
#> 6 25.6                    0.201  30       0
```

We can clearly see the eight features which will be used to predict the ninth feature, Outcome. Our final logistic model will be in the form of: Outcome \~ ., data = df.

By far the most common way to do this is using Generalized Linear Models, and we will begin there. We will follow our well established method for building a model, which we used with numerical and classification data:

> Load the library
>
> Set initial values to 0
>
> Create the function
>
> Set up random resampling
>
> Break the data into train and test
>
> Fit the model on the training data, make predictions and measure error on the test data
>
> Return the results
>
> Check for errors or warnings
>
> Test on a different data set

For this set of examples we are also going to add the results of the ROC curve, so the ROC curve is printed automatically.


``` r

# Load the library
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
```

``` r
library(pROC)
#> Type 'citation("pROC")' for a citation.
#> 
#> Attaching package: 'pROC'
#> 
#> The following objects are masked from 'package:stats':
#> 
#>     cov, smooth, var
```

``` r

# Set initial values to 0

glm_train_accuracy <- 0
glm_test_accuracy <- 0
glm_holdout_accuracy <- 0
glm_duration <- 0
glm_table_total <- 0

# Create the function

glm_1 <- function(data, colnum, numresamples, train_amount, test_amount){
  
colnames(data)[colnum] <- "y"

df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right

df <- df[sample(1:nrow(df)), ] # randomizes the rows
  
# Set up random resampling

for (i in 1:numresamples) {
  
index <- sample(c(1:2), nrow(df), replace = TRUE, prob = c(train_amount, test_amount))

train <- df[index == 1, ]
test <- df[index == 2, ]

y_train <- train$y
y_test <- test$y

# Fit the model to the training data, make predictions on the holdout data

glm_train_fit <- stats::glm(y ~ ., data = train, family = binomial(link = "logit"))

glm_train_pred <- stats::predict(glm_train_fit, train, type = "response")
glm_train_predictions <- ifelse(glm_train_pred > 0.5, 1, 0)
glm_train_table <- table(glm_train_predictions, y_train)
glm_train_accuracy[i] <- (glm_train_table[1, 1] + glm_train_table[2, 2]) / sum(glm_train_table)
glm_train_accuracy_mean <- mean(glm_train_accuracy)

glm_test_pred <- stats::predict(glm_train_fit, test, type = "response")
glm_test_predictions <- ifelse(glm_test_pred > 0.5, 1, 0)
glm_test_table <- table(glm_test_predictions, y_test)
glm_test_accuracy[i] <- (glm_test_table[1, 1] + glm_test_table[2, 2]) / sum(glm_test_table)
glm_test_accuracy_mean <- mean(glm_test_accuracy)

glm_holdout_accuracy_mean <- mean(glm_test_accuracy)

glm_roc_obj <- pROC::roc(as.numeric(c(test$y)), c(glm_test_pred))
glm_auc <- round((pROC::auc(c(test$y), as.numeric(c(glm_test_pred)) - 1)), 4)
print(pROC::ggroc(glm_roc_obj, color = "steelblue", size = 2) +
  ggplot2::ggtitle(paste0("Generalized Linear Models ", "(AUC = ", glm_auc, ")")) +
  ggplot2::labs(x = "Specificity", y = "Sensitivity") +
  ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
)

return(glm_holdout_accuracy_mean)

} # closing brace for numresamples 
  
} # Closing brace for the function

# Test the function
glm_1(data = diabetes, colnum = 9, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="06-Individual_Logistic_Models_files/figure-html/Our first logistic model-1.png" width="672" />

```
#> [1] 0.7718631
```

``` r

# Check for any errors
warnings()
```

The results are consistent with similar results using Generalized Linear Models for this data set.

### How to use non-GLM models in logistic analysis

The authors of the excellent book, Introduction to Statistical Learning, describe and demonstrate how non-GLM methods may be used in logistic analysis. They investigated Linear Discriminant Analysis, Quadratic Discriminant Analysis and K-Nearest Neighbors. We will look at a total of ten methods, though many more are possible.

![Logistic warning](Images/GLM_warning.jpg)

### Eight individual models for logistic data

### Adaboost


``` r

# Load the library
library(MachineShop)
#> 
#> Attaching package: 'MachineShop'
#> The following object is masked from 'package:pROC':
#> 
#>     auc
#> The following object is masked from 'package:purrr':
#> 
#>     lift
#> The following object is masked from 'package:stats':
#> 
#>     ppr
```

``` r
library(tidyverse)
library(pROC)

# Set initial values to 0

adaboost_train_accuracy <- 0
adaboost_test_accuracy <- 0
adaboost_holdout_accuracy <- 0
adaboost_duration <- 0
adaboost_table_total <- 0

# Create the function

adaboost_1 <- function(data, colnum, numresamples, train_amount, test_amount){
  
  colnames(data)[colnum] <- "y"
  
  df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
  
  df <- df[sample(1:nrow(df)), ] # randomizes the rows
  
  # Set up random resampling
  
  for (i in 1:numresamples) {
    
    index <- sample(c(1:2), nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
    
    train <- df[index == 1, ]
    test <- df[index == 2, ]
    
    y_train <- train$y
    y_test <- test$y
    
    # Fit the model to the training data, make predictions on the holdout data
adaboost_train_fit <- MachineShop::fit(formula = as.factor(y) ~ ., data = train, model = "AdaBoostModel")

adaboost_train_pred <- stats::predict(adaboost_train_fit, train, type = "prob")
adaboost_train_predictions <- ifelse(adaboost_train_pred > 0.5, 1, 0)
adaboost_train_table <- table(adaboost_train_predictions, y_train)
adaboost_train_accuracy[i] <- (adaboost_train_table[1, 1] + adaboost_train_table[2, 2]) / sum(adaboost_train_table)
adaboost_train_accuracy_mean <- mean(adaboost_train_accuracy)
    
adaboost_test_pred <- stats::predict(adaboost_train_fit, test, type = "prob")
adaboost_test_predictions <- ifelse(adaboost_test_pred > 0.5, 1, 0)
adaboost_test_table <- table(adaboost_test_predictions, y_test)
adaboost_test_accuracy[i] <- (adaboost_test_table[1, 1] + adaboost_test_table[2, 2]) / sum(adaboost_test_table)
adaboost_test_accuracy_mean <- mean(adaboost_test_accuracy)

adaboost_roc_obj <- pROC::roc(as.numeric(c(test$y)), c(adaboost_test_pred))
adaboost_auc <- round((pROC::auc(c(test$y), as.numeric(c(adaboost_test_pred)) - 1)), 4)
print(pROC::ggroc(adaboost_roc_obj, color = "steelblue", size = 2) +
            ggplot2::ggtitle(paste0("ADAboost Models ", "(AUC = ", adaboost_auc, ")")) +
            ggplot2::labs(x = "Specificity", y = "Sensitivity") +
            ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
    )
    
return(adaboost_test_accuracy_mean)
    
  } # closing brace for numresamples 
  
} # Closing brace for the function

# Test the function
adaboost_1(data = diabetes, colnum = 9, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="06-Individual_Logistic_Models_files/figure-html/ADABoost for logistic data-1.png" width="672" />

```
#> [1] 0.7484076
```

``` r

# Check for any errors
warnings()
```

### BayesGLM


``` r

# Load the library
library(arm)
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
library(tidyverse)
library(pROC)

# Set initial values to 0

bayesglm_train_accuracy <- 0
bayesglm_test_accuracy <- 0
bayesglm_holdout_accuracy <- 0
bayesglm_duration <- 0
bayesglm_table_total <- 0

# Create the function

bayesglm_1 <- function(data, colnum, numresamples, train_amount, test_amount){
  
  colnames(data)[colnum] <- "y"
  
  df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
  
  df <- df[sample(1:nrow(df)), ] # randomizes the rows
  
  # Set up random resampling
  
  for (i in 1:numresamples) {
    
    index <- sample(c(1:2), nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
    
    train <- df[index == 1, ]
    test <- df[index == 2, ]
    
    y_train <- train$y
    y_test <- test$y
    
    # Fit the model to the training data, make predictions on the holdout data
bayesglm_train_fit <- arm::bayesglm(y ~ ., data = train, family = binomial)
    
bayesglm_train_pred <- stats::predict(bayesglm_train_fit, train, type = "response")
bayesglm_train_predictions <- ifelse(bayesglm_train_pred > 0.5, 1, 0)
bayesglm_train_table <- table(bayesglm_train_predictions, y_train)
bayesglm_train_accuracy[i] <- (bayesglm_train_table[1, 1] + bayesglm_train_table[2, 2]) / sum(bayesglm_train_table)
bayesglm_train_accuracy_mean <- mean(bayesglm_train_accuracy)

bayesglm_test_pred <- stats::predict(bayesglm_train_fit, test, type = "response")
bayesglm_test_predictions <- ifelse(bayesglm_test_pred > 0.5, 1, 0)
bayesglm_test_table <- table(bayesglm_test_predictions, y_test)

bayesglm_test_accuracy[i] <- (bayesglm_test_table[1, 1] + bayesglm_test_table[2, 2]) / sum(bayesglm_test_table)
bayesglm_test_accuracy_mean <- mean(bayesglm_test_accuracy)

bayesglm_roc_obj <- pROC::roc(as.numeric(c(test$y)), c(bayesglm_test_pred))
bayesglm_auc <- round((pROC::auc(c(test$y), as.numeric(c(bayesglm_test_pred)) - 1)), 4)
print(pROC::ggroc(bayesglm_roc_obj, color = "steelblue", size = 2) +
            ggplot2::ggtitle(paste0("Bayesglm Models ", "(AUC = ", bayesglm_auc, ")")) +
            ggplot2::labs(x = "Specificity", y = "Sensitivity") +
            ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
    )
    
return(bayesglm_test_accuracy_mean)
    
  } # closing brace for numresamples 
  
} # Closing brace for the function

# Test the function
bayesglm_1(data = diabetes, colnum = 9, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="06-Individual_Logistic_Models_files/figure-html/BayesGLM model for logistic data-1.png" width="672" />

```
#> [1] 0.7824675
```

``` r

# Check for any errors
warnings()
```

### C50


``` r

# Load the library
library(C50)
library(tidyverse)
library(pROC)

# Set initial values to 0

C50_train_accuracy <- 0
C50_test_accuracy <- 0
C50_holdout_accuracy <- 0
C50_duration <- 0
C50_table_total <- 0

# Create the function

C50_1 <- function(data, colnum, numresamples, train_amount, test_amount){
  
  colnames(data)[colnum] <- "y"
  
  df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
  
  df <- df[sample(1:nrow(df)), ] # randomizes the rows
  
  # Set up random resampling
  
  for (i in 1:numresamples) {
    
    index <- sample(c(1:2), nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
    
    train <- df[index == 1, ]
    test <- df[index == 2, ]
    
    y_train <- train$y
    y_test <- test$y
    
    # Fit the model to the training data, make predictions on the holdout data
C50_train_fit <- C50::C5.0(as.factor(y_train) ~ ., data = train)

C50_train_pred <- stats::predict(C50_train_fit, train, type = "prob")
C50_train_predictions <- ifelse(C50_train_pred[, 2] > 0.5, 1, 0)
C50_train_table <- table(C50_train_predictions, y_train)
C50_train_accuracy[i] <- (C50_train_table[1, 1] + C50_train_table[2, 2]) / sum(C50_train_table)
C50_train_accuracy_mean <- mean(C50_train_accuracy)

C50_test_pred <- stats::predict(C50_train_fit, test, type = "prob")
C50_test_predictions <- ifelse(C50_test_pred[, 2] > 0.5, 1, 0)
C50_test_table <- table(C50_test_predictions, y_test)
C50_test_accuracy[i] <- (C50_test_table[1, 1] + C50_test_table[2, 2]) / sum(C50_test_table)
C50_test_accuracy_mean <- mean(C50_test_accuracy)

C50_roc_obj <- pROC::roc(as.numeric(c(test$y)), as.numeric(c(C50_test_predictions)))
C50_auc <- round((pROC::auc(c(test$y), as.numeric(c(C50_test_predictions)) - 1)), 4)
print(pROC::ggroc(C50_roc_obj, color = "steelblue", size = 2) +
  ggplot2::ggtitle(paste0("C50 ROC curve ", "(AUC = ", C50_auc, ")")) +
  ggplot2::labs(x = "Specificity", y = "Sensitivity") +
  ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
    )
    
return(C50_test_accuracy_mean)
    
  } # closing brace for numresamples 
  
} # Closing brace for the function

# Test the function
C50_1(data = diabetes, colnum = 9, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="06-Individual_Logistic_Models_files/figure-html/C50 models for logistic data-1.png" width="672" />

```
#> [1] 1
```

``` r

# Check for any errors
warnings()
```

### Cubist


``` r

# Load the library
library(Cubist)
#> Loading required package: lattice
```

``` r
library(tidyverse)
library(pROC)

# Set initial values to 0

cubist_train_accuracy <- 0
cubist_test_accuracy <- 0
cubist_holdout_accuracy <- 0
cubist_duration <- 0
cubist_table_total <- 0

# Create the function

cubist_1 <- function(data, colnum, numresamples, train_amount, test_amount){
  
  colnames(data)[colnum] <- "y"
  
  df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
  
  df <- df[sample(1:nrow(df)), ] # randomizes the rows
  
  # Set up random resampling
  
  for (i in 1:numresamples) {
    
    index <- sample(c(1:2), nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
    
    train <- df[index == 1, ]
    test <- df[index == 2, ]
    
    y_train <- train$y
    y_test <- test$y
    
    # Fit the model to the training data, make predictions on the holdout data
cubist_train_fit <- Cubist::cubist(x = as.data.frame(train), y = train$y)
    
cubist_train_pred <- stats::predict(cubist_train_fit, train, type = "prob")
cubist_train_table <- table(cubist_train_pred, y_train)
cubist_train_accuracy[i] <- (cubist_train_table[1, 1] + cubist_train_table[2, 2]) / sum(cubist_train_table)
cubist_train_accuracy_mean <- mean(cubist_train_accuracy)

cubist_test_pred <- stats::predict(cubist_train_fit, test, type = "prob")
cubist_test_table <- table(cubist_test_pred, y_test)
cubist_test_accuracy[i] <- (cubist_test_table[1, 1] + cubist_test_table[2, 2]) / sum(cubist_test_table)
cubist_test_accuracy_mean <- mean(cubist_test_accuracy)


cubist_roc_obj <- pROC::roc(as.numeric(c(test$y)), as.numeric(c(cubist_test_pred)))
cubist_auc <- round((pROC::auc(c(test$y), as.numeric(c(cubist_test_pred)) - 1)), 4)
print(pROC::ggroc(cubist_roc_obj, color = "steelblue", size = 2) +
  ggplot2::ggtitle(paste0("Cubist ROC curve ", "(AUC = ", cubist_auc, ")")) +
  ggplot2::labs(x = "Specificity", y = "Sensitivity") +
  ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
    )
    
return(cubist_test_accuracy_mean)
    
  } # closing brace for numresamples 
  
} # Closing brace for the function

# Test the function
cubist_1(data = diabetes, colnum = 9, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="06-Individual_Logistic_Models_files/figure-html/Cubist model for logistic data-1.png" width="672" />

```
#> [1] 1
```

``` r

# Check for any errors
warnings()
```

### Gradient Boosted


``` r

# Load the library
library(gbm)
#> Loaded gbm 2.1.9
#> This version of gbm is no longer under development. Consider transitioning to gbm3, https://github.com/gbm-developers/gbm3
```

``` r
library(tidyverse)
library(pROC)

# Set initial values to 0

gb_train_accuracy <- 0
gb_test_accuracy <- 0
gb_holdout_accuracy <- 0
gb_duration <- 0
gb_table_total <- 0

# Create the function

gb_1 <- function(data, colnum, numresamples, train_amount, test_amount){
  
  colnames(data)[colnum] <- "y"
  
  df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
  
  df <- df[sample(1:nrow(df)), ] # randomizes the rows
  
  # Set up random resampling
  
  for (i in 1:numresamples) {
    
    index <- sample(c(1:2), nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
    
    train <- df[index == 1, ]
    test <- df[index == 2, ]
    
    y_train <- train$y
    y_test <- test$y
    
    # Fit the model to the training data, make predictions on the holdout data
gb_train_fit <- gbm::gbm(train$y ~ ., data = train, distribution = "gaussian", n.trees = 100, shrinkage = 0.1, interaction.depth = 10)
    
gb_train_pred <- stats::predict(gb_train_fit, train, type = "response")
gb_train_predictions <- ifelse(gb_train_pred > 0.5, 1, 0)
gb_train_table <- table(gb_train_predictions, y_train)
gb_train_accuracy[i] <- (gb_train_table[1, 1] + gb_train_table[2, 2]) / sum(gb_train_table)
gb_train_accuracy_mean <- mean(gb_train_accuracy)

gb_test_pred <- stats::predict(gb_train_fit, test, type = "response")
gb_test_predictions <- ifelse(gb_test_pred > 0.5, 1, 0)
gb_test_table <- table(gb_test_predictions, y_test)
gb_test_accuracy[i] <- (gb_test_table[1, 1] + gb_test_table[2, 2]) / sum(gb_test_table)
gb_test_accuracy_mean <- mean(gb_test_accuracy)

gb_roc_obj <- pROC::roc(as.numeric(c(test$y)), as.numeric(c(gb_test_pred)))
gb_auc <- round((pROC::auc(c(test$y), as.numeric(c(gb_test_pred)) - 1)), 4)
print(pROC::ggroc(gb_roc_obj, color = "steelblue", size = 2) +
  ggplot2::ggtitle(paste0("Gradient Boosted ROC curve ", "(AUC = ", gb_auc, ")")) +
  ggplot2::labs(x = "Specificity", y = "Sensitivity") +
  ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
    )
    
return(gb_test_accuracy_mean)
    
  } # closing brace for numresamples 
  
} # Closing brace for the function

# Test the function
gb_1(data = diabetes, colnum = 9, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#> Using 100 trees...
#> Using 100 trees...
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="06-Individual_Logistic_Models_files/figure-html/Gradient Boosted model for logistic data-1.png" width="672" />

```
#> [1] 0.7012987
```

``` r

# Check for any errors
warnings()
```

### Random Forest


``` r

# Load the library
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
```

``` r
library(tidyverse)
library(pROC)

# Set initial values to 0

rf_train_accuracy <- 0
rf_test_accuracy <- 0
rf_holdout_accuracy <- 0
rf_duration <- 0
rf_table_total <- 0

# Create the function

rf_1 <- function(data, colnum, numresamples, train_amount, test_amount){
  
  colnames(data)[colnum] <- "y"
  
  df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
  
  df <- df[sample(1:nrow(df)), ] # randomizes the rows
  
  # Set up random resampling
  
  for (i in 1:numresamples) {
    
    index <- sample(c(1:2), nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
    
    train <- df[index == 1, ]
    test <- df[index == 2, ]
    
    y_train <- train$y
    y_test <- test$y
    
    # Fit the model to the training data, make predictions on the holdout data
rf_train_fit <- randomForest(x = train, y = as.factor(y_train), data = df)
    
rf_train_pred <- stats::predict(rf_train_fit, train, type = "prob")
rf_train_probabilities <- ifelse(rf_train_pred > 0.50, 1, 0)[, 2]
rf_train_table <- table(rf_train_probabilities, y_train)
rf_train_accuracy[i] <- (rf_train_table[1, 1] + rf_train_table[2, 2]) / sum(rf_train_table)
rf_train_accuracy_mean <- mean(rf_train_accuracy)

rf_test_pred <- stats::predict(rf_train_fit, test, type = "prob")
rf_test_probabilities <- ifelse(rf_test_pred > 0.50, 1, 0)[, 2]
rf_test_table <- table(rf_test_probabilities, y_test)
rf_test_accuracy[i] <- (rf_test_table[1, 1] + rf_test_table[2, 2]) / sum(rf_test_table)
rf_test_accuracy_mean <- mean(rf_test_accuracy)

rf_roc_obj <- pROC::roc(as.numeric(c(test$y)), as.numeric(c(rf_test_probabilities)))
rf_auc <- round((pROC::auc(c(test$y), as.numeric(c(rf_test_probabilities)) - 1)), 4)
print(pROC::ggroc(rf_roc_obj, color = "steelblue", size = 2) +
  ggplot2::ggtitle(paste0("Random Forest ", "(AUC = ", rf_auc, ")")) +
  ggplot2::labs(x = "Specificity", y = "Sensitivity") +
  ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
    )
    
return(rf_test_accuracy_mean)
    
  } # closing brace for numresamples 
  
} # Closing brace for the function

# Test the function
rf_1(data = diabetes, colnum = 9, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="06-Individual_Logistic_Models_files/figure-html/Random Forest for logistic data-1.png" width="672" />

```
#> [1] 1
```

``` r

# Check for any errors
warnings()
```

### Support Vector Machines


``` r

# Load the library
library(e1071)
library(tidyverse)
library(pROC)

# Set initial values to 0

svm_train_accuracy <- 0
svm_test_accuracy <- 0
svm_holdout_accuracy <- 0
svm_duration <- 0
svm_table_total <- 0

# Create the function

svm_1 <- function(data, colnum, numresamples, train_amount, test_amount){
  
  colnames(data)[colnum] <- "y"
  
  df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
  
  df <- df[sample(1:nrow(df)), ] # randomizes the rows
  
  # Set up random resampling
  
  for (i in 1:numresamples) {
    
    index <- sample(c(1:2), nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
    
    train <- df[index == 1, ]
    test <- df[index == 2, ]
    
    y_train <- train$y
    y_test <- test$y
    
    # Fit the model to the training data, make predictions on the holdout data
svm_train_fit <- e1071::svm(as.factor(y) ~ ., data = train)
    
svm_train_pred <- stats::predict(svm_train_fit, train, type = "prob")
svm_train_table <- table(svm_train_pred, y_train)
svm_train_accuracy[i] <- (svm_train_table[1, 1] + svm_train_table[2, 2]) / sum(svm_train_table)
svm_train_accuracy_mean <- mean(svm_train_accuracy)

svm_test_pred <- stats::predict(svm_train_fit, test, type = "prob")
svm_test_table <- table(svm_test_pred, y_test)
svm_test_accuracy[i] <- (svm_test_table[1, 1] + svm_test_table[2, 2]) / sum(svm_test_table)
svm_test_accuracy_mean <- mean(svm_test_accuracy)

svm_roc_obj <- pROC::roc(as.numeric(c(test$y)), as.numeric(c(svm_test_pred)))
svm_auc <- round((pROC::auc(c(test$y), as.numeric(c(svm_test_pred)) - 1)), 4)
print(pROC::ggroc(svm_roc_obj, color = "steelblue", size = 2) +
  ggplot2::ggtitle(paste0("Random Forest ", "(AUC = ", svm_auc, ")")) +
  ggplot2::labs(x = "Specificity", y = "Sensitivity") +
  ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
    )
    
return(svm_test_accuracy_mean)
    
  } # closing brace for numresamples 
  
} # Closing brace for the function

# Test the function
svm_1(data = diabetes, colnum = 9, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="06-Individual_Logistic_Models_files/figure-html/Support Vector Machines for logistic data-1.png" width="672" />

```
#> [1] 0.7317881
```

``` r

# Check for any errors
warnings()
```

### XGBoost


``` r

# Load the library
library(xgboost)
#> 
#> Attaching package: 'xgboost'
#> The following object is masked from 'package:dplyr':
#> 
#>     slice
```

``` r
library(tidyverse)
library(pROC)

# Set initial values to 0

xgb_train_accuracy <- 0
xgb_test_accuracy <- 0
xgb_holdout_accuracy <- 0
xgb_duration <- 0
xgb_table_total <- 0

# Create the function

xgb_1 <- function(data, colnum, numresamples, train_amount, test_amount){
  
  colnames(data)[colnum] <- "y"
  
  df <- data %>% dplyr::relocate(y, .after = last_col()) # Moves the target column to the last column on the right
  
  df <- df[sample(1:nrow(df)), ] # randomizes the rows
  
  # Set up random resampling
  
  for (i in 1:numresamples) {
    
    index <- sample(c(1:2), nrow(df), replace = TRUE, prob = c(train_amount, test_amount))
    
    train <- df[index == 1, ]
    test <- df[index == 2, ]
    
    y_train <- train$y
    y_test <- test$y
    
# Fit the model to the training data, make predictions on the holdout data
train_x <- data.matrix(train[, -ncol(train)])
train_y <- train[, ncol(train)]
    
# define predictor and response variables in test set
test_x <- data.matrix(test[, -ncol(test)])
test_y <- test[, ncol(test)]
    
# define final train and test sets
xgb_train <- xgboost::xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgboost::xgb.DMatrix(data = test_x, label = test_y)

# define watchlist
watchlist <- list(train = xgb_train)
watchlist_test <- list(train = xgb_train, test = xgb_test)

xgb_model <- xgboost::xgb.train(data = xgb_train, max.depth = 3, watchlist = watchlist_test, nrounds = 70)
    
xgb_min <- which.min(xgb_model$evaluation_log$validation_rmse)
    
xgb_train_pred <- stats::predict(object = xgb_model, newdata = train_x, type = "prob")
xgb_train_predictions <- ifelse(xgb_train_pred > 0.5, 1, 0)
xgb_train_table <- table(xgb_train_predictions, y_train)
xgb_train_accuracy[i] <- (xgb_train_table[1, 1] + xgb_train_table[2, 2]) / sum(xgb_train_table)
xgb_train_accuracy_mean <- mean(xgb_train_accuracy)

xgb_test_pred <- stats::predict(object = xgb_model, newdata = test_x, type = "prob")
xgb_test_predictions <- ifelse(xgb_test_pred > 0.5, 1, 0)
xgb_test_table <- table(xgb_test_predictions, y_test)
xgb_test_accuracy[i] <- (xgb_test_table[1, 1] + xgb_test_table[2, 2]) / sum(xgb_test_table)
xgb_test_accuracy_mean <- mean(xgb_test_accuracy)

xgb_roc_obj <- pROC::roc(as.numeric(c(test$y)), as.numeric(c(xgb_test_pred)))
xgb_auc <- round((pROC::auc(c(test$y), as.numeric(c(xgb_test_pred)) - 1)), 4)
print(pROC::ggroc(xgb_roc_obj, color = "steelblue", size = 2) +
  ggplot2::ggtitle(paste0("XGBoost ", "(AUC = ", xgb_auc, ")")) +
  ggplot2::labs(x = "Specificity", y = "Sensitivity") +
  ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
    )
    
return(xgb_test_accuracy_mean)
    
  } # closing brace for numresamples 
  
} # Closing brace for the function

# Test the function
xgb_1(data = diabetes, colnum = 9, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#> [1]	train-rmse:0.443820	test-rmse:0.451114 
#> [2]	train-rmse:0.410144	test-rmse:0.424842 
#> [3]	train-rmse:0.389456	test-rmse:0.414245 
#> [4]	train-rmse:0.375497	test-rmse:0.406926 
#> [5]	train-rmse:0.361120	test-rmse:0.404512 
#> [6]	train-rmse:0.351706	test-rmse:0.402855 
#> [7]	train-rmse:0.344964	test-rmse:0.402907 
#> [8]	train-rmse:0.338652	test-rmse:0.403506 
#> [9]	train-rmse:0.332740	test-rmse:0.402499 
#> [10]	train-rmse:0.329630	test-rmse:0.404849 
#> [11]	train-rmse:0.326029	test-rmse:0.403572 
#> [12]	train-rmse:0.322103	test-rmse:0.403685 
#> [13]	train-rmse:0.316984	test-rmse:0.402375 
#> [14]	train-rmse:0.314421	test-rmse:0.402665 
#> [15]	train-rmse:0.312002	test-rmse:0.402012 
#> [16]	train-rmse:0.306025	test-rmse:0.401865 
#> [17]	train-rmse:0.300422	test-rmse:0.406396 
#> [18]	train-rmse:0.296890	test-rmse:0.407696 
#> [19]	train-rmse:0.294799	test-rmse:0.407525 
#> [20]	train-rmse:0.293248	test-rmse:0.409466 
#> [21]	train-rmse:0.290883	test-rmse:0.408838 
#> [22]	train-rmse:0.289344	test-rmse:0.407957 
#> [23]	train-rmse:0.286014	test-rmse:0.405951 
#> [24]	train-rmse:0.284558	test-rmse:0.408597 
#> [25]	train-rmse:0.282312	test-rmse:0.409822 
#> [26]	train-rmse:0.279900	test-rmse:0.409098 
#> [27]	train-rmse:0.279171	test-rmse:0.409240 
#> [28]	train-rmse:0.277008	test-rmse:0.409459 
#> [29]	train-rmse:0.275484	test-rmse:0.409977 
#> [30]	train-rmse:0.270146	test-rmse:0.409051 
#> [31]	train-rmse:0.268145	test-rmse:0.409530 
#> [32]	train-rmse:0.267432	test-rmse:0.408804 
#> [33]	train-rmse:0.266643	test-rmse:0.408577 
#> [34]	train-rmse:0.265945	test-rmse:0.408396 
#> [35]	train-rmse:0.262627	test-rmse:0.408979 
#> [36]	train-rmse:0.261944	test-rmse:0.409056 
#> [37]	train-rmse:0.261083	test-rmse:0.410124 
#> [38]	train-rmse:0.259975	test-rmse:0.410166 
#> [39]	train-rmse:0.257546	test-rmse:0.409997 
#> [40]	train-rmse:0.254742	test-rmse:0.406248 
#> [41]	train-rmse:0.253515	test-rmse:0.405945 
#> [42]	train-rmse:0.253009	test-rmse:0.405861 
#> [43]	train-rmse:0.248366	test-rmse:0.406695 
#> [44]	train-rmse:0.245926	test-rmse:0.405393 
#> [45]	train-rmse:0.245340	test-rmse:0.405159 
#> [46]	train-rmse:0.241983	test-rmse:0.404499 
#> [47]	train-rmse:0.238854	test-rmse:0.403837 
#> [48]	train-rmse:0.237425	test-rmse:0.402440 
#> [49]	train-rmse:0.234284	test-rmse:0.403601 
#> [50]	train-rmse:0.232766	test-rmse:0.403870 
#> [51]	train-rmse:0.230626	test-rmse:0.404637 
#> [52]	train-rmse:0.230082	test-rmse:0.404204 
#> [53]	train-rmse:0.229541	test-rmse:0.404657 
#> [54]	train-rmse:0.226442	test-rmse:0.403503 
#> [55]	train-rmse:0.222275	test-rmse:0.404375 
#> [56]	train-rmse:0.221854	test-rmse:0.404138 
#> [57]	train-rmse:0.219353	test-rmse:0.405571 
#> [58]	train-rmse:0.217785	test-rmse:0.405259 
#> [59]	train-rmse:0.215515	test-rmse:0.404085 
#> [60]	train-rmse:0.214636	test-rmse:0.403443 
#> [61]	train-rmse:0.213945	test-rmse:0.403807 
#> [62]	train-rmse:0.211350	test-rmse:0.404858 
#> [63]	train-rmse:0.209434	test-rmse:0.404904 
#> [64]	train-rmse:0.208350	test-rmse:0.404777 
#> [65]	train-rmse:0.206028	test-rmse:0.404528 
#> [66]	train-rmse:0.205355	test-rmse:0.404718 
#> [67]	train-rmse:0.203309	test-rmse:0.405660 
#> [68]	train-rmse:0.201764	test-rmse:0.404970 
#> [69]	train-rmse:0.199831	test-rmse:0.404663 
#> [70]	train-rmse:0.198291	test-rmse:0.405125
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="06-Individual_Logistic_Models_files/figure-html/XGBoost model for logistic data-1.png" width="672" />

```
#> [1] 0.7429577
```

``` r

# Check for any errors
warnings()
```
