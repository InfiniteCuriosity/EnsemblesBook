# Advice to Lebron James (and everyone who does talent analytics): Logistic ensembles

In this section we're going to take the lessons of the previous chapter and move them into making ensembles of models. The process is extremely similar, and follows these steps:

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

Logistic ensembles can be used in an extremely wide range of fields. Previously we modeled diabetes in Pima Indian women. This chapter's example will be the performance on the court of Lebron James.

Here's an image of Lebron in play.

![Lebron James](_book/images/LeBron_James_small.jpg)

There is a LOT of data in sports and HR analytics (which are extremely similar in some ways). A lot of the data is set up as logistic data. For example, here is a data set of the performance of Lebron James. The main column of interest is "result", which is either 1 or 0. Thus it perfectly fits our requirements for logistic analysis.


``` r

library(Ensembles)
#> Loading required package: arm
#> Loading required package: MASS
#> Loading required package: Matrix
#> Loading required package: lme4
#> 
#> arm (Version 1.14-4, built: 2024-4-1)
#> Working directory is /Users/russellconte/Library/Mobile Documents/com~apple~CloudDocs/Documents/Machine Learning templates in R/EnsemblesBook
#> Loading required package: brnn
#> Loading required package: Formula
#> Loading required package: truncnorm
#> Loading required package: broom
#> Loading required package: C50
#> Loading required package: caret
#> Loading required package: ggplot2
#> Loading required package: lattice
#> Loading required package: class
#> Loading required package: corrplot
#> corrplot 0.92 loaded
#> 
#> Attaching package: 'corrplot'
#> The following object is masked from 'package:arm':
#> 
#>     corrplot
#> Loading required package: Cubist
#> Loading required package: doParallel
#> Loading required package: foreach
#> Loading required package: iterators
#> Loading required package: parallel
#> Loading required package: dplyr
#> 
#> Attaching package: 'dplyr'
#> The following object is masked from 'package:MASS':
#> 
#>     select
#> The following objects are masked from 'package:stats':
#> 
#>     filter, lag
#> The following objects are masked from 'package:base':
#> 
#>     intersect, setdiff, setequal, union
#> Loading required package: e1071
#> Loading required package: fable
#> Loading required package: fabletools
#> Registered S3 method overwritten by 'tsibble':
#>   method          from
#>   format.interval inum
#> 
#> Attaching package: 'fabletools'
#> The following object is masked from 'package:e1071':
#> 
#>     interpolate
#> The following objects are masked from 'package:caret':
#> 
#>     MAE, RMSE
#> The following object is masked from 'package:lme4':
#> 
#>     refit
#> Loading required package: fable.prophet
#> Loading required package: Rcpp
#> Loading required package: feasts
#> Loading required package: gam
#> Loading required package: splines
#> Loaded gam 1.22-3
#> Loading required package: gbm
#> Loaded gbm 2.1.9
#> This version of gbm is no longer under development. Consider transitioning to gbm3, https://github.com/gbm-developers/gbm3
#> Loading required package: GGally
#> Registered S3 method overwritten by 'GGally':
#>   method from   
#>   +.gg   ggplot2
#> Loading required package: glmnet
#> Loaded glmnet 4.1-8
#> Loading required package: gridExtra
#> 
#> Attaching package: 'gridExtra'
#> The following object is masked from 'package:dplyr':
#> 
#>     combine
#> Loading required package: gt
#> Loading required package: gtExtras
#> 
#> Attaching package: 'gtExtras'
#> The following object is masked from 'package:MASS':
#> 
#>     select
#> Loading required package: ipred
#> Loading required package: kernlab
#> 
#> Attaching package: 'kernlab'
#> The following object is masked from 'package:ggplot2':
#> 
#>     alpha
#> Loading required package: klaR
#> Loading required package: leaps
#> Loading required package: MachineShop
#> 
#> Attaching package: 'MachineShop'
#> The following objects are masked from 'package:fabletools':
#> 
#>     accuracy, response
#> The following objects are masked from 'package:caret':
#> 
#>     calibration, lift, precision, recall, rfe,
#>     sensitivity, specificity
#> The following object is masked from 'package:stats':
#> 
#>     ppr
#> Loading required package: magrittr
#> Loading required package: mda
#> Loaded mda 0.5-4
#> 
#> Attaching package: 'mda'
#> The following object is masked from 'package:MachineShop':
#> 
#>     confusion
#> Loading required package: Metrics
#> 
#> Attaching package: 'Metrics'
#> The following objects are masked from 'package:MachineShop':
#> 
#>     accuracy, auc, mae, mse, msle, precision, recall,
#>     rmse, rmsle
#> The following object is masked from 'package:fabletools':
#> 
#>     accuracy
#> The following objects are masked from 'package:caret':
#> 
#>     precision, recall
#> Loading required package: neuralnet
#> 
#> Attaching package: 'neuralnet'
#> The following object is masked from 'package:dplyr':
#> 
#>     compute
#> Loading required package: pls
#> 
#> Attaching package: 'pls'
#> The following object is masked from 'package:corrplot':
#> 
#>     corrplot
#> The following object is masked from 'package:caret':
#> 
#>     R2
#> The following objects are masked from 'package:arm':
#> 
#>     coefplot, corrplot
#> The following object is masked from 'package:stats':
#> 
#>     loadings
#> Loading required package: pROC
#> Type 'citation("pROC")' for a citation.
#> 
#> Attaching package: 'pROC'
#> The following object is masked from 'package:Metrics':
#> 
#>     auc
#> The following object is masked from 'package:MachineShop':
#> 
#>     auc
#> The following objects are masked from 'package:stats':
#> 
#>     cov, smooth, var
#> Loading required package: purrr
#> 
#> Attaching package: 'purrr'
#> The following object is masked from 'package:magrittr':
#> 
#>     set_names
#> The following object is masked from 'package:MachineShop':
#> 
#>     lift
#> The following object is masked from 'package:kernlab':
#> 
#>     cross
#> The following objects are masked from 'package:foreach':
#> 
#>     accumulate, when
#> The following object is masked from 'package:caret':
#> 
#>     lift
#> Loading required package: randomForest
#> randomForest 4.7-1.1
#> Type rfNews() to see new features/changes/bug fixes.
#> 
#> Attaching package: 'randomForest'
#> The following object is masked from 'package:gridExtra':
#> 
#>     combine
#> The following object is masked from 'package:dplyr':
#> 
#>     combine
#> The following object is masked from 'package:ggplot2':
#> 
#>     margin
#> Loading required package: reactable
#> Loading required package: reactablefmtr
#> 
#> Attaching package: 'reactablefmtr'
#> The following object is masked from 'package:randomForest':
#> 
#>     margin
#> The following objects are masked from 'package:gt':
#> 
#>     google_font, html
#> The following object is masked from 'package:ggplot2':
#> 
#>     margin
#> Loading required package: readr
#> Loading required package: rpart
#> Loading required package: scales
#> 
#> Attaching package: 'scales'
#> The following object is masked from 'package:readr':
#> 
#>     col_factor
#> The following object is masked from 'package:purrr':
#> 
#>     discard
#> The following object is masked from 'package:kernlab':
#> 
#>     alpha
#> The following object is masked from 'package:arm':
#> 
#>     rescale
#> Loading required package: tibble
#> Loading required package: tidyr
#> 
#> Attaching package: 'tidyr'
#> The following object is masked from 'package:magrittr':
#> 
#>     extract
#> The following objects are masked from 'package:Matrix':
#> 
#>     expand, pack, unpack
#> Loading required package: tree
#> Loading required package: tsibble
#> 
#> Attaching package: 'tsibble'
#> The following objects are masked from 'package:base':
#> 
#>     intersect, setdiff, union
#> Loading required package: xgboost
#> 
#> Attaching package: 'xgboost'
#> The following object is masked from 'package:dplyr':
#> 
#>     slice
```

``` r
head(lebron, n = 20)
#>    top left  date qtr time_remaining result shot_type
#> 1  310  203 19283   2            566      0         3
#> 2  213  259 19283   2            518      0         2
#> 3  143  171 19283   2            490      0         2
#> 4   68  215 19283   2            324      1         2
#> 5   66  470 19283   2             62      0         3
#> 6   63  239 19283   4            690      1         2
#> 7  230   54 19283   4            630      0         3
#> 8   53  224 19283   4            605      1         2
#> 9  241   67 19283   4            570      0         3
#> 10 273  113 19283   4            535      0         3
#> 11  62  224 19283   4            426      0         2
#> 12  63  249 19283   4            233      1         2
#> 13 103  236 19283   4            154      0         2
#> 14  54  249 19283   4            108      1         2
#> 15  53  240 19283   4             58      0         2
#> 16 230   71 19283   5            649      1         3
#> 17 231  358 19283   5            540      0         2
#> 18  61  240 19283   5            524      1         2
#> 19  59  235 19283   5             71      1         2
#> 20 299  188 19283   5              6      1         3
#>    distance_ft lead lebron_team_score opponent_team_score
#> 1           26    0                 2                   2
#> 2           16    0                 4                   5
#> 3           11    0                 4                   7
#> 4            3    0                12                  19
#> 5           23    0                22                  23
#> 6            1    0                24                  25
#> 7           26    0                24                  27
#> 8            2    0                26                  27
#> 9           26    0                26                  29
#> 10          25    0                26                  32
#> 11           2    0                31                  39
#> 12           1    0                39                  49
#> 13           5    0                39                  51
#> 14           1    0                44                  53
#> 15           0    0                46                  55
#> 16          25    0                58                  63
#> 17          21    0                60                  70
#> 18           1    0                62                  70
#> 19           1    0                68                  91
#> 20          25    0                71                  91
#>    opponent
#> 1         9
#> 2         9
#> 3         9
#> 4         9
#> 5         9
#> 6         9
#> 7         9
#> 8         9
#> 9         9
#> 10        9
#> 11        9
#> 12        9
#> 13        9
#> 14        9
#> 15        9
#> 16        9
#> 17        9
#> 18        9
#> 19        9
#> 20        9
```

Let's look at the structure of the data:


``` r
lebron <- Ensembles::lebron
str(Ensembles::lebron)
#> 'data.frame':	1533 obs. of  12 variables:
#>  $ top                : int  310 213 143 68 66 63 230 53 241 273 ...
#>  $ left               : int  203 259 171 215 470 239 54 224 67 113 ...
#>  $ date               : num  19283 19283 19283 19283 19283 ...
#>  $ qtr                : num  2 2 2 2 2 4 4 4 4 4 ...
#>  $ time_remaining     : num  566 518 490 324 62 690 630 605 570 535 ...
#>  $ result             : num  0 0 0 1 0 1 0 1 0 0 ...
#>  $ shot_type          : int  3 2 2 2 3 2 3 2 3 3 ...
#>  $ distance_ft        : int  26 16 11 3 23 1 26 2 26 25 ...
#>  $ lead               : num  0 0 0 0 0 0 0 0 0 0 ...
#>  $ lebron_team_score  : int  2 4 4 12 22 24 24 26 26 26 ...
#>  $ opponent_team_score: int  2 5 7 19 23 25 27 27 29 32 ...
#>  $ opponent           : num  9 9 9 9 9 9 9 9 9 9 ...
```

We see that all of these are numbers. It might be easier if "qtr" and "opponent" were changed to factors, so we'll do that first.


``` r

lebron$qtr <- as.factor(lebron$qtr)
lebron$opponent <- as.factor(lebron$opponent)
```

Now we're ready to create an ensemble of models to make predictions about Lebron's future performance. The new skill in this chapter will be saving all trained models to the Environment. This will allow us to look at the trained models, and use them to make the strongest evidence based recommendations.

We will use the following individual models and ensemble of models:

Individual models:

AdaBoost

BayesGLM

C50

Cubist

Generalized Linear Models (GLM)

Random Forest

XGBoost

We will make an ensemble of the predictions from those five models, and then use that ensemble to model predictions for Lebron's performance.

We will also show the ROC curves for each of the results, and save all the trained models at the end.


``` r

# Load libraries - note these will work with individual and ensemble models
library(arm) # to use with BayesGLM
library(C50) # To use with C50
library(Cubist) # To use with Cubist modeling
library(MachineShop)# To use with ADABoost
library(randomForest) # Random Forest models
library(tidyverse) # My favorite tool for data science!
#> ── Attaching core tidyverse packages ──── tidyverse 2.0.0 ──
#> ✔ forcats   1.0.0     ✔ stringr   1.5.1
#> ✔ lubridate 1.9.3     
#> ── Conflicts ────────────────────── tidyverse_conflicts() ──
#> ✖ purrr::accumulate()     masks foreach::accumulate()
#> ✖ scales::alpha()         masks kernlab::alpha(), ggplot2::alpha()
#> ✖ scales::col_factor()    masks readr::col_factor()
#> ✖ randomForest::combine() masks gridExtra::combine(), dplyr::combine()
#> ✖ neuralnet::compute()    masks dplyr::compute()
#> ✖ purrr::cross()          masks kernlab::cross()
#> ✖ scales::discard()       masks purrr::discard()
#> ✖ tidyr::expand()         masks Matrix::expand()
#> ✖ tidyr::extract()        masks magrittr::extract()
#> ✖ dplyr::filter()         masks stats::filter()
#> ✖ lubridate::interval()   masks tsibble::interval()
#> ✖ dplyr::lag()            masks stats::lag()
#> ✖ purrr::lift()           masks MachineShop::lift(), caret::lift()
#> ✖ reactablefmtr::margin() masks randomForest::margin(), ggplot2::margin()
#> ✖ tidyr::pack()           masks Matrix::pack()
#> ✖ gtExtras::select()      masks dplyr::select(), MASS::select()
#> ✖ purrr::set_names()      masks magrittr::set_names()
#> ✖ xgboost::slice()        masks dplyr::slice()
#> ✖ tidyr::unpack()         masks Matrix::unpack()
#> ✖ purrr::when()           masks foreach::when()
#> ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
```

``` r
library(pROC) # To print ROC curves

# Set initial values to 0
adaboost_train_accuracy <- 0
adaboost_test_accuracy <- 0
adaboost_holdout_accuracy <- 0
adaboost_duration <- 0
adaboost_table_total <- 0

bayesglm_train_accuracy <- 0
bayesglm_test_accuracy <- 0
bayesglm_holdout_accuracy <- 0
bayesglm_duration <- 0
bayesglm_table_total <- 0

C50_train_accuracy <- 0
C50_test_accuracy <- 0
C50_holdout_accuracy <- 0
C50_duration <- 0
C50_table_total <- 0

cubist_train_accuracy <- 0
cubist_test_accuracy <- 0
cubist_holdout_accuracy <- 0
cubist_duration <- 0
cubist_table_total <- 0

rf_train_accuracy <- 0
rf_test_accuracy <- 0
rf_holdout_accuracy <- 0
rf_duration <- 0
rf_table_total <- 0

xgb_train_accuracy <- 0
xgb_test_accuracy <- 0
xgb_holdout_accuracy <- 0
xgb_duration <- 0
xgb_table_total <- 0


ensemble_adaboost_train_accuracy <- 0
ensemble_adaboost_test_accuracy <- 0
ensemble_adaboost_holdout_accuracy <- 0
ensemble_adaboost_duration <- 0
ensemble_adaboost_table_total <- 0
ensemble_adaboost_train_pred <- 0

ensemble_bayesglm_train_accuracy <- 0
ensemble_bayesglm_test_accuracy <- 0
ensemble_bayesglm_holdout_accuracy <- 0
ensemble_bayesglm_duration <- 0
ensemble_bayesglm_table_total <- 0

ensemble_C50_train_accuracy <- 0
ensemble_C50_test_accuracy <- 0
ensemble_C50_holdout_accuracy <- 0
ensemble_C50_duration <- 0
ensemble_C50_table_total <- 0

ensemble_rf_train_accuracy <- 0
ensemble_rf_test_accuracy <- 0
ensemble_rf_holdout_accuracy <- 0
ensemble_rf_duration <- 0
ensemble_rf_table_total <- 0

ensemble_xgb_train_accuracy <- 0
ensemble_xgb_test_accuracy <- 0
ensemble_xgb_holdout_accuracy <- 0
ensemble_xgb_duration <- 0
ensemble_xgb_table_total <- 0

# Create the function

logistic_1 <- function(data, colnum, numresamples, train_amount, test_amount){
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
  
  
# ADABoost model
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


# BayesGLM
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

# C50 model

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

# Cubist
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

# Random Forest
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

# XGBoost
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

# Ensemble

ensemble1 <- data.frame(
  'ADABoost' = adaboost_test_predictions,
  'BayesGLM'= bayesglm_test_predictions,
  'C50' = C50_test_predictions,
  'Cubist' = cubist_test_pred,
  'Random_Forest' = rf_test_pred,
  'XGBoost' = xgb_test_predictions,
  'y' = test$y
)

ensemble_index <- sample(c(1:2), nrow(ensemble1), replace = TRUE, prob = c(train_amount, test_amount))
ensemble_train <- ensemble1[ensemble_index == 1, ]
ensemble_test <- ensemble1[ensemble_index == 2, ]
ensemble_y_train <- ensemble_train$y
ensemble_y_test <- ensemble_test$y

# Ensemble ADABoost
ensemble_adaboost_train_fit <- MachineShop::fit(as.factor(y) ~ ., data = ensemble_train, model = "AdaBoostModel")
    
ensemble_adaboost_train_pred <- stats::predict(ensemble_adaboost_train_fit, ensemble_train, type = "prob")
ensemble_adaboost_train_probabilities <- ifelse(ensemble_adaboost_train_pred > 0.5, 1, 0)
ensemble_adaboost_train_table <- table(ensemble_adaboost_train_probabilities, ensemble_y_train)
ensemble_adaboost_train_accuracy[i] <- (ensemble_adaboost_train_table[1, 1] + ensemble_adaboost_train_table[2, 2]) / sum(ensemble_adaboost_train_table)
ensemble_adaboost_train_accuracy_mean <- mean(ensemble_adaboost_train_accuracy)
    
ensemble_adaboost_test_pred <- stats::predict(ensemble_adaboost_train_fit, ensemble_test, type = "prob")
ensemble_adaboost_test_probabilities <- ifelse(ensemble_adaboost_test_pred > 0.5, 1, 0)
ensemble_adaboost_test_table <- table(ensemble_adaboost_test_probabilities, ensemble_y_test)
ensemble_adaboost_test_accuracy[i] <- (ensemble_adaboost_test_table[1, 1] + ensemble_adaboost_test_table[2, 2]) / sum(ensemble_adaboost_test_table)
ensemble_adaboost_test_accuracy_mean <- mean(ensemble_adaboost_test_accuracy)
    
ensemble_adaboost_holdout_accuracy_mean <- mean(ensemble_adaboost_test_accuracy)
    
ensemble_adaboost_roc_obj <- pROC::roc(as.numeric(c(ensemble_test$y)), as.numeric(c(ensemble_adaboost_test_pred)))
ensemble_adaboost_auc <- round((pROC::auc(c(ensemble_test$y), as.numeric(c(ensemble_adaboost_test_pred)) - 1)), 4)
print(pROC::ggroc(ensemble_adaboost_roc_obj, color = "steelblue", size = 2) +
            ggplot2::ggtitle(paste0("Ensemble Adaboostoost ", "(AUC = ", ensemble_adaboost_auc, ")")) +
            ggplot2::labs(x = "Specificity", y = "Sensitivity") +
            ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
    )


# Ensembles using C50
ensemble_C50_train_fit <- C50::C5.0(as.factor(ensemble_y_train) ~ ., data = ensemble_train)

ensemble_C50_train_pred <- stats::predict(ensemble_C50_train_fit, ensemble_train, type = "prob")
ensemble_C50_train_probabilities <- ifelse(ensemble_C50_train_pred[, 2] > 0.5, 1, 0)
ensemble_C50_train_table <- table(ensemble_C50_train_probabilities, ensemble_y_train)
ensemble_C50_train_accuracy[i] <- (ensemble_C50_train_table[1, 1] + ensemble_C50_train_table[2, 2]) / sum(ensemble_C50_train_table)
ensemble_C50_train_accuracy_mean <- mean(ensemble_C50_train_accuracy)

ensemble_C50_test_pred <- stats::predict(ensemble_C50_train_fit, ensemble_test, type = "prob")
ensemble_C50_test_probabilities <- ifelse(ensemble_C50_test_pred[, 2] > 0.5, 1, 0)
ensemble_C50_test_table <- table(ensemble_C50_test_probabilities, ensemble_y_test)
ensemble_C50_test_accuracy[i] <- (ensemble_C50_test_table[1, 1] + ensemble_C50_test_table[2, 2]) / sum(ensemble_C50_test_table)
ensemble_C50_test_accuracy_mean <- mean(ensemble_C50_test_accuracy)

ensemble_C50_roc_obj <- pROC::roc(as.numeric(c(ensemble_test$y)), as.numeric(c(ensemble_C50_test_pred[, 2])))
ensemble_C50_auc <- round((pROC::auc(c(ensemble_test$y), as.numeric(c(ensemble_C50_test_pred[, 2])) - 1)), 4)
print(pROC::ggroc(ensemble_C50_roc_obj, color = "steelblue", size = 2) +
  ggplot2::ggtitle(paste0("Ensemble_C50 ", "(AUC = ", ensemble_C50_auc, ")")) +
  ggplot2::labs(x = "Specificity", y = "Sensitivity") +
  ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
)

# Ensemble Random Forest

ensemble_rf_train_fit <- randomForest(x = ensemble_train, y = as.factor(ensemble_y_train), data = ensemble1)

ensemble_rf_train_pred <- stats::predict(ensemble_rf_train_fit, ensemble_train, type = "prob")
ensemble_rf_train_predictions <- ifelse(ensemble_rf_train_pred > 0.50, 1, 0)[, 2]
ensemble_rf_train_table <- table(ensemble_rf_train_predictions, ensemble_y_train)
ensemble_rf_train_accuracy[i] <- (ensemble_rf_train_table[1, 1] + ensemble_rf_train_table[2, 2]) / sum(ensemble_rf_train_table)
ensemble_rf_train_accuracy_mean <- mean(ensemble_rf_train_accuracy)

ensemble_rf_test_pred <- stats::predict(ensemble_rf_train_fit, ensemble_test, type = "prob")
ensemble_rf_test_predictions <- ifelse(ensemble_rf_test_pred > 0.50, 1, 0)[, 2]
ensemble_rf_test_table <- table(ensemble_rf_test_predictions, ensemble_y_test)
ensemble_rf_test_accuracy[i] <- (ensemble_rf_test_table[1, 1] + ensemble_rf_test_table[2, 2]) / sum(ensemble_rf_test_table)
ensemble_rf_test_accuracy_mean <- mean(ensemble_rf_test_accuracy)

ensemble_rf_roc_obj <- pROC::roc(as.numeric(c(ensemble_test$y)), as.numeric(c(ensemble_rf_test_predictions)))
ensemble_rf_auc <- round((pROC::auc(c(ensemble_test$y), as.numeric(c(ensemble_rf_test_predictions)) - 1)), 4)
print(pROC::ggroc(ensemble_rf_roc_obj, color = "steelblue", size = 2) +
        ggplot2::ggtitle(paste0("Ensemble_rf ", "(AUC = ", ensemble_rf_auc, ")")) +
        ggplot2::labs(x = "Specificity", y = "Sensitivity") +
        ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
)

# Ensemble XGBoost

ensemble_train_x <- data.matrix(ensemble_train[, -ncol(ensemble_train)])
ensemble_train_y <- ensemble_train[, ncol(ensemble_train)]

  # define predictor and response variables in test set
ensemble_test_x <- data.matrix(ensemble_test[, -ncol(ensemble_test)])
ensemble_test_y <- ensemble_test[, ncol(ensemble_test)]

# define final train and test sets
ensemble_xgb_train <- xgboost::xgb.DMatrix(data = ensemble_train_x, label = ensemble_train_y)
ensemble_xgb_test <- xgboost::xgb.DMatrix(data = ensemble_test_x, label = ensemble_test_y)

# define watchlist
ensemble_watchlist <- list(train = ensemble_xgb_train)
ensemble_watchlist_test <- list(train = ensemble_xgb_train, test = ensemble_xgb_test)

ensemble_xgb_model <- xgboost::xgb.train(data = ensemble_xgb_train, max.depth = 3, watchlist = ensemble_watchlist_test, nrounds = 70)

ensemble_xgboost_min <- which.min(ensemble_xgb_model$evaluation_log$validation_rmse)

ensemble_xgb_train_pred <- predict(object = ensemble_xgb_model, newdata = ensemble_train_x, type = "response")
ensemble_xgb_train_probabilities <- ifelse(ensemble_xgb_train_pred > 0.5, 1, 0)
ensemble_xgb_train_table <- table(ensemble_xgb_train_probabilities, ensemble_y_train)
ensemble_xgb_train_accuracy[i] <- (ensemble_xgb_train_table[1, 1] + ensemble_xgb_train_table[2, 2]) / sum(ensemble_xgb_train_table)
ensemble_xgb_train_accuracy_mean <- mean(ensemble_xgb_train_accuracy)

ensemble_xgb_test_pred <- predict(object = ensemble_xgb_model, newdata = ensemble_test_x, type = "response")
ensemble_xgb_test_probabilities <- ifelse(ensemble_xgb_test_pred > 0.5, 1, 0)
ensemble_xgb_test_table <- table(ensemble_xgb_test_probabilities, ensemble_y_test)
ensemble_xgb_test_accuracy[i] <- (ensemble_xgb_test_table[1, 1] + ensemble_xgb_test_table[2, 2]) / sum(ensemble_xgb_test_table)
ensemble_xgb_test_accuracy_mean <- mean(ensemble_xgb_test_accuracy)

ensemble_xgb_roc_obj <- pROC::roc(as.numeric(c(ensemble_test$y)), as.numeric(c(ensemble_xgb_test_pred)))
ensemble_xgb_auc <- round((pROC::auc(c(ensemble_test$y), as.numeric(c(ensemble_xgb_test_pred)) - 1)), 4)
print(pROC::ggroc(ensemble_xgb_roc_obj, color = "steelblue", size = 2) +
  ggplot2::ggtitle(paste0("Ensemble XGBoost ", "(AUC = ", ensemble_xgb_auc, ")")) +
  ggplot2::labs(x = "Specificity", y = "Sensitivity") +
  ggplot2::annotate("segment", x = 1, xend = 0, y = 0, yend = 1, color = "grey")
)
  
  
# Save all trained models to the Environment
adaboost_train_fit <<- adaboost_train_fit
bayesglm_train_fit <<- bayesglm_train_fit
C50_train_fit<<- C50_train_fit
cubist_train_fit <<- cubist_train_fit
rf_train_fit <<- rf_train_fit
xgb_model <<- xgb_model
ensemble_adaboost_train_fit <<- ensemble_adaboost_train_fit
ensemble_C50_train_fit <<- ensemble_C50_train_fit
ensemble_rf_train_fit <<- ensemble_rf_train_fit
ensemble_xgb_model <<- ensemble_xgb_model

results <- data.frame(
  'Model'= c('ADABoost', 'BayesGLM', 'C50', 'Cubist', 'Random_Forest', 'XGBoost', 'Ensemble_ADABoost', 'Ensemble_C50', 'Ensemble_Random_Forest', 'Ensemble_XGBoost'),
  'Accuracy' = c(adaboost_test_accuracy_mean, bayesglm_test_accuracy_mean, C50_test_accuracy_mean, cubist_test_accuracy_mean, rf_test_accuracy_mean, xgb_test_accuracy_mean, ensemble_adaboost_holdout_accuracy_mean, ensemble_C50_test_accuracy_mean, ensemble_rf_test_accuracy_mean, ensemble_xgb_test_accuracy_mean)
)

results <- results %>% arrange(desc(Accuracy), Model)

} # Closing loop for numresamples
return(results)

} # Closing loop for the function

logistic_1(data = lebron, colnum = 6, numresamples = 5, train_amount = 0.60, test_amount = 0.40)
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-1.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-2.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-3.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-4.png" width="672" />

```
#> [1]	train-rmse:0.473260	test-rmse:0.480928 
#> [2]	train-rmse:0.456710	test-rmse:0.470992 
#> [3]	train-rmse:0.447663	test-rmse:0.467005 
#> [4]	train-rmse:0.440178	test-rmse:0.465661 
#> [5]	train-rmse:0.434177	test-rmse:0.466090 
#> [6]	train-rmse:0.428811	test-rmse:0.467722 
#> [7]	train-rmse:0.423727	test-rmse:0.466571 
#> [8]	train-rmse:0.421580	test-rmse:0.466593 
#> [9]	train-rmse:0.417858	test-rmse:0.467885 
#> [10]	train-rmse:0.413965	test-rmse:0.467015 
#> [11]	train-rmse:0.410669	test-rmse:0.466459 
#> [12]	train-rmse:0.405800	test-rmse:0.467934 
#> [13]	train-rmse:0.402774	test-rmse:0.468156 
#> [14]	train-rmse:0.399552	test-rmse:0.468364 
#> [15]	train-rmse:0.396837	test-rmse:0.469123 
#> [16]	train-rmse:0.395868	test-rmse:0.468896 
#> [17]	train-rmse:0.393769	test-rmse:0.469476 
#> [18]	train-rmse:0.392662	test-rmse:0.469459 
#> [19]	train-rmse:0.388343	test-rmse:0.471196 
#> [20]	train-rmse:0.385616	test-rmse:0.473240 
#> [21]	train-rmse:0.382860	test-rmse:0.472697 
#> [22]	train-rmse:0.379975	test-rmse:0.472335 
#> [23]	train-rmse:0.376858	test-rmse:0.472360 
#> [24]	train-rmse:0.375869	test-rmse:0.472726 
#> [25]	train-rmse:0.373040	test-rmse:0.472583 
#> [26]	train-rmse:0.371312	test-rmse:0.472935 
#> [27]	train-rmse:0.370105	test-rmse:0.473267 
#> [28]	train-rmse:0.367494	test-rmse:0.473389 
#> [29]	train-rmse:0.364031	test-rmse:0.474244 
#> [30]	train-rmse:0.363426	test-rmse:0.474258 
#> [31]	train-rmse:0.361620	test-rmse:0.475409 
#> [32]	train-rmse:0.360143	test-rmse:0.477001 
#> [33]	train-rmse:0.357859	test-rmse:0.477171 
#> [34]	train-rmse:0.355857	test-rmse:0.476392 
#> [35]	train-rmse:0.353645	test-rmse:0.476856 
#> [36]	train-rmse:0.350245	test-rmse:0.477487 
#> [37]	train-rmse:0.349426	test-rmse:0.478119 
#> [38]	train-rmse:0.346989	test-rmse:0.479051 
#> [39]	train-rmse:0.344126	test-rmse:0.478877 
#> [40]	train-rmse:0.342363	test-rmse:0.479401 
#> [41]	train-rmse:0.340939	test-rmse:0.480583 
#> [42]	train-rmse:0.339069	test-rmse:0.481435 
#> [43]	train-rmse:0.338761	test-rmse:0.481577 
#> [44]	train-rmse:0.336075	test-rmse:0.482866 
#> [45]	train-rmse:0.334078	test-rmse:0.485063 
#> [46]	train-rmse:0.332140	test-rmse:0.485620 
#> [47]	train-rmse:0.329294	test-rmse:0.486408 
#> [48]	train-rmse:0.326892	test-rmse:0.487116 
#> [49]	train-rmse:0.325316	test-rmse:0.487400 
#> [50]	train-rmse:0.322989	test-rmse:0.487219 
#> [51]	train-rmse:0.322548	test-rmse:0.487497 
#> [52]	train-rmse:0.320879	test-rmse:0.486324 
#> [53]	train-rmse:0.319337	test-rmse:0.485726 
#> [54]	train-rmse:0.317468	test-rmse:0.485902 
#> [55]	train-rmse:0.316826	test-rmse:0.485982 
#> [56]	train-rmse:0.314655	test-rmse:0.486821 
#> [57]	train-rmse:0.312849	test-rmse:0.487355 
#> [58]	train-rmse:0.312045	test-rmse:0.487324 
#> [59]	train-rmse:0.310744	test-rmse:0.487221 
#> [60]	train-rmse:0.308556	test-rmse:0.488889 
#> [61]	train-rmse:0.306704	test-rmse:0.488557 
#> [62]	train-rmse:0.305717	test-rmse:0.488832 
#> [63]	train-rmse:0.302922	test-rmse:0.489763 
#> [64]	train-rmse:0.300995	test-rmse:0.490850 
#> [65]	train-rmse:0.299050	test-rmse:0.491371 
#> [66]	train-rmse:0.297149	test-rmse:0.492290 
#> [67]	train-rmse:0.295700	test-rmse:0.492574 
#> [68]	train-rmse:0.295242	test-rmse:0.492833 
#> [69]	train-rmse:0.294881	test-rmse:0.492807 
#> [70]	train-rmse:0.293163	test-rmse:0.493658
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-5.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-6.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-7.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-8.png" width="672" />

```
#> [1]	train-rmse:0.350802	test-rmse:0.350804 
#> [2]	train-rmse:0.246124	test-rmse:0.246126 
#> [3]	train-rmse:0.172682	test-rmse:0.172684 
#> [4]	train-rmse:0.121154	test-rmse:0.121156 
#> [5]	train-rmse:0.085002	test-rmse:0.085004 
#> [6]	train-rmse:0.059638	test-rmse:0.059640 
#> [7]	train-rmse:0.041842	test-rmse:0.041844 
#> [8]	train-rmse:0.029357	test-rmse:0.029358 
#> [9]	train-rmse:0.020597	test-rmse:0.020598 
#> [10]	train-rmse:0.014451	test-rmse:0.014451 
#> [11]	train-rmse:0.010139	test-rmse:0.010139 
#> [12]	train-rmse:0.007113	test-rmse:0.007114 
#> [13]	train-rmse:0.004991	test-rmse:0.004991 
#> [14]	train-rmse:0.003502	test-rmse:0.003502 
#> [15]	train-rmse:0.002457	test-rmse:0.002457 
#> [16]	train-rmse:0.001724	test-rmse:0.001724 
#> [17]	train-rmse:0.001209	test-rmse:0.001209 
#> [18]	train-rmse:0.000848	test-rmse:0.000849 
#> [19]	train-rmse:0.000595	test-rmse:0.000595 
#> [20]	train-rmse:0.000418	test-rmse:0.000418 
#> [21]	train-rmse:0.000293	test-rmse:0.000293 
#> [22]	train-rmse:0.000206	test-rmse:0.000206 
#> [23]	train-rmse:0.000144	test-rmse:0.000144 
#> [24]	train-rmse:0.000101	test-rmse:0.000101 
#> [25]	train-rmse:0.000071	test-rmse:0.000071 
#> [26]	train-rmse:0.000050	test-rmse:0.000050 
#> [27]	train-rmse:0.000050	test-rmse:0.000050 
#> [28]	train-rmse:0.000050	test-rmse:0.000050 
#> [29]	train-rmse:0.000050	test-rmse:0.000050 
#> [30]	train-rmse:0.000050	test-rmse:0.000050 
#> [31]	train-rmse:0.000050	test-rmse:0.000050 
#> [32]	train-rmse:0.000050	test-rmse:0.000050 
#> [33]	train-rmse:0.000050	test-rmse:0.000050 
#> [34]	train-rmse:0.000050	test-rmse:0.000050 
#> [35]	train-rmse:0.000050	test-rmse:0.000050 
#> [36]	train-rmse:0.000050	test-rmse:0.000050 
#> [37]	train-rmse:0.000050	test-rmse:0.000050 
#> [38]	train-rmse:0.000050	test-rmse:0.000050 
#> [39]	train-rmse:0.000050	test-rmse:0.000050 
#> [40]	train-rmse:0.000050	test-rmse:0.000050 
#> [41]	train-rmse:0.000050	test-rmse:0.000050 
#> [42]	train-rmse:0.000050	test-rmse:0.000050 
#> [43]	train-rmse:0.000050	test-rmse:0.000050 
#> [44]	train-rmse:0.000050	test-rmse:0.000050 
#> [45]	train-rmse:0.000050	test-rmse:0.000050 
#> [46]	train-rmse:0.000050	test-rmse:0.000050 
#> [47]	train-rmse:0.000050	test-rmse:0.000050 
#> [48]	train-rmse:0.000050	test-rmse:0.000050 
#> [49]	train-rmse:0.000050	test-rmse:0.000050 
#> [50]	train-rmse:0.000050	test-rmse:0.000050 
#> [51]	train-rmse:0.000050	test-rmse:0.000050 
#> [52]	train-rmse:0.000050	test-rmse:0.000050 
#> [53]	train-rmse:0.000050	test-rmse:0.000050 
#> [54]	train-rmse:0.000050	test-rmse:0.000050 
#> [55]	train-rmse:0.000050	test-rmse:0.000050 
#> [56]	train-rmse:0.000050	test-rmse:0.000050 
#> [57]	train-rmse:0.000050	test-rmse:0.000050 
#> [58]	train-rmse:0.000050	test-rmse:0.000050 
#> [59]	train-rmse:0.000050	test-rmse:0.000050 
#> [60]	train-rmse:0.000050	test-rmse:0.000050 
#> [61]	train-rmse:0.000050	test-rmse:0.000050 
#> [62]	train-rmse:0.000050	test-rmse:0.000050 
#> [63]	train-rmse:0.000050	test-rmse:0.000050 
#> [64]	train-rmse:0.000050	test-rmse:0.000050 
#> [65]	train-rmse:0.000050	test-rmse:0.000050 
#> [66]	train-rmse:0.000050	test-rmse:0.000050 
#> [67]	train-rmse:0.000050	test-rmse:0.000050 
#> [68]	train-rmse:0.000050	test-rmse:0.000050 
#> [69]	train-rmse:0.000050	test-rmse:0.000050 
#> [70]	train-rmse:0.000050	test-rmse:0.000050
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-9.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-10.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-11.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-12.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-13.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-14.png" width="672" />

```
#> [1]	train-rmse:0.475826	test-rmse:0.481500 
#> [2]	train-rmse:0.459657	test-rmse:0.472443 
#> [3]	train-rmse:0.450397	test-rmse:0.465385 
#> [4]	train-rmse:0.442450	test-rmse:0.462061 
#> [5]	train-rmse:0.434957	test-rmse:0.462729 
#> [6]	train-rmse:0.429407	test-rmse:0.461732 
#> [7]	train-rmse:0.424977	test-rmse:0.462985 
#> [8]	train-rmse:0.420224	test-rmse:0.462965 
#> [9]	train-rmse:0.415484	test-rmse:0.462735 
#> [10]	train-rmse:0.413699	test-rmse:0.462835 
#> [11]	train-rmse:0.408541	test-rmse:0.464830 
#> [12]	train-rmse:0.405177	test-rmse:0.462881 
#> [13]	train-rmse:0.403092	test-rmse:0.464161 
#> [14]	train-rmse:0.400489	test-rmse:0.465113 
#> [15]	train-rmse:0.399295	test-rmse:0.465720 
#> [16]	train-rmse:0.396391	test-rmse:0.467182 
#> [17]	train-rmse:0.393456	test-rmse:0.469140 
#> [18]	train-rmse:0.390611	test-rmse:0.468779 
#> [19]	train-rmse:0.388942	test-rmse:0.468704 
#> [20]	train-rmse:0.384047	test-rmse:0.470124 
#> [21]	train-rmse:0.381096	test-rmse:0.470402 
#> [22]	train-rmse:0.380055	test-rmse:0.470720 
#> [23]	train-rmse:0.377648	test-rmse:0.470391 
#> [24]	train-rmse:0.374757	test-rmse:0.470524 
#> [25]	train-rmse:0.372815	test-rmse:0.470675 
#> [26]	train-rmse:0.372197	test-rmse:0.471392 
#> [27]	train-rmse:0.370336	test-rmse:0.472008 
#> [28]	train-rmse:0.369157	test-rmse:0.472623 
#> [29]	train-rmse:0.367782	test-rmse:0.473329 
#> [30]	train-rmse:0.365917	test-rmse:0.473106 
#> [31]	train-rmse:0.363801	test-rmse:0.474664 
#> [32]	train-rmse:0.361328	test-rmse:0.474565 
#> [33]	train-rmse:0.359178	test-rmse:0.474916 
#> [34]	train-rmse:0.357302	test-rmse:0.475243 
#> [35]	train-rmse:0.354064	test-rmse:0.476413 
#> [36]	train-rmse:0.351931	test-rmse:0.477776 
#> [37]	train-rmse:0.350072	test-rmse:0.478116 
#> [38]	train-rmse:0.349207	test-rmse:0.478193 
#> [39]	train-rmse:0.346550	test-rmse:0.479690 
#> [40]	train-rmse:0.343628	test-rmse:0.478700 
#> [41]	train-rmse:0.343019	test-rmse:0.478809 
#> [42]	train-rmse:0.340757	test-rmse:0.480682 
#> [43]	train-rmse:0.338739	test-rmse:0.481500 
#> [44]	train-rmse:0.337990	test-rmse:0.481779 
#> [45]	train-rmse:0.335377	test-rmse:0.482366 
#> [46]	train-rmse:0.334332	test-rmse:0.482275 
#> [47]	train-rmse:0.332337	test-rmse:0.484199 
#> [48]	train-rmse:0.331546	test-rmse:0.484268 
#> [49]	train-rmse:0.330369	test-rmse:0.484690 
#> [50]	train-rmse:0.329295	test-rmse:0.484644 
#> [51]	train-rmse:0.326743	test-rmse:0.486167 
#> [52]	train-rmse:0.324712	test-rmse:0.486878 
#> [53]	train-rmse:0.322148	test-rmse:0.487101 
#> [54]	train-rmse:0.320578	test-rmse:0.486838 
#> [55]	train-rmse:0.319698	test-rmse:0.487108 
#> [56]	train-rmse:0.318283	test-rmse:0.487043 
#> [57]	train-rmse:0.315816	test-rmse:0.486793 
#> [58]	train-rmse:0.314033	test-rmse:0.487445 
#> [59]	train-rmse:0.312157	test-rmse:0.487611 
#> [60]	train-rmse:0.311287	test-rmse:0.487606 
#> [61]	train-rmse:0.310028	test-rmse:0.487976 
#> [62]	train-rmse:0.308745	test-rmse:0.489266 
#> [63]	train-rmse:0.307336	test-rmse:0.489580 
#> [64]	train-rmse:0.306699	test-rmse:0.490017 
#> [65]	train-rmse:0.306391	test-rmse:0.490565 
#> [66]	train-rmse:0.305921	test-rmse:0.490665 
#> [67]	train-rmse:0.303994	test-rmse:0.490936 
#> [68]	train-rmse:0.302247	test-rmse:0.490894 
#> [69]	train-rmse:0.301744	test-rmse:0.491388 
#> [70]	train-rmse:0.299744	test-rmse:0.492478
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-15.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-16.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-17.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-18.png" width="672" />

```
#> [1]	train-rmse:0.350798	test-rmse:0.350798 
#> [2]	train-rmse:0.246118	test-rmse:0.246119 
#> [3]	train-rmse:0.172676	test-rmse:0.172676 
#> [4]	train-rmse:0.121148	test-rmse:0.121149 
#> [5]	train-rmse:0.084997	test-rmse:0.084998 
#> [6]	train-rmse:0.059634	test-rmse:0.059634 
#> [7]	train-rmse:0.041839	test-rmse:0.041839 
#> [8]	train-rmse:0.029354	test-rmse:0.029354 
#> [9]	train-rmse:0.020595	test-rmse:0.020595 
#> [10]	train-rmse:0.014449	test-rmse:0.014449 
#> [11]	train-rmse:0.010137	test-rmse:0.010138 
#> [12]	train-rmse:0.007112	test-rmse:0.007112 
#> [13]	train-rmse:0.004990	test-rmse:0.004990 
#> [14]	train-rmse:0.003501	test-rmse:0.003501 
#> [15]	train-rmse:0.002456	test-rmse:0.002456 
#> [16]	train-rmse:0.001723	test-rmse:0.001723 
#> [17]	train-rmse:0.001209	test-rmse:0.001209 
#> [18]	train-rmse:0.000848	test-rmse:0.000848 
#> [19]	train-rmse:0.000595	test-rmse:0.000595 
#> [20]	train-rmse:0.000418	test-rmse:0.000418 
#> [21]	train-rmse:0.000293	test-rmse:0.000293 
#> [22]	train-rmse:0.000206	test-rmse:0.000206 
#> [23]	train-rmse:0.000144	test-rmse:0.000144 
#> [24]	train-rmse:0.000101	test-rmse:0.000101 
#> [25]	train-rmse:0.000071	test-rmse:0.000071 
#> [26]	train-rmse:0.000050	test-rmse:0.000050 
#> [27]	train-rmse:0.000050	test-rmse:0.000050 
#> [28]	train-rmse:0.000050	test-rmse:0.000050 
#> [29]	train-rmse:0.000050	test-rmse:0.000050 
#> [30]	train-rmse:0.000050	test-rmse:0.000050 
#> [31]	train-rmse:0.000050	test-rmse:0.000050 
#> [32]	train-rmse:0.000050	test-rmse:0.000050 
#> [33]	train-rmse:0.000050	test-rmse:0.000050 
#> [34]	train-rmse:0.000050	test-rmse:0.000050 
#> [35]	train-rmse:0.000050	test-rmse:0.000050 
#> [36]	train-rmse:0.000050	test-rmse:0.000050 
#> [37]	train-rmse:0.000050	test-rmse:0.000050 
#> [38]	train-rmse:0.000050	test-rmse:0.000050 
#> [39]	train-rmse:0.000050	test-rmse:0.000050 
#> [40]	train-rmse:0.000050	test-rmse:0.000050 
#> [41]	train-rmse:0.000050	test-rmse:0.000050 
#> [42]	train-rmse:0.000050	test-rmse:0.000050 
#> [43]	train-rmse:0.000050	test-rmse:0.000050 
#> [44]	train-rmse:0.000050	test-rmse:0.000050 
#> [45]	train-rmse:0.000050	test-rmse:0.000050 
#> [46]	train-rmse:0.000050	test-rmse:0.000050 
#> [47]	train-rmse:0.000050	test-rmse:0.000050 
#> [48]	train-rmse:0.000050	test-rmse:0.000050 
#> [49]	train-rmse:0.000050	test-rmse:0.000050 
#> [50]	train-rmse:0.000050	test-rmse:0.000050 
#> [51]	train-rmse:0.000050	test-rmse:0.000050 
#> [52]	train-rmse:0.000050	test-rmse:0.000050 
#> [53]	train-rmse:0.000050	test-rmse:0.000050 
#> [54]	train-rmse:0.000050	test-rmse:0.000050 
#> [55]	train-rmse:0.000050	test-rmse:0.000050 
#> [56]	train-rmse:0.000050	test-rmse:0.000050 
#> [57]	train-rmse:0.000050	test-rmse:0.000050 
#> [58]	train-rmse:0.000050	test-rmse:0.000050 
#> [59]	train-rmse:0.000050	test-rmse:0.000050 
#> [60]	train-rmse:0.000050	test-rmse:0.000050 
#> [61]	train-rmse:0.000050	test-rmse:0.000050 
#> [62]	train-rmse:0.000050	test-rmse:0.000050 
#> [63]	train-rmse:0.000050	test-rmse:0.000050 
#> [64]	train-rmse:0.000050	test-rmse:0.000050 
#> [65]	train-rmse:0.000050	test-rmse:0.000050 
#> [66]	train-rmse:0.000050	test-rmse:0.000050 
#> [67]	train-rmse:0.000050	test-rmse:0.000050 
#> [68]	train-rmse:0.000050	test-rmse:0.000050 
#> [69]	train-rmse:0.000050	test-rmse:0.000050 
#> [70]	train-rmse:0.000050	test-rmse:0.000050
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-19.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-20.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-21.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-22.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-23.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-24.png" width="672" />

```
#> [1]	train-rmse:0.472769	test-rmse:0.482654 
#> [2]	train-rmse:0.455927	test-rmse:0.474974 
#> [3]	train-rmse:0.445050	test-rmse:0.471813 
#> [4]	train-rmse:0.437652	test-rmse:0.470455 
#> [5]	train-rmse:0.432534	test-rmse:0.472091 
#> [6]	train-rmse:0.427086	test-rmse:0.471085 
#> [7]	train-rmse:0.424693	test-rmse:0.470329 
#> [8]	train-rmse:0.423074	test-rmse:0.469975 
#> [9]	train-rmse:0.419267	test-rmse:0.468250 
#> [10]	train-rmse:0.416963	test-rmse:0.468207 
#> [11]	train-rmse:0.413444	test-rmse:0.467577 
#> [12]	train-rmse:0.410447	test-rmse:0.468505 
#> [13]	train-rmse:0.406754	test-rmse:0.468206 
#> [14]	train-rmse:0.403631	test-rmse:0.469170 
#> [15]	train-rmse:0.402640	test-rmse:0.469300 
#> [16]	train-rmse:0.401049	test-rmse:0.469462 
#> [17]	train-rmse:0.397761	test-rmse:0.469707 
#> [18]	train-rmse:0.395251	test-rmse:0.469630 
#> [19]	train-rmse:0.392317	test-rmse:0.468573 
#> [20]	train-rmse:0.389907	test-rmse:0.468384 
#> [21]	train-rmse:0.386263	test-rmse:0.467207 
#> [22]	train-rmse:0.385092	test-rmse:0.466582 
#> [23]	train-rmse:0.382947	test-rmse:0.468251 
#> [24]	train-rmse:0.379480	test-rmse:0.469186 
#> [25]	train-rmse:0.377447	test-rmse:0.469316 
#> [26]	train-rmse:0.374702	test-rmse:0.469646 
#> [27]	train-rmse:0.371630	test-rmse:0.469389 
#> [28]	train-rmse:0.369563	test-rmse:0.469242 
#> [29]	train-rmse:0.367447	test-rmse:0.469919 
#> [30]	train-rmse:0.364762	test-rmse:0.468792 
#> [31]	train-rmse:0.362708	test-rmse:0.469501 
#> [32]	train-rmse:0.359730	test-rmse:0.469661 
#> [33]	train-rmse:0.357905	test-rmse:0.469505 
#> [34]	train-rmse:0.356490	test-rmse:0.470354 
#> [35]	train-rmse:0.355746	test-rmse:0.470424 
#> [36]	train-rmse:0.353715	test-rmse:0.470998 
#> [37]	train-rmse:0.352081	test-rmse:0.471469 
#> [38]	train-rmse:0.348582	test-rmse:0.472291 
#> [39]	train-rmse:0.347687	test-rmse:0.472313 
#> [40]	train-rmse:0.345912	test-rmse:0.472007 
#> [41]	train-rmse:0.345686	test-rmse:0.471909 
#> [42]	train-rmse:0.343115	test-rmse:0.470923 
#> [43]	train-rmse:0.341945	test-rmse:0.470430 
#> [44]	train-rmse:0.340025	test-rmse:0.470453 
#> [45]	train-rmse:0.338328	test-rmse:0.470725 
#> [46]	train-rmse:0.337134	test-rmse:0.471166 
#> [47]	train-rmse:0.335030	test-rmse:0.471723 
#> [48]	train-rmse:0.333215	test-rmse:0.472744 
#> [49]	train-rmse:0.332176	test-rmse:0.472636 
#> [50]	train-rmse:0.330275	test-rmse:0.472985 
#> [51]	train-rmse:0.328481	test-rmse:0.473944 
#> [52]	train-rmse:0.328104	test-rmse:0.474638 
#> [53]	train-rmse:0.326592	test-rmse:0.474324 
#> [54]	train-rmse:0.325568	test-rmse:0.474741 
#> [55]	train-rmse:0.323967	test-rmse:0.474859 
#> [56]	train-rmse:0.322216	test-rmse:0.475386 
#> [57]	train-rmse:0.320249	test-rmse:0.477011 
#> [58]	train-rmse:0.318805	test-rmse:0.477852 
#> [59]	train-rmse:0.318007	test-rmse:0.478533 
#> [60]	train-rmse:0.314959	test-rmse:0.478792 
#> [61]	train-rmse:0.313552	test-rmse:0.478426 
#> [62]	train-rmse:0.311992	test-rmse:0.477958 
#> [63]	train-rmse:0.309770	test-rmse:0.477936 
#> [64]	train-rmse:0.308589	test-rmse:0.477971 
#> [65]	train-rmse:0.307905	test-rmse:0.478684 
#> [66]	train-rmse:0.305894	test-rmse:0.479162 
#> [67]	train-rmse:0.304314	test-rmse:0.479473 
#> [68]	train-rmse:0.302703	test-rmse:0.479354 
#> [69]	train-rmse:0.301543	test-rmse:0.479396 
#> [70]	train-rmse:0.301132	test-rmse:0.479597
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-25.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-26.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-27.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-28.png" width="672" />

```
#> [1]	train-rmse:0.350787	test-rmse:0.350788 
#> [2]	train-rmse:0.246104	test-rmse:0.246105 
#> [3]	train-rmse:0.172660	test-rmse:0.172661 
#> [4]	train-rmse:0.121134	test-rmse:0.121135 
#> [5]	train-rmse:0.084985	test-rmse:0.084986 
#> [6]	train-rmse:0.059623	test-rmse:0.059624 
#> [7]	train-rmse:0.041830	test-rmse:0.041831 
#> [8]	train-rmse:0.029347	test-rmse:0.029347 
#> [9]	train-rmse:0.020589	test-rmse:0.020590 
#> [10]	train-rmse:0.014445	test-rmse:0.014445 
#> [11]	train-rmse:0.010134	test-rmse:0.010134 
#> [12]	train-rmse:0.007110	test-rmse:0.007110 
#> [13]	train-rmse:0.004988	test-rmse:0.004988 
#> [14]	train-rmse:0.003500	test-rmse:0.003500 
#> [15]	train-rmse:0.002455	test-rmse:0.002455 
#> [16]	train-rmse:0.001722	test-rmse:0.001723 
#> [17]	train-rmse:0.001208	test-rmse:0.001209 
#> [18]	train-rmse:0.000848	test-rmse:0.000848 
#> [19]	train-rmse:0.000595	test-rmse:0.000595 
#> [20]	train-rmse:0.000417	test-rmse:0.000417 
#> [21]	train-rmse:0.000293	test-rmse:0.000293 
#> [22]	train-rmse:0.000205	test-rmse:0.000205 
#> [23]	train-rmse:0.000144	test-rmse:0.000144 
#> [24]	train-rmse:0.000101	test-rmse:0.000101 
#> [25]	train-rmse:0.000071	test-rmse:0.000071 
#> [26]	train-rmse:0.000050	test-rmse:0.000050 
#> [27]	train-rmse:0.000050	test-rmse:0.000050 
#> [28]	train-rmse:0.000050	test-rmse:0.000050 
#> [29]	train-rmse:0.000050	test-rmse:0.000050 
#> [30]	train-rmse:0.000050	test-rmse:0.000050 
#> [31]	train-rmse:0.000050	test-rmse:0.000050 
#> [32]	train-rmse:0.000050	test-rmse:0.000050 
#> [33]	train-rmse:0.000050	test-rmse:0.000050 
#> [34]	train-rmse:0.000050	test-rmse:0.000050 
#> [35]	train-rmse:0.000050	test-rmse:0.000050 
#> [36]	train-rmse:0.000050	test-rmse:0.000050 
#> [37]	train-rmse:0.000050	test-rmse:0.000050 
#> [38]	train-rmse:0.000050	test-rmse:0.000050 
#> [39]	train-rmse:0.000050	test-rmse:0.000050 
#> [40]	train-rmse:0.000050	test-rmse:0.000050 
#> [41]	train-rmse:0.000050	test-rmse:0.000050 
#> [42]	train-rmse:0.000050	test-rmse:0.000050 
#> [43]	train-rmse:0.000050	test-rmse:0.000050 
#> [44]	train-rmse:0.000050	test-rmse:0.000050 
#> [45]	train-rmse:0.000050	test-rmse:0.000050 
#> [46]	train-rmse:0.000050	test-rmse:0.000050 
#> [47]	train-rmse:0.000050	test-rmse:0.000050 
#> [48]	train-rmse:0.000050	test-rmse:0.000050 
#> [49]	train-rmse:0.000050	test-rmse:0.000050 
#> [50]	train-rmse:0.000050	test-rmse:0.000050 
#> [51]	train-rmse:0.000050	test-rmse:0.000050 
#> [52]	train-rmse:0.000050	test-rmse:0.000050 
#> [53]	train-rmse:0.000050	test-rmse:0.000050 
#> [54]	train-rmse:0.000050	test-rmse:0.000050 
#> [55]	train-rmse:0.000050	test-rmse:0.000050 
#> [56]	train-rmse:0.000050	test-rmse:0.000050 
#> [57]	train-rmse:0.000050	test-rmse:0.000050 
#> [58]	train-rmse:0.000050	test-rmse:0.000050 
#> [59]	train-rmse:0.000050	test-rmse:0.000050 
#> [60]	train-rmse:0.000050	test-rmse:0.000050 
#> [61]	train-rmse:0.000050	test-rmse:0.000050 
#> [62]	train-rmse:0.000050	test-rmse:0.000050 
#> [63]	train-rmse:0.000050	test-rmse:0.000050 
#> [64]	train-rmse:0.000050	test-rmse:0.000050 
#> [65]	train-rmse:0.000050	test-rmse:0.000050 
#> [66]	train-rmse:0.000050	test-rmse:0.000050 
#> [67]	train-rmse:0.000050	test-rmse:0.000050 
#> [68]	train-rmse:0.000050	test-rmse:0.000050 
#> [69]	train-rmse:0.000050	test-rmse:0.000050 
#> [70]	train-rmse:0.000050	test-rmse:0.000050
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-29.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-30.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-31.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-32.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-33.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-34.png" width="672" />

```
#> [1]	train-rmse:0.475532	test-rmse:0.479722 
#> [2]	train-rmse:0.461715	test-rmse:0.469507 
#> [3]	train-rmse:0.450855	test-rmse:0.461037 
#> [4]	train-rmse:0.443802	test-rmse:0.459167 
#> [5]	train-rmse:0.438438	test-rmse:0.457998 
#> [6]	train-rmse:0.434583	test-rmse:0.458561 
#> [7]	train-rmse:0.431620	test-rmse:0.455979 
#> [8]	train-rmse:0.428429	test-rmse:0.455833 
#> [9]	train-rmse:0.426257	test-rmse:0.456038 
#> [10]	train-rmse:0.422599	test-rmse:0.456239 
#> [11]	train-rmse:0.418804	test-rmse:0.455760 
#> [12]	train-rmse:0.417226	test-rmse:0.455697 
#> [13]	train-rmse:0.414559	test-rmse:0.456316 
#> [14]	train-rmse:0.411412	test-rmse:0.455462 
#> [15]	train-rmse:0.408283	test-rmse:0.455988 
#> [16]	train-rmse:0.407567	test-rmse:0.455761 
#> [17]	train-rmse:0.405276	test-rmse:0.456471 
#> [18]	train-rmse:0.403403	test-rmse:0.457584 
#> [19]	train-rmse:0.400764	test-rmse:0.457745 
#> [20]	train-rmse:0.399792	test-rmse:0.458031 
#> [21]	train-rmse:0.396750	test-rmse:0.458245 
#> [22]	train-rmse:0.393334	test-rmse:0.459049 
#> [23]	train-rmse:0.391641	test-rmse:0.460172 
#> [24]	train-rmse:0.390145	test-rmse:0.459686 
#> [25]	train-rmse:0.386848	test-rmse:0.461748 
#> [26]	train-rmse:0.383815	test-rmse:0.462019 
#> [27]	train-rmse:0.382860	test-rmse:0.461958 
#> [28]	train-rmse:0.380752	test-rmse:0.462361 
#> [29]	train-rmse:0.379025	test-rmse:0.463454 
#> [30]	train-rmse:0.376613	test-rmse:0.463279 
#> [31]	train-rmse:0.375179	test-rmse:0.462394 
#> [32]	train-rmse:0.372078	test-rmse:0.462297 
#> [33]	train-rmse:0.370390	test-rmse:0.461997 
#> [34]	train-rmse:0.367370	test-rmse:0.461258 
#> [35]	train-rmse:0.365171	test-rmse:0.461642 
#> [36]	train-rmse:0.364556	test-rmse:0.462097 
#> [37]	train-rmse:0.362250	test-rmse:0.463210 
#> [38]	train-rmse:0.360822	test-rmse:0.463464 
#> [39]	train-rmse:0.359736	test-rmse:0.464817 
#> [40]	train-rmse:0.357757	test-rmse:0.465533 
#> [41]	train-rmse:0.355711	test-rmse:0.464467 
#> [42]	train-rmse:0.353179	test-rmse:0.464570 
#> [43]	train-rmse:0.350284	test-rmse:0.465779 
#> [44]	train-rmse:0.348136	test-rmse:0.466365 
#> [45]	train-rmse:0.346818	test-rmse:0.466977 
#> [46]	train-rmse:0.344784	test-rmse:0.466445 
#> [47]	train-rmse:0.343106	test-rmse:0.466392 
#> [48]	train-rmse:0.340745	test-rmse:0.466662 
#> [49]	train-rmse:0.338533	test-rmse:0.466695 
#> [50]	train-rmse:0.337674	test-rmse:0.466918 
#> [51]	train-rmse:0.335752	test-rmse:0.467161 
#> [52]	train-rmse:0.333461	test-rmse:0.467704 
#> [53]	train-rmse:0.331401	test-rmse:0.467842 
#> [54]	train-rmse:0.329978	test-rmse:0.468242 
#> [55]	train-rmse:0.328074	test-rmse:0.469040 
#> [56]	train-rmse:0.327299	test-rmse:0.469125 
#> [57]	train-rmse:0.326539	test-rmse:0.469162 
#> [58]	train-rmse:0.325743	test-rmse:0.469178 
#> [59]	train-rmse:0.323197	test-rmse:0.468921 
#> [60]	train-rmse:0.321051	test-rmse:0.469180 
#> [61]	train-rmse:0.318822	test-rmse:0.468901 
#> [62]	train-rmse:0.317071	test-rmse:0.468493 
#> [63]	train-rmse:0.315341	test-rmse:0.469553 
#> [64]	train-rmse:0.313901	test-rmse:0.470131 
#> [65]	train-rmse:0.312277	test-rmse:0.470533 
#> [66]	train-rmse:0.310928	test-rmse:0.470303 
#> [67]	train-rmse:0.309997	test-rmse:0.470076 
#> [68]	train-rmse:0.309665	test-rmse:0.470048 
#> [69]	train-rmse:0.308328	test-rmse:0.470354 
#> [70]	train-rmse:0.306956	test-rmse:0.470402
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-35.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-36.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-37.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-38.png" width="672" />

```
#> [1]	train-rmse:0.350813	test-rmse:0.350813 
#> [2]	train-rmse:0.246140	test-rmse:0.246140 
#> [3]	train-rmse:0.172698	test-rmse:0.172698 
#> [4]	train-rmse:0.121169	test-rmse:0.121170 
#> [5]	train-rmse:0.085016	test-rmse:0.085016 
#> [6]	train-rmse:0.059649	test-rmse:0.059649 
#> [7]	train-rmse:0.041851	test-rmse:0.041851 
#> [8]	train-rmse:0.029364	test-rmse:0.029364 
#> [9]	train-rmse:0.020603	test-rmse:0.020603 
#> [10]	train-rmse:0.014455	test-rmse:0.014455 
#> [11]	train-rmse:0.010142	test-rmse:0.010142 
#> [12]	train-rmse:0.007116	test-rmse:0.007116 
#> [13]	train-rmse:0.004993	test-rmse:0.004993 
#> [14]	train-rmse:0.003503	test-rmse:0.003503 
#> [15]	train-rmse:0.002458	test-rmse:0.002458 
#> [16]	train-rmse:0.001724	test-rmse:0.001724 
#> [17]	train-rmse:0.001210	test-rmse:0.001210 
#> [18]	train-rmse:0.000849	test-rmse:0.000849 
#> [19]	train-rmse:0.000596	test-rmse:0.000596 
#> [20]	train-rmse:0.000418	test-rmse:0.000418 
#> [21]	train-rmse:0.000293	test-rmse:0.000293 
#> [22]	train-rmse:0.000206	test-rmse:0.000206 
#> [23]	train-rmse:0.000144	test-rmse:0.000144 
#> [24]	train-rmse:0.000101	test-rmse:0.000101 
#> [25]	train-rmse:0.000071	test-rmse:0.000071 
#> [26]	train-rmse:0.000050	test-rmse:0.000050 
#> [27]	train-rmse:0.000050	test-rmse:0.000050 
#> [28]	train-rmse:0.000050	test-rmse:0.000050 
#> [29]	train-rmse:0.000050	test-rmse:0.000050 
#> [30]	train-rmse:0.000050	test-rmse:0.000050 
#> [31]	train-rmse:0.000050	test-rmse:0.000050 
#> [32]	train-rmse:0.000050	test-rmse:0.000050 
#> [33]	train-rmse:0.000050	test-rmse:0.000050 
#> [34]	train-rmse:0.000050	test-rmse:0.000050 
#> [35]	train-rmse:0.000050	test-rmse:0.000050 
#> [36]	train-rmse:0.000050	test-rmse:0.000050 
#> [37]	train-rmse:0.000050	test-rmse:0.000050 
#> [38]	train-rmse:0.000050	test-rmse:0.000050 
#> [39]	train-rmse:0.000050	test-rmse:0.000050 
#> [40]	train-rmse:0.000050	test-rmse:0.000050 
#> [41]	train-rmse:0.000050	test-rmse:0.000050 
#> [42]	train-rmse:0.000050	test-rmse:0.000050 
#> [43]	train-rmse:0.000050	test-rmse:0.000050 
#> [44]	train-rmse:0.000050	test-rmse:0.000050 
#> [45]	train-rmse:0.000050	test-rmse:0.000050 
#> [46]	train-rmse:0.000050	test-rmse:0.000050 
#> [47]	train-rmse:0.000050	test-rmse:0.000050 
#> [48]	train-rmse:0.000050	test-rmse:0.000050 
#> [49]	train-rmse:0.000050	test-rmse:0.000050 
#> [50]	train-rmse:0.000050	test-rmse:0.000050 
#> [51]	train-rmse:0.000050	test-rmse:0.000050 
#> [52]	train-rmse:0.000050	test-rmse:0.000050 
#> [53]	train-rmse:0.000050	test-rmse:0.000050 
#> [54]	train-rmse:0.000050	test-rmse:0.000050 
#> [55]	train-rmse:0.000050	test-rmse:0.000050 
#> [56]	train-rmse:0.000050	test-rmse:0.000050 
#> [57]	train-rmse:0.000050	test-rmse:0.000050 
#> [58]	train-rmse:0.000050	test-rmse:0.000050 
#> [59]	train-rmse:0.000050	test-rmse:0.000050 
#> [60]	train-rmse:0.000050	test-rmse:0.000050 
#> [61]	train-rmse:0.000050	test-rmse:0.000050 
#> [62]	train-rmse:0.000050	test-rmse:0.000050 
#> [63]	train-rmse:0.000050	test-rmse:0.000050 
#> [64]	train-rmse:0.000050	test-rmse:0.000050 
#> [65]	train-rmse:0.000050	test-rmse:0.000050 
#> [66]	train-rmse:0.000050	test-rmse:0.000050 
#> [67]	train-rmse:0.000050	test-rmse:0.000050 
#> [68]	train-rmse:0.000050	test-rmse:0.000050 
#> [69]	train-rmse:0.000050	test-rmse:0.000050 
#> [70]	train-rmse:0.000050	test-rmse:0.000050
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-39.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-40.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-41.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-42.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-43.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-44.png" width="672" />

```
#> [1]	train-rmse:0.473850	test-rmse:0.479414 
#> [2]	train-rmse:0.459094	test-rmse:0.469794 
#> [3]	train-rmse:0.449354	test-rmse:0.461877 
#> [4]	train-rmse:0.443149	test-rmse:0.459174 
#> [5]	train-rmse:0.437627	test-rmse:0.459811 
#> [6]	train-rmse:0.432413	test-rmse:0.460721 
#> [7]	train-rmse:0.428558	test-rmse:0.462380 
#> [8]	train-rmse:0.425258	test-rmse:0.462570 
#> [9]	train-rmse:0.423077	test-rmse:0.461556 
#> [10]	train-rmse:0.418896	test-rmse:0.462989 
#> [11]	train-rmse:0.416189	test-rmse:0.464328 
#> [12]	train-rmse:0.414683	test-rmse:0.463995 
#> [13]	train-rmse:0.411538	test-rmse:0.462447 
#> [14]	train-rmse:0.410403	test-rmse:0.462092 
#> [15]	train-rmse:0.407792	test-rmse:0.463728 
#> [16]	train-rmse:0.404779	test-rmse:0.464408 
#> [17]	train-rmse:0.402084	test-rmse:0.464279 
#> [18]	train-rmse:0.398376	test-rmse:0.464539 
#> [19]	train-rmse:0.395787	test-rmse:0.464975 
#> [20]	train-rmse:0.393919	test-rmse:0.466032 
#> [21]	train-rmse:0.391891	test-rmse:0.465225 
#> [22]	train-rmse:0.390346	test-rmse:0.464729 
#> [23]	train-rmse:0.387206	test-rmse:0.465898 
#> [24]	train-rmse:0.384101	test-rmse:0.465737 
#> [25]	train-rmse:0.383148	test-rmse:0.465977 
#> [26]	train-rmse:0.382018	test-rmse:0.465669 
#> [27]	train-rmse:0.378659	test-rmse:0.467162 
#> [28]	train-rmse:0.376657	test-rmse:0.469619 
#> [29]	train-rmse:0.373677	test-rmse:0.468774 
#> [30]	train-rmse:0.371718	test-rmse:0.469019 
#> [31]	train-rmse:0.368742	test-rmse:0.469019 
#> [32]	train-rmse:0.365627	test-rmse:0.470020 
#> [33]	train-rmse:0.363830	test-rmse:0.472418 
#> [34]	train-rmse:0.360158	test-rmse:0.473165 
#> [35]	train-rmse:0.357578	test-rmse:0.472734 
#> [36]	train-rmse:0.356505	test-rmse:0.473035 
#> [37]	train-rmse:0.353059	test-rmse:0.472369 
#> [38]	train-rmse:0.351706	test-rmse:0.472498 
#> [39]	train-rmse:0.351316	test-rmse:0.472638 
#> [40]	train-rmse:0.350547	test-rmse:0.473132 
#> [41]	train-rmse:0.347650	test-rmse:0.473525 
#> [42]	train-rmse:0.345427	test-rmse:0.475680 
#> [43]	train-rmse:0.343986	test-rmse:0.476514 
#> [44]	train-rmse:0.342900	test-rmse:0.477205 
#> [45]	train-rmse:0.341841	test-rmse:0.478580 
#> [46]	train-rmse:0.340976	test-rmse:0.478855 
#> [47]	train-rmse:0.338885	test-rmse:0.478731 
#> [48]	train-rmse:0.336912	test-rmse:0.478227 
#> [49]	train-rmse:0.333370	test-rmse:0.479379 
#> [50]	train-rmse:0.331511	test-rmse:0.480517 
#> [51]	train-rmse:0.328791	test-rmse:0.480654 
#> [52]	train-rmse:0.327809	test-rmse:0.480366 
#> [53]	train-rmse:0.325429	test-rmse:0.481282 
#> [54]	train-rmse:0.324589	test-rmse:0.481796 
#> [55]	train-rmse:0.323163	test-rmse:0.482088 
#> [56]	train-rmse:0.322723	test-rmse:0.482131 
#> [57]	train-rmse:0.320544	test-rmse:0.482739 
#> [58]	train-rmse:0.318653	test-rmse:0.484235 
#> [59]	train-rmse:0.316577	test-rmse:0.484282 
#> [60]	train-rmse:0.315825	test-rmse:0.484865 
#> [61]	train-rmse:0.315498	test-rmse:0.484841 
#> [62]	train-rmse:0.313831	test-rmse:0.485849 
#> [63]	train-rmse:0.313020	test-rmse:0.486368 
#> [64]	train-rmse:0.311073	test-rmse:0.487467 
#> [65]	train-rmse:0.309329	test-rmse:0.488293 
#> [66]	train-rmse:0.308715	test-rmse:0.488110 
#> [67]	train-rmse:0.307195	test-rmse:0.487954 
#> [68]	train-rmse:0.304213	test-rmse:0.488320 
#> [69]	train-rmse:0.301896	test-rmse:0.487969 
#> [70]	train-rmse:0.299934	test-rmse:0.489176
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-45.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-46.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-47.png" width="672" />

```
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-48.png" width="672" />

```
#> [1]	train-rmse:0.350824	test-rmse:0.350826 
#> [2]	train-rmse:0.246155	test-rmse:0.246158 
#> [3]	train-rmse:0.172714	test-rmse:0.172718 
#> [4]	train-rmse:0.121185	test-rmse:0.121188 
#> [5]	train-rmse:0.085029	test-rmse:0.085032 
#> [6]	train-rmse:0.059660	test-rmse:0.059663 
#> [7]	train-rmse:0.041861	test-rmse:0.041863 
#> [8]	train-rmse:0.029371	test-rmse:0.029373 
#> [9]	train-rmse:0.020608	test-rmse:0.020610 
#> [10]	train-rmse:0.014460	test-rmse:0.014461 
#> [11]	train-rmse:0.010146	test-rmse:0.010146 
#> [12]	train-rmse:0.007119	test-rmse:0.007119 
#> [13]	train-rmse:0.004995	test-rmse:0.004995 
#> [14]	train-rmse:0.003505	test-rmse:0.003505 
#> [15]	train-rmse:0.002459	test-rmse:0.002459 
#> [16]	train-rmse:0.001725	test-rmse:0.001726 
#> [17]	train-rmse:0.001211	test-rmse:0.001211 
#> [18]	train-rmse:0.000849	test-rmse:0.000850 
#> [19]	train-rmse:0.000596	test-rmse:0.000596 
#> [20]	train-rmse:0.000418	test-rmse:0.000418 
#> [21]	train-rmse:0.000293	test-rmse:0.000293 
#> [22]	train-rmse:0.000206	test-rmse:0.000206 
#> [23]	train-rmse:0.000144	test-rmse:0.000144 
#> [24]	train-rmse:0.000101	test-rmse:0.000101 
#> [25]	train-rmse:0.000071	test-rmse:0.000071 
#> [26]	train-rmse:0.000050	test-rmse:0.000050 
#> [27]	train-rmse:0.000050	test-rmse:0.000050 
#> [28]	train-rmse:0.000050	test-rmse:0.000050 
#> [29]	train-rmse:0.000050	test-rmse:0.000050 
#> [30]	train-rmse:0.000050	test-rmse:0.000050 
#> [31]	train-rmse:0.000050	test-rmse:0.000050 
#> [32]	train-rmse:0.000050	test-rmse:0.000050 
#> [33]	train-rmse:0.000050	test-rmse:0.000050 
#> [34]	train-rmse:0.000050	test-rmse:0.000050 
#> [35]	train-rmse:0.000050	test-rmse:0.000050 
#> [36]	train-rmse:0.000050	test-rmse:0.000050 
#> [37]	train-rmse:0.000050	test-rmse:0.000050 
#> [38]	train-rmse:0.000050	test-rmse:0.000050 
#> [39]	train-rmse:0.000050	test-rmse:0.000050 
#> [40]	train-rmse:0.000050	test-rmse:0.000050 
#> [41]	train-rmse:0.000050	test-rmse:0.000050 
#> [42]	train-rmse:0.000050	test-rmse:0.000050 
#> [43]	train-rmse:0.000050	test-rmse:0.000050 
#> [44]	train-rmse:0.000050	test-rmse:0.000050 
#> [45]	train-rmse:0.000050	test-rmse:0.000050 
#> [46]	train-rmse:0.000050	test-rmse:0.000050 
#> [47]	train-rmse:0.000050	test-rmse:0.000050 
#> [48]	train-rmse:0.000050	test-rmse:0.000050 
#> [49]	train-rmse:0.000050	test-rmse:0.000050 
#> [50]	train-rmse:0.000050	test-rmse:0.000050 
#> [51]	train-rmse:0.000050	test-rmse:0.000050 
#> [52]	train-rmse:0.000050	test-rmse:0.000050 
#> [53]	train-rmse:0.000050	test-rmse:0.000050 
#> [54]	train-rmse:0.000050	test-rmse:0.000050 
#> [55]	train-rmse:0.000050	test-rmse:0.000050 
#> [56]	train-rmse:0.000050	test-rmse:0.000050 
#> [57]	train-rmse:0.000050	test-rmse:0.000050 
#> [58]	train-rmse:0.000050	test-rmse:0.000050 
#> [59]	train-rmse:0.000050	test-rmse:0.000050 
#> [60]	train-rmse:0.000050	test-rmse:0.000050 
#> [61]	train-rmse:0.000050	test-rmse:0.000050 
#> [62]	train-rmse:0.000050	test-rmse:0.000050 
#> [63]	train-rmse:0.000050	test-rmse:0.000050 
#> [64]	train-rmse:0.000050	test-rmse:0.000050 
#> [65]	train-rmse:0.000050	test-rmse:0.000050 
#> [66]	train-rmse:0.000050	test-rmse:0.000050 
#> [67]	train-rmse:0.000050	test-rmse:0.000050 
#> [68]	train-rmse:0.000050	test-rmse:0.000050 
#> [69]	train-rmse:0.000050	test-rmse:0.000050 
#> [70]	train-rmse:0.000050	test-rmse:0.000050
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
#> Setting levels: control = 0, case = 1
#> Setting direction: controls < cases
```

<img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-49.png" width="672" /><img src="07-Ensembles_of_Logistic_Models_files/figure-html/Logistic ensemble-50.png" width="672" />

```
#>                     Model  Accuracy
#> 1                     C50 1.0000000
#> 2                  Cubist 1.0000000
#> 3       Ensemble_ADABoost 1.0000000
#> 4            Ensemble_C50 1.0000000
#> 5  Ensemble_Random_Forest 1.0000000
#> 6        Ensemble_XGBoost 1.0000000
#> 7           Random_Forest 1.0000000
#> 8                BayesGLM 0.6479583
#> 9                 XGBoost 0.6437120
#> 10               ADABoost 0.6191110
```

``` r
warnings()
```
