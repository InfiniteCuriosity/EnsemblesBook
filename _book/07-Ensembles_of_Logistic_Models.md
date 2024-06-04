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
#> [1]	train-rmse:0.476144	test-rmse:0.478562 
#> [2]	train-rmse:0.463574	test-rmse:0.467503 
#> [3]	train-rmse:0.454959	test-rmse:0.460474 
#> [4]	train-rmse:0.448993	test-rmse:0.455963 
#> [5]	train-rmse:0.444821	test-rmse:0.452101 
#> [6]	train-rmse:0.440419	test-rmse:0.451403 
#> [7]	train-rmse:0.436667	test-rmse:0.450788 
#> [8]	train-rmse:0.433511	test-rmse:0.449871 
#> [9]	train-rmse:0.431663	test-rmse:0.450169 
#> [10]	train-rmse:0.427720	test-rmse:0.450046 
#> [11]	train-rmse:0.424788	test-rmse:0.450895 
#> [12]	train-rmse:0.422744	test-rmse:0.449402 
#> [13]	train-rmse:0.419597	test-rmse:0.450385 
#> [14]	train-rmse:0.417344	test-rmse:0.450237 
#> [15]	train-rmse:0.413427	test-rmse:0.450686 
#> [16]	train-rmse:0.411022	test-rmse:0.450659 
#> [17]	train-rmse:0.409211	test-rmse:0.450818 
#> [18]	train-rmse:0.405904	test-rmse:0.452264 
#> [19]	train-rmse:0.403075	test-rmse:0.452364 
#> [20]	train-rmse:0.402247	test-rmse:0.453080 
#> [21]	train-rmse:0.398977	test-rmse:0.452725 
#> [22]	train-rmse:0.397407	test-rmse:0.453944 
#> [23]	train-rmse:0.393930	test-rmse:0.453229 
#> [24]	train-rmse:0.389994	test-rmse:0.452871 
#> [25]	train-rmse:0.388467	test-rmse:0.453027 
#> [26]	train-rmse:0.386738	test-rmse:0.452534 
#> [27]	train-rmse:0.384467	test-rmse:0.452899 
#> [28]	train-rmse:0.382570	test-rmse:0.453100 
#> [29]	train-rmse:0.379445	test-rmse:0.454196 
#> [30]	train-rmse:0.377415	test-rmse:0.454774 
#> [31]	train-rmse:0.376504	test-rmse:0.455492 
#> [32]	train-rmse:0.375392	test-rmse:0.455794 
#> [33]	train-rmse:0.373634	test-rmse:0.455489 
#> [34]	train-rmse:0.371433	test-rmse:0.455733 
#> [35]	train-rmse:0.368571	test-rmse:0.456959 
#> [36]	train-rmse:0.367051	test-rmse:0.457008 
#> [37]	train-rmse:0.363892	test-rmse:0.458813 
#> [38]	train-rmse:0.361112	test-rmse:0.458801 
#> [39]	train-rmse:0.359866	test-rmse:0.458274 
#> [40]	train-rmse:0.358560	test-rmse:0.457887 
#> [41]	train-rmse:0.358106	test-rmse:0.458127 
#> [42]	train-rmse:0.355921	test-rmse:0.458705 
#> [43]	train-rmse:0.353788	test-rmse:0.459461 
#> [44]	train-rmse:0.353182	test-rmse:0.459651 
#> [45]	train-rmse:0.352354	test-rmse:0.459765 
#> [46]	train-rmse:0.350558	test-rmse:0.460158 
#> [47]	train-rmse:0.348136	test-rmse:0.458902 
#> [48]	train-rmse:0.346172	test-rmse:0.459073 
#> [49]	train-rmse:0.345309	test-rmse:0.460300 
#> [50]	train-rmse:0.342630	test-rmse:0.460079 
#> [51]	train-rmse:0.341311	test-rmse:0.461742 
#> [52]	train-rmse:0.339357	test-rmse:0.462133 
#> [53]	train-rmse:0.337334	test-rmse:0.462307 
#> [54]	train-rmse:0.334896	test-rmse:0.463136 
#> [55]	train-rmse:0.333012	test-rmse:0.463257 
#> [56]	train-rmse:0.332788	test-rmse:0.463209 
#> [57]	train-rmse:0.330628	test-rmse:0.464458 
#> [58]	train-rmse:0.328841	test-rmse:0.464871 
#> [59]	train-rmse:0.327336	test-rmse:0.466450 
#> [60]	train-rmse:0.325394	test-rmse:0.467173 
#> [61]	train-rmse:0.323361	test-rmse:0.467378 
#> [62]	train-rmse:0.321633	test-rmse:0.467970 
#> [63]	train-rmse:0.319833	test-rmse:0.468894 
#> [64]	train-rmse:0.317659	test-rmse:0.470018 
#> [65]	train-rmse:0.316241	test-rmse:0.470120 
#> [66]	train-rmse:0.314436	test-rmse:0.471113 
#> [67]	train-rmse:0.312160	test-rmse:0.471866 
#> [68]	train-rmse:0.311475	test-rmse:0.472474 
#> [69]	train-rmse:0.309364	test-rmse:0.473537 
#> [70]	train-rmse:0.308724	test-rmse:0.473588
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
#> [1]	train-rmse:0.350824	test-rmse:0.350826 
#> [2]	train-rmse:0.246155	test-rmse:0.246158 
#> [3]	train-rmse:0.172714	test-rmse:0.172718 
#> [4]	train-rmse:0.121185	test-rmse:0.121188 
#> [5]	train-rmse:0.085029	test-rmse:0.085032 
#> [6]	train-rmse:0.059661	test-rmse:0.059663 
#> [7]	train-rmse:0.041861	test-rmse:0.041863 
#> [8]	train-rmse:0.029372	test-rmse:0.029373 
#> [9]	train-rmse:0.020608	test-rmse:0.020610 
#> [10]	train-rmse:0.014460	test-rmse:0.014461 
#> [11]	train-rmse:0.010146	test-rmse:0.010146 
#> [12]	train-rmse:0.007119	test-rmse:0.007119 
#> [13]	train-rmse:0.004995	test-rmse:0.004995 
#> [14]	train-rmse:0.003505	test-rmse:0.003505 
#> [15]	train-rmse:0.002459	test-rmse:0.002459 
#> [16]	train-rmse:0.001725	test-rmse:0.001726 
#> [17]	train-rmse:0.001211	test-rmse:0.001211 
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
#> [1]	train-rmse:0.470376	test-rmse:0.479959 
#> [2]	train-rmse:0.453387	test-rmse:0.471480 
#> [3]	train-rmse:0.442872	test-rmse:0.467979 
#> [4]	train-rmse:0.435073	test-rmse:0.467341 
#> [5]	train-rmse:0.431218	test-rmse:0.466991 
#> [6]	train-rmse:0.425888	test-rmse:0.467568 
#> [7]	train-rmse:0.423895	test-rmse:0.466266 
#> [8]	train-rmse:0.419954	test-rmse:0.466441 
#> [9]	train-rmse:0.417434	test-rmse:0.465230 
#> [10]	train-rmse:0.413718	test-rmse:0.466888 
#> [11]	train-rmse:0.410281	test-rmse:0.466853 
#> [12]	train-rmse:0.407585	test-rmse:0.468813 
#> [13]	train-rmse:0.406435	test-rmse:0.468586 
#> [14]	train-rmse:0.405209	test-rmse:0.469302 
#> [15]	train-rmse:0.401530	test-rmse:0.468692 
#> [16]	train-rmse:0.398126	test-rmse:0.468722 
#> [17]	train-rmse:0.395889	test-rmse:0.469836 
#> [18]	train-rmse:0.392163	test-rmse:0.470516 
#> [19]	train-rmse:0.387945	test-rmse:0.471069 
#> [20]	train-rmse:0.386932	test-rmse:0.471384 
#> [21]	train-rmse:0.384878	test-rmse:0.472052 
#> [22]	train-rmse:0.382496	test-rmse:0.472522 
#> [23]	train-rmse:0.379117	test-rmse:0.473605 
#> [24]	train-rmse:0.375649	test-rmse:0.473835 
#> [25]	train-rmse:0.373457	test-rmse:0.474532 
#> [26]	train-rmse:0.370817	test-rmse:0.474590 
#> [27]	train-rmse:0.368762	test-rmse:0.476130 
#> [28]	train-rmse:0.366161	test-rmse:0.476836 
#> [29]	train-rmse:0.363188	test-rmse:0.477363 
#> [30]	train-rmse:0.361532	test-rmse:0.476964 
#> [31]	train-rmse:0.360264	test-rmse:0.477921 
#> [32]	train-rmse:0.357790	test-rmse:0.477276 
#> [33]	train-rmse:0.355311	test-rmse:0.478001 
#> [34]	train-rmse:0.352196	test-rmse:0.477349 
#> [35]	train-rmse:0.350114	test-rmse:0.477405 
#> [36]	train-rmse:0.349005	test-rmse:0.477194 
#> [37]	train-rmse:0.347719	test-rmse:0.477163 
#> [38]	train-rmse:0.346784	test-rmse:0.477026 
#> [39]	train-rmse:0.345636	test-rmse:0.477004 
#> [40]	train-rmse:0.343623	test-rmse:0.477453 
#> [41]	train-rmse:0.340926	test-rmse:0.477200 
#> [42]	train-rmse:0.339610	test-rmse:0.477063 
#> [43]	train-rmse:0.338114	test-rmse:0.476603 
#> [44]	train-rmse:0.337161	test-rmse:0.477081 
#> [45]	train-rmse:0.335310	test-rmse:0.477718 
#> [46]	train-rmse:0.333155	test-rmse:0.478709 
#> [47]	train-rmse:0.331283	test-rmse:0.479318 
#> [48]	train-rmse:0.328594	test-rmse:0.479546 
#> [49]	train-rmse:0.327513	test-rmse:0.480445 
#> [50]	train-rmse:0.327141	test-rmse:0.481062 
#> [51]	train-rmse:0.325709	test-rmse:0.481752 
#> [52]	train-rmse:0.323339	test-rmse:0.482711 
#> [53]	train-rmse:0.320991	test-rmse:0.483422 
#> [54]	train-rmse:0.320335	test-rmse:0.483969 
#> [55]	train-rmse:0.318912	test-rmse:0.484782 
#> [56]	train-rmse:0.317346	test-rmse:0.485648 
#> [57]	train-rmse:0.316029	test-rmse:0.485568 
#> [58]	train-rmse:0.314046	test-rmse:0.486434 
#> [59]	train-rmse:0.311916	test-rmse:0.487268 
#> [60]	train-rmse:0.309432	test-rmse:0.487138 
#> [61]	train-rmse:0.308339	test-rmse:0.487302 
#> [62]	train-rmse:0.307715	test-rmse:0.487245 
#> [63]	train-rmse:0.306562	test-rmse:0.487641 
#> [64]	train-rmse:0.305473	test-rmse:0.487510 
#> [65]	train-rmse:0.303432	test-rmse:0.487882 
#> [66]	train-rmse:0.301222	test-rmse:0.488623 
#> [67]	train-rmse:0.298861	test-rmse:0.489104 
#> [68]	train-rmse:0.297184	test-rmse:0.490484 
#> [69]	train-rmse:0.295459	test-rmse:0.490403 
#> [70]	train-rmse:0.294919	test-rmse:0.490425
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
#> [1]	train-rmse:0.350822	test-rmse:0.350821 
#> [2]	train-rmse:0.246152	test-rmse:0.246151 
#> [3]	train-rmse:0.172711	test-rmse:0.172710 
#> [4]	train-rmse:0.121182	test-rmse:0.121180 
#> [5]	train-rmse:0.085026	test-rmse:0.085025 
#> [6]	train-rmse:0.059658	test-rmse:0.059657 
#> [7]	train-rmse:0.041859	test-rmse:0.041858 
#> [8]	train-rmse:0.029370	test-rmse:0.029369 
#> [9]	train-rmse:0.020607	test-rmse:0.020607 
#> [10]	train-rmse:0.014459	test-rmse:0.014459 
#> [11]	train-rmse:0.010145	test-rmse:0.010145 
#> [12]	train-rmse:0.007118	test-rmse:0.007118 
#> [13]	train-rmse:0.004994	test-rmse:0.004994 
#> [14]	train-rmse:0.003504	test-rmse:0.003504 
#> [15]	train-rmse:0.002459	test-rmse:0.002459 
#> [16]	train-rmse:0.001725	test-rmse:0.001725 
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
#> [1]	train-rmse:0.471391	test-rmse:0.481189 
#> [2]	train-rmse:0.455547	test-rmse:0.469744 
#> [3]	train-rmse:0.445057	test-rmse:0.466894 
#> [4]	train-rmse:0.438664	test-rmse:0.466114 
#> [5]	train-rmse:0.432583	test-rmse:0.464660 
#> [6]	train-rmse:0.427073	test-rmse:0.466880 
#> [7]	train-rmse:0.421625	test-rmse:0.464989 
#> [8]	train-rmse:0.417663	test-rmse:0.464809 
#> [9]	train-rmse:0.415027	test-rmse:0.465373 
#> [10]	train-rmse:0.409836	test-rmse:0.466984 
#> [11]	train-rmse:0.405816	test-rmse:0.469168 
#> [12]	train-rmse:0.403440	test-rmse:0.470523 
#> [13]	train-rmse:0.401912	test-rmse:0.471266 
#> [14]	train-rmse:0.398924	test-rmse:0.472400 
#> [15]	train-rmse:0.395019	test-rmse:0.472489 
#> [16]	train-rmse:0.393648	test-rmse:0.471867 
#> [17]	train-rmse:0.391375	test-rmse:0.473183 
#> [18]	train-rmse:0.388984	test-rmse:0.472980 
#> [19]	train-rmse:0.388284	test-rmse:0.473698 
#> [20]	train-rmse:0.386324	test-rmse:0.474739 
#> [21]	train-rmse:0.385213	test-rmse:0.474960 
#> [22]	train-rmse:0.382278	test-rmse:0.474498 
#> [23]	train-rmse:0.378827	test-rmse:0.476668 
#> [24]	train-rmse:0.377884	test-rmse:0.477570 
#> [25]	train-rmse:0.374871	test-rmse:0.477621 
#> [26]	train-rmse:0.372707	test-rmse:0.477877 
#> [27]	train-rmse:0.370127	test-rmse:0.477583 
#> [28]	train-rmse:0.367759	test-rmse:0.476749 
#> [29]	train-rmse:0.366066	test-rmse:0.476368 
#> [30]	train-rmse:0.363867	test-rmse:0.476622 
#> [31]	train-rmse:0.362045	test-rmse:0.477304 
#> [32]	train-rmse:0.358841	test-rmse:0.476316 
#> [33]	train-rmse:0.356543	test-rmse:0.476612 
#> [34]	train-rmse:0.353660	test-rmse:0.476929 
#> [35]	train-rmse:0.352077	test-rmse:0.477073 
#> [36]	train-rmse:0.351153	test-rmse:0.477482 
#> [37]	train-rmse:0.348208	test-rmse:0.477965 
#> [38]	train-rmse:0.346105	test-rmse:0.477946 
#> [39]	train-rmse:0.344032	test-rmse:0.477134 
#> [40]	train-rmse:0.341681	test-rmse:0.477221 
#> [41]	train-rmse:0.338970	test-rmse:0.478715 
#> [42]	train-rmse:0.336918	test-rmse:0.479090 
#> [43]	train-rmse:0.335588	test-rmse:0.479170 
#> [44]	train-rmse:0.333278	test-rmse:0.480013 
#> [45]	train-rmse:0.332120	test-rmse:0.480946 
#> [46]	train-rmse:0.329920	test-rmse:0.481960 
#> [47]	train-rmse:0.328341	test-rmse:0.482355 
#> [48]	train-rmse:0.326399	test-rmse:0.483102 
#> [49]	train-rmse:0.324738	test-rmse:0.484285 
#> [50]	train-rmse:0.324118	test-rmse:0.484037 
#> [51]	train-rmse:0.321977	test-rmse:0.485103 
#> [52]	train-rmse:0.320736	test-rmse:0.485361 
#> [53]	train-rmse:0.318751	test-rmse:0.487033 
#> [54]	train-rmse:0.316191	test-rmse:0.488110 
#> [55]	train-rmse:0.314172	test-rmse:0.487651 
#> [56]	train-rmse:0.313694	test-rmse:0.488266 
#> [57]	train-rmse:0.311423	test-rmse:0.488294 
#> [58]	train-rmse:0.310172	test-rmse:0.488361 
#> [59]	train-rmse:0.308348	test-rmse:0.488932 
#> [60]	train-rmse:0.306982	test-rmse:0.489724 
#> [61]	train-rmse:0.305282	test-rmse:0.490384 
#> [62]	train-rmse:0.303546	test-rmse:0.490633 
#> [63]	train-rmse:0.302081	test-rmse:0.490621 
#> [64]	train-rmse:0.301427	test-rmse:0.490028 
#> [65]	train-rmse:0.300181	test-rmse:0.490678 
#> [66]	train-rmse:0.298538	test-rmse:0.492055 
#> [67]	train-rmse:0.297295	test-rmse:0.492493 
#> [68]	train-rmse:0.296920	test-rmse:0.492876 
#> [69]	train-rmse:0.295868	test-rmse:0.493036 
#> [70]	train-rmse:0.295132	test-rmse:0.493147
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
#> [1]	train-rmse:0.350732	test-rmse:0.350737 
#> [2]	train-rmse:0.246025	test-rmse:0.246032 
#> [3]	train-rmse:0.172578	test-rmse:0.172585 
#> [4]	train-rmse:0.121057	test-rmse:0.121064 
#> [5]	train-rmse:0.084917	test-rmse:0.084923 
#> [6]	train-rmse:0.059566	test-rmse:0.059571 
#> [7]	train-rmse:0.041784	test-rmse:0.041788 
#> [8]	train-rmse:0.029310	test-rmse:0.029313 
#> [9]	train-rmse:0.020560	test-rmse:0.020562 
#> [10]	train-rmse:0.014422	test-rmse:0.014424 
#> [11]	train-rmse:0.010116	test-rmse:0.010118 
#> [12]	train-rmse:0.007096	test-rmse:0.007097 
#> [13]	train-rmse:0.004978	test-rmse:0.004979 
#> [14]	train-rmse:0.003492	test-rmse:0.003492 
#> [15]	train-rmse:0.002449	test-rmse:0.002450 
#> [16]	train-rmse:0.001718	test-rmse:0.001718 
#> [17]	train-rmse:0.001205	test-rmse:0.001205 
#> [18]	train-rmse:0.000845	test-rmse:0.000846 
#> [19]	train-rmse:0.000593	test-rmse:0.000593 
#> [20]	train-rmse:0.000416	test-rmse:0.000416 
#> [21]	train-rmse:0.000292	test-rmse:0.000292 
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
#> [34]	train-rmse:0.000049	test-rmse:0.000050 
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
#> [1]	train-rmse:0.474806	test-rmse:0.477327 
#> [2]	train-rmse:0.460194	test-rmse:0.468567 
#> [3]	train-rmse:0.451103	test-rmse:0.464446 
#> [4]	train-rmse:0.444766	test-rmse:0.461353 
#> [5]	train-rmse:0.439933	test-rmse:0.459735 
#> [6]	train-rmse:0.434587	test-rmse:0.458148 
#> [7]	train-rmse:0.432074	test-rmse:0.456628 
#> [8]	train-rmse:0.428070	test-rmse:0.456919 
#> [9]	train-rmse:0.425809	test-rmse:0.457126 
#> [10]	train-rmse:0.423424	test-rmse:0.457745 
#> [11]	train-rmse:0.421371	test-rmse:0.457869 
#> [12]	train-rmse:0.418730	test-rmse:0.457889 
#> [13]	train-rmse:0.415050	test-rmse:0.457260 
#> [14]	train-rmse:0.410685	test-rmse:0.458697 
#> [15]	train-rmse:0.406654	test-rmse:0.458959 
#> [16]	train-rmse:0.402437	test-rmse:0.459619 
#> [17]	train-rmse:0.398835	test-rmse:0.460915 
#> [18]	train-rmse:0.396471	test-rmse:0.461584 
#> [19]	train-rmse:0.395539	test-rmse:0.461310 
#> [20]	train-rmse:0.392449	test-rmse:0.462386 
#> [21]	train-rmse:0.388385	test-rmse:0.463716 
#> [22]	train-rmse:0.386142	test-rmse:0.464954 
#> [23]	train-rmse:0.383937	test-rmse:0.466150 
#> [24]	train-rmse:0.383189	test-rmse:0.467036 
#> [25]	train-rmse:0.381766	test-rmse:0.467289 
#> [26]	train-rmse:0.379216	test-rmse:0.467775 
#> [27]	train-rmse:0.376576	test-rmse:0.468056 
#> [28]	train-rmse:0.374455	test-rmse:0.468093 
#> [29]	train-rmse:0.372813	test-rmse:0.468241 
#> [30]	train-rmse:0.371925	test-rmse:0.468451 
#> [31]	train-rmse:0.370339	test-rmse:0.468939 
#> [32]	train-rmse:0.368483	test-rmse:0.469352 
#> [33]	train-rmse:0.365769	test-rmse:0.470102 
#> [34]	train-rmse:0.363748	test-rmse:0.470834 
#> [35]	train-rmse:0.362648	test-rmse:0.470980 
#> [36]	train-rmse:0.359854	test-rmse:0.471540 
#> [37]	train-rmse:0.358168	test-rmse:0.472813 
#> [38]	train-rmse:0.355880	test-rmse:0.472769 
#> [39]	train-rmse:0.353999	test-rmse:0.472446 
#> [40]	train-rmse:0.351588	test-rmse:0.473530 
#> [41]	train-rmse:0.349368	test-rmse:0.472972 
#> [42]	train-rmse:0.348816	test-rmse:0.473473 
#> [43]	train-rmse:0.346445	test-rmse:0.474530 
#> [44]	train-rmse:0.344027	test-rmse:0.474449 
#> [45]	train-rmse:0.341265	test-rmse:0.476748 
#> [46]	train-rmse:0.338787	test-rmse:0.477893 
#> [47]	train-rmse:0.336858	test-rmse:0.478158 
#> [48]	train-rmse:0.335004	test-rmse:0.479104 
#> [49]	train-rmse:0.334594	test-rmse:0.478971 
#> [50]	train-rmse:0.333463	test-rmse:0.478899 
#> [51]	train-rmse:0.332069	test-rmse:0.479791 
#> [52]	train-rmse:0.331328	test-rmse:0.481001 
#> [53]	train-rmse:0.329724	test-rmse:0.481301 
#> [54]	train-rmse:0.327866	test-rmse:0.481444 
#> [55]	train-rmse:0.325509	test-rmse:0.481943 
#> [56]	train-rmse:0.323790	test-rmse:0.482850 
#> [57]	train-rmse:0.322412	test-rmse:0.483244 
#> [58]	train-rmse:0.320115	test-rmse:0.483289 
#> [59]	train-rmse:0.319143	test-rmse:0.483689 
#> [60]	train-rmse:0.318132	test-rmse:0.483346 
#> [61]	train-rmse:0.317056	test-rmse:0.483574 
#> [62]	train-rmse:0.314811	test-rmse:0.484470 
#> [63]	train-rmse:0.313338	test-rmse:0.485339 
#> [64]	train-rmse:0.310976	test-rmse:0.485383 
#> [65]	train-rmse:0.309806	test-rmse:0.486243 
#> [66]	train-rmse:0.307857	test-rmse:0.486291 
#> [67]	train-rmse:0.306005	test-rmse:0.486813 
#> [68]	train-rmse:0.303882	test-rmse:0.486972 
#> [69]	train-rmse:0.303206	test-rmse:0.487281 
#> [70]	train-rmse:0.301917	test-rmse:0.487293
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
#> [1]	train-rmse:0.350811	test-rmse:0.350811 
#> [2]	train-rmse:0.246136	test-rmse:0.246136 
#> [3]	train-rmse:0.172695	test-rmse:0.172695 
#> [4]	train-rmse:0.121166	test-rmse:0.121166 
#> [5]	train-rmse:0.085013	test-rmse:0.085013 
#> [6]	train-rmse:0.059647	test-rmse:0.059647 
#> [7]	train-rmse:0.041850	test-rmse:0.041850 
#> [8]	train-rmse:0.029363	test-rmse:0.029363 
#> [9]	train-rmse:0.020601	test-rmse:0.020601 
#> [10]	train-rmse:0.014454	test-rmse:0.014454 
#> [11]	train-rmse:0.010141	test-rmse:0.010141 
#> [12]	train-rmse:0.007115	test-rmse:0.007115 
#> [13]	train-rmse:0.004992	test-rmse:0.004992 
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
#> [1]	train-rmse:0.471577	test-rmse:0.480744 
#> [2]	train-rmse:0.454265	test-rmse:0.471583 
#> [3]	train-rmse:0.444762	test-rmse:0.468178 
#> [4]	train-rmse:0.438154	test-rmse:0.465017 
#> [5]	train-rmse:0.434042	test-rmse:0.464051 
#> [6]	train-rmse:0.429515	test-rmse:0.463162 
#> [7]	train-rmse:0.424879	test-rmse:0.462757 
#> [8]	train-rmse:0.422605	test-rmse:0.463300 
#> [9]	train-rmse:0.418741	test-rmse:0.463332 
#> [10]	train-rmse:0.415888	test-rmse:0.465554 
#> [11]	train-rmse:0.412331	test-rmse:0.465948 
#> [12]	train-rmse:0.409270	test-rmse:0.466338 
#> [13]	train-rmse:0.407598	test-rmse:0.464630 
#> [14]	train-rmse:0.405619	test-rmse:0.465577 
#> [15]	train-rmse:0.403221	test-rmse:0.465341 
#> [16]	train-rmse:0.401821	test-rmse:0.465707 
#> [17]	train-rmse:0.400564	test-rmse:0.466053 
#> [18]	train-rmse:0.399474	test-rmse:0.465748 
#> [19]	train-rmse:0.397441	test-rmse:0.465750 
#> [20]	train-rmse:0.394472	test-rmse:0.466224 
#> [21]	train-rmse:0.393116	test-rmse:0.466412 
#> [22]	train-rmse:0.390620	test-rmse:0.465684 
#> [23]	train-rmse:0.387416	test-rmse:0.466095 
#> [24]	train-rmse:0.386378	test-rmse:0.466550 
#> [25]	train-rmse:0.384605	test-rmse:0.466656 
#> [26]	train-rmse:0.381410	test-rmse:0.466764 
#> [27]	train-rmse:0.378212	test-rmse:0.465584 
#> [28]	train-rmse:0.377111	test-rmse:0.465691 
#> [29]	train-rmse:0.374755	test-rmse:0.466208 
#> [30]	train-rmse:0.371775	test-rmse:0.466713 
#> [31]	train-rmse:0.370571	test-rmse:0.466772 
#> [32]	train-rmse:0.369029	test-rmse:0.468559 
#> [33]	train-rmse:0.368610	test-rmse:0.468287 
#> [34]	train-rmse:0.367808	test-rmse:0.468524 
#> [35]	train-rmse:0.367062	test-rmse:0.468352 
#> [36]	train-rmse:0.366186	test-rmse:0.468407 
#> [37]	train-rmse:0.363883	test-rmse:0.467376 
#> [38]	train-rmse:0.361354	test-rmse:0.468753 
#> [39]	train-rmse:0.359755	test-rmse:0.469330 
#> [40]	train-rmse:0.357923	test-rmse:0.469506 
#> [41]	train-rmse:0.354185	test-rmse:0.469816 
#> [42]	train-rmse:0.352034	test-rmse:0.471337 
#> [43]	train-rmse:0.348321	test-rmse:0.471297 
#> [44]	train-rmse:0.344673	test-rmse:0.471338 
#> [45]	train-rmse:0.341964	test-rmse:0.471260 
#> [46]	train-rmse:0.339466	test-rmse:0.471782 
#> [47]	train-rmse:0.338032	test-rmse:0.471911 
#> [48]	train-rmse:0.335915	test-rmse:0.471929 
#> [49]	train-rmse:0.333468	test-rmse:0.472566 
#> [50]	train-rmse:0.331806	test-rmse:0.472157 
#> [51]	train-rmse:0.328948	test-rmse:0.472021 
#> [52]	train-rmse:0.326076	test-rmse:0.472872 
#> [53]	train-rmse:0.323846	test-rmse:0.473045 
#> [54]	train-rmse:0.321768	test-rmse:0.473011 
#> [55]	train-rmse:0.320399	test-rmse:0.472836 
#> [56]	train-rmse:0.318842	test-rmse:0.472671 
#> [57]	train-rmse:0.316777	test-rmse:0.473612 
#> [58]	train-rmse:0.316078	test-rmse:0.473772 
#> [59]	train-rmse:0.315315	test-rmse:0.473332 
#> [60]	train-rmse:0.313517	test-rmse:0.473501 
#> [61]	train-rmse:0.312084	test-rmse:0.474112 
#> [62]	train-rmse:0.310038	test-rmse:0.473522 
#> [63]	train-rmse:0.309017	test-rmse:0.473653 
#> [64]	train-rmse:0.307556	test-rmse:0.473114 
#> [65]	train-rmse:0.305566	test-rmse:0.472764 
#> [66]	train-rmse:0.303763	test-rmse:0.472985 
#> [67]	train-rmse:0.302499	test-rmse:0.473220 
#> [68]	train-rmse:0.300772	test-rmse:0.474625 
#> [69]	train-rmse:0.298926	test-rmse:0.475909 
#> [70]	train-rmse:0.297109	test-rmse:0.475763
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
#> [1]	train-rmse:0.350845	test-rmse:0.350845 
#> [2]	train-rmse:0.246185	test-rmse:0.246185 
#> [3]	train-rmse:0.172745	test-rmse:0.172745 
#> [4]	train-rmse:0.121214	test-rmse:0.121214 
#> [5]	train-rmse:0.085054	test-rmse:0.085055 
#> [6]	train-rmse:0.059682	test-rmse:0.059682 
#> [7]	train-rmse:0.041878	test-rmse:0.041878 
#> [8]	train-rmse:0.029385	test-rmse:0.029386 
#> [9]	train-rmse:0.020620	test-rmse:0.020620 
#> [10]	train-rmse:0.014469	test-rmse:0.014469 
#> [11]	train-rmse:0.010152	test-rmse:0.010152 
#> [12]	train-rmse:0.007124	test-rmse:0.007124 
#> [13]	train-rmse:0.004999	test-rmse:0.004999 
#> [14]	train-rmse:0.003508	test-rmse:0.003508 
#> [15]	train-rmse:0.002461	test-rmse:0.002461 
#> [16]	train-rmse:0.001727	test-rmse:0.001727 
#> [17]	train-rmse:0.001212	test-rmse:0.001212 
#> [18]	train-rmse:0.000850	test-rmse:0.000850 
#> [19]	train-rmse:0.000597	test-rmse:0.000597 
#> [20]	train-rmse:0.000419	test-rmse:0.000419 
#> [21]	train-rmse:0.000294	test-rmse:0.000294 
#> [22]	train-rmse:0.000206	test-rmse:0.000206 
#> [23]	train-rmse:0.000145	test-rmse:0.000145 
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
#> 8                BayesGLM 0.6455523
#> 9                 XGBoost 0.6441496
#> 10               ADABoost 0.6086781
```

``` r
warnings()
```
