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


```r

library(Ensembles)
#> Loading required package: arm
#> Loading required package: MASS
#> Loading required package: Matrix
#> Loading required package: lme4
#> 
#> arm (Version 1.14-4, built: 2024-4-1)
#> Working directory is /Users/russconte/Library/Mobile Documents/com~apple~CloudDocs/Documents/Machine Learning templates in R/EnsemblesBook
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


```r
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


```r

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


```r

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
#> Warning in rgl.init(initValue, onlyNULL): RGL: unable to
#> open X11 display
#> Warning: 'rgl.init' failed, running with 'rgl.useNULL =
#> TRUE'.
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
#> [1]	train-rmse:0.475192	test-rmse:0.478079 
#> [2]	train-rmse:0.460745	test-rmse:0.465107 
#> [3]	train-rmse:0.451388	test-rmse:0.459080 
#> [4]	train-rmse:0.445489	test-rmse:0.455906 
#> [5]	train-rmse:0.438896	test-rmse:0.454753 
#> [6]	train-rmse:0.435170	test-rmse:0.454940 
#> [7]	train-rmse:0.431544	test-rmse:0.454370 
#> [8]	train-rmse:0.427571	test-rmse:0.452858 
#> [9]	train-rmse:0.424162	test-rmse:0.453291 
#> [10]	train-rmse:0.420299	test-rmse:0.454480 
#> [11]	train-rmse:0.416102	test-rmse:0.456759 
#> [12]	train-rmse:0.411797	test-rmse:0.457650 
#> [13]	train-rmse:0.410482	test-rmse:0.457898 
#> [14]	train-rmse:0.408634	test-rmse:0.459100 
#> [15]	train-rmse:0.406348	test-rmse:0.460187 
#> [16]	train-rmse:0.402368	test-rmse:0.458932 
#> [17]	train-rmse:0.399812	test-rmse:0.458677 
#> [18]	train-rmse:0.397695	test-rmse:0.459359 
#> [19]	train-rmse:0.394028	test-rmse:0.461203 
#> [20]	train-rmse:0.391255	test-rmse:0.462953 
#> [21]	train-rmse:0.390029	test-rmse:0.463780 
#> [22]	train-rmse:0.386454	test-rmse:0.464364 
#> [23]	train-rmse:0.383933	test-rmse:0.464019 
#> [24]	train-rmse:0.380405	test-rmse:0.465352 
#> [25]	train-rmse:0.378997	test-rmse:0.465155 
#> [26]	train-rmse:0.378035	test-rmse:0.465882 
#> [27]	train-rmse:0.376297	test-rmse:0.467501 
#> [28]	train-rmse:0.373678	test-rmse:0.468427 
#> [29]	train-rmse:0.370551	test-rmse:0.469632 
#> [30]	train-rmse:0.368614	test-rmse:0.469647 
#> [31]	train-rmse:0.367087	test-rmse:0.468969 
#> [32]	train-rmse:0.365240	test-rmse:0.469454 
#> [33]	train-rmse:0.364351	test-rmse:0.469551 
#> [34]	train-rmse:0.363537	test-rmse:0.470564 
#> [35]	train-rmse:0.362598	test-rmse:0.470403 
#> [36]	train-rmse:0.360605	test-rmse:0.470439 
#> [37]	train-rmse:0.359448	test-rmse:0.471352 
#> [38]	train-rmse:0.357047	test-rmse:0.472299 
#> [39]	train-rmse:0.356753	test-rmse:0.472468 
#> [40]	train-rmse:0.356337	test-rmse:0.472618 
#> [41]	train-rmse:0.354152	test-rmse:0.472050 
#> [42]	train-rmse:0.353494	test-rmse:0.472129 
#> [43]	train-rmse:0.351278	test-rmse:0.473007 
#> [44]	train-rmse:0.349063	test-rmse:0.474251 
#> [45]	train-rmse:0.347140	test-rmse:0.475184 
#> [46]	train-rmse:0.344705	test-rmse:0.476496 
#> [47]	train-rmse:0.343971	test-rmse:0.476474 
#> [48]	train-rmse:0.342816	test-rmse:0.477032 
#> [49]	train-rmse:0.341562	test-rmse:0.476662 
#> [50]	train-rmse:0.338655	test-rmse:0.477149 
#> [51]	train-rmse:0.336345	test-rmse:0.478140 
#> [52]	train-rmse:0.334830	test-rmse:0.478396 
#> [53]	train-rmse:0.332630	test-rmse:0.479673 
#> [54]	train-rmse:0.330085	test-rmse:0.480057 
#> [55]	train-rmse:0.328973	test-rmse:0.480978 
#> [56]	train-rmse:0.327095	test-rmse:0.480589 
#> [57]	train-rmse:0.326709	test-rmse:0.480824 
#> [58]	train-rmse:0.325985	test-rmse:0.481004 
#> [59]	train-rmse:0.323679	test-rmse:0.480972 
#> [60]	train-rmse:0.322220	test-rmse:0.480484 
#> [61]	train-rmse:0.320199	test-rmse:0.480786 
#> [62]	train-rmse:0.318155	test-rmse:0.481231 
#> [63]	train-rmse:0.317449	test-rmse:0.482030 
#> [64]	train-rmse:0.316382	test-rmse:0.482857 
#> [65]	train-rmse:0.314298	test-rmse:0.484149 
#> [66]	train-rmse:0.313270	test-rmse:0.485328 
#> [67]	train-rmse:0.312403	test-rmse:0.485504 
#> [68]	train-rmse:0.310201	test-rmse:0.485571 
#> [69]	train-rmse:0.309894	test-rmse:0.485966 
#> [70]	train-rmse:0.308192	test-rmse:0.486661
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
#> [1]	train-rmse:0.350836	test-rmse:0.350838 
#> [2]	train-rmse:0.246171	test-rmse:0.246174 
#> [3]	train-rmse:0.172731	test-rmse:0.172734 
#> [4]	train-rmse:0.121201	test-rmse:0.121204 
#> [5]	train-rmse:0.085043	test-rmse:0.085046 
#> [6]	train-rmse:0.059672	test-rmse:0.059674 
#> [7]	train-rmse:0.041870	test-rmse:0.041872 
#> [8]	train-rmse:0.029379	test-rmse:0.029381 
#> [9]	train-rmse:0.020615	test-rmse:0.020616 
#> [10]	train-rmse:0.014465	test-rmse:0.014466 
#> [11]	train-rmse:0.010149	test-rmse:0.010150 
#> [12]	train-rmse:0.007122	test-rmse:0.007122 
#> [13]	train-rmse:0.004997	test-rmse:0.004997 
#> [14]	train-rmse:0.003506	test-rmse:0.003507 
#> [15]	train-rmse:0.002460	test-rmse:0.002460 
#> [16]	train-rmse:0.001726	test-rmse:0.001726 
#> [17]	train-rmse:0.001211	test-rmse:0.001211 
#> [18]	train-rmse:0.000850	test-rmse:0.000850 
#> [19]	train-rmse:0.000596	test-rmse:0.000596 
#> [20]	train-rmse:0.000418	test-rmse:0.000419 
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
#> [1]	train-rmse:0.471288	test-rmse:0.480597 
#> [2]	train-rmse:0.454148	test-rmse:0.471956 
#> [3]	train-rmse:0.444279	test-rmse:0.468135 
#> [4]	train-rmse:0.436460	test-rmse:0.466477 
#> [5]	train-rmse:0.430706	test-rmse:0.465734 
#> [6]	train-rmse:0.425711	test-rmse:0.466878 
#> [7]	train-rmse:0.420403	test-rmse:0.466368 
#> [8]	train-rmse:0.415413	test-rmse:0.466983 
#> [9]	train-rmse:0.411158	test-rmse:0.468684 
#> [10]	train-rmse:0.408749	test-rmse:0.468528 
#> [11]	train-rmse:0.405317	test-rmse:0.468140 
#> [12]	train-rmse:0.401817	test-rmse:0.468375 
#> [13]	train-rmse:0.399431	test-rmse:0.469036 
#> [14]	train-rmse:0.395495	test-rmse:0.469593 
#> [15]	train-rmse:0.392010	test-rmse:0.470457 
#> [16]	train-rmse:0.388653	test-rmse:0.472501 
#> [17]	train-rmse:0.387409	test-rmse:0.472397 
#> [18]	train-rmse:0.383724	test-rmse:0.473634 
#> [19]	train-rmse:0.382593	test-rmse:0.474546 
#> [20]	train-rmse:0.378619	test-rmse:0.476287 
#> [21]	train-rmse:0.376242	test-rmse:0.476496 
#> [22]	train-rmse:0.374736	test-rmse:0.474994 
#> [23]	train-rmse:0.372731	test-rmse:0.476013 
#> [24]	train-rmse:0.369985	test-rmse:0.476399 
#> [25]	train-rmse:0.367332	test-rmse:0.477766 
#> [26]	train-rmse:0.364767	test-rmse:0.478064 
#> [27]	train-rmse:0.362684	test-rmse:0.478989 
#> [28]	train-rmse:0.360203	test-rmse:0.478573 
#> [29]	train-rmse:0.358111	test-rmse:0.478864 
#> [30]	train-rmse:0.357035	test-rmse:0.479190 
#> [31]	train-rmse:0.354377	test-rmse:0.480162 
#> [32]	train-rmse:0.351908	test-rmse:0.480480 
#> [33]	train-rmse:0.351209	test-rmse:0.480823 
#> [34]	train-rmse:0.348424	test-rmse:0.480542 
#> [35]	train-rmse:0.346924	test-rmse:0.480054 
#> [36]	train-rmse:0.344238	test-rmse:0.481170 
#> [37]	train-rmse:0.343246	test-rmse:0.481543 
#> [38]	train-rmse:0.342540	test-rmse:0.481763 
#> [39]	train-rmse:0.339574	test-rmse:0.481122 
#> [40]	train-rmse:0.337354	test-rmse:0.482510 
#> [41]	train-rmse:0.335757	test-rmse:0.482954 
#> [42]	train-rmse:0.333766	test-rmse:0.484287 
#> [43]	train-rmse:0.331758	test-rmse:0.484819 
#> [44]	train-rmse:0.331201	test-rmse:0.485416 
#> [45]	train-rmse:0.329451	test-rmse:0.485938 
#> [46]	train-rmse:0.327174	test-rmse:0.485647 
#> [47]	train-rmse:0.325412	test-rmse:0.486074 
#> [48]	train-rmse:0.323566	test-rmse:0.486410 
#> [49]	train-rmse:0.322551	test-rmse:0.487852 
#> [50]	train-rmse:0.320895	test-rmse:0.488615 
#> [51]	train-rmse:0.319681	test-rmse:0.489755 
#> [52]	train-rmse:0.318106	test-rmse:0.490562 
#> [53]	train-rmse:0.315893	test-rmse:0.490989 
#> [54]	train-rmse:0.314484	test-rmse:0.491501 
#> [55]	train-rmse:0.313154	test-rmse:0.492249 
#> [56]	train-rmse:0.310960	test-rmse:0.492431 
#> [57]	train-rmse:0.308151	test-rmse:0.492892 
#> [58]	train-rmse:0.305982	test-rmse:0.493646 
#> [59]	train-rmse:0.304090	test-rmse:0.493688 
#> [60]	train-rmse:0.301510	test-rmse:0.493719 
#> [61]	train-rmse:0.300066	test-rmse:0.494916 
#> [62]	train-rmse:0.297987	test-rmse:0.496096 
#> [63]	train-rmse:0.296870	test-rmse:0.496458 
#> [64]	train-rmse:0.296605	test-rmse:0.496525 
#> [65]	train-rmse:0.294505	test-rmse:0.497734 
#> [66]	train-rmse:0.292709	test-rmse:0.498562 
#> [67]	train-rmse:0.291528	test-rmse:0.498973 
#> [68]	train-rmse:0.289977	test-rmse:0.499720 
#> [69]	train-rmse:0.288232	test-rmse:0.500854 
#> [70]	train-rmse:0.286875	test-rmse:0.501465
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
#> [1]	train-rmse:0.350806	test-rmse:0.350810 
#> [2]	train-rmse:0.246130	test-rmse:0.246136 
#> [3]	train-rmse:0.172688	test-rmse:0.172694 
#> [4]	train-rmse:0.121160	test-rmse:0.121166 
#> [5]	train-rmse:0.085008	test-rmse:0.085012 
#> [6]	train-rmse:0.059642	test-rmse:0.059646 
#> [7]	train-rmse:0.041846	test-rmse:0.041849 
#> [8]	train-rmse:0.029360	test-rmse:0.029362 
#> [9]	train-rmse:0.020599	test-rmse:0.020601 
#> [10]	train-rmse:0.014453	test-rmse:0.014454 
#> [11]	train-rmse:0.010140	test-rmse:0.010141 
#> [12]	train-rmse:0.007114	test-rmse:0.007115 
#> [13]	train-rmse:0.004992	test-rmse:0.004992 
#> [14]	train-rmse:0.003502	test-rmse:0.003503 
#> [15]	train-rmse:0.002457	test-rmse:0.002458 
#> [16]	train-rmse:0.001724	test-rmse:0.001724 
#> [17]	train-rmse:0.001210	test-rmse:0.001210 
#> [18]	train-rmse:0.000849	test-rmse:0.000849 
#> [19]	train-rmse:0.000595	test-rmse:0.000596 
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
#> [1]	train-rmse:0.473801	test-rmse:0.480014 
#> [2]	train-rmse:0.458215	test-rmse:0.469864 
#> [3]	train-rmse:0.449741	test-rmse:0.466168 
#> [4]	train-rmse:0.442625	test-rmse:0.464386 
#> [5]	train-rmse:0.437502	test-rmse:0.463313 
#> [6]	train-rmse:0.433406	test-rmse:0.465118 
#> [7]	train-rmse:0.429744	test-rmse:0.465226 
#> [8]	train-rmse:0.427812	test-rmse:0.464808 
#> [9]	train-rmse:0.425196	test-rmse:0.465825 
#> [10]	train-rmse:0.422256	test-rmse:0.466677 
#> [11]	train-rmse:0.418247	test-rmse:0.467299 
#> [12]	train-rmse:0.414233	test-rmse:0.465835 
#> [13]	train-rmse:0.412075	test-rmse:0.467204 
#> [14]	train-rmse:0.410693	test-rmse:0.467586 
#> [15]	train-rmse:0.409503	test-rmse:0.468581 
#> [16]	train-rmse:0.408755	test-rmse:0.469685 
#> [17]	train-rmse:0.406969	test-rmse:0.469987 
#> [18]	train-rmse:0.404962	test-rmse:0.470371 
#> [19]	train-rmse:0.403384	test-rmse:0.471426 
#> [20]	train-rmse:0.400175	test-rmse:0.473795 
#> [21]	train-rmse:0.399097	test-rmse:0.473812 
#> [22]	train-rmse:0.396074	test-rmse:0.473160 
#> [23]	train-rmse:0.393537	test-rmse:0.472550 
#> [24]	train-rmse:0.390971	test-rmse:0.473480 
#> [25]	train-rmse:0.387452	test-rmse:0.473622 
#> [26]	train-rmse:0.384494	test-rmse:0.473356 
#> [27]	train-rmse:0.381861	test-rmse:0.474089 
#> [28]	train-rmse:0.377919	test-rmse:0.475059 
#> [29]	train-rmse:0.376196	test-rmse:0.474627 
#> [30]	train-rmse:0.375485	test-rmse:0.474749 
#> [31]	train-rmse:0.373193	test-rmse:0.475647 
#> [32]	train-rmse:0.372183	test-rmse:0.475989 
#> [33]	train-rmse:0.370839	test-rmse:0.475017 
#> [34]	train-rmse:0.368011	test-rmse:0.474879 
#> [35]	train-rmse:0.365786	test-rmse:0.476053 
#> [36]	train-rmse:0.361786	test-rmse:0.478188 
#> [37]	train-rmse:0.358594	test-rmse:0.479465 
#> [38]	train-rmse:0.357436	test-rmse:0.480038 
#> [39]	train-rmse:0.355393	test-rmse:0.480862 
#> [40]	train-rmse:0.352473	test-rmse:0.482852 
#> [41]	train-rmse:0.350967	test-rmse:0.483703 
#> [42]	train-rmse:0.349832	test-rmse:0.483414 
#> [43]	train-rmse:0.349549	test-rmse:0.483538 
#> [44]	train-rmse:0.347342	test-rmse:0.483293 
#> [45]	train-rmse:0.345605	test-rmse:0.483945 
#> [46]	train-rmse:0.344805	test-rmse:0.483469 
#> [47]	train-rmse:0.342493	test-rmse:0.483422 
#> [48]	train-rmse:0.339428	test-rmse:0.483950 
#> [49]	train-rmse:0.339030	test-rmse:0.484049 
#> [50]	train-rmse:0.336944	test-rmse:0.483367 
#> [51]	train-rmse:0.333893	test-rmse:0.482459 
#> [52]	train-rmse:0.331244	test-rmse:0.483001 
#> [53]	train-rmse:0.329855	test-rmse:0.482918 
#> [54]	train-rmse:0.329137	test-rmse:0.482874 
#> [55]	train-rmse:0.328487	test-rmse:0.483778 
#> [56]	train-rmse:0.327071	test-rmse:0.484555 
#> [57]	train-rmse:0.324390	test-rmse:0.485547 
#> [58]	train-rmse:0.322206	test-rmse:0.486124 
#> [59]	train-rmse:0.319892	test-rmse:0.487004 
#> [60]	train-rmse:0.317308	test-rmse:0.488051 
#> [61]	train-rmse:0.315291	test-rmse:0.487747 
#> [62]	train-rmse:0.314257	test-rmse:0.487359 
#> [63]	train-rmse:0.313193	test-rmse:0.487545 
#> [64]	train-rmse:0.310592	test-rmse:0.488525 
#> [65]	train-rmse:0.310393	test-rmse:0.488563 
#> [66]	train-rmse:0.308776	test-rmse:0.489347 
#> [67]	train-rmse:0.306880	test-rmse:0.489801 
#> [68]	train-rmse:0.306194	test-rmse:0.489683 
#> [69]	train-rmse:0.303963	test-rmse:0.490333 
#> [70]	train-rmse:0.302760	test-rmse:0.491298
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
#> [1]	train-rmse:0.350779	test-rmse:0.350780 
#> [2]	train-rmse:0.246092	test-rmse:0.246093 
#> [3]	train-rmse:0.172648	test-rmse:0.172649 
#> [4]	train-rmse:0.121123	test-rmse:0.121123 
#> [5]	train-rmse:0.084975	test-rmse:0.084975 
#> [6]	train-rmse:0.059615	test-rmse:0.059615 
#> [7]	train-rmse:0.041823	test-rmse:0.041824 
#> [8]	train-rmse:0.029341	test-rmse:0.029342 
#> [9]	train-rmse:0.020585	test-rmse:0.020585 
#> [10]	train-rmse:0.014441	test-rmse:0.014442 
#> [11]	train-rmse:0.010131	test-rmse:0.010132 
#> [12]	train-rmse:0.007108	test-rmse:0.007108 
#> [13]	train-rmse:0.004987	test-rmse:0.004987 
#> [14]	train-rmse:0.003498	test-rmse:0.003498 
#> [15]	train-rmse:0.002454	test-rmse:0.002454 
#> [16]	train-rmse:0.001722	test-rmse:0.001722 
#> [17]	train-rmse:0.001208	test-rmse:0.001208 
#> [18]	train-rmse:0.000847	test-rmse:0.000847 
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
#> [1]	train-rmse:0.469345	test-rmse:0.483306 
#> [2]	train-rmse:0.451129	test-rmse:0.475073 
#> [3]	train-rmse:0.438859	test-rmse:0.472677 
#> [4]	train-rmse:0.430394	test-rmse:0.474792 
#> [5]	train-rmse:0.424651	test-rmse:0.474480 
#> [6]	train-rmse:0.419686	test-rmse:0.476138 
#> [7]	train-rmse:0.416104	test-rmse:0.477385 
#> [8]	train-rmse:0.409485	test-rmse:0.479152 
#> [9]	train-rmse:0.407226	test-rmse:0.477947 
#> [10]	train-rmse:0.403165	test-rmse:0.478425 
#> [11]	train-rmse:0.401277	test-rmse:0.479130 
#> [12]	train-rmse:0.396918	test-rmse:0.479524 
#> [13]	train-rmse:0.392692	test-rmse:0.481042 
#> [14]	train-rmse:0.389309	test-rmse:0.482407 
#> [15]	train-rmse:0.385995	test-rmse:0.483304 
#> [16]	train-rmse:0.382852	test-rmse:0.484639 
#> [17]	train-rmse:0.381596	test-rmse:0.485118 
#> [18]	train-rmse:0.379741	test-rmse:0.485999 
#> [19]	train-rmse:0.377448	test-rmse:0.486745 
#> [20]	train-rmse:0.373995	test-rmse:0.487018 
#> [21]	train-rmse:0.371236	test-rmse:0.487427 
#> [22]	train-rmse:0.367078	test-rmse:0.487029 
#> [23]	train-rmse:0.364324	test-rmse:0.487024 
#> [24]	train-rmse:0.361429	test-rmse:0.486409 
#> [25]	train-rmse:0.358355	test-rmse:0.486373 
#> [26]	train-rmse:0.356879	test-rmse:0.486452 
#> [27]	train-rmse:0.356182	test-rmse:0.486853 
#> [28]	train-rmse:0.353620	test-rmse:0.487637 
#> [29]	train-rmse:0.352210	test-rmse:0.487617 
#> [30]	train-rmse:0.349787	test-rmse:0.488029 
#> [31]	train-rmse:0.347052	test-rmse:0.488152 
#> [32]	train-rmse:0.344668	test-rmse:0.489566 
#> [33]	train-rmse:0.342204	test-rmse:0.489855 
#> [34]	train-rmse:0.340754	test-rmse:0.489307 
#> [35]	train-rmse:0.340022	test-rmse:0.489012 
#> [36]	train-rmse:0.337948	test-rmse:0.489216 
#> [37]	train-rmse:0.336364	test-rmse:0.489918 
#> [38]	train-rmse:0.335238	test-rmse:0.490907 
#> [39]	train-rmse:0.333505	test-rmse:0.490977 
#> [40]	train-rmse:0.331388	test-rmse:0.491889 
#> [41]	train-rmse:0.328998	test-rmse:0.492251 
#> [42]	train-rmse:0.327470	test-rmse:0.491634 
#> [43]	train-rmse:0.326239	test-rmse:0.492068 
#> [44]	train-rmse:0.324830	test-rmse:0.492400 
#> [45]	train-rmse:0.322450	test-rmse:0.492293 
#> [46]	train-rmse:0.321218	test-rmse:0.492800 
#> [47]	train-rmse:0.318764	test-rmse:0.493141 
#> [48]	train-rmse:0.316828	test-rmse:0.493773 
#> [49]	train-rmse:0.314391	test-rmse:0.493342 
#> [50]	train-rmse:0.312233	test-rmse:0.493425 
#> [51]	train-rmse:0.310441	test-rmse:0.493546 
#> [52]	train-rmse:0.309609	test-rmse:0.493152 
#> [53]	train-rmse:0.308001	test-rmse:0.493657 
#> [54]	train-rmse:0.306697	test-rmse:0.493582 
#> [55]	train-rmse:0.304640	test-rmse:0.493931 
#> [56]	train-rmse:0.303244	test-rmse:0.494351 
#> [57]	train-rmse:0.302286	test-rmse:0.494680 
#> [58]	train-rmse:0.301160	test-rmse:0.494775 
#> [59]	train-rmse:0.300158	test-rmse:0.494663 
#> [60]	train-rmse:0.297504	test-rmse:0.494739 
#> [61]	train-rmse:0.295155	test-rmse:0.495895 
#> [62]	train-rmse:0.293799	test-rmse:0.496655 
#> [63]	train-rmse:0.292051	test-rmse:0.496780 
#> [64]	train-rmse:0.288926	test-rmse:0.497473 
#> [65]	train-rmse:0.287236	test-rmse:0.497843 
#> [66]	train-rmse:0.285369	test-rmse:0.498784 
#> [67]	train-rmse:0.283905	test-rmse:0.499510 
#> [68]	train-rmse:0.283018	test-rmse:0.499894 
#> [69]	train-rmse:0.281572	test-rmse:0.499873 
#> [70]	train-rmse:0.281403	test-rmse:0.499882
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
#> [1]	train-rmse:0.350752	test-rmse:0.350751 
#> [2]	train-rmse:0.246054	test-rmse:0.246052 
#> [3]	train-rmse:0.172608	test-rmse:0.172606 
#> [4]	train-rmse:0.121085	test-rmse:0.121083 
#> [5]	train-rmse:0.084942	test-rmse:0.084940 
#> [6]	train-rmse:0.059587	test-rmse:0.059586 
#> [7]	train-rmse:0.041800	test-rmse:0.041799 
#> [8]	train-rmse:0.029323	test-rmse:0.029322 
#> [9]	train-rmse:0.020570	test-rmse:0.020570 
#> [10]	train-rmse:0.014430	test-rmse:0.014430 
#> [11]	train-rmse:0.010123	test-rmse:0.010122 
#> [12]	train-rmse:0.007101	test-rmse:0.007101 
#> [13]	train-rmse:0.004982	test-rmse:0.004981 
#> [14]	train-rmse:0.003495	test-rmse:0.003494 
#> [15]	train-rmse:0.002451	test-rmse:0.002451 
#> [16]	train-rmse:0.001720	test-rmse:0.001720 
#> [17]	train-rmse:0.001206	test-rmse:0.001206 
#> [18]	train-rmse:0.000846	test-rmse:0.000846 
#> [19]	train-rmse:0.000594	test-rmse:0.000594 
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
#> [34]	train-rmse:0.000050	test-rmse:0.000050 
#> [35]	train-rmse:0.000050	test-rmse:0.000050 
#> [36]	train-rmse:0.000050	test-rmse:0.000050 
#> [37]	train-rmse:0.000050	test-rmse:0.000050 
#> [38]	train-rmse:0.000050	test-rmse:0.000050 
#> [39]	train-rmse:0.000050	test-rmse:0.000049 
#> [40]	train-rmse:0.000050	test-rmse:0.000049 
#> [41]	train-rmse:0.000050	test-rmse:0.000049 
#> [42]	train-rmse:0.000050	test-rmse:0.000049 
#> [43]	train-rmse:0.000050	test-rmse:0.000049 
#> [44]	train-rmse:0.000050	test-rmse:0.000049 
#> [45]	train-rmse:0.000050	test-rmse:0.000049 
#> [46]	train-rmse:0.000050	test-rmse:0.000049 
#> [47]	train-rmse:0.000050	test-rmse:0.000049 
#> [48]	train-rmse:0.000050	test-rmse:0.000049 
#> [49]	train-rmse:0.000050	test-rmse:0.000049 
#> [50]	train-rmse:0.000050	test-rmse:0.000049 
#> [51]	train-rmse:0.000050	test-rmse:0.000049 
#> [52]	train-rmse:0.000050	test-rmse:0.000049 
#> [53]	train-rmse:0.000050	test-rmse:0.000049 
#> [54]	train-rmse:0.000050	test-rmse:0.000049 
#> [55]	train-rmse:0.000050	test-rmse:0.000049 
#> [56]	train-rmse:0.000050	test-rmse:0.000049 
#> [57]	train-rmse:0.000050	test-rmse:0.000049 
#> [58]	train-rmse:0.000050	test-rmse:0.000049 
#> [59]	train-rmse:0.000050	test-rmse:0.000049 
#> [60]	train-rmse:0.000050	test-rmse:0.000049 
#> [61]	train-rmse:0.000050	test-rmse:0.000049 
#> [62]	train-rmse:0.000050	test-rmse:0.000049 
#> [63]	train-rmse:0.000050	test-rmse:0.000049 
#> [64]	train-rmse:0.000050	test-rmse:0.000049 
#> [65]	train-rmse:0.000050	test-rmse:0.000049 
#> [66]	train-rmse:0.000050	test-rmse:0.000049 
#> [67]	train-rmse:0.000050	test-rmse:0.000049 
#> [68]	train-rmse:0.000050	test-rmse:0.000049 
#> [69]	train-rmse:0.000050	test-rmse:0.000049 
#> [70]	train-rmse:0.000050	test-rmse:0.000049
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
#> [1]	train-rmse:0.469983	test-rmse:0.481825 
#> [2]	train-rmse:0.453292	test-rmse:0.474652 
#> [3]	train-rmse:0.442160	test-rmse:0.470276 
#> [4]	train-rmse:0.434314	test-rmse:0.469678 
#> [5]	train-rmse:0.429381	test-rmse:0.468875 
#> [6]	train-rmse:0.425673	test-rmse:0.470175 
#> [7]	train-rmse:0.422683	test-rmse:0.469549 
#> [8]	train-rmse:0.419515	test-rmse:0.471713 
#> [9]	train-rmse:0.418412	test-rmse:0.471700 
#> [10]	train-rmse:0.412741	test-rmse:0.473357 
#> [11]	train-rmse:0.410944	test-rmse:0.473120 
#> [12]	train-rmse:0.408424	test-rmse:0.473023 
#> [13]	train-rmse:0.405919	test-rmse:0.472233 
#> [14]	train-rmse:0.404462	test-rmse:0.472746 
#> [15]	train-rmse:0.403653	test-rmse:0.472729 
#> [16]	train-rmse:0.399715	test-rmse:0.472551 
#> [17]	train-rmse:0.396624	test-rmse:0.472368 
#> [18]	train-rmse:0.393517	test-rmse:0.471737 
#> [19]	train-rmse:0.392274	test-rmse:0.471105 
#> [20]	train-rmse:0.391304	test-rmse:0.471421 
#> [21]	train-rmse:0.388356	test-rmse:0.472729 
#> [22]	train-rmse:0.386881	test-rmse:0.473676 
#> [23]	train-rmse:0.383574	test-rmse:0.474620 
#> [24]	train-rmse:0.379972	test-rmse:0.475196 
#> [25]	train-rmse:0.379027	test-rmse:0.474874 
#> [26]	train-rmse:0.377911	test-rmse:0.475319 
#> [27]	train-rmse:0.374023	test-rmse:0.477312 
#> [28]	train-rmse:0.371324	test-rmse:0.477651 
#> [29]	train-rmse:0.369694	test-rmse:0.477372 
#> [30]	train-rmse:0.367384	test-rmse:0.476881 
#> [31]	train-rmse:0.365231	test-rmse:0.477780 
#> [32]	train-rmse:0.362514	test-rmse:0.477771 
#> [33]	train-rmse:0.361889	test-rmse:0.477999 
#> [34]	train-rmse:0.358611	test-rmse:0.477956 
#> [35]	train-rmse:0.357341	test-rmse:0.477890 
#> [36]	train-rmse:0.354910	test-rmse:0.477889 
#> [37]	train-rmse:0.352434	test-rmse:0.478723 
#> [38]	train-rmse:0.349446	test-rmse:0.479393 
#> [39]	train-rmse:0.348320	test-rmse:0.479751 
#> [40]	train-rmse:0.346057	test-rmse:0.479579 
#> [41]	train-rmse:0.344301	test-rmse:0.479321 
#> [42]	train-rmse:0.342503	test-rmse:0.479055 
#> [43]	train-rmse:0.341741	test-rmse:0.479281 
#> [44]	train-rmse:0.340278	test-rmse:0.479698 
#> [45]	train-rmse:0.338905	test-rmse:0.479804 
#> [46]	train-rmse:0.338534	test-rmse:0.479388 
#> [47]	train-rmse:0.338150	test-rmse:0.479594 
#> [48]	train-rmse:0.336896	test-rmse:0.479947 
#> [49]	train-rmse:0.334466	test-rmse:0.481629 
#> [50]	train-rmse:0.332821	test-rmse:0.482458 
#> [51]	train-rmse:0.330561	test-rmse:0.483253 
#> [52]	train-rmse:0.328776	test-rmse:0.482987 
#> [53]	train-rmse:0.327713	test-rmse:0.483217 
#> [54]	train-rmse:0.325673	test-rmse:0.483616 
#> [55]	train-rmse:0.323789	test-rmse:0.484530 
#> [56]	train-rmse:0.321965	test-rmse:0.485015 
#> [57]	train-rmse:0.321327	test-rmse:0.484828 
#> [58]	train-rmse:0.320789	test-rmse:0.484955 
#> [59]	train-rmse:0.319977	test-rmse:0.485544 
#> [60]	train-rmse:0.319336	test-rmse:0.485246 
#> [61]	train-rmse:0.317920	test-rmse:0.485508 
#> [62]	train-rmse:0.316384	test-rmse:0.486216 
#> [63]	train-rmse:0.314147	test-rmse:0.486477 
#> [64]	train-rmse:0.313547	test-rmse:0.487216 
#> [65]	train-rmse:0.311705	test-rmse:0.488294 
#> [66]	train-rmse:0.310588	test-rmse:0.488662 
#> [67]	train-rmse:0.308651	test-rmse:0.487913 
#> [68]	train-rmse:0.306966	test-rmse:0.488149 
#> [69]	train-rmse:0.305213	test-rmse:0.489127 
#> [70]	train-rmse:0.303103	test-rmse:0.489845
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
#> [1]	train-rmse:0.350759	test-rmse:0.350760 
#> [2]	train-rmse:0.246064	test-rmse:0.246065 
#> [3]	train-rmse:0.172619	test-rmse:0.172619 
#> [4]	train-rmse:0.121095	test-rmse:0.121096 
#> [5]	train-rmse:0.084951	test-rmse:0.084951 
#> [6]	train-rmse:0.059595	test-rmse:0.059595 
#> [7]	train-rmse:0.041807	test-rmse:0.041807 
#> [8]	train-rmse:0.029328	test-rmse:0.029328 
#> [9]	train-rmse:0.020574	test-rmse:0.020574 
#> [10]	train-rmse:0.014433	test-rmse:0.014433 
#> [11]	train-rmse:0.010125	test-rmse:0.010125 
#> [12]	train-rmse:0.007103	test-rmse:0.007103 
#> [13]	train-rmse:0.004983	test-rmse:0.004983 
#> [14]	train-rmse:0.003496	test-rmse:0.003496 
#> [15]	train-rmse:0.002452	test-rmse:0.002452 
#> [16]	train-rmse:0.001720	test-rmse:0.001720 
#> [17]	train-rmse:0.001207	test-rmse:0.001207 
#> [18]	train-rmse:0.000847	test-rmse:0.000847 
#> [19]	train-rmse:0.000594	test-rmse:0.000594 
#> [20]	train-rmse:0.000417	test-rmse:0.000417 
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
#> 8                BayesGLM 0.6430284
#> 9                 XGBoost 0.6361022
#> 10               ADABoost 0.6176642
warnings()
```
