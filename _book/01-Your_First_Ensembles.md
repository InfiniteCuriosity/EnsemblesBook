---
editor_options: 
  markdown: 
    wrap: 72
---

# Introduction and your first ensembles

## How a Chicago blizzard led to the very unlikely story of the best solutions to supervised data

My journey to the most advanced AI in the world started with an actual
blizzard in Chicago. It might seem like Chicago would never get a
blizzard, but we did in 2011, and it was incredibly intense, as this
video shows:

<https://www.youtube.com/watch?v=cPiFn52ztd8>

What does the [Chicago 2011
Snomageddon](https://en.wikipedia.org/wiki/2011_Groundhog_Day_blizzard)
have to do with the creation of the most advanced AI? Everything. Here's
the story.

At the time of the 2011 Blizzard I worked a Recruiter for [Kelly
Services](https://en.wikipedia.org/wiki/Kelly_Services), where I had
worked since 1996. I agreed to work out of the Kelly Services office in
Frankfort, Illinois at this time, though I worked out of nearly every
Kelly Services office at one time or another. The trip to Frankfort
involved a daily commute to the office, but I was able to make the best
use of the time on the road.

My manager at the time let me know several days in advance that there
was a very large amount of snow forecast, and that I might want to be
prepared. The most recent forecasts for large amounts of snow in the
Chicago area all amounted to nothing. They were perfectly normal days in
the Chicago area, so I predicted this storm would also be nothing, based
on the most recent results. This was a great example of a prior
prediction not transferring well to a current situation.

That morning I went to work as normal, and did not even look at the
weather forecast. Around 2:45 pm my manager came out of her office and
said "Russ, you need to come here and look at the weather radar!". I
walked into her office, and saw a map of a winter storm that was
incredibly huge. She had the image zoomed out, so it was possible to see
several states. From what I could tell, the massive snow storm was
barreling down on Chicago, and was about 15 minutes away from our
location.

I told the candidate I was interviewing that I was leaving immediately,
and that he is not allowed to stay. He has to get home as fast as
possible for his own safety.

The storm started dropping snow on my trip north back home. The commute
took around 50% longer than normal due to the rapidly falling snow.

As I later learned, the storm was forecast to start in the Chicago area
around 3:00 pm, finish up between 11:00 am - 1:00 pm two days later, and
leave 17 - 19 inches of snow.

How bad was it? Even City of Chicago snow plows were stopped by the
snow:

![[Chicago snow
plow](https://en.wikipedia.org/wiki/2011_Groundhog_Day_blizzard#/media/File:Stuck_Salt_Truck_on_Lake_Shore_drive_Chicago_Feb_2_2011_storm.JPG)
stuck on Lake Shore Drive in the 2011 snow
storm](_book/images/Chicago_Snow_Plow_stuck_on_Lake_Shore_drive_Chicago_Feb_2_2011_storm.jpg)

To see what the forecasts looked like, check out this news report from
the day:

<https://www.nbcchicago.com/news/local/blizzard-unleashes-winter-fury/2096753/>

It turns out all three predictions of the blizzard were accurate to a
level that almost seemed uncanny to me: Start time, accumulation, and
end time were all spot on. This is the first time I recall ever seeing a
prediction at this level of accuracy. I had no idea this type of
predictive accuracy was even possible. This level of accuracy in
predicting results totally blew me away. I had never seen anything with
this level of accuracy, and now I wanted to know how it was done.

I searched and searched for how the accuracy was so high for this
forecast.

The power of the method—whatever it was—was obvious to me. I realized
that if it could work for the weather, the solution method could work in
an incredibly broad range of situations. A few of many other areas
include business forecasts, production work, modeling prices, and much,
much more. But that this point I had no idea how the accurate prediction
was done.

Some months later a person wrote to [Tom
Skilling](https://en.wikipedia.org/wiki/Tom_Skilling), chief
meteorologist for WGN TV in Chicago. Tom posted an answer that opened up
the solution for me. Here is the relevant part of [Tom Skilling's
answer](https://www.facebook.com/TomSkilling/posts/531146448370303) to a
2011 storm how the forecast was so accurate:

> The Weather Service has developed an interesting "SNOWFALL ENSEMBLE
> FORECAST PROBABILITY SYSTEM" which draws upon a wide range of snow
> accumulation forecasts from a whole set of different computer models.
> By "blending" these model projections, probability of snowfalls
> falling within certain ranges becomes possible. Also, this "blending"
> of multiple forecasts "smooths" the sometimes huge model disparities
> in the amounts being predicted. The resulting probabilities therefore
> represent a "best case" forecast.

So that was the first step. Ensembles were the way they achieved such
extraordinary prediction accuracy.

My next goal was to figure out how ensembles were made. As I looked up
information, it became obvious that ensembles had been used for a while,
such as the winning entry in the Netflix Prize Competition:

![Netflix Prize Competition](_book/images/netflix_prize.jpg)

The [Netflix Prize
Competition](https://en.wikipedia.org/wiki/Netflix_Prize) was sponsored
by Netflix to create a method to accurately predict user ratings on
films. The minimum winning score needed to beat the Netflix method
(named Cinematch) by at least 10%. Several years of work went into
solving this problem, and the results even included several published
papers. The winning solution was an ensemble of methods that beat the
Cinematch results by 10.09%.

So it was now clear to me that ensembles were the path forward. However,
I had no idea how to make ensembles.

I went to graduate school to study data science and predictive
analytics. My degree was completed in 2017, from Northwestern
University. However, I still was not sure how ensembles of models were
built, nor could I find any clear methods to build them (except for
pre-made methods, such as random forests). While it is true there were
packages that could do some of the work, nothing I found did what I was
looking for: How to build ensembles of models in general. Despite
playing with the idea and looking online, I was not able to build the
ensembles I wanted to build.

## Saturday, October 15, 2022 at 4:58 pm. The exact birth of the Ensembles system

Everything changed on Saturday, October 15, 2022 at 4:58 pm. I was
playing with various methods to make an ensemble, and got an ensemble
that worked for the very first time. While the results were extremely
modest by any standards, it was clear to me that the foundation was
there to build a general solution that can work in an extremely wide
range of areas. Here is my journal entry:

![Birth of the ensembles method (typo of W0w in the
original)](_book/images/Birth_of_ensembles.jpg)

You might be asking yourself how I know the day and time. That is a very
reasonable question. I've been keeping a journal since I was 19 years
old, and have thousands of entries. As soon as I realized how to
correctly build ensembles, I made this entry, which contains the key
elements to make an ensemble, and we will do these steps in just a
moment. Notice that the subject line in the journal matches the text
above.

One of the ways to improve your skills is to keep a journal, and we'll
be looking at that in more depth in this chapter and future chapters.
The journal I use is [MacJournal](https://danschimpf.com/), though there
are a large number of other options available on the market.

![](_book/images/Journal_birth_of_ensembles.png)

Birth of ensembles, Saturday, October 15, 2022 at 4:58 pm

![Keep a journal](_book/images/Keep_a_journal.jpg)

## Here is what an ensemble of models looks like at the most basic level, using the Boston Housing data set as an example:

### Head of Boston Housing data set

![Head of Boston Housing data
set](_book/images/Boston_Housing_data_head.png)

We will start our first ensemble with a data set that only has numerical
values. Our first example will use the Boston Housing data set, from the
MASS package. While the Boston Housing data set is controversial (and we
will discuss some of the controversies in our example making
professional quality reports for the C-Suite), for now it works as a
very well known data set to begin our journey into ensembles.

Overview of the most basic steps to make an ensemble:

We will be using the Boston Housing data set, so let's have a look at
some Boston images:

![Boston](_book/images/Boston.png)

## The steps to build your first ensemble from scratch

-   Load the packages we will need (MASS, tree)

-   Load the Boston Housing data set, and split it into train (60%) and
    test (40%) sections.

-   Create a linear model by fitting the linear model on the training
    data, and make predictions on the Boston Housing test data. Measure
    the accuracy of the predictions against the actual values.

-   Create a model using trees by fitting the tree model on the training
    data, and making predictions on the Boston Housing test data.
    Measure the accuracy of the predictions against the actual values.

-   Make a new data frame. This will be our ensemble of model
    predictions. One column will be the linear predictions, and one will
    be the tree predictions.

-   Make a new column for the true values—these are the true values in
    the Boston Housing test data set

-   Once we have the new ensemble data set, it's simply another data
    set. No different in many ways from any other data set (except how
    it was made).

-   Break the ensemble data set into train (60%) and test (40%)
    sections.

-   Fit a linear model to the ensemble training data. Make predictions
    using the testing data, and measure the accuracy of the predictions
    against the test data.

-   Summarize the results.

**I suggest reading the over of the most basic steps to make an ensemble
a couple of times, to make sure you are very familiar with the steps.**

## Building the first actual ensemble

Load the packages we will need (MASS, tree):


``` r
library(MASS) # for the Boston Housing data set
library(tree) # To make models using trees
library(Metrics) # To calculate error rate (root mean squared error)
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

Load the Boston Housing data set, and split it into train (60%) and test
(40%) sections.


``` r
df <- MASS::Boston
train <- df[1:400, ]
test <- df[401:505, ]

# Let's have a quick look at the train and test sets
head(train)
#>      crim zn indus chas   nox    rm  age    dis rad tax
#> 1 0.00632 18  2.31    0 0.538 6.575 65.2 4.0900   1 296
#> 2 0.02731  0  7.07    0 0.469 6.421 78.9 4.9671   2 242
#> 3 0.02729  0  7.07    0 0.469 7.185 61.1 4.9671   2 242
#> 4 0.03237  0  2.18    0 0.458 6.998 45.8 6.0622   3 222
#> 5 0.06905  0  2.18    0 0.458 7.147 54.2 6.0622   3 222
#> 6 0.02985  0  2.18    0 0.458 6.430 58.7 6.0622   3 222
#>   ptratio  black lstat medv
#> 1    15.3 396.90  4.98 24.0
#> 2    17.8 396.90  9.14 21.6
#> 3    17.8 392.83  4.03 34.7
#> 4    18.7 394.63  2.94 33.4
#> 5    18.7 396.90  5.33 36.2
#> 6    18.7 394.12  5.21 28.7
```

``` r
head(test)
#>         crim zn indus chas   nox    rm   age    dis rad tax
#> 401 25.04610  0  18.1    0 0.693 5.987 100.0 1.5888  24 666
#> 402 14.23620  0  18.1    0 0.693 6.343 100.0 1.5741  24 666
#> 403  9.59571  0  18.1    0 0.693 6.404 100.0 1.6390  24 666
#> 404 24.80170  0  18.1    0 0.693 5.349  96.0 1.7028  24 666
#> 405 41.52920  0  18.1    0 0.693 5.531  85.4 1.6074  24 666
#> 406 67.92080  0  18.1    0 0.693 5.683 100.0 1.4254  24 666
#>     ptratio  black lstat medv
#> 401    20.2 396.90 26.77  5.6
#> 402    20.2 396.90 20.32  7.2
#> 403    20.2 376.11 20.31 12.1
#> 404    20.2 396.90 19.77  8.3
#> 405    20.2 329.46 27.38  8.5
#> 406    20.2 384.97 22.98  5.0
```

Create a linear model by fitting the linear model on the training data,
and make predictions on the Boston Housing test data. Measure the
accuracy of the predictions against the actual values.


``` r
Boston_lm <- lm(medv ~ ., data = train) # Fit the model to the training data
Boston_lm_predictions <- predict(object = Boston_lm, newdata = test)

# Let's have a quick look at the model predictions
head(Boston_lm_predictions)
#>       401       402       403       404       405       406 
#> 12.618507 19.785728 20.919370 13.014507  6.946392  5.123039
```

Calculate the error for the model


``` r
Boston_linear_RMSE <- Metrics::rmse(actual = test$medv, predicted = Boston_lm_predictions)
Boston_linear_RMSE
#> [1] 6.108005
```

The error rate for the linear model is 6.108005. Let's do the same using
the tree method.

Create a model using trees by fitting the tree model on the training
data, and making predictions on the Boston Housing test data. Measure
the accuracy of the predictions against the actual values.


``` r
Boston_tree <- tree(medv ~ ., data = train) # Fit the model to the training data
Boston_tree_predictions <- predict(object = Boston_tree, newdata = test)

# Let's have a quick look at the predictions:
head(Boston_tree_predictions)
#>      401      402      403      404      405      406 
#> 13.30769 13.30769 13.30769 13.30769 13.30769 13.30769
```

Calculate the error rate for the tree model:


``` r
Boston_tree_RMSE <- Metrics::rmse(actual = test$medv, predicted = Boston_tree_predictions)
Boston_tree_RMSE
#> [1] 5.478017
```

The error rate for the tree model is lower (which is better). The error
rate for the tree model is 5.478017.

## We're ready to make our first ensemble!!

Make a new data frame. This will be our ensemble of model predictions,
and one column for the true values. One column will be the linear
predictions, and one will be the tree predictions. We'll make a third
column, the true values.

Make a new column for the true values—these are the true values in the
Boston Housing test data set


``` r
ensemble <- data.frame(
  'linear' = Boston_lm_predictions,
  'tree' = Boston_tree_predictions,
  'y' = test$medv
)

# Let's have a look at the ensemble:
head(ensemble)
#>        linear     tree    y
#> 401 12.618507 13.30769  5.6
#> 402 19.785728 13.30769  7.2
#> 403 20.919370 13.30769 12.1
#> 404 13.014507 13.30769  8.3
#> 405  6.946392 13.30769  8.5
#> 406  5.123039 13.30769  5.0
```

``` r
dim(ensemble)
#> [1] 105   3
```

Once we have the new ensemble data set, it's simply another data set. No
different in many ways from any other data set (except how it was made).

Break the ensemble data set into train (60%) and test (40%) sections.
There is nothing special about the 60/40 split here, you may use any
numbers you wish.


``` r
ensemble_train <- ensemble[1:60, ]
ensemble_test <- ensemble[61:105, ]

head(ensemble_train)
#>        linear     tree    y
#> 401 12.618507 13.30769  5.6
#> 402 19.785728 13.30769  7.2
#> 403 20.919370 13.30769 12.1
#> 404 13.014507 13.30769  8.3
#> 405  6.946392 13.30769  8.5
#> 406  5.123039 13.30769  5.0
```

``` r
head(ensemble_test)
#>       linear     tree    y
#> 461 23.88984 13.30769 16.4
#> 462 23.29129 13.30769 17.7
#> 463 22.54055 21.84327 19.5
#> 464 25.50940 21.84327 20.2
#> 465 22.71231 21.84327 21.4
#> 466 20.83810 21.84327 19.9
```

Fit a linear model to the ensemble training data. Make predictions using
the testing data, and measure the accuracy of the predictions against
the test data. Notice how similar this is to our linear and tree models.


``` r
# Fit the model to the training data
ensemble_lm <- lm(y ~ ., data = ensemble_train)

# Make predictions using the model on the test data
ensemble_lm_predictions <- predict(object = ensemble_lm, newdata = ensemble_test)

# Calculate error rate for the ensemble predictions
ensemble_lm_rmse <- Metrics::rmse(actual = ensemble_test$y, predicted = ensemble_lm_predictions)

# Report the error rate for the ensemble
ensemble_lm_rmse
#> [1] 4.826962
```

Summarize the results.


``` r
results <- data.frame(
  'Model' = c('Linear', 'Tree', 'Ensemble'),
  'Error' = c(Boston_linear_RMSE, Boston_tree_RMSE, ensemble_lm_rmse)
)

results
#>      Model    Error
#> 1   Linear 6.108005
#> 2     Tree 5.478017
#> 3 Ensemble 4.826962
```

Clearly the ensemble had the lowest error rate of the three models. The
ensemble is easily the best of the three models because it has the
lowest error rate of all the models.

### Try it yourself: Make an ensemble where the ensemble is made using trees instead of linear models.


``` r
# Fit the model to the training data
ensemble_tree <- tree(y ~ ., data = ensemble_train)

# Make predictions using the model on the test data
ensemble_tree_predict <- predict(object = ensemble_tree, newdata = ensemble_test)

# Let's look at the predictions
head(ensemble_tree_predict)
#>      461      462      463      464      465      466 
#> 14.80000 14.80000 18.94286 18.94286 18.94286 18.94286
```

``` r

# Calculate the error rate
ensemble_tree_rmse <- Metrics::rmse(actual = ensemble_test$y, predicted = ensemble_tree_predict)

ensemble_tree_rmse
#> [1] 5.322011
```

How does this compare to our three other results? Let's update the
results table


``` r
results <- data.frame(
  'Model' = c('Linear', 'Tree', 'Ensemble_Linear', 'Ensemble_Tree'),
  'Error' = c(Boston_linear_RMSE, Boston_tree_RMSE, ensemble_lm_rmse, ensemble_tree_rmse)
)

results <- results %>% arrange(Error)

results
#>             Model    Error
#> 1 Ensemble_Linear 4.826962
#> 2   Ensemble_Tree 5.322011
#> 3            Tree 5.478017
#> 4          Linear 6.108005
```

### Both of the ensemble models beat both of the individual models in this example

## Principle: What is one improvement that can be made? Use a diverse set of models and ensembles to get the best possible result

As we shall see when we go through and learn how to build ensembles, the
numerical method we will use will build 27 individual models and 13
ensembles for a total of 40 results. When the goal is to get the best
possible results, a diverse set of models and ensembles, such as the 40
results for numerical data, will produce much better results than a
limited number of models and ensembles.

We will do the same principal when we are looking at classification
data, logistic, data, and time series forecasting data. We will use a
large number of individual models and ensembles with the goal of
achieving the best possible result.

## Principle: Randomizing the data before the analysis will make the results more general (and is very easy to do!)


``` r
df <- df[sample(nrow(df)),] # Randomize the rows before the analysis
```

## Try it yourself: Repeat the previous analysis, but randomize the rows before the analysis. Otherwise keep the process the same. Share your results on social media.

We'll follow the exact same steps, except for randomizing the rows
first.

• Randomize the rows

• Break the data into train and test sets

• Fit the model to the training set

• Make predictions and calculate error from the model on the test set


``` r
df <- df[sample(nrow(df)),] # Randomize the rows before the analysis

train <- df[1:400, ]
test <- df[401:505, ]

# Fit the model to the training data
Boston_lm <- lm(medv ~ ., data = train)

# Make predictions using the model on the test data
Boston_lm_predictions <- predict(object = Boston_lm, newdata = test)

# Let's have a quick look at the linear model predictions:

head(Boston_lm_predictions)
#>      311      429      258       91      498      128 
#> 18.32366 14.22723 42.37954 26.94967 18.94004 15.20354
```


``` r
Boston_linear_rmse <- Metrics::rmse(actual = test$medv, predicted = Boston_lm_predictions)

Boston_tree <- tree(medv ~ ., data = train)
Boston_tree_predictions <- predict(object = Boston_tree, newdata = test)
Boston_tree_rmse <- Metrics::rmse(actual = test$medv, predicted = Boston_tree_predictions)

# Let's have a quick look at the tree model predictions:

head(Boston_tree_predictions)
#>      311      429      258       91      498      128 
#> 21.72692 10.57826 47.08421 21.72692 21.72692 15.42821
```


``` r
ensemble <- data.frame( 'linear' = Boston_lm_predictions, 'tree' = Boston_tree_predictions, 'y_ensemble' = test$medv )

ensemble <- ensemble[sample(nrow(ensemble)), ] # Randomizes the rows of the ensemble

ensemble_train <- ensemble[1:60, ]
ensemble_test <- ensemble[61:105, ]
```


``` r
ensemble_lm <- lm(y_ensemble ~ ., data = ensemble_train)

# Predictions for the ensemble linear model

ensemble_prediction <- predict(ensemble_lm, newdata = ensemble_test)

# Root mean squared error for the ensemble linear model

ensemble_lm_rmse <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = ensemble_prediction)

# Same for tree models

ensemble_tree <- tree(y_ensemble ~ ., data = ensemble_train)
ensemble_tree_predictions <- predict(object = ensemble_tree, newdata = ensemble_test)
ensemble_tree_rmse <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = ensemble_tree_predictions)

results <- list( 'Linear' = Boston_linear_rmse, 'Trees' = Boston_tree_rmse, 'Ensembles_Linear' = ensemble_lm_rmse, 'Ensemble_Tree' = ensemble_tree_rmse )

results
#> $Linear
#> [1] 3.793948
#> 
#> $Trees
#> [1] 5.02707
#> 
#> $Ensembles_Linear
#> [1] 4.823033
#> 
#> $Ensemble_Tree
#> [1] 4.103466
```

The fact that our results are a bit different from our first ensemble is
useful. This gives us another solid principle to use in our analysis
methods:

## The more we can randomize the data, the more our results will match nature

Just watch: Repeat the results 100 times, return the mean of the results
(hint: It's two small changes)


``` r
for (i in 1:100) {

# First the linear model with randomized data

df <- df[sample(nrow(df)),] # Randomize the rows before the analysis

train <- df[1:400, ]
test <- df[401:505, ]

Boston_lm <- lm(medv ~ ., data = train)
Boston_lm_predictions <- predict(object = Boston_lm, newdata = test)

# Let's have a quick look at the linear model predictions:

head(Boston_lm_predictions)

# Let's calculate the root mean squared error rate of the predictions:

Boston_linear_rmse[i] <- Metrics::rmse(actual = test$medv, predicted = Boston_lm_predictions)

Boston_linear_rmse_mean <- mean(Boston_linear_rmse)

# Let's use tree models

Boston_tree <- tree(medv ~ ., data = train)

Boston_tree_predictions <- predict(object = Boston_tree, newdata = test)

# Let's have a quick look at the tree model predictions:

head(Boston_tree_predictions)

# Let's calculate the root mean squared error rate of the predictions:

Boston_tree_rmse[i] <- Metrics::rmse(actual = test$medv, predicted = Boston_tree_predictions) 
Boston_tree_rmse_mean <- mean(Boston_tree_rmse)

ensemble <- data.frame('linear' = Boston_lm_predictions, 'tree' = Boston_tree_predictions, 'y_ensemble' = test$medv )

ensemble <- ensemble[sample(nrow(ensemble)), ] # Randomizes the rows of the ensemble

ensemble_train <- ensemble[1:60, ]
ensemble_test <- ensemble[61:105, ]

# Ensemble linear modeling

ensemble_lm <- lm(y_ensemble ~ ., data = ensemble_train)

# Predictions for the ensemble linear model

ensemble_prediction <- predict(ensemble_lm, newdata = ensemble_test)

# Root mean squared error for the ensemble linear model

ensemble_lm_rmse[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = ensemble_prediction)

ensemble_lm_rmse_mean <- mean(ensemble_lm_rmse)

ensemble_tree <- tree(y_ensemble ~ ., data = ensemble_train)

ensemble_tree_predictions <- predict(object = ensemble_tree, newdata = ensemble_test) 

ensemble_tree_rmse[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = 
ensemble_tree_predictions)

ensemble_tree_rmse_mean <- mean(ensemble_tree_rmse)

results <- data.frame(
  'Linear' = Boston_linear_rmse_mean,
  'Trees' = Boston_tree_rmse_mean,
  'Ensembles_Linear' = ensemble_lm_rmse_mean,
  'Ensemble_Tree' = ensemble_tree_rmse_mean )

}

results
#>     Linear   Trees Ensembles_Linear Ensemble_Tree
#> 1 4.838286 4.66105         4.211636      5.199095
```

``` r
warnings() # No warnings!
```

![Automate as much as
possible](_book/images/automate_as_much_as_possible.jpg)

## Principle: "Is this my very best work?"

This is your best work to build ensembles at this stage of your skills.
We are going to make a number of improvements to the solutions we see
here, so our final result will be much stronger than what we have here
so far. Always strive to do your very best work, without any excuses.

## "Where do I get help with errors or warnings?"

It is extremely useful to check if your code returns any errors or
warnings, and fix those as fast as possible. There are numerous sites to
help address errors in your code:

<https://stackoverflow.com>

<https://forum.posit.co>

<https://www.r-project.org/help.html>

## Is there an easy way to save all trained models?

Absolutely! We will simply add the code at the end of this section that
saves the four trained models (linear, tree, ensemble_linear and
ensemble_tree), as follows:


``` r
library(MASS)
library(Metrics)
library(tree)

ensemble_lm_rmse <- 0
ensemble_tree_rmse <- 0

for (i in 1:100) {

# Fit the linear model with randomized data

df <- df[sample(nrow(df)),] # Randomize the rows before the analysis

train <- df[1:400, ]
test <- df[401:505, ]

Boston_lm <- lm(medv ~ ., data = train)

Boston_lm_predictions <- predict(object = Boston_lm, newdata = test)

# Let's have a quick look at the linear model predictions:

head(Boston_lm_predictions)

# Let's calculate the root mean squared error rate of the predictions:

Boston_linear_rmse[i] <- Metrics::rmse(actual = test$medv, predicted = Boston_lm_predictions) 
Boston_linear_rmse_mean <- mean(Boston_linear_rmse)

# Let's use tree models

Boston_tree <- tree(medv ~ ., data = train)

Boston_tree_predictions <- predict(object = Boston_tree, newdata = test)

# Let's have a quick look at the tree model predictions:

head(Boston_tree_predictions)

# Let's calculate the root mean squared error rate of the predictions:

Boston_tree_rmse[i] <- Metrics::rmse(actual = test$medv, predicted = Boston_tree_predictions) 
Boston_tree_rmse_mean <- mean(Boston_tree_rmse)

ensemble <- data.frame( 'linear' = Boston_lm_predictions, 'tree' = Boston_tree_predictions, 'y_ensemble' = test$medv )

ensemble <- ensemble[sample(nrow(ensemble)), ] # Randomizes the rows of the ensemble

ensemble_train <- ensemble[1:60, ]

ensemble_test <- ensemble[61:105, ]

# Ensemble linear modeling

ensemble_lm <- lm(y_ensemble ~ ., data = ensemble_train)

# Predictions for the ensemble linear model

ensemble_prediction <- predict(ensemble_lm, newdata = ensemble_test)

# Root mean squared error for the ensemble linear model

ensemble_lm_rmse[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = ensemble_prediction)

ensemble_lm_rmse_mean <- mean(ensemble_lm_rmse)

ensemble_tree <- tree(y_ensemble ~ ., data = ensemble_train)

ensemble_tree_predictions <- predict(object = ensemble_tree, newdata = ensemble_test) 

ensemble_tree_rmse[i] <- Metrics::rmse(actual = ensemble_test$y_ensemble, predicted = ensemble_tree_predictions)

ensemble_tree_rmse_mean <- mean(ensemble_tree_rmse)

results <- list( 'Linear' = Boston_linear_rmse_mean, 'Trees' = Boston_tree_rmse_mean, 'Ensembles_Linear' = ensemble_lm_rmse_mean, 'Ensemble_Tree' = ensemble_tree_rmse_mean )

}

results
#> $Linear
#> [1] 4.854109
#> 
#> $Trees
#> [1] 4.712546
#> 
#> $Ensembles_Linear
#> [1] 4.260444
#> 
#> $Ensemble_Tree
#> [1] 5.180599
```

``` r
warnings()
```


``` r

Boston_lm <- Boston_lm
Boston_tree <- Boston_tree
ensemble_lm <- ensemble_lm
ensemble_tree <- ensemble_tree
```

### What about classification, logistic and time series data?

In subsequent chapters we will do similar processes with classification,
logistic and time series data. It's possible to build ensembles with all
these types of data. The results are extremely similar to the results
we've seen here with numerical data: While the ensembles won't always
have the best results, it is best to have a diverse set of models and
ensembles to get the best possible results.

### Principle: Ensembles can work with many types of data, and we will do that in this book

### Can it make predictions on totally new data from the trained models---including the ensembles?

The solutions in this book are independent of the use of the data. We
will look at everything from housing prices to business analysis to HR
analytics to research in medicine. One of our later examples will do
exactly what this question is asking—build individual and ensemble
models from data, then use those pre-trained models to make predictions
on totally unseen data. You will develop this set of skills later in the
book, but it's a minor extension of what you're already seen and
completed.

### The way I was taught how to write code was totally wrong for me: The best way for me is to start at the end and work backward from there. Do not start coding looking for a solution, instead, start with the ending and work backwards from there.

![Start at the end and work backwards](_book/images/Steve%20Jobs.jpg)

[Start at the end and work backwards from
there](https://www.youtube.com/watch?v=oeqPrUmVz-o)

The biggest lesson for me in all of this work is how to make ensembles.
You've already seen some of the steps, and there are more results to
come. The second biggest lesson is that everything I was taught about
how to do data science and AI was backwards to what actually works for
me in real life. I've learned how I learn, and applied that skill
(learning how I learn) to a wide range of skills, including:

• Running a multi-million dollar division of a Fortune 1000 company,
including full profit and loss responsibility

• Performing at a professional level on many musical instruments

• Able to communicate in English, Spanish and sign language in a
professional setting

• Earning the #1 place on the annual undergradate university mathematics
competition—twice

• Completing a Master's degree in Guidance and Counseling, allowing me
to help many people in their path toward a healthier life

• Leader of the Oak Park, Illinois chapter of Amnesty International for
ten years, helping to release several Prisoners of Conscience

• President of the Chicago Apple User Group for ten years, helping many
people do extremely good work with their hardware and software

• Leg press 1,000 pounds ten times in a row

• Climbed a mountain in Colorado

• Completed multiple skydives (and looking forward to doing more)

The point here is that I have learned how I learn, and I've applied that
skill to many areas. When I started learning data science/AI/coding, it
was all very different from the way I was being creative my whole life.
The way that works for me is to start at the end, work backward from
there, and never give up. Maybe the best evidence of the success of this
method is this fact:

**When I started to write the code that led to the Ensembles package, I
followed those steps: Start at the end, work backward from there, and
never give up. I wound up writing an average of 1,000 lines of clean,
error free code per month for 15 months. The Ensembles package is around
15,000 lines of clean, error free code.**

I found my attitude was much more important than my skill set, by a long
shot.

### How I stuck with it all the way to the end: The best career advice I ever received was from a homeless man I never met, and answers the question of what most strongly predicts success.

![Ashford and Simpson](_book/images/Ashford_and_Simpson.png)

Ashford and Simpson

Learning about building ensembles will help you make more accurate
predictions. That's an extrdmely good skill to have in any setting. But
I found the most important thing to predict is success. This has been
studied, and there are quite a few good works on the subject, both
academic and for the general population.

My favorite career advice—which I listened to nearly every day as I
worked on the Ensembles project—is from a man who was homeless at the
time he came up with the words.

Nick Ashford was from Willow Run, Michigan. He moved to New York, hoping
to get into the entertainment world as a dancer. Unfortunately he ended
up homeless on the streets of New York. He slept on park benches, and
got food from soup kitchens.

He heard that the people at White Rock Baptist Church would feed him (a
homeless man) a normal meal, so Nick went there one Sunday morning. He
met the people, especially the choir members, and started working with
the piano player in the choir. Her name is Valerie Simpson.

Soon Nick and Valerie were writing songs for the church choir. Nick
mentioned that while he was homeless, he realized that New York wasn't
going to "do me in". He was determined. The words he put down say:

**Ain't no mountain high enough**

**Ain't no valley low enough**

**Ain't no river wide enough**

Valerie took those words, and set them to music. They sent that song to
Motown, who released it with Marvin Gaye and Tammy Terrell covering the
vocals. It was later re-done by Ashford and Simpson and Paul Riser, with
Diana Ross singing the lead.

Here is a short video that summarizes that experience, and concludes
with the finale of the 1970 version of the song. This attitude that
Ashford and Simpson expressed in song is extremely highly predictive of
success, no matter what the field of endeavor. I found this extremely
motivating, and used it to overcome any obstacles and challenges I had
while on the journey.

While I have the skill of knowing how I learn (which I will continue to
share with you in this book), this attitude of working no matter how
high the mountain or long the valley or wide the river, gives me the how
and the why to keep moving toward success, until that success is fully
achieved.

Later on we will look at how to make presentations, consider this as an
example of the level of quality that can be done:

[https://www.icloud.com/iclouddrive/002bNfVreagRYCYHAZ9GyQ02w#Ain't%5FNo%5FMountain%5FHigh%5FEnough](https://www.icloud.com/iclouddrive/002bNfVreagRYCYHAZ9GyQ02w#Ain't%5FNo%5FMountain%5FHigh%5FEnough){.uri}

### Exercises:

1.  Find your data science Genesis. The data science idea that totally
    excites you and gets you out of bed every day. The idea that leads
    to the creation of many other ideas. The biggest and boldest dreams
    you can possibly have. The idea that is so strong that you have to
    do it. Not for yourself, but for the benefit of all who will use it
    and receive all the good it will create.
2.  Keep a journal of your progress. It's much easier to see results
    over time when there is a record. Set the journal up today (or this
    week). I did not use Github as a journal. My journal was for crazy
    ideas, contradictory evidence, writing down my frustrations and
    successes, inspiration, the one next thing I worked on, and having a
    rock solid record of the path to success. Seeing the path I
    traversed was a huge motivation to finishing the project.
3.  Do your best to add journal entries to your regular schedule.
4.  Make an ensemble using the Boston Housing data set. Model any of the
    other 13 columns of data, not the median value of the home (14th
    column) which we have been working on in this chapter.
5.  Start planning for your comprehensive project. What types of data
    are you most interested in? What patterns would you like to
    discover? Begin looking online now for possible data sets, and so a
    little basic research. More examples will be provided as we get
    closer to that section of the book.
