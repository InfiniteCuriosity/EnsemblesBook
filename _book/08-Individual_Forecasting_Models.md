# How to Make 27 Individual Forecasting Models

This chapter builds time series models. This is also known as forecasting, the professional organization is named the International Institute of Forecasters, and their website is <https://forecasters.org>. I strongly recommend checking the IIF, I've found it to be a very good source of skills and knowledge when it comes to forecasting.

In this chapter we are going to build 16 forecasting models. There are a few large groups of models, with variations within each of the groups. For example, we will use (or not use) seasonality in the model making process.

We'll follow the same pattern/process we've been following in the previous sections:

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

The first step is to load the library In the case of time series forecasting, the library is the excellent FPP3 library. There is an excellent book available that guides the learner through the time series process. The book is Forecasting Principles and Practice. It is currently in its third edition, and I recommend it very highly. The website for the book is:

<https://otexts.com/fpp3/>

The time series data we will use is the most important data that is published on a regular bases by the United States federal government: The monthly labor report. There is a large set of time series data sets at the Bureau of Labor Statistics website:

<https://www.bls.gov>

The top picks of time series data are here:

<https://data.bls.gov/cgi-bin/surveymost?ce>

For our work here we will only be looking at one data set, but it's by far the most watched result: Total nonfarm employment. The data can be found here:

<https://data.bls.gov/timeseries/CES0000000001>

I have the data stored on my Github repository, and w will be accessing that data, but there are other ways the data may be retrieved. If you plan to use this a lot, consider registering for the Application Program Interface (API) for the time series data. More information about the API and directions to register is available at:

<https://www.bls.gov/developers/>

![All data is about people](_book/images/All_data_connects_to_people.jpg)

## 12 Individual Time Series Models

### Arima 1


``` r

library(fpp3)
#> ── Attaching packages ────────────────────────── fpp3 0.5 ──
#> ✔ tibble      3.2.1     ✔ tsibble     1.1.4
#> ✔ dplyr       1.1.4     ✔ tsibbledata 0.4.1
#> ✔ tidyr       1.3.1     ✔ feasts      0.3.2
#> ✔ lubridate   1.9.3     ✔ fable       0.3.4
#> ✔ ggplot2     3.5.1     ✔ fabletools  0.4.2
#> ── Conflicts ───────────────────────────── fpp3_conflicts ──
#> ✖ lubridate::date()    masks base::date()
#> ✖ dplyr::filter()      masks stats::filter()
#> ✖ tsibble::intersect() masks base::intersect()
#> ✖ tsibble::interval()  masks lubridate::interval()
#> ✖ dplyr::lag()         masks stats::lag()
#> ✖ tsibble::setdiff()   masks base::setdiff()
#> ✖ tsibble::union()     masks base::union()
```

``` r

# Set initial values to 0

# Set up function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {

# Determine if the data is quarterly, monthly or weekly from the input:
if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }

# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]

# Build the model
Arima1_model = fable::ARIMA(Difference ~ season() + trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE)

# Calculate error rate
Arima1_test_error <- time_series_train %>%
    fabletools::model(Arima1_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)

# Make predictions on the holdout/test data
Arima1_predictions <- time_series_test %>%
    fabletools::model(
      Arima1_model,
    ) %>%
    fabletools::forecast(h = number)

# Report the predictions
Arima1_prediction_model <- Arima1_predictions[1]
Arima1_prediction_date<- Arima1_predictions[2]
Arima1_prediction_range <- Arima1_predictions[3]
Arima1_prediction_mean <-Arima1_predictions[4]

results <- data.frame(
  'Model' = Arima1_predictions[1],
  'Error' = Arima1_test_error$RMSE,
  'Date' = Arima1_predictions[2],
  'Forecast' = Arima1_predictions[4]
)

return(results)

}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>         .model    Error     Date     .mean
#> 1 Arima1_model 58.39161 2024 May  740.0623
#> 2 Arima1_model 58.39161 2024 Jun 1029.3480
#> 3 Arima1_model 58.39161 2024 Jul  586.4908
```

``` r
warnings()
```

### Arima 2


``` r

library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {

# Determine if the data is quarterly, monthly or weekly from the input:
if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }

# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]

Arima2_model <- fable::ARIMA(Difference ~ season(), stepwise = TRUE, greedy = TRUE, approximation = TRUE)

# Calculate error rate
Arima2_test_error <- time_series_train %>%
  fabletools::model(Arima2_model) %>%
  fabletools::forecast(h = number) %>%
  fabletools::accuracy(time_series_test)

# Make predictions on the holdout/test data
Arima2_predictions <- time_series_test %>%
  fabletools::model(
    Arima2_model,
  ) %>%
  fabletools::forecast(h = number)

# Report the predictions
Arima2_prediction_model <- Arima2_predictions[1]
Arima2_prediction_date<- Arima2_predictions[2]
Arima2_prediction_range <- Arima2_predictions[3]
Arima2_prediction_mean <-Arima2_predictions[4]

results <- data.frame(
  'Model' = Arima2_predictions[1],
  'Error' = Arima2_test_error$RMSE,
  'Date' = Arima2_predictions[2],
  'Forecast' = Arima2_predictions[4]
)

return(results)

}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>         .model    Error     Date    .mean
#> 1 Arima2_model 54.98322 2024 May 623.5714
#> 2 Arima2_model 54.98322 2024 Jun 912.8571
#> 3 Arima2_model 54.98322 2024 Jul 470.0000
```

``` r
warnings()
```

### Arima3


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {

# Determine if the data is quarterly, monthly or weekly from the input:
if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }

# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]

# Create the model:
Arima3_model <- fable::ARIMA(Difference ~ trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE)

# Calculate the error:
Arima3_test_error <- time_series_train %>%
    fabletools::model(Arima3_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)

# Calculate the forecast:
Arima3_predictions <- time_series_test %>%
    fabletools::model(
      Arima3_model,
    ) %>%
    fabletools::forecast(h = number)

# Report the predictions:
results <- data.frame(
  'Model' = Arima3_predictions[1],
  'Error' = Arima3_test_error$RMSE,
  'Date' = Arima3_predictions[2],
  'Forecast' = Arima3_predictions[4]
)

return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>         .model    Error     Date    .mean
#> 1 Arima3_model 46.96308 2024 May 196.6640
#> 2 Arima3_model 46.96308 2024 Jun 197.5579
#> 3 Arima3_model 46.96308 2024 Jul 198.4518
```

``` r
warnings()
```

### Arima4


``` r

library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
# Create the model:
Arima4_model <- fable::ARIMA(Difference ~ trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE)
  
  # Calculate the error:
  Arima4_test_error <- time_series_train %>%
    fabletools::model(Arima4_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Arima4_predictions <- time_series_test %>%
    fabletools::model(
      Arima4_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Arima4_predictions[1],
    'Error' = Arima4_test_error$RMSE,
    'Date' = Arima4_predictions[2],
    'Forecast' = Arima4_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>         .model    Error     Date    .mean
#> 1 Arima4_model 46.96308 2024 May 196.6640
#> 2 Arima4_model 46.96308 2024 Jun 197.5579
#> 3 Arima4_model 46.96308 2024 Jul 198.4518
```

``` r
warnings()
```

### Deterministic


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
Deterministic_model <- fable::ARIMA(Difference ~  1 + pdq(d = 0))
  
# Calculate the error:
Deterministic_test_error <- time_series_train %>%
    fabletools::model(Deterministic_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
# Calculate the forecast:
Deterministic_predictions <- time_series_test %>%
    fabletools::model(
      Deterministic_model,
    ) %>%
    fabletools::forecast(h = number)
  
# Report the predictions:
results <- data.frame(
    'Model' = Deterministic_predictions[1],
    'Error' = Deterministic_test_error$RMSE,
    'Date' = Deterministic_predictions[2],
    'Forecast' = Deterministic_predictions[4]
  )
  
return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>                .model    Error     Date    .mean
#> 1 Deterministic_model 42.86143 2024 May 146.3409
#> 2 Deterministic_model 42.86143 2024 Jun 146.3409
#> 3 Deterministic_model 42.86143 2024 Jul 146.3409
```

``` r
warnings()
```

### Drift


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
# Create the model:
Drift_model <- fable::SNAIVE(Difference ~ drift())

# Calculate the error:
Drift_test_error <- time_series_train %>%
    fabletools::model(Drift_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
# Calculate the forecast:
Drift_predictions <- time_series_test %>%
    fabletools::model(
      Drift_model,
    ) %>%
    fabletools::forecast(h = number)
  
# Report the predictions:
results <- data.frame(
    'Model' = Drift_predictions[1],
    'Error' = Drift_test_error$RMSE,
    'Date' = Drift_predictions[2],
    'Range' = Drift_predictions[3],
    'Value' = Drift_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>        .model    Error     Date            Difference
#> 1 Drift_model 99.09185 2024 May N(287.3684, 12520161)
#> 2 Drift_model 99.09185 2024 Jun N(191.3684, 12520161)
#> 3 Drift_model 99.09185 2024 Jul N(193.3684, 12520161)
#>      .mean
#> 1 287.3684
#> 2 191.3684
#> 3 193.3684
```

``` r
warnings()
```

### ETS1


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
# Create the model:
ETS1_model <-   fable::ETS(Difference ~ season() + trend())

# Calculate the error:
ETS1_test_error <- time_series_train %>%
    fabletools::model(ETS1_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
# Calculate the forecast:
ETS1_predictions <- time_series_test %>%
    fabletools::model(
      ETS1_model,
    ) %>%
    fabletools::forecast(h = number)
  
# Report the predictions:
results <- data.frame(
    'Model' = ETS1_predictions[1],
    'Error' = ETS1_test_error$RMSE,
    'Date' = ETS1_predictions[2],
    'Value' = ETS1_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>       .model    Error     Date    .mean
#> 1 ETS1_model 42.98491 2024 May 189.8662
#> 2 ETS1_model 42.98491 2024 Jun 189.8662
#> 3 ETS1_model 42.98491 2024 Jul 189.8662
```

``` r
warnings()
```

### ETS2


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
# Create the model:
ETS2_model <- fable::ETS(Difference ~ trend())

# Calculate the error:
ETS2_test_error <- time_series_train %>%
    fabletools::model(ETS2_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
# Calculate the forecast:
ETS2_predictions <- time_series_test %>%
    fabletools::model(
      ETS2_model,
    ) %>%
    fabletools::forecast(h = number)
  
# Report the predictions:
results <- data.frame(
    'Model' = ETS2_predictions[1],
    'Error' = ETS2_test_error$RMSE,
    'Date' = ETS2_predictions[2],
    'Value' = ETS2_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>       .model    Error     Date    .mean
#> 1 ETS2_model 42.98491 2024 May 189.8662
#> 2 ETS2_model 42.98491 2024 Jun 189.8662
#> 3 ETS2_model 42.98491 2024 Jul 189.8662
```

``` r
warnings()
```

### ETS3


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
# Create the model:
ETS3_model <- fable::ETS(Difference ~ season())

# Calculate the error:
ETS3_test_error <- time_series_train %>%
    fabletools::model(ETS3_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
# Calculate the forecast:
ETS3_predictions <- time_series_test %>%
    fabletools::model(
      ETS3_model,
    ) %>%
    fabletools::forecast(h = number)
  
# Report the predictions:
results <- data.frame(
    'Model' = ETS3_predictions[1],
    'Error' = ETS3_test_error$RMSE,
    'Date' = ETS3_predictions[2],
    'Value' = ETS3_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>       .model    Error     Date    .mean
#> 1 ETS3_model 42.98491 2024 May 189.8662
#> 2 ETS3_model 42.98491 2024 Jun 189.8662
#> 3 ETS3_model 42.98491 2024 Jul 189.8662
```

``` r
warnings()
```

### ETS4


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
# Create the model:
ETS4_model <- fable::ETS(Difference)

# Calculate the error:
ETS4_test_error <- time_series_train %>%
    fabletools::model(ETS4_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
# Calculate the forecast:
ETS4_predictions <- time_series_test %>%
    fabletools::model(
      ETS4_model,
    ) %>%
    fabletools::forecast(h = number)
  
# Report the predictions:
results <- data.frame(
    'Model' = ETS4_predictions[1],
    'Error' = ETS4_test_error$RMSE,
    'Date' = ETS4_predictions[2],
    'Value' = ETS4_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>       .model    Error     Date    .mean
#> 1 ETS4_model 42.98491 2024 May 189.8662
#> 2 ETS4_model 42.98491 2024 Jun 189.8662
#> 3 ETS4_model 42.98491 2024 Jul 189.8662
```

``` r
warnings()
```

### Holt-Winters Additive


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
# Create the model:
Holt_Winters_Additive_model <- fable::ETS(Difference ~ error("A") + trend("A") + season("A"))

# Calculate the error:
Holt_Winters_Additive_test_error <- time_series_train %>%
    fabletools::model(Holt_Winters_Additive_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
# Calculate the forecast:
Holt_Winters_Additive_predictions <- time_series_test %>%
    fabletools::model(
      Holt_Winters_Additive_model,
    ) %>%
    fabletools::forecast(h = number)
  
# Report the predictions:
results <- data.frame(
    'Model' = Holt_Winters_Additive_predictions[1],
    'Error' = Holt_Winters_Additive_test_error$RMSE,
    'Date' = Holt_Winters_Additive_predictions[2],
    'Value' = Holt_Winters_Additive_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>                        .model    Error     Date    .mean
#> 1 Holt_Winters_Additive_model 52.74689 2024 May 467.2980
#> 2 Holt_Winters_Additive_model 52.74689 2024 Jun 802.6268
#> 3 Holt_Winters_Additive_model 52.74689 2024 Jul 229.5501
```

``` r
warnings()
```

### Holt-Winters Damped


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
# Create the model:
Holt_Winters_Damped_model <- fable::ETS(Difference ~ error("M") + trend("Ad") + season("M"))

# Calculate the error:
Holt_Winters_Damped_test_error <- time_series_train %>%
    fabletools::model(Holt_Winters_Damped_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
# Calculate the forecast:
Holt_Winters_Damped_predictions <- time_series_test %>%
    fabletools::model(
      Holt_Winters_Damped_model,
    ) %>%
    fabletools::forecast(h = number)
  
# Report the predictions:
results <- data.frame(
    'Model' = Holt_Winters_Damped_predictions[1],
    'Error' = Holt_Winters_Damped_test_error$RMSE,
    'Date' = Holt_Winters_Damped_predictions[2],
    'Value' = Holt_Winters_Damped_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>                      .model    Error     Date     .mean
#> 1 Holt_Winters_Damped_model 481.2281 2024 May 137.36755
#> 2 Holt_Winters_Damped_model 481.2281 2024 Jun 116.40494
#> 3 Holt_Winters_Damped_model 481.2281 2024 Jul  97.76673
```

``` r
warnings()
```

### Holt-Winters Multiplicative


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
# Create the model:
Holt_Winters_Multiplicative_model <- fable::ETS(Difference ~ error("M") + trend("A") + season("M"))

# Calculate the error:
Holt_Winters_Multiplicative_test_error <- time_series_train %>%
    fabletools::model(Holt_Winters_Multiplicative_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
# Calculate the forecast:
Holt_Winters_Multiplicative_predictions <- time_series_test %>%
    fabletools::model(
      Holt_Winters_Multiplicative_model,
    ) %>%
    fabletools::forecast(h = number)
  
# Report the predictions:
results <- data.frame(
    'Model' = Holt_Winters_Multiplicative_predictions[1],
    'Error' = Holt_Winters_Multiplicative_test_error$RMSE,
    'Date' = Holt_Winters_Multiplicative_predictions[2],
    'Value' = Holt_Winters_Multiplicative_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>                              .model    Error     Date
#> 1 Holt_Winters_Multiplicative_model 470.7401 2024 May
#> 2 Holt_Winters_Multiplicative_model 470.7401 2024 Jun
#> 3 Holt_Winters_Multiplicative_model 470.7401 2024 Jul
#>      .mean
#> 1 122.6052
#> 2 128.6342
#> 3 108.2321
```

``` r
warnings()
```

### Linear 1


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
# Split the data into train and test:
time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
# Create the model:
Linear1_model <- fable::ETS(Difference ~ error("M") + trend("Ad") + season("M"))

# Calculate the error:
Linear1_test_error <- time_series_train %>%
    fabletools::model(Linear1_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
# Calculate the forecast:
Linear1_predictions <- time_series_test %>%
    fabletools::model(
      Linear1_model,
    ) %>%
    fabletools::forecast(h = number)
  
# Report the predictions:
results <- data.frame(
    'Model' = Linear1_predictions[1],
    'Error' = Linear1_test_error$RMSE,
    'Date' = Linear1_predictions[2],
    'Value' = Linear1_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>          .model    Error     Date     .mean
#> 1 Linear1_model 481.2281 2024 May 137.36755
#> 2 Linear1_model 481.2281 2024 Jun 116.40494
#> 3 Linear1_model 481.2281 2024 Jul  97.76673
```

``` r
warnings()
```

### Linear 2


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Linear2_model <- fable::TSLM(Difference)
  
  # Calculate the error:
  Linear2_test_error <- time_series_train %>%
    fabletools::model(Linear2_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Linear2_predictions <- time_series_test %>%
    fabletools::model(
      Linear2_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Linear2_predictions[1],
    'Error' = Linear2_test_error$RMSE,
    'Date' = Linear2_predictions[2],
    'Value' = Linear2_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>          .model    Error     Date    .mean
#> 1 Linear2_model 120.6944 2024 May 146.3409
#> 2 Linear2_model 120.6944 2024 Jun 146.3409
#> 3 Linear2_model 120.6944 2024 Jul 146.3409
```

``` r
warnings()
```

### Linear 3


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Linear2_model <- fable::TSLM(Difference)
  
  # Calculate the error:
  Linear2_test_error <- time_series_train %>%
    fabletools::model(Linear2_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Linear2_predictions <- time_series_test %>%
    fabletools::model(
      Linear2_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Linear2_predictions[1],
    'Error' = Linear2_test_error$RMSE,
    'Date' = Linear2_predictions[2],
    'Value' = Linear2_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>          .model    Error     Date    .mean
#> 1 Linear2_model 120.6944 2024 May 146.3409
#> 2 Linear2_model 120.6944 2024 Jun 146.3409
#> 3 Linear2_model 120.6944 2024 Jul 146.3409
```

``` r
warnings()
```

### Linear 4


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Linear4_model <- fable::TSLM(Difference ~ trend())
  
  # Calculate the error:
  Linear4_test_error <- time_series_train %>%
    fabletools::model(Linear4_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Linear4_predictions <- time_series_test %>%
    fabletools::model(
      Linear4_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Linear4_predictions[1],
    'Error' = Linear4_test_error$RMSE,
    'Date' = Linear4_predictions[2],
    'Value' = Linear4_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>          .model    Error     Date    .mean
#> 1 Linear4_model 87.83978 2024 May 313.7312
#> 2 Linear4_model 87.83978 2024 Jun 317.4928
#> 3 Linear4_model 87.83978 2024 Jul 321.2543
```

``` r
warnings()
```

### Mean


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Mean_model <- fable::MEAN(Difference)
  
  # Calculate the error:
  Mean_test_error <- time_series_train %>%
    fabletools::model(Mean_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Mean_predictions <- time_series_test %>%
    fabletools::model(
      Mean_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Mean_predictions[1],
    'Error' = Mean_test_error$RMSE,
    'Date' = Mean_predictions[2],
    'Value' = Mean_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>       .model    Error     Date    .mean
#> 1 Mean_model 120.6944 2024 May 146.3409
#> 2 Mean_model 120.6944 2024 Jun 146.3409
#> 3 Mean_model 120.6944 2024 Jul 146.3409
```

``` r
warnings()
```

### Naive


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Naive_model <- fable::NAIVE(Difference)
  
  # Calculate the error:
  Naive_test_error <- time_series_train %>%
    fabletools::model(Naive_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Naive_predictions <- time_series_test %>%
    fabletools::model(
      Naive_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Naive_predictions[1],
    'Error' = Naive_test_error$RMSE,
    'Date' = Naive_predictions[2],
    'Value' = Naive_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>        .model    Error     Date .mean
#> 1 Naive_model 52.38957 2024 May   175
#> 2 Naive_model 52.38957 2024 Jun   175
#> 3 Naive_model 52.38957 2024 Jul   175
```

``` r
warnings()
```

### Neuralnet 1


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Neuralnet1_model <- fable::NNETAR(Difference ~ season() + trend())
  
  # Calculate the error:
  Neuralnet1_test_error <- time_series_train %>%
    fabletools::model(Neuralnet1_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Neuralnet1_predictions <- time_series_test %>%
    fabletools::model(
      Neuralnet1_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Neuralnet1_predictions[1],
    'Error' = Neuralnet1_test_error$RMSE,
    'Date' = Neuralnet1_predictions[2],
    'Value' = Neuralnet1_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>             .model    Error     Date      .mean
#> 1 Neuralnet1_model 80.01379 2024 May  254.58940
#> 2 Neuralnet1_model 80.01379 2024 Jun   31.16869
#> 3 Neuralnet1_model 80.01379 2024 Jul -149.61350
```

``` r
warnings()
```

### Neuralnet 2


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Neuralnet2_model <- fable::NNETAR(Difference ~ trend())
  
  # Calculate the error:
  Neuralnet2_test_error <- time_series_train %>%
    fabletools::model(Neuralnet2_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Neuralnet2_predictions <- time_series_test %>%
    fabletools::model(
      Neuralnet2_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Neuralnet2_predictions[1],
    'Error' = Neuralnet2_test_error$RMSE,
    'Date' = Neuralnet2_predictions[2],
    'Value' = Neuralnet2_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>             .model    Error     Date      .mean
#> 1 Neuralnet2_model 53.91726 2024 May   234.1566
#> 2 Neuralnet2_model 53.91726 2024 Jun -1894.7670
#> 3 Neuralnet2_model 53.91726 2024 Jul -4940.7453
```

``` r
warnings()
```

### Neuralnet 3


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Neuralnet3_model <- fable::NNETAR(Difference ~ season())
  
  # Calculate the error:
  Neuralnet3_test_error <- time_series_train %>%
    fabletools::model(Neuralnet3_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Neuralnet3_predictions <- time_series_test %>%
    fabletools::model(
      Neuralnet3_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Neuralnet3_predictions[1],
    'Error' = Neuralnet3_test_error$RMSE,
    'Date' = Neuralnet3_predictions[2],
    'Value' = Neuralnet3_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>             .model    Error     Date    .mean
#> 1 Neuralnet3_model 32.00607 2024 May 277.4085
#> 2 Neuralnet3_model 32.00607 2024 Jun 425.5440
#> 3 Neuralnet3_model 32.00607 2024 Jul 249.4818
```

``` r
warnings()
```

### Neuralnet 4


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Neuralnet3_model <- fable::NNETAR(Difference ~ season())
  
  # Calculate the error:
  Neuralnet3_test_error <- time_series_train %>%
    fabletools::model(Neuralnet3_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Neuralnet3_predictions <- time_series_test %>%
    fabletools::model(
      Neuralnet3_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Neuralnet3_predictions[1],
    'Error' = Neuralnet3_test_error$RMSE,
    'Date' = Neuralnet3_predictions[2],
    'Value' = Neuralnet3_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>             .model    Error     Date     .mean
#> 1 Neuralnet3_model 47.50821 2024 May 263.52302
#> 2 Neuralnet3_model 47.50821 2024 Jun 240.83646
#> 3 Neuralnet3_model 47.50821 2024 Jul -38.30823
```

``` r
warnings()
```

### Prophet Additive


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Prophet_Additive_model <- fable.prophet::prophet(Difference ~ season(period = 12, type = "additive"))
  
  # Calculate the error:
  Prophet_Additive_test_error <- time_series_train %>%
    fabletools::model(Prophet_Additive_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Prophet_Additive_predictions <- time_series_test %>%
    fabletools::model(
      Prophet_Additive_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Prophet_Additive_predictions[1],
    'Error' = Prophet_Additive_test_error$RMSE,
    'Date' = Prophet_Additive_predictions[2],
    'Value' = Prophet_Additive_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>                   .model   Error     Date    .mean
#> 1 Prophet_Additive_model 98.6034 2024 May 2686.376
#> 2 Prophet_Additive_model 98.6034 2024 Jun 3176.369
#> 3 Prophet_Additive_model 98.6034 2024 Jul 1016.474
```

``` r
warnings()
```

### Prophet Multiplicative


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Prophet_Multiplicative_model <- fable.prophet::prophet(Difference ~ season(period = 12, type = "multiplicative"))
  
  # Calculate the error:
  Prophet_Multiplicative_test_error <- time_series_train %>%
    fabletools::model(Prophet_Multiplicative_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Prophet_Multiplicative_predictions <- time_series_test %>%
    fabletools::model(
      Prophet_Multiplicative_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Prophet_Multiplicative_predictions[1],
    'Error' = Prophet_Multiplicative_test_error$RMSE,
    'Date' = Prophet_Multiplicative_predictions[2],
    'Value' = Prophet_Multiplicative_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>                         .model    Error     Date     .mean
#> 1 Prophet_Multiplicative_model 75.97023 2024 May -37.12521
#> 2 Prophet_Multiplicative_model 75.97023 2024 Jun -65.45028
#> 3 Prophet_Multiplicative_model 75.97023 2024 Jul -30.47203
```

``` r
warnings()
```

### Seasonal Naive


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  SNaive_model <- fable::SNAIVE(Difference)
  
  # Calculate the error:
  SNaive_test_error <- time_series_train %>%
    fabletools::model(SNaive_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  SNaive_predictions <- time_series_test %>%
    fabletools::model(
      SNaive_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = SNaive_predictions[1],
    'Error' = SNaive_test_error$RMSE,
    'Date' = SNaive_predictions[2],
    'Value' = SNaive_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>         .model    Error     Date .mean
#> 1 SNaive_model 98.94106 2024 May   281
#> 2 SNaive_model 98.94106 2024 Jun   185
#> 3 SNaive_model 98.94106 2024 Jul   187
```

``` r
warnings()
```

### Stochastic


``` r
library(fpp3)

# Set up the function:
forecasting <- function(time_series_data, train_amount, number, time_interval = c("Q", "M", "W")) {
  
  # Determine if the data is quarterly, monthly or weekly from the input:
  if (time_interval == "Q") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearquarter(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "M") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearmonth(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  if (time_interval == "W") {
    time_series_data <- time_series_data %>%
      dplyr::mutate(Date = tsibble::yearweek(Label), Value = Value, Difference = tsibble::difference(Value)) %>%
      dplyr::select(Date, Value, Difference) %>%
      tsibble::as_tsibble(index = Date) %>%
      dplyr::slice(-c(1))
  }
  
  # Split the data into train and test:
  time_series_train <- time_series_data[1:round(train_amount*(nrow(time_series_data))),]
  time_series_test <- time_series_data[(round(train_amount*(nrow(time_series_data))) +1):nrow(time_series_data),]
  
  # Create the model:
  Stochastic_model <- fable::ARIMA(Difference ~ pdq(d = 1), stepwise = TRUE, greedy = TRUE, approximation = TRUE)
  
  # Calculate the error:
  Stochastic_test_error <- time_series_train %>%
    fabletools::model(Stochastic_model) %>%
    fabletools::forecast(h = number) %>%
    fabletools::accuracy(time_series_test)
  
  # Calculate the forecast:
  Stochastic_predictions <- time_series_test %>%
    fabletools::model(
      Stochastic_model,
    ) %>%
    fabletools::forecast(h = number)
  
  # Report the predictions:
  results <- data.frame(
    'Model' = Stochastic_predictions[1],
    'Error' = Stochastic_test_error$RMSE,
    'Date' = Stochastic_predictions[2],
    'Value' = Stochastic_predictions[4]
  )
  
  return(results)
}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#>             .model    Error     Date    .mean
#> 1 Stochastic_model 42.98025 2024 May 239.7312
#> 2 Stochastic_model 42.98025 2024 Jun 256.0506
#> 3 Stochastic_model 42.98025 2024 Jul 241.0237
```

``` r
warnings()

```


``` r
summary_table <- data.frame()
```
