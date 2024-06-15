# Ensembles of 26 Forecasting Models

Once you know how to make the 27 individual time series forecasting models, the ensemble simply puts all of those 27 models together.


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

# Fit the ensemble model on the training data
Ensembles_model <- time_series_train %>%
    fabletools::model(
      Ensemble = (
      fable::TSLM(Value ~ season() + trend()) +
      fable::TSLM(Value) + fable::TSLM(Value ~ season()) +
      fable::TSLM(Value ~ trend()) +
      fable::ARIMA(Value ~ season() + trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE) +
      fable::ARIMA(Value ~ season(),stepwise = TRUE, greedy = TRUE, approximation = TRUE) +
      fable::ARIMA(Value ~ trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE) +         fable::ARIMA(Value) + fable::ARIMA(Value ~ pdq(d = 1), stepwise = TRUE, greedy = TRUE, approximation = TRUE) +
      fable::ETS(Value ~ season() + trend()) + fable::ETS(Value ~ trend()) + fable::ETS(Value ~ season()) +
      fable::ETS(Value) +
      fable::ETS(Value ~ error("A") + trend("A") + season("A")) + fable::ETS(Value ~ error("M") + trend("A") + season("M")) +
      fable::ETS(Value ~ error("M") + trend("Ad") + season("M")) +
      fable::MEAN(Value) +
      fable::NAIVE(Value) +
      fable::SNAIVE(Value) +
      fable::SNAIVE(Value ~ drift()) +
      fable.prophet::prophet(Value ~ season(period = 12, type = "multiplicative")) +
      fable.prophet::prophet(Value ~ season(period = 12, type = "additive")) +
      fable::NNETAR(Value ~ season() + trend()) +
      fable::NNETAR(Value ~ trend()) +
      fable::NNETAR(Value ~ season()) +
      fable::NNETAR(Value))/26
    )

# # Make predicitons:
# Ensemble_predictions <- time_series_test %>% 
#   model(Ensemble_model) %>%
#     fabletools::forecast(h = number)

Ensemble_predictions <- time_series_test %>%
  fabletools::model(
    Ensemble = (
      fable::TSLM(Difference ~ season() + trend()) +
      fable::TSLM(Difference) +
      fable::TSLM(Difference ~ season()) +
      fable::TSLM(Difference ~ trend()) +
      fable::ARIMA(Difference ~ season() + trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE) +
      fable::ARIMA(Difference ~ season(),stepwise = TRUE, greedy = TRUE, approximation = TRUE) +
      fable::ARIMA(Difference ~ trend(),stepwise = TRUE, greedy = TRUE, approximation = TRUE) +
      fable::ARIMA(Difference) +
      fable::ARIMA(Difference ~ pdq(d = 1), stepwise = TRUE, greedy = TRUE, approximation = TRUE) +
      fable::ETS(Difference ~ season() + trend()) +
      fable::ETS(Difference ~ trend()) +
      fable::ETS(Difference ~ season()) +
      fable::ETS(Difference) +
      fable::ETS(Difference ~ error("A") + trend("A") + season("A")) +
      fable::ETS(Difference ~ error("M") + trend("A") + season("M")) +
      fable::ETS(Difference ~ error("M") + trend("Ad") + season("M")) +
      fable::MEAN(Difference) +
      fable::NAIVE(Difference) +
      fable::SNAIVE(Difference) +
      fable::SNAIVE(Difference ~ drift()) +
      fable.prophet::prophet(Difference ~ season(period = 12, type = "multiplicative")) +
      fable.prophet::prophet(Difference ~ season(period = 12, type = "additive")) +
      fable::NNETAR(Difference ~ season() + trend()) +
      fable::NNETAR(Difference ~ trend()) +
      fable::NNETAR(Difference ~ season()) +
      fable::NNETAR(Difference)/26
    )
  ) %>%
  fabletools::forecast(h = number)

results <- data.frame(
  'Model' = Ensemble_predictions[1],
  'Date' = Ensemble_predictions[2],
  'Forecast' = Ensemble_predictions[4]
)

return(results)

}

# Test the function:
time_series_data <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/forecasting_jobs/main/Total_Nonfarm.csv')

forecasting(time_series_data = time_series_data, train_amount = 0.60, number = 3, time_interval = "M")
#> Warning in sqrt(diag(best$var.coef)): NaNs produced
#>     .model     Date     .mean
#> 1 Ensemble 2024 May  9786.223
#> 2 Ensemble 2024 Jun  8727.396
#> 3 Ensemble 2024 Jul -1303.570
```

``` r
warnings()
```
