# ===========================
# REQUIRED PACKAGES
# ===========================
library(tidyverse)
library(lubridate)
library(forecast)
library(tseries)
library(corrplot)
library(gridExtra)
library(astsa)
library(xgboost)

# ===========================
# DATA PREPARATION
# ===========================

# Load datasets
cocoa_prices <- read.csv("data/Daily Prices_ICCO.csv")
ghana_weather <- read.csv("data/Ghana_data.csv")
exchange_rate <- read.csv("data/USD_GHS_1994_2024.csv") %>%
  select(Date, ExchangeRate = Price)

# Format date columns
cocoa_prices$Date <- as.Date(cocoa_prices$Date, format = "%d/%m/%Y")
ghana_weather$DATE <- as.Date(ghana_weather$DATE)
exchange_rate$Date <- as.Date(exchange_rate$Date)

# Clean and prepare cocoa price data
cocoa_prices$Price <- as.numeric(gsub(",", "", cocoa_prices$ICCO.daily.price..US..tonne.))
cocoa_prices <- cocoa_prices %>%
  select(Date, Price) %>%
  arrange(Date)

# Aggregate weather data by date
ghana_weather <- ghana_weather %>%
  group_by(DATE) %>%
  summarise(across(c(PRCP, TAVG, TMAX, TMIN), mean, na.rm = TRUE)) %>%
  rename(Date = DATE)

# Merge all datasets
cocoa_data <- cocoa_prices %>%
  left_join(ghana_weather, by = "Date") %>%
  left_join(exchange_rate, by = "Date") %>%
  drop_na()

# Feature engineering: log & differencing
cocoa_data <- cocoa_data %>%
  mutate(log_price = log(Price),
         diff_log_price = c(NA, diff(log_price))) %>%
  drop_na()

# Create lag features for modeling
create_lags <- function(data, lags = 1:7) {
  for (lag in lags) {
    data[[paste0("lag_", lag)]] <- dplyr::lag(data$log_price, lag)
  }
  return(data)
}

cocoa_data <- create_lags(cocoa_data) %>%
  drop_na() %>%
  mutate(across(where(is.numeric) & !where(lubridate::is.Date), ~ round(.x, 3)))

# ===========================
# EXPLORATORY DATA ANALYSIS
# ===========================

# 1. Summary Statistics
summary(cocoa_data %>% select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate))

# 2. Histogram and Density Plots for Each Variable
predictor_vars <- cocoa_data %>%
  select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value")

ggplot(predictor_vars, aes(x = Value)) +
  geom_histogram(fill = "darkgreen", bins = 30, alpha = 0.7) +
  facet_wrap(~ Variable, scales = "free", ncol = 3) +
  labs(title = "Distributions of Predictor Variables", x = "Value", y = "Count") +
  theme_minimal()

# 3. Correlation Heatmap
cor_matrix <- cor(cocoa_data %>% select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate))
corrplot(cor_matrix, method = "color", addCoef.col = "black", tl.cex = 1, number.cex = 0.8)

# 4. Cocoa Price Over Time
ggplot(cocoa_data, aes(x = Date, y = Price)) +
  geom_line(color = "blue", linewidth = 1) +
  labs(title = "Cocoa Price Over Time", x = "Date", y = "Cocoa Price (USD/tonne)") +
  theme_minimal()

# 5. Seasonal Decomposition of Cocoa Price
price_ts <- ts(cocoa_data$Price, start = c(year(min(cocoa_data$Date)), month(min(cocoa_data$Date))), frequency = 12)
decomp <- stl(price_ts, s.window = "periodic")
plot(decomp)

# ===========================
# CROSS-CORRELATION ANALYSIS
# ===========================
ts_PRCP <- ts(cocoa_data$PRCP)
ts_Price <- ts(cocoa_data$Price)
ccf(ts_Price, ts_PRCP, lag.max = 12, main = "CCF: Cocoa Price vs Rainfall")

# ===========================
# PERIODICITY ANALYSIS
# ===========================
mvspec(cocoa_data$Price, log = "no", main = "Periodogram of Cocoa Price")
mvspec(cocoa_data$Price, kernel("daniell", 3), log = "no", main = "Smoothed Periodogram")

# Additional weather lag feature
cocoa_data <- cocoa_data %>%
  mutate(PRCP_Lag1 = lag(PRCP, 5)) %>%
  mutate(PRCP_Lag1 = replace_na(PRCP_Lag1, 0))

# Create lagged dataset for modeling
cocoa_data_lagged <- create_lags(cocoa_data) %>%
  drop_na() %>%
  mutate(across(where(is.numeric) & !where(lubridate::is.Date), ~ round(.x, 3)))

# ===========================
# ADD DATE-BASED FEATURES FOR XGBOOST
# ===========================
cocoa_data_lagged <- cocoa_data_lagged %>%
  mutate(month = month(Date),
         day_of_year = yday(Date))

# ===========================
# MODEL FITTING & FORECASTING
# ===========================

# Split data into training and testing sets (80/20) using the non-lagged cocoa_data
train_size <- floor(0.8 * nrow(cocoa_data))
train_data <- cocoa_data[1:train_size, ]
test_data <- cocoa_data[(train_size + 1):nrow(cocoa_data), ]

# External regressors
external_vars <- c("PRCP", "TAVG", "TMAX", "TMIN", "ExchangeRate")
external_regressors_train <- train_data %>% select(all_of(external_vars))
external_regressors_test <- test_data %>% select(all_of(external_vars))

# Fit ETS model (no exogenous regressors)
ets_model <- ets(train_data$Price)

# Fit ARIMAX model (no seasonal component)
arimax_model <- auto.arima(train_data$Price, xreg = as.matrix(external_regressors_train), seasonal = FALSE)

# Fit SARIMAX model (with seasonal component)
sarimax_model <- auto.arima(train_data$Price, xreg = as.matrix(external_regressors_train), seasonal = TRUE)

# Forecast with each model
ets_forecast <- forecast(ets_model, h = nrow(test_data))
arimax_forecast <- forecast(arimax_model, xreg = as.matrix(external_regressors_test), h = nrow(test_data))
sarimax_forecast <- forecast(sarimax_model, xreg = as.matrix(external_regressors_test), h = nrow(test_data))

# ===========================
# IMPROVED XGBOOST WALK-FORWARD FORECAST
# ===========================
initial_size <- floor(0.8 * nrow(cocoa_data_lagged))
forecast_horizon <- nrow(cocoa_data_lagged) - initial_size

xgb_predictions <- c()
xgb_actuals <- c()
xgb_dates <- c()

for (i in 1:forecast_horizon) {
  xgb_train <- cocoa_data_lagged[1:(initial_size + i - 1), ]
  xgb_test  <- cocoa_data_lagged[(initial_size + i), ]
  
  # Use lag features plus external regressors and new date-based features
  x_train <- xgb_train %>% 
    select(starts_with("lag_"), PRCP, TAVG, TMAX, TMIN, ExchangeRate, month, day_of_year)
  y_train <- xgb_train$log_price
  x_test  <- xgb_test %>% 
    select(starts_with("lag_"), PRCP, TAVG, TMAX, TMIN, ExchangeRate, month, day_of_year)
  
  dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
  dtest  <- xgb.DMatrix(data = as.matrix(x_test))
  
  # Define improved hyperparameters
  params <- list(
    objective = "reg:squarederror",
    max_depth = 6,
    eta = 0.05,
    subsample = 0.9,
    colsample_bytree = 0.8
  )
  
  # Cross-validation to find the best number of rounds
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 200,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  best_nrounds <- cv$best_iteration
  
  # Train the model with best_nrounds
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 0
  )
  
  pred_log <- predict(model, dtest)
  pred_price <- exp(pred_log)
  
  xgb_predictions <- c(xgb_predictions, pred_price)
  xgb_actuals <- c(xgb_actuals, exp(xgb_test$log_price))
  xgb_dates <- c(xgb_dates, xgb_test$Date)
}

# ===========================
# COMBINE FORECASTS
# ===========================
# XGBoost forecast DF
xgb_df <- tibble(
  Date = as.Date(xgb_dates),
  XGBoost = xgb_predictions
)

# Align other models' forecasts
forecast_df <- tibble(
  Date = test_data$Date,
  Actual = test_data$Price,
  ETS = as.numeric(ets_forecast$mean),
  ARIMAX = as.numeric(arimax_forecast$mean),
  SARIMAX = as.numeric(sarimax_forecast$mean)
)

# Merge for plotting (long format)
plot_df <- forecast_df %>%
  left_join(xgb_df, by = "Date") %>%
  pivot_longer(cols = -Date, names_to = "Model", values_to = "Price")

# ===========================
# PLOT ALL MODEL FORECASTS
# ===========================
ggplot(plot_df, aes(x = Date, y = Price, color = Model)) +
  geom_line(linewidth = 0.5) +
  labs(
    title = "Forecast vs Actual Cocoa Prices (All Models)",
    x = "Date",
    y = "Cocoa Price (USD/tonne)",
    color = "Model"
  ) +
  scale_color_manual(values = c(
    "Actual" = "black",
    "ETS" = "red",
    "ARIMAX" = "green",
    "SARIMAX" = "purple",
    "XGBoost" = "orange"
  )) +
  theme_minimal()

# ===========================
# FORECAST ACCURACY METRICS
# ===========================
ets_accuracy <- forecast::accuracy(ets_forecast, test_data$Price)
arimax_accuracy <- forecast::accuracy(arimax_forecast, test_data$Price)
sarimax_accuracy <- forecast::accuracy(sarimax_forecast, test_data$Price)

# Compute XGBoost RMSE using merged forecasts
xgb_merged <- xgb_df %>%
  left_join(forecast_df %>% select(Date, Actual), by = "Date")
xgb_rmse <- sqrt(mean((xgb_merged$XGBoost - xgb_merged$Actual)^2, na.rm = TRUE))

# Print results
print("ETS Accuracy:")
print(ets_accuracy)

print("ARIMAX Accuracy:")
print(arimax_accuracy)

print("SARIMAX Accuracy:")
print(sarimax_accuracy)

print(paste("Improved XGBoost RMSE:", round(xgb_rmse, 2)))
