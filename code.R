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
cocoa_prices <- read.csv("data/raw_data/Daily Prices_ICCO.csv")
ghana_weather <- read.csv("data/raw_data/Ghana_data.csv")
exchange_rate <- read.csv("data/raw_data/USD_GHS_1994_2024.csv") %>%
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
  mutate(log_price = log(Price)) %>%
  drop_na()

write.csv(cocoa_data, "data/clean_data/cocoa_data.csv", row.names = FALSE)

# Create lag features for modeling
create_lags <- function(data, lags = 1:7) {
  for (lag in lags) {
    data[[paste0("lag_", lag)]] <- dplyr::lag(data$log_price, lag)
  }
  return(data)
}

cocoa_data_lagged <- create_lags(cocoa_data) %>%
  drop_na() %>%
  mutate(across(where(is.numeric) & !where(lubridate::is.Date), ~ round(.x, 3)))

write.csv(cocoa_data_lagged, "data/clean_data/cocoa_data_lagged.csv", row.names = FALSE)

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
# ACF PLOT 
# ===========================
acf(cocoa_data$Price, main = "ACF of Cocoa Prices")


# ===========================
# PERIODICITY ANALYSIS
# ===========================
mvspec(cocoa_data$Price, log = "no", main = "Periodogram of Cocoa Price")
mvspec(cocoa_data$Price, kernel("daniell", 3), log = "no", main = "Smoothed Periodogram")

# ===========================
# MODEL FITTING & FORECASTING
# ===========================

# 1. Split data into training and testing sets (80/20)
train_size <- floor(0.8 * nrow(cocoa_data))
train_data <- cocoa_data[1:train_size, ]
test_data <- cocoa_data[(train_size + 1):nrow(cocoa_data), ]

# 2. External regressors
external_vars <- c("PRCP", "TAVG", "TMAX", "TMIN", "ExchangeRate")
external_regressors_train <- train_data %>% select(all_of(external_vars))
external_regressors_test <- test_data %>% select(all_of(external_vars))

# 3. Fit ETS model (no exogenous regressors)
ets_model <- ets(train_data$Price)

# 4. Fit ARIMAX model (no seasonal component)
arimax_model <- auto.arima(train_data$Price, xreg = as.matrix(external_regressors_train), seasonal = FALSE)

# 5. Fit SARIMAX model (with seasonal component)
sarimax_model <- auto.arima(train_data$Price, xreg = as.matrix(external_regressors_train), seasonal = TRUE)

# 6. Forecast with each model
ets_forecast <- forecast(ets_model, h = nrow(test_data))
arimax_forecast <- forecast(arimax_model, xreg = as.matrix(external_regressors_test), h = nrow(test_data))
sarimax_forecast <- forecast(sarimax_model, xreg = as.matrix(external_regressors_test), h = nrow(test_data))

# ===========================
# XGBOOST Walk-Forward Forecast
# ===========================

initial_size <- floor(0.8 * nrow(cocoa_data_lagged))
forecast_horizon <- nrow(cocoa_data_lagged) - initial_size

xgb_predictions <- c()
xgb_actuals <- c()
xgb_dates <- c()

for (i in 1:forecast_horizon) {
  xgb_train <- cocoa_data_lagged[1:(initial_size + i - 1), ]
  xgb_test <- cocoa_data_lagged[(initial_size + i), ]
  
  x_train <- xgb_train %>% select(starts_with("lag_"), PRCP, TAVG, TMAX, TMIN, ExchangeRate)
  y_train <- xgb_train$log_price
  x_test <- xgb_test %>% select(starts_with("lag_"), PRCP, TAVG, TMAX, TMIN, ExchangeRate)
  
  dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
  dtest <- xgb.DMatrix(data = as.matrix(x_test))
  
  model <- xgboost(data = dtrain, nrounds = 100, objective = "reg:squarederror", verbose = 0)
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
  Date = as.Date(xgb_dates, origin = "1970-01-01"),
  XGBoost = xgb_predictions
)
saveRDS(xgb_df, "data/clean_data/xgb_df.rds")


# Align other models' forecasts
forecast_df <- tibble(
  Date = test_data$Date,
  Actual = test_data$Price,
  ETS = as.numeric(ets_forecast$mean),
  ARIMAX = as.numeric(arimax_forecast$mean),
  SARIMAX = as.numeric(sarimax_forecast$mean)
)

# Merge all into a long format for plotting
plot_df <- forecast_df %>%
  left_join(xgb_df, by = "Date", relationship = "many-to-many") %>%
  pivot_longer(cols = -Date, names_to = "Model", values_to = "Price")

saveRDS(plot_df, "data/clean_data/plot_df.rds")


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

print(paste("XGBoost RMSE:", round(xgb_rmse, 2)))


# ===========================
# RESIDUAL DIAGNOSTICS FOR XGBOOST
# ===========================

# Residuals
xgb_walk_df <- tibble(
  Date = as.Date(xgb_dates, origin = "1970-01-01"),
  Actual = xgb_actuals,
  Predicted = xgb_predictions
)

residuals_xgb <- xgb_walk_df$Actual - xgb_walk_df$Predicted



# 1. Plot residuals over time
ggplot(data.frame(Date = xgb_walk_df$Date, Residuals = residuals_xgb), 
       aes(x = Date, y = Residuals)) +
  geom_line(color = "darkred") +
  labs(title = "XGBoost Residuals Over Time", y = "Residual", x = "Date") +
  theme_minimal()

# 2. Check for autocorrelation
acf(residuals_xgb, main = "ACF of XGBoost Residuals")

# 3. Ljung-Box Test for autocorrelation
ljung_xgboox <- Box.test(residuals_xgb, lag = 12, type = "Ljung-Box")

# 4. Optional: Histogram + Q-Q Plot for normality
hist(residuals_xgb, breaks = 30, main = "Histogram of XGBoost Residuals", col = "gray")

qqnorm(residuals_xgb)
qqline(residuals_xgb, col = "blue")


# =====================================================
# ADDITIONAL: RESIDUAL ANALYSIS FOR ETS, ARIMAX, SARIMAX
# =====================================================

# Compute residuals
residuals_ets <- ets_forecast$mean - test_data$log_price
residuals_arimax <- arimax_forecast$mean - test_data$log_price
residuals_sarimax <- sarimax_forecast$mean - test_data$log_price

# Combine with dates
residuals_df <- tibble(
  Date = test_data$Date,
  ETS = residuals_ets,
  ARIMAX = residuals_arimax,
  SARIMAX = residuals_sarimax
)

# ===========================
# 1. Residual Plots Over Time
# ===========================
residuals_long <- residuals_df %>%
  pivot_longer(cols = -Date, names_to = "Model", values_to = "Residual")

ggplot(residuals_long, aes(x = Date, y = Residual, color = Model)) +
  geom_line(linewidth = 0.4) +
  labs(title = "Residuals Over Time (ETS, ARIMAX, SARIMAX)", x = "Date", y = "Residual") +
  theme_minimal()



# ===========================
# 2. ACF of Residuals
# ===========================
par(mfrow = c(1, 3))
acf(residuals_ets, main = "ACF: ETS Residuals")
acf(residuals_arimax, main = "ACF: ARIMAX Residuals")
acf(residuals_sarimax, main = "ACF: SARIMAX Residuals")
par(mfrow = c(1, 1))

# ===========================
# 3. Ljung-Box Test
# ===========================
ljung_ets <- Box.test(residuals_ets, lag = 12, type = "Ljung-Box")
ljung_arimax <- Box.test(residuals_arimax, lag = 12, type = "Ljung-Box")
ljung_sarimax <- Box.test(residuals_sarimax, lag = 12, type = "Ljung-Box")

print("Ljung-Box Test (ETS):")
print(ljung_ets)

print("Ljung-Box Test (ARIMAX):")
print(ljung_arimax)

print("Ljung-Box Test (SARIMAX):")
print(ljung_sarimax)

print(ljung_xgboox)

# ===========================
# 4. Histograms + QQ Plots
# ===========================
par(mfrow = c(3, 2))

# ETS
hist(residuals_ets, breaks = 30, main = "Histogram: ETS", col = "gray")
qqnorm(residuals_ets); qqline(residuals_ets, col = "blue")

# ARIMAX
hist(residuals_arimax, breaks = 30, main = "Histogram: ARIMAX", col = "gray")
qqnorm(residuals_arimax); qqline(residuals_arimax, col = "blue")

# SARIMAX
hist(residuals_sarimax, breaks = 30, main = "Histogram: SARIMAX", col = "gray")
qqnorm(residuals_sarimax); qqline(residuals_sarimax, col = "blue")







