# ===========================
# REQUIRED PACKAGES
# ===========================
library(tidyverse)
library(lubridate)
library(forecast)
library(tseries)
library(corrplot)
library(gridExtra)

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

# Clean price data
cocoa_prices$Price <- as.numeric(gsub(",", "", cocoa_prices$ICCO.daily.price..US..tonne.))
cocoa_prices <- cocoa_prices %>% select(Date, Price) %>% arrange(Date)

# Aggregate cocoa price data to monthly mean
cocoa_monthly <- cocoa_prices %>%
  mutate(Date = floor_date(Date, "month")) %>%
  group_by(Date) %>%
  summarise(Price = mean(Price, na.rm = TRUE)) %>%
  ungroup()

# Aggregate weather data to monthly mean (set PRCP NA to 0)
ghana_weather$PRCP[is.na(ghana_weather$PRCP)] <- 0
weather_monthly <- ghana_weather %>%
  mutate(Date = floor_date(DATE, "month")) %>%
  group_by(Date) %>%
  summarise(
    PRCP = mean(PRCP, na.rm = TRUE),
    TAVG = mean(TAVG, na.rm = TRUE),
    TMAX = mean(TMAX, na.rm = TRUE),
    TMIN = mean(TMIN, na.rm = TRUE)
  ) %>%
  ungroup()

# Aggregate exchange rate data to monthly mean & compute monthly change
exchange_monthly <- exchange_rate %>%
  mutate(Date = floor_date(Date, "month")) %>%
  group_by(Date) %>%
  summarise(ExchangeRate = mean(ExchangeRate, na.rm = TRUE)) %>%
  mutate(MonthlyChange = ExchangeRate - lag(ExchangeRate)) %>%
  mutate(MonthlyChange = replace_na(MonthlyChange, 0)) %>%
  ungroup()

# Merge datasets and round all numeric columns (except Date)
cocoa_data <- cocoa_monthly %>%
  left_join(weather_monthly, by = "Date") %>%
  left_join(exchange_monthly, by = "Date") %>%
  mutate(across(where(is.numeric) & !where(lubridate::is.Date), ~ round(.x, 3))) %>%
  drop_na()

# ===========================
# EXPLORATORY DATA ANALYSIS
# ===========================

# 1. Summary Statistics
summary(cocoa_data %>% select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate, MonthlyChange))

# 2. Histogram and Density Plots for Each Variable
predictor_vars <- cocoa_data %>%
  select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate, MonthlyChange) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value")

ggplot(predictor_vars, aes(x = Value)) +
  geom_histogram(fill = "darkgreen", bins = 30, alpha = 0.7) +
  facet_wrap(~ Variable, scales = "free", ncol = 3) +
  labs(title = "Distributions of Predictor Variables", x = "Value", y = "Count") +
  theme_minimal()

# 3. Correlation Heatmap
cor_matrix <- cor(cocoa_data %>% select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate, MonthlyChange))
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

# Visual check of lagged relationships
ts_PRCP <- ts(cocoa_data$PRCP)
ts_Price <- ts(cocoa_data$Price)
ccf(ts_Price, ts_PRCP, lag.max = 12, main = "CCF: Cocoa Price vs Rainfall")

# ===========================
# PERIODICITY ANALYSIS
# ===========================

# Raw periodogram to detect cycles
mvspec(cocoa_data$Price, log = "no", main = "Periodogram of Cocoa Price")

# Smoothed periodogram using Daniell kernel
mvspec(cocoa_data$Price, kernel("daniell", 3), log = "no", main = "Smoothed Periodogram")

cocoa_data <- cocoa_data %>%
  mutate(PRCP_Lag1 = lag(PRCP, 5)) %>%
  mutate(PRCP_Lag1 = replace_na(PRCP_Lag1, 0))


# ===========================
# MODEL FITTING & FORECASTING
# ===========================

# 1. Split data into training and testing sets (80/20)
train_size <- floor(0.8 * nrow(cocoa_data))
train_data <- cocoa_data[1:train_size, ]
test_data <- cocoa_data[(train_size + 1):nrow(cocoa_data), ]

# 2. External regressors
external_vars <- c("PRCP_Lag1", "TAVG", "TMAX", "TMIN", "ExchangeRate", "MonthlyChange")
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

# 7. Create a dataframe for plotting forecast results
forecast_df <- data.frame(
  Date = test_data$Date,
  Actual = test_data$Price,
  ETS_Forecast = as.numeric(ets_forecast$mean),
  ARIMAX_Forecast = as.numeric(arimax_forecast$mean),
  SARIMAX_Forecast = as.numeric(sarimax_forecast$mean)
)

# 8. Plot forecast vs actual prices
ggplot(forecast_df, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual Price"), linewidth = 1) +
  geom_line(aes(y = ETS_Forecast, color = "ETS Forecast"), linetype = "dashed", linewidth = 1) +
  geom_line(aes(y = ARIMAX_Forecast, color = "ARIMAX Forecast"), linetype = "dotted", linewidth = 1) +
  geom_line(aes(y = SARIMAX_Forecast, color = "SARIMAX Forecast"), linetype = "dotdash", linewidth = 1) +
  labs(title = "Monthly Cocoa Price Forecast vs Actual Prices",
       x = "Date",
       y = "Cocoa Price (USD/tonne)",
       color = "Legend") +
  scale_color_manual(values = c("Actual Price" = "blue", 
                                "ETS Forecast" = "red", 
                                "ARIMAX Forecast" = "green", 
                                "SARIMAX Forecast" = "purple")) +
  theme_minimal()

# 9. Forecast accuracy
ets_accuracy <- accuracy(ets_forecast, test_data$Price)
arimax_accuracy <- accuracy(arimax_forecast, test_data$Price)
sarimax_accuracy <- accuracy(sarimax_forecast, test_data$Price)

# 10. Print accuracy metrics
ets_accuracy
arimax_accuracy
sarimax_accuracy



