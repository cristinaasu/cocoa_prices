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
  select(Date, ExchangeRate = Price) %>%
  mutate(
    Date = as.Date(Date),
    DailyChange = ExchangeRate - lag(ExchangeRate)
  )

# Format date columns
cocoa_prices$Date <- as.Date(cocoa_prices$Date, format = "%d/%m/%Y")
ghana_weather$DATE <- as.Date(ghana_weather$DATE)

# Clean price data
cocoa_prices$Price <- as.numeric(gsub(",", "", cocoa_prices$ICCO.daily.price..US..tonne.))
cocoa_prices <- cocoa_prices %>% select(Date, Price) %>% arrange(Date)

# Aggregate weather data (daily mean, rounded)
ghana_weather <- ghana_weather %>%
  group_by(DATE) %>%
  summarise(across(c(PRCP, TAVG, TMAX, TMIN), ~ round(mean(.x, na.rm = TRUE), 0))) %>%
  ungroup()

# Merge datasets and create lagged rainfall variable
cocoa_data <- left_join(cocoa_prices, ghana_weather, by = c("Date" = "DATE")) %>%
  arrange(Date) %>%
  left_join(exchange_rate, by = "Date") %>%
  drop_na()

# ===========================
# EXPLORATORY DATA ANALYSIS
# ===========================

# 1. Summary Statistics
summary(cocoa_data %>% select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate, DailyChange))

# 2. Histogram and Density Plots for Each Variable
predictor_vars <- cocoa_data %>%
  select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate, DailyChange) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value")

ggplot(predictor_vars, aes(x = Value)) +
  geom_histogram(fill = "darkgreen", bins = 30, alpha = 0.7) +
  facet_wrap(~ Variable, scales = "free", ncol = 3) +
  labs(title = "Distributions of Predictor Variables", x = "Value", y = "Count") +
  theme_minimal()

# 3. Correlation Heatmap
cor_matrix <- cor(cocoa_data %>% select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate, DailyChange))
corrplot(cor_matrix, method = "color", addCoef.col = "black", tl.cex = 1, number.cex = 0.8)

# 4. Cocoa Price Over Time
ggplot(cocoa_data, aes(x = Date, y = Price)) +
  geom_line(color = "blue", linewidth = 1) +
  labs(title = "Cocoa Price Over Time", x = "Date", y = "Cocoa Price (USD/tonne)") +
  theme_minimal()

# 5. Seasonal Decomposition of Cocoa Price
price_ts <- ts(cocoa_data$Price, start = c(year(min(cocoa_data$Date)), month(min(cocoa_data$Date))), frequency = 365)
decomp <- stl(price_ts, s.window = "periodic")
plot(decomp)

# ===========================
# MODEL FITTING & FORECASTING
# ===========================

# 1. Split data into training and testing sets (80/20)
train_size <- floor(0.8 * nrow(cocoa_data))
train_data <- cocoa_data[1:train_size, ]
test_data <- cocoa_data[(train_size + 1):nrow(cocoa_data), ]

# 2. External regressors
external_vars <- c("PRCP", "TAVG", "TMAX", "TMIN", "ExchangeRate", "DailyChange")
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
  labs(title = "Cocoa Price Forecast vs Actual Prices",
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








