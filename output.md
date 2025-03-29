> # DATA PREPARATION
> # ===========================
> 
> # Load datasets
> cocoa_prices <- read.csv("data/Daily Prices_ICCO.csv")
> ghana_weather <- read.csv("data/Ghana_data.csv")
> exchange_rate <- read.csv("data/USD_GHS_1994_2024.csv") %>%
+   select(Date, ExchangeRate = Price)
> 
> # Format date columns
> cocoa_prices$Date <- as.Date(cocoa_prices$Date, format = "%d/%m/%Y")
> ghana_weather$DATE <- as.Date(ghana_weather$DATE)
> exchange_rate$Date <- as.Date(exchange_rate$Date)
> 
> # Clean and prepare cocoa price data
> cocoa_prices$Price <- as.numeric(gsub(",", "", cocoa_prices$ICCO.daily.price..US..tonne.))
> cocoa_prices <- cocoa_prices %>%
+   select(Date, Price) %>%
+   arrange(Date)
> 
> # Aggregate weather data by date
> ghana_weather <- ghana_weather %>%
+   group_by(DATE) %>%
+   summarise(across(c(PRCP, TAVG, TMAX, TMIN), mean, na.rm = TRUE)) %>%
+   rename(Date = DATE)
Warning message:
There was 1 warning in `summarise()`.
ℹ In argument: `across(c(PRCP, TAVG, TMAX, TMIN), mean, na.rm = TRUE)`.
ℹ In group 1: `DATE = 1990-01-01`.
Caused by warning:
! The `...` argument of `across()` is deprecated as of dplyr 1.1.0.
Supply arguments directly to `.fns` through an anonymous function instead.

  # Previously
  across(a:b, mean, na.rm = TRUE)

  # Now
  across(a:b, \(x) mean(x, na.rm = TRUE))
This warning is displayed once every 8 hours.
Call `lifecycle::last_lifecycle_warnings()` to see where this warning was generated. 
> 
> # Merge all datasets
> cocoa_data <- cocoa_prices %>%
+   left_join(ghana_weather, by = "Date") %>%
+   left_join(exchange_rate, by = "Date") %>%
+   drop_na()
> 
> # Feature engineering: log & differencing
> cocoa_data <- cocoa_data %>%
+   mutate(log_price = log(Price),
+          diff_log_price = c(NA, diff(log_price))) %>%
+   drop_na()
> 
> # Create lag features for modeling
> create_lags <- function(data, lags = 1:7) {
+   for (lag in lags) {
+     data[[paste0("lag_", lag)]] <- dplyr::lag(data$log_price, lag)
+   }
+   return(data)
+ }
> 
> cocoa_data <- create_lags(cocoa_data) %>%
+   drop_na() %>%
+   mutate(across(where(is.numeric) & !where(lubridate::is.Date), ~ round(.x, 3)))
> 
> # ===========================
> # EXPLORATORY DATA ANALYSIS
> # ===========================
> 
> # 1. Summary Statistics
> summary(cocoa_data %>% select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate))
     Price              PRCP              TAVG            TMAX             TMIN        ExchangeRate   
 Min.   :  778.4   Min.   : 0.0000   Min.   :73.60   Min.   : 76.50   Min.   :61.00   Min.   : 0.105  
 1st Qu.: 1692.7   1st Qu.: 0.0000   1st Qu.:78.56   1st Qu.: 85.50   1st Qu.:72.60   1st Qu.: 0.892  
 Median : 2333.5   Median : 0.0780   Median :80.50   Median : 88.67   Median :73.71   Median : 3.540  
 Mean   : 2592.9   Mean   : 0.2476   Mean   :80.62   Mean   : 88.44   Mean   :73.86   Mean   : 4.199  
 3rd Qu.: 2933.2   3rd Qu.: 0.3010   3rd Qu.:82.60   3rd Qu.: 91.27   3rd Qu.:75.00   3rd Qu.: 5.740  
 Max.   :10690.7   Max.   :10.2800   Max.   :88.00   Max.   :101.00   Max.   :82.00   Max.   :16.300  
> 
> # 2. Histogram and Density Plots for Each Variable
> predictor_vars <- cocoa_data %>%
+   select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate) %>%
+   pivot_longer(everything(), names_to = "Variable", values_to = "Value")
> 
> ggplot(predictor_vars, aes(x = Value)) +
+   geom_histogram(fill = "darkgreen", bins = 30, alpha = 0.7) +
+   facet_wrap(~ Variable, scales = "free", ncol = 3) +
+   labs(title = "Distributions of Predictor Variables", x = "Value", y = "Count") +
+   theme_minimal()
> 
> # 3. Correlation Heatmap
> cor_matrix <- cor(cocoa_data %>% select(Price, PRCP, TAVG, TMAX, TMIN, ExchangeRate))
> corrplot(cor_matrix, method = "color", addCoef.col = "black", tl.cex = 1, number.cex = 0.8)
> 
> # 4. Cocoa Price Over Time
> ggplot(cocoa_data, aes(x = Date, y = Price)) +
+   geom_line(color = "blue", linewidth = 1) +
+   labs(title = "Cocoa Price Over Time", x = "Date", y = "Cocoa Price (USD/tonne)") +
+   theme_minimal()
> 
> # 5. Seasonal Decomposition of Cocoa Price
> price_ts <- ts(cocoa_data$Price, start = c(year(min(cocoa_data$Date)), month(min(cocoa_data$Date))), frequency = 12)
> decomp <- stl(price_ts, s.window = "periodic")
> plot(decomp)
> 
> # ===========================
> # CROSS-CORRELATION ANALYSIS
> # ===========================
> ts_PRCP <- ts(cocoa_data$PRCP)
> ts_Price <- ts(cocoa_data$Price)
> ccf(ts_Price, ts_PRCP, lag.max = 12, main = "CCF: Cocoa Price vs Rainfall")
> 
> # ===========================
> # PERIODICITY ANALYSIS
> # ===========================
> mvspec(cocoa_data$Price, log = "no", main = "Periodogram of Cocoa Price")
> mvspec(cocoa_data$Price, kernel("daniell", 3), log = "no", main = "Smoothed Periodogram")
Bandwidth: 0.002 | Degrees of Freedom: 13.94 | split taper: 0% 
> 
> # Additional weather lag feature
> cocoa_data <- cocoa_data %>%
+   mutate(PRCP_Lag1 = lag(PRCP, 5)) %>%
+   mutate(PRCP_Lag1 = replace_na(PRCP_Lag1, 0))
> 
> # Create lagged dataset for modeling
> cocoa_data_lagged <- create_lags(cocoa_data) %>%
+   drop_na() %>%
+   mutate(across(where(is.numeric) & !where(lubridate::is.Date), ~ round(.x, 3)))
> 
> # ===========================
> # ADD DATE-BASED FEATURES FOR XGBOOST
> # ===========================
> cocoa_data_lagged <- cocoa_data_lagged %>%
+   mutate(month = month(Date),
+          day_of_year = yday(Date))
> 
> # ===========================
> # MODEL FITTING & FORECASTING
> # ===========================
> 
> # Split data into training and testing sets (80/20) using the non-lagged cocoa_data
> train_size <- floor(0.8 * nrow(cocoa_data))
> train_data <- cocoa_data[1:train_size, ]
> test_data <- cocoa_data[(train_size + 1):nrow(cocoa_data), ]
> 
> # External regressors
> external_vars <- c("PRCP", "TAVG", "TMAX", "TMIN", "ExchangeRate")
> external_regressors_train <- train_data %>% select(all_of(external_vars))
> external_regressors_test <- test_data %>% select(all_of(external_vars))
> 
> # Fit ETS model (no exogenous regressors)
> ets_model <- ets(train_data$Price)
> 
> # Fit ARIMAX model (no seasonal component)
> arimax_model <- auto.arima(train_data$Price, xreg = as.matrix(external_regressors_train), seasonal = FALSE)
> 
> # Fit SARIMAX model (with seasonal component)
> sarimax_model <- auto.arima(train_data$Price, xreg = as.matrix(external_regressors_train), seasonal = TRUE)
> 
> # Forecast with each model
> ets_forecast <- forecast(ets_model, h = nrow(test_data))
> arimax_forecast <- forecast(arimax_model, xreg = as.matrix(external_regressors_test), h = nrow(test_data))
> sarimax_forecast <- forecast(sarimax_model, xreg = as.matrix(external_regressors_test), h = nrow(test_data))
> 
> # ===========================
> # XGBOOST WALK-FORWARD FORECAST
> # ===========================
> initial_size <- floor(0.8 * nrow(cocoa_data_lagged))
> forecast_horizon <- nrow(cocoa_data_lagged) - initial_size
> 
> xgb_predictions <- c()
> xgb_actuals <- c()
> xgb_dates <- c()
> 
> for (i in 1:forecast_horizon) {
+   xgb_train <- cocoa_data_lagged[1:(initial_size + i - 1), ]
+   xgb_test  <- cocoa_data_lagged[(initial_size + i), ]
+   
+   # Use lag features plus external regressors and new date-based features
+   x_train <- xgb_train %>% 
+     select(starts_with("lag_"), PRCP, TAVG, TMAX, TMIN, ExchangeRate, month, day_of_year)
+   y_train <- xgb_train$log_price
+   x_test  <- xgb_test %>% 
+     select(starts_with("lag_"), PRCP, TAVG, TMAX, TMIN, ExchangeRate, month, day_of_year)
+   
+   dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
+   dtest  <- xgb.DMatrix(data = as.matrix(x_test))
+   
+   # Define improved hyperparameters
+   params <- list(
+     objective = "reg:squarederror",
+     max_depth = 6,
+     eta = 0.05,
+     subsample = 0.9,
+     colsample_bytree = 0.8
+   )
+   
+   # Cross-validation to find the best number of rounds
+   cv <- xgb.cv(
+     params = params,
+     data = dtrain,
+     nrounds = 200,
+     nfold = 5,
+     early_stopping_rounds = 10,
+     verbose = 0
+   )
+   
+   best_nrounds <- cv$best_iteration
+   
+   # Train the model with best_nrounds
+   model <- xgb.train(
+     params = params,
+     data = dtrain,
+     nrounds = best_nrounds,
+     verbose = 0
+   )
+   
+   pred_log <- predict(model, dtest)
+   pred_price <- exp(pred_log)
+   
+   xgb_predictions <- c(xgb_predictions, pred_price)
+   xgb_actuals <- c(xgb_actuals, exp(xgb_test$log_price))
+   xgb_dates <- c(xgb_dates, xgb_test$Date)
+ }
> 
> # ===========================
> # COMBINE FORECASTS
> # ===========================
> # XGBoost forecast DF
> xgb_df <- tibble(
+   Date = as.Date(xgb_dates),
+   XGBoost = xgb_predictions
+ )
> 
> # Align other models' forecasts
> forecast_df <- tibble(
+   Date = test_data$Date,
+   Actual = test_data$Price,
+   ETS = as.numeric(ets_forecast$mean),
+   ARIMAX = as.numeric(arimax_forecast$mean),
+   SARIMAX = as.numeric(sarimax_forecast$mean)
+ )
> 
> # Merge for plotting (long format)
> plot_df <- forecast_df %>%
+   left_join(xgb_df, by = "Date") %>%
+   pivot_longer(cols = -Date, names_to = "Model", values_to = "Price")
Warning message:
In left_join(., xgb_df, by = "Date") :
  Detected an unexpected many-to-many relationship between `x` and `y`.
ℹ Row 405 of `x` matches multiple rows in `y`.
ℹ Row 404 of `y` matches multiple rows in `x`.
ℹ If a many-to-many relationship is expected, set `relationship = "many-to-many"` to silence this
  warning.
> 
> # ===========================
> # PLOT ALL MODEL FORECASTS
> # ===========================
> ggplot(plot_df, aes(x = Date, y = Price, color = Model)) +
+   geom_line(linewidth = 0.5) +
+   labs(
+     title = "Forecast vs Actual Cocoa Prices (All Models)",
+     x = "Date",
+     y = "Cocoa Price (USD/tonne)",
+     color = "Model"
+   ) +
+   scale_color_manual(values = c(
+     "Actual" = "black",
+     "ETS" = "red",
+     "ARIMAX" = "green",
+     "SARIMAX" = "purple",
+     "XGBoost" = "orange"
+   )) +
+   theme_minimal()
Warning message:
Removed 1 row containing missing values or values outside the scale range (`geom_line()`). 
> 
> # ===========================
> # FORECAST ACCURACY METRICS
> # ===========================
> ets_accuracy <- forecast::accuracy(ets_forecast, test_data$Price)
> arimax_accuracy <- forecast::accuracy(arimax_forecast, test_data$Price)
> sarimax_accuracy <- forecast::accuracy(sarimax_forecast, test_data$Price)
> 
> # Compute XGBoost RMSE using merged forecasts
> xgb_merged <- xgb_df %>%
+   left_join(forecast_df %>% select(Date, Actual), by = "Date")
Warning message:
In left_join(., forecast_df %>% select(Date, Actual), by = "Date") :
  Detected an unexpected many-to-many relationship between `x` and `y`.
ℹ Row 404 of `x` matches multiple rows in `y`.
ℹ Row 405 of `y` matches multiple rows in `x`.
ℹ If a many-to-many relationship is expected, set `relationship = "many-to-many"` to silence this
  warning.
> xgb_rmse <- sqrt(mean((xgb_merged$XGBoost - xgb_merged$Actual)^2, na.rm = TRUE))
> 
> # Print results
> print("ETS Accuracy:")
[1] "ETS Accuracy:"
> print(ets_accuracy)
                       ME       RMSE        MAE         MPE      MAPE       MASE       ACF1
Training set    0.4480274   60.90867   36.40954 -0.01791881  1.673149  0.9999366 0.06622727
Test set     1665.5994361 2793.78832 1770.74573 24.73666088 29.320761 48.6310311         NA
> 
> print("ARIMAX Accuracy:")
[1] "ARIMAX Accuracy:"
> print(arimax_accuracy)
                       ME       RMSE        MAE         MPE      MAPE       MASE          ACF1
Training set    0.3802087   60.33331   36.36043 -0.02428569  1.679339  0.9985878 -0.0005149814
Test set     1496.7187418 2634.39010 1661.60840 20.66101027 27.793632 45.6337285            NA
> 
> print("SARIMAX Accuracy:")
[1] "SARIMAX Accuracy:"
> print(sarimax_accuracy)
                       ME       RMSE        MAE         MPE      MAPE       MASE          ACF1
Training set    0.3802087   60.33331   36.36043 -0.02428569  1.679339  0.9985878 -0.0005149814
Test set     1496.7187418 2634.39010 1661.60840 20.66101027 27.793632 45.6337285            NA
> 
> print(paste("XGBoost RMSE:", round(xgb_rmse, 2)))
[1] "XGBoost RMSE: 268.32"