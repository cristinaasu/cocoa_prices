# Required packages
library(tidyverse)
library(lubridate)
library(forecast)
library(tseries)

# Read dataset
cocoa_prices <- read.csv("data/Daily Prices_ICCO.csv")
ghana_weather <- read.csv("data/Ghana_data.csv")

# Turn variable date into date format
cocoa_prices$Date <- as.Date(cocoa_prices$Date, format='%d/%m/%Y')
ghana_weather$DATE <- as.Date(ghana_weather$DATE)

# Remove commas and convert to numeric)
cocoa_prices$Price <- as.numeric(gsub(",", "", cocoa_prices$ICCO.daily.price..US..tonne.))

# Select and arrange data
cocoa_prices <- cocoa_prices %>% select(Date, Price) %>% arrange(Date)

# Aggregate weather data by date (mean values for daily observations)
ghana_weather <- ghana_weather %>%
  group_by(DATE) %>%
  summarise(across(c(PRCP, TAVG, TMAX, TMIN), ~ round(mean(.x, na.rm = TRUE), 0)))

# Merge cocoa prices with weather data
cocoa_data <- left_join(cocoa_prices, ghana_weather, by = c("Date" = "DATE"))
cocoa_data <- na.omit(cocoa_data) 