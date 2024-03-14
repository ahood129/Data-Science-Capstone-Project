# Load required packages---------------------------------------------------------
# Define packages to be used, install them if not already installed, and open them with library
packages <- c("caret", "randomForest", "dplyr", "ggplot2", "knitr", "openxlsx", "rmarkdown")

for (package in packages) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package)
    library(package, character.only = TRUE)
  }
}

# Load the data--------------------------------------------------------
# Batting dataset
# https://www.kaggle.com/datasets/open-source-sports/baseball-databank?select=Batting.csv

# Download the data
dl <- tempfile()
download.file("https://www.kaggle.com/datasets/open-source-sports/baseball-databank?select=Batting.csv", dl)

# Read the downloaded CSV file into a dataframe
data <- read.csv("baseball/Batting.csv")

# Select desired predictors
batting <- data %>% 
  select(HR, playerID, yearID, G, AB, BB, SO) %>% 
  mutate(playerID = as.character(playerID))

# Data analysis------------------------------------------------------------------

# Years 
year_avg <- batting %>%
  group_by(yearID) %>%
  summarize(avg_hr = mean(HR)) %>%
  ungroup()

y_model <- lm(avg_hr ~ yearID, data = year_avg)

rsquared <- summary(y_model)$r.squared

ggplot(year_avg, aes(x = yearID, y = avg_hr)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Average HR per Year", x = "Year", y = "Average HR") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  annotate("text", x = max(year_avg$yearID), y = min(year_avg$avg_hr), 
           label = paste("R-squared = ", round(rsquared, digits = 3)), hjust = 1, vjust = -1)

# Games
game_avg <- batting %>%
  group_by(G) %>%
  summarize(avg_hr = mean(HR)) %>%
  ungroup()

g_model <- lm(avg_hr ~ G, data = game_avg)

rsquared <- summary(g_model)$r.squared

ggplot(game_avg, aes(x = G, y = avg_hr)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Average HR by Games Played", x = "Games Played", y = "Average HR") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  annotate("text", x = max(game_avg$G), y = min(game_avg$avg_hr), 
           label = paste("R-squared = ", round(rsquared, digits = 3)), hjust = 1, vjust = -1)

# At-bats
ab_avg <- batting %>%
  group_by(AB) %>%
  summarize(avg_hr = mean(HR)) %>%
  ungroup()

ab_model <- lm(avg_hr ~ AB, data = ab_avg)

rsquared <- summary(ab_model)$r.squared

ggplot(ab_avg, aes(x = AB, y = avg_hr)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Average HR by Number of At-bats", x = "At-Bats", y = "Average HR") +
  theme_minimal() +
  annotate("text", x = max(ab_avg$AB), y = min(ab_avg$avg_hr), 
           label = paste("R-squared = ", round(rsquared, digits = 3)), hjust = 1, vjust = -1)

# Walks
bb_avg <- batting %>%
  group_by(BB) %>%
  summarize(avg_hr = mean(HR)) %>%
  ungroup()

bb_model <- lm(avg_hr ~ BB, data = bb_avg)

rsquared <- summary(bb_model)$r.squared

ggplot(bb_avg, aes(x = BB, y = avg_hr)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Average HR vs Number of Walks", x = "Walks", y = "Average HR") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  annotate("text", x = max(bb_avg$BB), y = min(bb_avg$avg_hr), 
           label = paste("R-squared = ", round(rsquared, digits = 3)), hjust = 1, vjust = -1)

# Strikeouts
so_avg <- batting %>% 
  group_by(SO) %>% 
  mutate(avg_hr = mean(HR)) %>% 
  select(SO, avg_hr) %>% 
  distinct(SO, .keep_all = TRUE)

so_model <- lm(avg_hr ~ SO, data = so_avg)

rsquared <- summary(so_model)$r.squared

ggplot(so_avg, aes(x = SO, y = avg_hr)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Average HR vs Number of Strikeouts", x = "Strikeouts", y = "Average HR") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  annotate("text", x = max(so_avg$SO), y = min(so_avg$avg_hr), 
           label = paste("R-squared = ", round(rsquared, digits = 3)), hjust = 1, vjust = -1)

# Methods------------------------------------------------------------------------
##Split the Data------------

# Find complete cases and replace non-completes with column medians
sum(!complete.cases(batting))

for(i in 1:ncol(batting)) {
  batting[ , i][is.na(batting[ , i])] <- median(batting[ , i], na.rm=TRUE)
}

# Split the data into train and test sets
set.seed(1)
train_index <- createDataPartition(batting$HR, p = 0.8, list = FALSE)
train <- batting[train_index, ]
test <- batting[-train_index, ]

# Define the predictors and target variable to predict
predictors <- c("yearID", "G", "AB", "BB", "SO")
target_variable <- "HR"

##Random Forest Model---------------------------

# Run the random forest
rf_model <- randomForest(HR ~ ., data = train, ntree = 100)

# Make predictions on the test set
predictions_rf <- predict(rf_model, newdata = test)

# Calculate evaluation metrics
eval_metrics_rf <- postResample(predictions_rf, test$HR)

# Print evaluation metrics
cat("Mean Squared Error:", eval_metrics_rf[["RMSE"]]^2, "\n")
cat("Root Mean Squared Error:", eval_metrics_rf[["RMSE"]], "\n")
cat("R-squared:", eval_metrics_rf[["Rsquared"]], "\n")

##Linear Regression-------------------------

# Run the Linear Regression
lr_model <- lm(HR ~ yearID + G + AB + BB + SO, data = train)

# Make the predictions
predictions_lr <- predict(lr_model, test)

# Calculate MSE
mse_lr <- mean((test$HR - predictions_lr)^2)

# Calculate RMSE
rmse_lr <- sqrt(mse_lr)

# Calculate R-squared
rsquared_lr <- summary(lr_model)$r.squared

# Display results
cat("MSE:", mse_lr, "\n")
cat("RMSE:", rmse_lr, "\n")
cat("R-squared:", rsquared_lr, "\n")

eval_metrics_lr <- data.frame(Method = "LR", MSE = mse_lr, RMSE = rmse_lr, R2 = rsquared_lr)

# Results------------------------------------------------------------------------
##Random Forest---------
results <- data.frame(Method = "RF", MSE = eval_metrics_rf[["RMSE"]]^2, RMSE = eval_metrics_rf[["RMSE"]], R2 = eval_metrics_rf[["Rsquared"]])

# Visualize the RMSE 
# combine actual and predicted values
rf_df <- data.frame(Actual = test$HR, Predicted = predictions_rf)

# Calculate RMSE
rmse_rf <- sqrt(mean((rf_df$Predicted - rf_df$Actual)^2))

# Plot the deviation
ggplot(rf_df, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "blue", linetype = "dashed") +
  labs(x = "Actual HR", y = "Predicted HR", title = "Actual vs Predicted HR") +
  geom_smooth(method = "lm", se = FALSE) +
  annotate("text", x = max(rf_df$Actual), y = min(rf_df$Predicted), 
           label = paste("RMSE =", round(rmse_rf, 2)), hjust = 1, vjust = -1)

##Linear Regression----------
# Visualize the RMSE 
# combine actual and predicted values
lr_df <- data.frame(Actual = test$HR, Predicted = predictions_lr)

# Plot the deviation
ggplot(lr_df, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "blue", linetype = "dashed") +
  labs(x = "Actual HR", y = "Predicted HR", title = "Actual vs Predicted HR") +
  geom_smooth(method = "lm", se = FALSE) +
  annotate("text", x = max(lr_df$Actual), y = min(lr_df$Predicted), 
           label = paste("RMSE =", round(rmse_lr, 2)), hjust = 1, vjust = -1)

##Comparison----------
 #display a table comparing the methods
final_results <- full_join(results, eval_metrics_lr)

print(final_results)
