#################################
## Choose Your Own-Project: Predict if someone will order from a website
## by the interaction on the website beforehand
#################################

#################################
## Install & Load Packages
#################################

if (!require(dslabs)) install.packages('dslabs')
library(dslabs)
if (!require(tidyverse)) install.packages('tidyverse')
library(tidyverse)
if (!require(lubridate)) install.packages('lubridate')
library(lubridate)
if (!require(caret)) install.packages('caret')
library(caret)
if (!require(data.table)) install.packages('data.table')
library(data.table)
if (!require(corrplot)) install.packages('corrplot')
library(corrplot)
if (!require(rpart)) install.packages('rpart')
library(rpart)
if (!require(rpart.plot)) install.packages('rpart.plot')
library(rpart.plot)
if (!require(readr)) install.packages('readr')
library(readr)
if (!require(randomForest)) install.packages('randomForest')
library(randomForest)

#################################
## Data Import
#################################

# This project uses the "Customer propensity to purchase dataset - A data set
# logging shoppers interactions on an online store" accessed from:
# https://www.kaggle.com/benpowis/customer-propensity-to-purchase-data/metadata.

# The provided "Testing Sample" will not be used as it does not contain any
# customers that have placed an order (testing_sample$ordered == 1) which makes
# it useless for this analysis. Instead, the provided training_sample has been split
# into two data sets.

# Test set
url_testing <- "https://raw.githubusercontent.com/KaHeimerl/ChooseYourOwn/main/training_sample_01.txt"
testing_sample <- read_csv(url(url_testing))

# Train set
url_training <- "https://raw.githubusercontent.com/KaHeimerl/ChooseYourOwn/main/training_sample_02.txt"
training_sample <- read_csv(url(url_training))

rm(url_testing, url_training)

#################################
## Data Exploration
#################################

# Data set
dim(training_sample)
head(training_sample)

# Is any UserID twice in the sample? -> No
duplicates <- duplicated(training_sample$UserID)
mean(duplicates)

# How many users have placed an order? -> ca. 4,2%
mean(training_sample$ordered)

# Check which independent variables (excl. UserID) correlate with the
# dependent variable "ordered"

# Visualizing the correlation matrix
correlations <- cor(training_sample[-1])
corrplot(correlations, method="color", order="hclust", tl.col="black", tl.srt=45, tl.cex=0.6)

# Calculate correlations
corr_ordered <- cor(training_sample[-1], training_sample$ordered)
print(corr_ordered[order(corr_ordered[,1],decreasing=TRUE),])
# Highest correlation for checked_delivery_detail, saw_checkout and sign_in
# Noticeable correlation also for basket_icon_click, basket_add_detail
# and basket_add_list
# Not considering saw_checkout for further analysis: step after ordering

# Check distribution of highly correlated variables / predictors in the sample
mean(training_sample$checked_delivery_detail)
mean(training_sample$sign_in)
mean(training_sample$basket_icon_click)
mean(training_sample$basket_add_detail)
mean(training_sample$basket_add_list)
# All centered around approx. 6-10% of the sample

# Do the actions / predictors occur together?
mean(training_sample$checked_delivery_detail & training_sample$sign_in)
mean(training_sample$basket_icon_click & training_sample$basket_add_detail)
mean(training_sample$basket_icon_click & training_sample$basket_add_list)
mean(training_sample$checked_delivery_detail & training_sample$basket_icon_click)
mean(training_sample$sign_in & training_sample$basket_icon_click)
mean(training_sample$checked_delivery_detail & training_sample$basket_add_detail)
mean(training_sample$sign_in & training_sample$basket_add_detail)
mean(training_sample$checked_delivery_detail & training_sample$basket_add_list)
mean(training_sample$sign_in & training_sample$basket_add_list)
# Given the overall low prevalence in the sample (around 6-10%), actions do occur
# rather often together, where making sense: approx. 3-5%

#################################
## Method
#################################

# Create a second train and test set in order to test several models before
# validating the final algorithm with the testing_sample
set.seed(1)
test_index <- createDataPartition(training_sample$ordered,
                                  times = 1, p = 0.1,
                                  list = FALSE)
test_set <- training_sample[test_index, ]
train_set <- training_sample[-test_index, ]

###########
# Model 1: Guessing the outcome
###########

set.seed(1)
y_guessing <- sample(c(0,1), length(test_index), replace = TRUE)
mean(y_guessing == test_set$ordered)

# Guessing the outcome with fixed probability
set.seed(1)
y_guessing_prob <- sample(c(0,1), length(test_index), replace = TRUE, prob=c(0.95,0.05))
mean(y_guessing_prob == test_set$ordered)
# Problem: We are guessing right which customers do NOT order (0), not necessarily who does order (1)
table(predicted = y_guessing_prob, actual = test_set$ordered)
# Confirmed: more than 20.000 correctly guessed 0s, but only less than 100 correctly guessed 1s

# Calculating the confusion matrix
confusionMatrix_guessing <- confusionMatrix(data = as.factor(y_guessing_prob),
                                   reference = as.factor(test_set$ordered),
                                   positive = "1")
print(confusionMatrix_guessing)
# Really low sensitivity due to low prevalence, balanced accuracy only 50%
# Will prevalence be the same if applied to other data sets?

# Using the F_meas function to compute the weighted harmonic average of sensitivity and
# specificity (F1-score) with beta=2 (higher importance of sensitivity) as a benchmark for optimization
fmeas_guessing <- F_meas(data = as.factor(y_guessing_prob),
                         reference = as.factor(test_set$ordered),
                         relevant = "1",
                         beta = 2)
print(fmeas_guessing)

# Create a table that stores all modeling results obtained
fmeas_results <- data.frame(method = "guessing", F1 = fmeas_guessing)

###########
# Model 2: Logistic regression
###########

fit_log_regression <- glm(ordered ~ sign_in + checked_delivery_detail,
                          data = train_set,
                          family = "binomial")
prob_log_regression <- predict(fit_log_regression, test_set, type = "response")
table(prob_log_regression)
y_log_regression <- ifelse(prob_log_regression > 0.11, 1, 0)

# Check Confusion Matrix
confusionMatrix_log_regression <- confusionMatrix(data = as.factor(y_log_regression),
                                                  reference = as.factor(test_set$ordered),
                                                  positive = "1")
print(confusionMatrix_log_regression)

# Check F1-score
fmeas_log_regression <- F_meas(data = as.factor(y_log_regression),
                               reference = as.factor(test_set$ordered),
                               relevant = "1",
                               beta = 2)
print(fmeas_log_regression)
# Logistic regression with highly better results than guessing
# Results depending upon chosen "cutoff" of prediction rule

# Add result to fmeas table
fmeas_results <- bind_rows(fmeas_results,
                           data.frame(method = "logistic regression",
                                      F1 = fmeas_log_regression))

###########
# Model 3: Linear discriminant analysis (lda)
###########

fit_lda <- train(as.factor(ordered) ~ checked_delivery_detail + sign_in,
                 method = "lda", data = train_set)
y_lda <- predict(fit_lda, test_set)

# Check Confusion Matrix
confusionMatrix_lda <- confusionMatrix(data = y_lda,
                                       reference = as.factor(test_set$ordered),
                                       positive = "1")
print(confusionMatrix_lda)

# Check F1-score
fmeas_lda <- F_meas(data = as.factor(y_lda),
                    reference = as.factor(test_set$ordered),
                    relevant = "1",
                    beta = 2)
print(fmeas_lda)

# Add result to fmeas table
fmeas_results <- bind_rows(fmeas_results,
                           data.frame(method = "lda", F1 = fmeas_lda))

###########
# Model 4: Classification tree
###########

fit_tree <- rpart(ordered ~ checked_delivery_detail + sign_in,
                  data = train_set,
                  method = 'class')

rpart.plot(fit_tree,
           extra = 106,
           yesno = 2,
           xflip = TRUE)

y_tree <- predict(fit_tree, test_set, type = 'class')

# Check Confusion Matrix
confusionMatrix_tree <- confusionMatrix(data = as.factor(y_tree),
                                        reference = as.factor(test_set$ordered),
                                        positive = "1")
print(confusionMatrix_tree)

# Check F1-score
fmeas_tree <- F_meas(data = as.factor(y_tree),
                     reference = as.factor(test_set$ordered),
                     relevant = "1",
                     beta = 2)
print(fmeas_tree)
# Classification tree with same results as logistic regression

# Add result to fmeas table
fmeas_results <- bind_rows(fmeas_results,
                           data.frame(method = "classification tree",
                                      F1 = fmeas_tree))

###########
# Model 5: Random Forest
###########

# Characteristic of the random forest is to include all possible predictor variables and
# ensure randomness by randomly selecting sets of variables for prediction

train_set_forest <- subset(train_set, select = -c(UserID, saw_checkout,
                                                  device_mobile, device_computer,
                                                  device_tablet, returning_user,
                                                  loc_uk))

# ! Attention: Takes long time to compute
memory.limit(9999999999)
fit_forest <- randomForest(as.factor(ordered) ~ ., data = train_set_forest)
y_forest <- predict(fit_forest, test_set)

# Check Confusion Matrix
confusionMatrix_forest <- confusionMatrix(data = as.factor(y_forest),
                                          reference = as.factor(test_set$ordered),
                                          positive = "1")
print(confusionMatrix_forest)

# Check F1-score
fmeas_forest <- F_meas(data = as.factor(y_forest),
                      reference = as.factor(test_set$ordered),
                      relevant = "1",
                      beta = 2)
print(fmeas_forest)
# Slightly worse results than for logistic regression / classification tree

# Add result to fmeas table
fmeas_results <- bind_rows(fmeas_results,
                           data.frame(method = "random forest",
                                      F1 = fmeas_forest))

# Check Variable Importance
variable_importance <- importance(fit_forest)
variable_importance[order(variable_importance[,1], decreasing = TRUE),]
# Already identified variables with the highest importance:
# checked_delivery_detail and sign_in
# But also high importance of basket_icon_click, basket_add_detail and basket_add_list

# Next Step: Try add those 3 variables to the best performing model to see if it improves
print(fmeas_results)

###########
# Model 6: Classification tree with 5 predictors
###########

fit_tree_5 <- rpart(ordered ~ checked_delivery_detail + sign_in +
                  basket_add_detail + basket_icon_click + basket_add_list,
                  data = train_set,
                  method = 'class')

rpart.plot(fit_tree_5,
           extra = 106,
           yesno = 2,
           xflip = TRUE)

y_tree_5 <- predict(fit_tree_5, test_set, type = 'class')

# Check Confusion Matrix
confusionMatrix_tree_5 <- confusionMatrix(data = as.factor(y_tree_5),
                                          reference = as.factor(test_set$ordered),
                                          positive = "1")
print(confusionMatrix_tree_5)

# Check F1-score
fmeas_tree_5 <- F_meas(data = as.factor(y_tree_5),
                     reference = as.factor(test_set$ordered),
                     relevant = "1",
                     beta = 2)
print(fmeas_tree_5)
# Adding the 3 more variables to classification tree does not change the F1 score

###########
# Final Model: Apply classification tree to testing_sample
###########

fit_tree_fin <- rpart(ordered ~ checked_delivery_detail + sign_in,
                      data = training_sample,
                      method = 'class')

rpart.plot(fit_tree_fin,
           extra = 106,
           yesno = 2,
           xflip = TRUE)

y_tree_fin <- predict(fit_tree_fin, testing_sample, type = 'class')

# Check Confusion Matrix
confusionMatrix_tree_fin <- confusionMatrix(data = as.factor(y_tree_fin),
                                            reference = as.factor(testing_sample$ordered),
                                            positive = "1")
print(confusionMatrix_tree_fin)

# Check F1-score
fmeas_tree_fin <- F_meas(data = as.factor(y_tree_fin),
                        reference = as.factor(testing_sample$ordered),
                        relevant = "1",
                        beta = 2)
print(fmeas_tree_fin)
# Very good results also with the testing sample

# Conclusion: It would, however, be interesting to see if this model
# would work as well with a completely different data set, e.g.
# with data for a different kind of web shop (e.g. clothing vs. electronics)