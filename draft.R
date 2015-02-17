# Read data
dataset <- read.csv("pml-training.csv", na.strings = c("NA", "", " "))

# Check for NAs
sum(!complete.cases(dataset))
summary(dataset)

# Omit columns with NAs
omit.variables <- sapply(dataset, function(x) {sum(is.na(x)) > 0})
dataset <- dataset[ , !omit.variables]

# Omit first columns (no useful info)
library(dplyr)
dataset <- select(dataset, -(X:num_window))

# Create training and test set
library(caret)
set.seed(12321)
inTrain <- createDataPartition(dataset$classe, p = 0.7, list = FALSE)
training <- dataset[inTrain, ]
testing <- dataset[-inTrain, ]

# Plot
qplot(classe, data = training, fill = classe)

# Training Random Forest
model.RF <- train(classe ~ ., data = training, method = "rf", prox = TRUE)
model.Boost <- train(classe ~ ., data = training, method = "gbm", verbose = FALSE)