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
set.seed(12321)
library(caret)
inTrain <- createDataPartition(dataset$classe, p = 0.7, list = FALSE)
training <- dataset[inTrain, ]
validation <- dataset[-inTrain, ]

# Plot
qplot(classe, data = training, fill = classe)

# Training Random Forest
library(randomForest)
#chosen <- sample(1:nrow(training), 100)
model.RF <- randomForest(classe ~ ., 
                         data = training,
                         ntree = 20)

# Cross-Validation
result <- rfcv(trainx = training[ , -53], trainy = training[ , 53], ntree = 20)
with(result, plot(n.var, 
                  error.cv, 
                  log = "x", 
                  type = "o", 
                  lwd = 2, 
                  xlab = "Number of variables", 
                  ylab = "CV Error"))

# Training with 13 top important variables
varImpPlot(model.RF, n.var = 13)
variables <- varImp(model.RF)
top.names <- rownames(variables)[order(variables, decreasing = TRUE)][1:13]

model.RF.top <- randomForest(x = training[ , c(top.names)],
                             y = training$classe,
                             ntree = 20)

# Validation
pred <- predict(model.RF.top, validation)
print(confusionMatrix(pred, validation$classe))

#model.Boost <- train(classe ~ ., data = training, method = "gbm", verbose = FALSE)

# Testing
testset <- read.csv("pml-testing.csv", na.strings = c("NA", "", " "))

# Omit columns with NAs
omit.variables <- sapply(testset, function(x) {sum(is.na(x)) > 0})
testset <- testset[ , !omit.variables]

# Omit first columns (no useful info)
testset <- select(testset, -(X:num_window))

pred.test <- predict(model.RF.top, testset)

# Submission
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

#pml_write_files(pred.test)