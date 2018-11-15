library(tidyverse)
library(readr)
library(xgboost)

# load data
train = read_csv("train.csv")
test = read_csv("test.csv")

# check column data types and convert data to numeric for xgboost
features <- names(train)[2:ncol(train)-1]

for (f in features) {
  if (class(train[[f]]) == "character") {
    levels = unique(c(train[[f]], test[[f]]))
    train[[f]] = as.integer(factor(train[[f]], levels = levels))
    test[[f]]  = as.integer(factor(test[[f]],  levels = levels))
  }
}

clf <- xgboost(data        = data.matrix(train[,features]),
               label       = train$Response,
               eta         = 0.025,
               depth       = 10,
               nrounds     = 2500,
               objective   = "reg:linear",
               eval_metric = "rmse")

# plotting rmse curve
# we see large diminishing returns around 100 rounds
e <- data.frame(clf$evaluation_log)
plot(e$iter, e$train_rmse, col = 'blue')

# feature importance
imp <- xgb.importance(colnames(features), model = clf)
xgb.plot.importance(imp)

# testing model on actual known values
preds = round(predict(clf, data.matrix(train[,features])))
preds = ifelse(preds > 8, 8, preds)
preds = ifelse(preds < 1, 1, preds)
table(Prediction = preds, Actual = train$Response)
# mean squared error 1.47 of train data after rounding
rmsetrain = sqrt(mean((preds - train$Response) ^ 2))

# test data prediction and submission
submission <- data.frame(Id=test$Id)
submission$Response <- as.integer(round(predict(clf, data.matrix(test[,features]))))

table(submission$Response)

submission$Response = ifelse(submission$Response > 8, 8, submission$Response)
submission$Response = ifelse(submission$Response < 1, 1, submission$Response)

write_csv(submission, "submission.csv")


