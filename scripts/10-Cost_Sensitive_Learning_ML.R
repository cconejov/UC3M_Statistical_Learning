#################################
# Cost sentitive learning
#################################

library(tidyverse)
library(caret)

load('rda/wineQuality.rda')
#load('rda/wineQualityLightVersion.rda')

wineQuality$quality.class <- factor(ifelse(wineQuality$quality < 6, "Low",
                                           ifelse(wineQuality$quality < 7, "Medium", "High")), 
                                    levels = c("Low", "Medium", "High"))

wineQuality$quality <- NULL

# New Categories
wineQuality$quality.class <- factor(ifelse(wineQuality$quality.class == "High", "High", "Regular"),
                                    levels = c("Regular","High"))


set.seed(42)
spl = createDataPartition(wineQuality$quality.class, p = 0.8, list = FALSE)  # 80% for training
wineQuality.Train = wineQuality[spl,]
wineQuality.Test = wineQuality[-spl,]
rm(spl)


load('rda/train_models_2class_ML.rda')


## LDA
lda_model_2_class <- train_models_2class[[1]]
lda_model_2_class_pred.test <-  predict(lda_model_2_class, wineQuality.Test)
lda_model_2_class_CM <- confusionMatrix(lda_model_2_class_pred.test,
                                        wineQuality.Test$quality.class)


RF_model_2_class <- train_models_2class[[4]]
RF_model_2_class_pred.test = predict(RF_model_2_class, wineQuality.Test)
RF_model_2_class_CM <- confusionMatrix(RF_model_2_class_pred.test,
                                       wineQuality.Test$quality.class)

RF_model_2_class_prob.test = predict(RF_model_2_class, wineQuality.Test, type = "prob")

threshold = 0.25
RF_model_2_class_prob_pred.test <- rep("Regular", nrow(wineQuality.Test))


RF_model_2_class_prob_pred.test[which(RF_model_2_class_prob.test[,2] > threshold)] = "High"
RF_model_2_class_prob_pred.test <- factor(RF_model_2_class_prob_pred.test, levels = c("Regular","High"))
RF_model_2_class_CM_prob <- confusionMatrix(RF_model_2_class_prob_pred.test, 
                                            wineQuality.Test$quality.class)


## Economic profit
price <- c(0,-0.5,0.1,1)
# Threshold 0.5
sum(price * lda_model_2_class_CM$table)
# Selected Threshold Task 1 for lda 0.35 with 67.4

RF_CM_BYR <- RF_model_2_class_CM$table
RF_CM_THR <- RF_model_2_class_CM_prob$table

sum(price * RF_CM_BYR)
sum(price * RF_CM_THR)

EconomicProfit <- function(data, lev = NULL, model = NULL) {
  y.pred = data$pred 
  y.true = data$obs
  CM = confusionMatrix(y.pred, y.true)$table
  out = sum(as.vector(CM)*price)
  names(out) <- c("EconomicProfit")
  out
}



load('rda/train_models_2class_econProfit_ML.rda')
RF_model_2_class_econProfit <- train_models_2class_econProfit[[1]]
xgb_model_2_class_econProfit <- train_models_2class_econProfit[[2]]
knn_model_2_class_econProfit <- train_models_2class_econProfit[[3]]
SVM_model_2_class_econProfit <- train_models_2class_econProfit[[4]]

# optimal BYR
RF_model_2_class_pred.test.econProfit <- predict(RF_model_2_class_econProfit,
                                                 newdata = wineQuality.Test)
RF_CM = confusionMatrix(factor(RF_model_2_class_pred.test.econProfit), 
                     wineQuality.Test$quality.class)$table
RF_profit = sum(as.vector(RF_CM)*price)
RF_profit


# What hapens if we change the threshoold
threshold = 0.25
RF_model_2_class_prob_econProfit.test = predict(RF_model_2_class_econProfit, 
                                                wineQuality.Test, 
                                                type = "prob")
RF_model_2_class_prob_pred_econProfit.test <- rep("Regular", nrow(wineQuality.Test))
RF_model_2_class_prob_pred_econProfit.test[which(RF_model_2_class_prob_econProfit.test[,2] > threshold)] = "High"
RF_model_2_class_prob_pred_econProfit.test <- factor(RF_model_2_class_prob_pred_econProfit.test, levels = c("Regular","High"))
RF_model_2_class_CM_prob_econProfit025 <- confusionMatrix(RF_model_2_class_prob_pred_econProfit.test, 
                                                       wineQuality.Test$quality.class)$table

RF_CM
RF_model_2_class_CM_prob_econProfit

sum(RF_model_2_class_CM_prob_econProfit025*price)



# optimal BYR
xgb_model_2_class_pred.test.econProfit <- predict(xgb_model_2_class_econProfit,
                                                 newdata = wineQuality.Test)
xgb_CM = confusionMatrix(factor(xgb_model_2_class_pred.test.econProfit), 
                        wineQuality.Test$quality.class)$table
xgb_profit = sum(as.vector(xgb_CM)*price)
#xgb_profit


# optimal BYR
knn_model_2_class_pred.test.econProfit <- predict(knn_model_2_class_econProfit,
                                                  newdata = wineQuality.Test)
knn_CM = confusionMatrix(factor(knn_model_2_class_pred.test.econProfit), 
                         wineQuality.Test$quality.class)$table
knn_profit = sum(as.vector(knn_CM)*price)
#knn_profit


# optimal BYR
SVM_model_2_class_pred.test.econProfit <- predict(SVM_model_2_class_econProfit,
                                                  newdata = wineQuality.Test)
SVM_CM = confusionMatrix(factor(xgb_model_2_class_pred.test.econProfit), 
                         wineQuality.Test$quality.class)$table
SVM_profit = sum(as.vector(xgb_CM)*price)
#SVM_profit