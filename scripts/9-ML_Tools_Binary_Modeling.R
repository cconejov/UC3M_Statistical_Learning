#################################
# WINE QUALITY: Classification Analysis. 2 Groups ML Tools
#################################

library(tidyverse)
library(caret)
library(MASS)

load('rda/wineQuality.rda')
#load('rda/wineQualityLightVersion.rda')

wineQuality$quality.class <- factor(ifelse(wineQuality$quality < 6, "Low",
                                           ifelse(wineQuality$quality < 7, "Medium", "High")), 
                                    levels = c("Low", "Medium", "High"))

wineQuality$quality <- NULL

# New Categories
wineQuality$quality.class <- factor(ifelse(wineQuality$quality.class == "High", "High", "Regular"),
                                    levels = c("Regular","High"))

summary(wineQuality)

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
#lda_model_2_class_CM$overall

## KNN
knn_model_2_class <- train_models_2class[[2]]
knn_model_2_class_pred.test = predict(knn_model_2_class, wineQuality.Test)
knn_model_2_class_CM <- confusionMatrix(knn_model_2_class_pred.test, 
                                        wineQuality.Test$quality.class)
#rm(knn_model_2_class_pred.test)
#knn_model_2_class_CM


## SVM
SVM_model_2_class <- train_models_2class[[3]]
SVM_model_2_class_pred.test = predict(SVM_model_2_class, wineQuality.Test)
SVM_model_2_class_CM <- confusionMatrix(SVM_model_2_class_pred.test,
                                        wineQuality.Test$quality.class)
rm(SVM_model_2_class_pred.test)
#SVM_model_2_class_CM

##RF

RF_model_2_class <- train_models_2class[[4]]
RF_model_2_class_pred.test = predict(RF_model_2_class, wineQuality.Test)
RF_model_2_class_CM <- confusionMatrix(RF_model_2_class_pred.test,
                                       wineQuality.Test$quality.class)
#rm(RF_model_2_class_pred.test)
#RF_model_2_class_CM

# Variable importance
RF_model_2_class_imp <- varImp(RF_model_2_class, scale = F)
plot(RF_model_2_class_imp, scales = list(y = list(cex = .95)))

## GradientBoosting

xgb_model_2_class <- train_models_2class[[5]]
xgb_model_2_class_pred.test = predict(xgb_model_2_class, wineQuality.Test)
xgb_model_2_class_CM <- confusionMatrix(xgb_model_2_class_pred.test,
                                        wineQuality.Test$quality.class)
#rm(xgb_model_2_class_pred.test)
#xgb_model_2_class_CM


## NN
nn_model_2_class <- train_models_2class[[6]]
nn_model_2_class_pred.test = predict(nn_model_2_class, wineQuality.Test)
nn_model_2_class_CM <- confusionMatrix(nn_model_2_class_pred.test,
                                       wineQuality.Test$quality.class)
rm(nn_model_2_class_pred.test)
#nn_model_2_class_CM


## DNN
dnn_model_2_class <- train_models_2class[[6]]
dnn_model_2_class_pred.test = predict(dnn_model_2_class, wineQuality.Test)
dnn_model_2_class_CM <- confusionMatrix(dnn_model_2_class_pred.test,
                                        wineQuality.Test$quality.class)
rm(dnn_model_2_class_pred.test)
#dnn_model_2_class_CM


# wINNER METHODS: RF/xgb/knn

# Create mode function
mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

ensemble.pred = apply(data.frame(RF_model_2_class_pred.test, xgb_model_2_class_pred.test, knn_model_2_class_pred.test), 1, mode) 
ensemble.pred <- factor(ensemble.pred, levels = c("Regular", "High"))
ensemble_2_class_model_CM = confusionMatrix(ensemble.pred, wineQuality.Test$quality.class)

# Summary # Ranking of the best model here!

Methods <- c("sparseLDA","knn", "SVM", "RF", "xgb", "nn", "dnn", "Ensemble")
Accuraracy <- c(round(100*lda_model_2_class_CM$overall[1],2),
                round(100*knn_model_2_class_CM$overall[1],2),
                round(100*SVM_model_2_class_CM$overall[1],2),
                round(100*RF_model_2_class_CM$overall[1],2),
                round(100*xgb_model_2_class_CM$overall[1],2),
                round(100*nn_model_2_class_CM$overall[1],2),
                round(100*dnn_model_2_class_CM$overall[1],2),
                round(100*ensemble_2_class_model_CM$overall[1],2))

kappa <- c(round(100*lda_model_2_class_CM$overall[2],2),
           round(100*knn_model_2_class_CM$overall[2],2),
           round(100*SVM_model_2_class_CM$overall[2],2),
           round(100*RF_model_2_class_CM$overall[2],2),
           round(100*xgb_model_2_class_CM$overall[2],2),
           round(100*nn_model_2_class_CM$overall[2],2),
           round(100*dnn_model_2_class_CM$overall[2],2),
           round(100*ensemble_2_class_model_CM$overall[2],2))

Specificity <- c(round(100*lda_model_2_class_CM$byClass[2],2),
                 round(100*knn_model_2_class_CM$byClass[2],2),
                 round(100*SVM_model_2_class_CM$byClass[2],2),
                 round(100*RF_model_2_class_CM$byClass[2],2),
                 round(100*xgb_model_2_class_CM$byClass[2],2),
                 round(100*nn_model_2_class_CM$byClass[2],2),
                 round(100*dnn_model_2_class_CM$byClass[2],2),
                 round(100*ensemble_2_class_model_CM$byClass[2],2))

Worst.Error <- c(lda_model_2_class_CM$table[1,2],
                 knn_model_2_class_CM$table[1,2],
                 SVM_model_2_class_CM$table[1,2],
                 RF_model_2_class_CM$table[1,2],
                 xgb_model_2_class_CM$table[1,2],
                 nn_model_2_class_CM$table[1,2],
                 dnn_model_2_class_CM$table[1,2],
                 ensemble_2_class_model_CM$table[1,2])

summary.methods <- data.frame(Methods,
                              Accuraracy,
                              kappa,
                              Specificity,
                              Worst.Error,
                              row.names = NULL)
rm(Methods,kappa,Accuraracy,Specificity,Worst.Error)
summary.methods


## Visual:

#set.seed(42)
#resamps <- resamples(list(LDA = lda_model_2_class,
#                          RF = SVM_model_2_class,
#                          NN = nn_model_2_class))
#resamps


#theme1 <- trellis.par.get()
#theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
#theme1$plot.symbol$pch = 16
#theme1$plot.line$col = rgb(1, 0, 0, .7)
#theme1$plot.line$lwd <- 2
#trellis.par.set(theme1)
#bwplot(resamps, layout = c(3, 1))


## BEST MODELS

set.seed(42)
models_compare = resamples(list(sparseLDA = lda_model_2_class,
                                knn = knn_model_2_class,
                                SVM = SVM_model_2_class,
                                RF = RF_model_2_class,
                                xgb = xgb_model_2_class,
                                nn = nn_model_2_class,
                                dnn = dnn_model_2_class))


models_scales = list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(models_compare, scales = models_scales, main = "Metrics for classiffication: 3 classes")


## 4.5 The ROC curve

### a) General

# ROC curve shows true positives vs false positives in relation with different thresholds
# y-axis = Sensitivity (TP)
# x-axis = Specificity (1-FP)

library(pROC)

RF_model_2_class_prob.test = predict(RF_model_2_class, wineQuality.Test, type = "prob")

plot.roc(wineQuality.Test$quality.class, 
         RF_model_2_class_prob.test[,2],
         col="darkblue", 
         print.auc = TRUE, 
         auc.polygon=TRUE, 
         grid=c(0.1, 0.2),
         grid.col=c("green", "red"), 
         max.auc.polygon=TRUE,
         auc.polygon.col="lightblue", 
         print.thres=TRUE)


## Recomendded Threshold 0.3

threshold = 0.3
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


# Considering the RF model we have:

RF_CM_BYR <- RF_model_2_class_CM$table
RF_CM_THR <- RF_model_2_class_CM_prob$table

sum(price * RF_CM_BYR)
sum(price * RF_CM_THR)


RF_model_2_class_pred.test = predict(RF_model_2_class, wineQuality.Test)

# Let's optimize now the hyper-parameters using Caret with a specific loss

EconomicProfit <- function(data, lev = NULL, model = NULL) {
  y.pred = data$pred 
  y.true = data$obs
  CM = confusionMatrix(y.pred, y.true)$table
  out = sum(as.vector(CM)*price)
  names(out) <- c("EconomicProfit")
  out
}


# trainControl number = 5, fitting 2
ctrl <- trainControl(method = "repeatedcv",
                     repeats = 5,
                     number = 10,
                     classProbs = TRUE, 
                     summaryFunction = EconomicProfit,
                     verboseIter = T)

# Bayes rule
EconomicProfit(data = data.frame(pred  = RF_model_2_class_pred.test, obs = wineQuality.Test$quality.class))

RF_model_2_class_econProfit <- train(quality.class ~., 
                                     method = "rf", 
                                     data = wineQuality.Train,
                                     preProcess = c("center", "scale"),
                                     #ntree = 200,
                                     #cutoff=c(0.7,0.3),
                                     #tuneGrid = expand.grid(mtry=c(6,8,10)), 
                                     metric = "EconomicProfit",
                                     trControl = ctrl)


#load('rda/train_models_2class_econProfit_ML.rda')
#RF_model_2_class_econProfit <- train_models_2class_econProfit[[1]]

train_models_2class_econProfit <- list()
train_models_2class_econProfit[[1]] <- RF_model_2_class_econProfit
save(train_models_2class_econProfit, file = 'rda/train_models_2class_econProfit_ML.rda')


# optimal BYR
RF_model_2_class_pred.test.econProfit <- predict(RF_model_2_class_econProfit,
                                                 newdata = wineQuality.Test)
CM = confusionMatrix(factor(RF_model_2_class_pred.test.econProfit), 
                     wineQuality.Test$quality.class)$table
profit = sum(as.vector(CM)*price)
profit


threshold = 0.15
RF_model_2_class_prob_econProfit.test = predict(RF_model_2_class_econProfit, 
                                                wineQuality.Test, 
                                                type = "prob")
RF_model_2_class_prob_pred_econProfit.test <- rep("Regular", nrow(wineQuality.Test))
RF_model_2_class_prob_pred_econProfit.test[which(RF_model_2_class_prob_econProfit.test[,2] > threshold)] = "High"
RF_model_2_class_prob_pred_econProfit.test <- factor(RF_model_2_class_prob_pred_econProfit.test, levels = c("Regular","High"))
RF_model_2_class_CM_prob_econProfit <- confusionMatrix(RF_model_2_class_prob_pred_econProfit.test, 
                                            wineQuality.Test$quality.class)

CM
RF_model_2_class_CM_prob_econProfit$table

RF_model_2_class_CM_prob_econProfit$table
sum(RF_model_2_class_CM_prob_econProfit$table*price)


# same with xgboost


# Let's try now xgboost with Caret:
xgb_grid = expand.grid(nrounds = c(500,1000),
                       eta = c(0.01, 0.001), # c(0.01,0.05,0.1)
                       max_depth = c(2, 4, 6),
                       gamma = 1,
                       colsample_bytree = c(0.2, 0.4),
                       min_child_weight = c(1,5),
                       subsample = 1)

set.seed(42)
xgb_model_2_class_econProfit = train(quality.class ~ .,
                          method = "xgbTree",
                          data = wineQuality.Train,
                          trControl = ctrl,
                          metric = "EconomicProfit",
                          #tuneGrid = xgb_grid,
                          preProcess = c("center", "scale")
)

train_models_2class_econProfit[[2]] <- xgb_model_2_class_econProfit
save(train_models_2class_econProfit, file = 'rda/train_models_2class_econProfit_ML.rda')



set.seed(42)
knn_model_2_class_econProfit <- train(quality.class ~ ., 
                           method = "knn", 
                           data = wineQuality.Train,
                           preProcess = c("center", "scale"),
                           metric = "EconomicProfit",
                           trControl = ctrl)



## b) Save / Load the model
train_models_2class_econProfit[[3]] <- knn_model_2_class_econProfit
save(train_models_2class_econProfit, file = 'rda/train_models_2class_econProfit_ML.rda')


set.seed(42)
SVM_model_2_class_econProfit <- train(quality.class ~ .,
                           method = "svmRadial",
                           data = wineQuality.Train,
                           preProcess = c("center", "scale"),
                           #tuneGrid = expand.grid(C = c(.25, .5, 1),
                           #                      sigma = c(0.01,.05)), 
                           metric = "EconomicProfit",
                           trControl = ctrl)

## b) Save / Load the model
train_models_2class_econProfit[[4]] <- SVM_model_2_class_econProfit
save(train_models_2class_econProfit, file = 'rda/train_models_2class_econProfit_ML.rda')
