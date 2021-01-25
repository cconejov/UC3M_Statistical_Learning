#################################
# WINE QUALITY: Classification Analysis. 3 Groups ML Tools
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

#proportions of categories

tb <- wineQuality %>% count(quality.class) %>% mutate(p = round(prop.table(n),2))
colnames(tb) <- c("Quality class", "Count", "Proportion")

# Training and testing split

set.seed(42)
spl <- createDataPartition(wineQuality$quality.class, p = 0.8, list = FALSE)  # 80% for training
wineQuality.Train <- wineQuality[spl,]
wineQuality.Test  <- wineQuality[-spl,]
rm(spl)

# Load models:
load('rda/train_models_3class_ML.rda')

# Actual values of quality wine
table(wineQuality.Test$quality.class)

lda_model <- train_models_3class[[1]]
lda_model_pred.test = predict(lda_model, wineQuality.Test)
lda_model_CM <- confusionMatrix(lda_model_pred.test,wineQuality.Test$quality.class)
lda_model_CM


knn_model <- train_models_3class[[2]]
knn_model_pred.test = predict(knn_model, wineQuality.Test)
knn_model_CM <- confusionMatrix(knn_model_pred.test,wineQuality.Test$quality.class)


SVM_model <- train_models_3class[[3]]
SVM_model_pred.test = predict(SVM_model, wineQuality.Test)
SVM_model_CM <- confusionMatrix(SVM_model_pred.test,wineQuality.Test$quality.class)

RF_model <- train_models_3class[[4]]
RF_model_pred.test = predict(RF_model, wineQuality.Test)
RF_model_CM <- confusionMatrix(RF_model_pred.test,wineQuality.Test$quality.class)


xgb_model <- train_models_3class[[5]]
xgb_model_pred.test = predict(xgb_model, wineQuality.Test)
xgb_model_CM <- confusionMatrix(xgb_model_pred.test,wineQuality.Test$quality.class)


nn_model <- train_models_3class[[6]]
nn_model_pred.test = predict(nn_model, wineQuality.Test)
nn_model_CM <- confusionMatrix(nn_model_pred.test,wineQuality.Test$quality.class)


dnn_model <- train_models_3class[[6]] #it 7
dnn_model_pred.test = predict(dnn_model, wineQuality.Test)
dnn_model_CM <- confusionMatrix(dnn_model_pred.test,wineQuality.Test$quality.class)


## Ensemble

### RF, xgbost, SVM 

# Create mode function
mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

ensemble.pred = apply(data.frame(RF_model_pred.test, xgb_model_pred.test, SVM_model_pred.test), 1, mode) 
ensemble.pred <- factor(ensemble.pred, levels = c("Low", "Medium", "High"))
ensemble_model_CM = confusionMatrix(ensemble.pred, wineQuality.Test$quality.class)

Methods <- c("sparseLDA","knn", "SVM", "RF", "xgb", "nn", "dnn", "Ensemble")
Accuraracy <- c(round(100*lda_model_CM$overall[1],2),
                round(100*knn_model_CM$overall[1],2),
                round(100*SVM_model_CM$overall[1],2),
                round(100*RF_model_CM$overall[1],2),
                round(100*xgb_model_CM$overall[1],2),
                round(100*nn_model_CM$overall[1],2),
                round(100*dnn_model_CM$overall[1],2),
                round(100*ensemble_model_CM$overall[1],2))

kappa <- c(round(100*lda_model_CM$overall[2],2),
           round(100*knn_model_CM$overall[2],2),
           round(100*SVM_model_CM$overall[2],2),
           round(100*RF_model_CM$overall[2],2),
           round(100*xgb_model_CM$overall[2],2),
           round(100*nn_model_CM$overall[2],2),
           round(100*dnn_model_CM$overall[2],2),
           round(100*ensemble_model_CM$overall[2],2))

High.Sensitivity <- c(round(100*lda_model_CM$byClass[3,1],2),
                      round(100*knn_model_CM$byClass[3,1],2),
                      round(100*SVM_model_CM$byClass[3,1],2),
                      round(100*RF_model_CM$byClass[3,1],2),
                      round(100*xgb_model_CM$byClass[3,1],2),
                      round(100*nn_model_CM$byClass[3,1],2),
                      round(100*dnn_model_CM$byClass[3,1],2),
                      round(100*ensemble_model_CM$byClass[3,1],2))

Worst.Error <- c(lda_model_CM$table[3,1],
                 knn_model_CM$table[3,1],
                 SVM_model_CM$table[3,1],
                 RF_model_CM$table[3,1],
                 xgb_model_CM$table[3,1],
                 nn_model_CM$table[3,1],
                 dnn_model_CM$table[3,1],
                 ensemble_model_CM$table[3,1])

summary.methods <- data.frame(Methods,
                              Accuraracy,
                              kappa,
                              High.Sensitivity,
                              Worst.Error,
                              row.names = NULL)
rm(Methods,kappa,Accuraracy,High.Sensitivity,Worst.Error)
summary.methods


## BEST MODELS

set.seed(42)
models_compare = resamples(list(sparseLDA = lda_model,
                                knn = knn_model,
                                SVM = SVM_model,
                                RF = RF_model,
                                xgb = xgb_model,
                                nn = nn_model,
                                dnn = dnn_model))


models_scales = list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(models_compare, scales = models_scales, main = "Metrics for classiffication: 3 classes")