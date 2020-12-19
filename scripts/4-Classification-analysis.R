#################################
# WINE QUALITY: Classification Analysis. 3 Groups
#################################

library(tidyverse)
library(caret)
library(VGAM)
library(MASS)

load('rda/wineQuality.rda')
table(wineQuality$quality)

load('rda/wineQualityLightVersion.rda')
table(wineQuality$quality)

# Load the models
load('rda/train_models_classification.rda')


wineQuality$quality.class <- factor(ifelse(wineQuality$quality < 6, "Low",
                                           ifelse(wineQuality$quality < 7, "Medium", "High")), 
                                    levels = c("Low", "Medium", "High"))

wineQuality$quality <- NULL

summary(wineQuality)

# Training and testing sets

set.seed(42)
spl <- createDataPartition(wineQuality$quality.class, p = 0.8, list = FALSE)  # 80% for training
wineQuality.Train <- wineQuality[spl,]
wineQuality.Test  <- wineQuality[-spl,]
rm(spl)

# save all models!!

train.models <- list()

################################
# SECTION 0: CLASIFICATION VIS
################################

colnames(wineQuality.Train)

featurePlot(x = wineQuality.Train[,-c(12,13)],
            y = factor(wineQuality.Train$quality.class),
            plot = "box",
            strip = strip.custom(par.strip.text = list(cex = 0.7)),
            scales = list(x = list(relation = "free"),
                          y = list(relation = "free")
            )
)

################################
# SECTION 1: LOGISTIC REGRESSION
################################

# 1.1. Simple Logistic regression


## a) Creating model
log_reg_model_1 <- vglm(quality.class ~ alcohol, 
                        family = multinomial(refLevel=1), 
                        data = wineQuality.Train)

slog_reg_model_1 <-  summary(log_reg_model_1)

coefoutplut <- log_reg_model_1@coefficients


str(slog_reg_model_1)

## b) Save / Load the model
#train.models[[1]] <- log_reg_model_1
#log_reg_model_1 <- train.models[[1]]

#saveRDS(log_reg_model_1, file = "rda/log_reg_model_1.rds")
#log_reg_model_z <- readRDS("rda/log_reg_model_1.rds")

## c) Summary / Predictions / CM
summary(log_reg_model_1)

log_reg_model_1_prob.test = predict(log_reg_model_1, 
                                    newdata = wineQuality.Test, 
                                    type = "response")

log_reg_model_1_pred.test <- factor(levels(wineQuality$quality.class)[max.col(log_reg_model_1_prob.test)], levels = c("Low", "Medium", "High"))

log_reg_model_1_CM <-  confusionMatrix(log_reg_model_1_pred.test,
                                       wineQuality.Test$quality.class)

rm(log_reg_model_1_prob.test,log_reg_model_1_pred.test)

## d) Statistics / metrics

# Real values
table(wineQuality.Test$quality.class)
# Predicted value
table(log_reg_model_1_pred.test)

log_reg_model_1_CM$table
log_reg_model_1_CM$byClass
log_reg_model_1_CM$overall


# Summary of metrics
## Accuracy
log_reg_model_1_CM$overall[1]*100     #(The greater the better)
## High.Sensitity
log_reg_model_1_CM$byClass[3,1]*100   #(The greater the better)
## Worst error in prediction
log_reg_model_1_CM$table[3,1]         #(The less the better, worse error!)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 1.2. Multiple Logistic regression

## a) Creating model
log_reg_model_2 <- vglm(quality.class ~ ., 
                        family = multinomial(refLevel=1), 
                        data = wineQuality.Train)

## b) Save / Load the model
train.models[[2]] <- log_reg_model_2
#log_reg_model_2 <- train.models[[2]] 

## c) Summary / Predictions / CM
summary(log_reg_model_2)

log_reg_model_2_prob.test <- predict(log_reg_model_2, 
                                    newdata = wineQuality.Test, 
                                    type = "response")

log_reg_model_2_pred.test <- factor(levels(wineQuality$quality.class)[max.col(log_reg_model_2_prob.test)], levels = c("Low", "Medium", "High"))

table(log_reg_model_2_pred.test)
log_reg_model_2_CM <-  confusionMatrix(log_reg_model_2_pred.test,
                                       wineQuality.Test$quality.class)

rm(log_reg_model_2_pred.test,log_reg_model_2_prob.test)
log_reg_model_2_CM


## d) Statistics / metrics
## Accuracy
log_reg_model_2_CM$overall[1]*100     #(The greater the better)
log_reg_model_2_CM$byClass[3,1]*100   #(The greater the better)
log_reg_model_2_CM$table[3,1]         #(The less the better, worse error!)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#  Caret Version ctrl function
ctrl <- trainControl(method = "repeatedcv", 
                     repeats = 5,
                     number = 10)

# 1.3. Penalized Logistic regression

## a) Creating model
set.seed(42)
log_reg_model_3 <- train(quality.class ~ ., 
                         method = "glmnet",
                         family = "multinomial",
                         data = wineQuality.Train,
                         preProcess = c("center", "scale"),
                         #tuneGrid = expand.grid(alpha = seq(0, 1, 0.1), lambda = seq(0, .1, 0.01)),
                         metric = "Accuracy",
                         #tuneLength = 20,
                         trControl = ctrl)

## b) Save / Load the model
train.models[[3]] <- log_reg_model_3
#log_reg_model_3 <- train.models[[3]] 

## c) Summary / Predictions / CM
print(log_reg_model_3)
log_reg_model_3_pred.test <- predict(log_reg_model_3, wineQuality.Test)
log_reg_model_3_CM <-  confusionMatrix(log_reg_model_3_pred.test,
                                       wineQuality.Test$quality.class)
rm(log_reg_model_3_pred.test)

## d) Statistics / metrics
## Accuracy
log_reg_model_3_CM$overall[1]*100     #(The greater the better)
log_reg_model_3_CM$byClass[3,1]*100   #(The greater the better)
log_reg_model_3_CM$table[3,1]         #(The less the better, worse error!)


log_reg_model_3_imp <- varImp(log_reg_model_3, scale = F)
plot(log_reg_model_3_imp, scales = list(y = list(cex = .95)))


#models_compare = resamples(list(ADABOOST = log_reg_model_3, RF = log_reg_model_3, XGBDART = log_reg_model_3))
#scales = list(x = list(relation = "free"), y = list(relation = "free"))
#bwplot(models_compare, scales = scales)


## Summary Regression models

Method.lr <- c("Simple.Log.Reg","Multiple.Log.Reg","Penalized.Log.Reg")
Accuraracy.lr <- c(round(100*log_reg_model_1_CM$overall[1],2),
                round(100*log_reg_model_2_CM$overall[1],2),
                round(100*log_reg_model_3_CM$overall[1],2))
High.Sensitivity.lr <- c(round(100*log_reg_model_1_CM$byClass[3,1],2),
                      round(100*log_reg_model_2_CM$byClass[3,1],2),
                      round(100*log_reg_model_3_CM$byClass[3,1],2))
Worst.Error.lr <- c(log_reg_model_1_CM$table[3,1] ,
                 log_reg_model_2_CM$table[3,1] ,
                 log_reg_model_3_CM$table[3,1] )
summary.methods.lr <- data.frame(Method.lr,
                              Accuraracy.lr,
                              High.Sensitivity.lr,
                              Worst.Error.lr,
                              row.names = NULL)
rm(Method.lr,Accuraracy.lr,High.Sensitivity.lr,Worst.Error.lr)
summary.methods.lr

save(train.models, file = "rda/train_models_log_regression.rda")

################################
# SECTION 2: BAYES
################################

# PREPOCESSING (Do it with ML)
#dummies_model = dummyVars(quality.class ~ ., data = wineQuality.Train)
#trainData_mat = predict(dummies_model, newdata = wineQuality.Train)
#wineQuality.Train = data.frame(trainData_mat)
#rm(trainData_mat, dummies_model)
#str(wineQuality.Train)


#range.model <- preProcess(wineQuality.Train, method = "range")
#wineQuality.Train <- predict(range.model, newdata = wineQuality.Train)
# Append the Y variable
#wineQuality.Train$quality.class <- y
#rm(range.model)
#summary(wineQuality.Train)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2.1 LDA
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## 2.1.1 LDA

## a) Creating model
set.seed(42)
lda_model_1 <- train(quality.class ~ ., 
                     method = "lda", 
                     data = wineQuality.Train,
                     preProcess = c("center", "scale"),
                     metric = "Accuracy",
                     trControl = ctrl)

## b) Save / Load the model
train.models[[4]] <- lda_model_1
#lda_model_1 <- train.models[[4]]

## c) Summary / Predictions / CM
lda_model_1_pred.test = predict(lda_model_1, wineQuality.Test)
lda_model_1_CM <- confusionMatrix(lda_model_1_pred.test,wineQuality.Test$quality.class)
rm(lda_model_1_pred.test)
lda_model_1_CM

## It will work if we proceed with preprocessing
#lda_model_1_imp <- varImp(lda_model_1, scale = F)
#plot(lr_imp, scales = list(y = list(cex = .95)))

## d) Statistics / metrics
## Accuracy
lda_model_1_CM$overall[1]*100     #(The greater the better)
lda_model_1_CM$byClass[3,1]*100   #(The greater the better)
lda_model_1_CM$table[3,1]         #(The less the better, worse error!)

## 2.1.2 sparseLDA

## a) Creating model
set.seed(42)
lda_model_2 <- train(quality.class ~ ., 
                     method = "sparseLDA", 
                     data = wineQuality.Train,
                     preProcess = c("center", "scale"),
                     metric = "Accuracy",
                     trControl = ctrl)

## b) Save / Load the model
train.models[[5]] <- lda_model_2
#lda_model_2 <- train.models[[5]]

## c) Summary / Predictions / CM
lda_model_2_pred.test = predict(lda_model_2, wineQuality.Test)
lda_model_2_CM <- confusionMatrix(lda_model_2_pred.test,wineQuality.Test$quality.class)
rm(lda_model_2_pred.test)
lda_model_2_CM

## d) Statistics / metrics
## Accuracy
lda_model_2_CM$overall[1]*100     #(The greater the better)
lda_model_2_CM$byClass[3,1]*100   #(The greater the better)
lda_model_2_CM$table[3,1] 


## 2.1.3 sparseLDA

## a) Creating model
set.seed(42)
lda_model_3 <- train(quality.class ~ .,
                     method = "stepLDA", 
                     data = wineQuality.Train,
                     preProcess = c("center", "scale"),
                     metric = "Accuracy",
                     trControl = ctrl)

## b) Save / Load the model
train.models[[6]] <- lda_model_3
#lda_model_3 <- train.models[[6]]

## c) Summary / Predictions / CM
lda_model_3_pred.test = predict(lda_model_3, wineQuality.Test)
lda_model_3_CM <- confusionMatrix(lda_model_3_pred.test,wineQuality.Test$quality.class)
rm(lda_model_3_pred.test)
lda_model_3_CM

## d) Statistics / metrics
## Accuracy
lda_model_3_CM$overall[1]*100     #(The greater the better)
lda_model_3_CM$byClass[3,1]*100   #(The greater the better)
lda_model_3_CM$table[3,1] 

## SUMARY LDA METRICS

lda_models_compare = resamples(list(LDA = lda_model_1,
                                    sparseLDA = lda_model_2, 
                                    stepLDA = lda_model_3))

lda_scales = list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(lda_models_compare, scales = lda_scales)


Method.lda <- c("lda","sparseLDA","stepLDA")
Accuraracy.lda <- c(round(100*lda_model_1_CM$overall[1],2),
                    round(100*lda_model_2_CM$overall[1],2),
                    round(100*lda_model_3_CM$overall[1],2))
High.Sensitivity.lda <- c(round(100*lda_model_1_CM$byClass[3,1],2),
                          round(100*lda_model_2_CM$byClass[3,1],2),
                          round(100*lda_model_3_CM$byClass[3,1],2))
Worst.Error.lda <- c(lda_model_1_CM$table[3,1] ,
                     lda_model_2_CM$table[3,1] ,
                     lda_model_3_CM$table[3,1] )
summary.methods.lda <- data.frame(Method.lda,
                                  Accuraracy.lda,
                                  High.Sensitivity.lda,
                                  Worst.Error.lda,
                                  row.names = NULL)
rm(Method.lda,Accuraracy.lda,High.Sensitivity.lda,Worst.Error.lda)
summary.methods.lda


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2.2 QDA
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 2.2.0 dummmy qda 

qda.model <- qda(quality.class ~ ., data = wineQuality.Train)
qda.model

## 2.2.1 QDA

## a) Creating model
set.seed(42)
qda_model_1 <- train(quality.class ~ ., 
                     method = "qda", 
                     data = wineQuality.Train,
                     preProcess = c("center", "scale"),
                     metric = "Accuracy",
                     trControl = ctrl)

## b) Save / Load the model
train.models[[7]] <- qda_model_1
#qda_model_1 <- train.models[[7]]

## c) Summary / Predictions / CM
qda_model_1_pred.test <-  predict(qda_model_1, wineQuality.Test)
qda_model_1_CM <- confusionMatrix(qda_model_1_pred.test,wineQuality.Test$quality.class)
rm(qda_model_1_pred.test)
qda_model_1_CM

## d) Statistics / metrics
## Accuracy
qda_model_1_CM$overall[1]*100     #(The greater the better)
qda_model_1_CM$byClass[3,1]*100   #(The greater the better)
qda_model_1_CM$table[3,1] 

## 2.2.2 stepQDA

## a) Creating model
set.seed(42)
qda_model_2 <- train(quality.class ~ ., 
                     method = "stepQDA", 
                     data = wineQuality.Train,
                     preProcess = c("center", "scale"),
                     metric = "Accuracy",
                     trControl = ctrl)

## b) Save / Load the model
train.models[[8]] <- qda_model_2
#qda_model_2 <- train.models[[8]]

## c) Summary / Predictions / CM
qda_model_2_pred.test <- predict(qda_model_2, wineQuality.Test)
qda_model_2_CM <- confusionMatrix(qda_model_2_pred.test,wineQuality.Test$quality.class)
rm(qda_model_2_pred.test)
qda_model_2_CM


## d) Statistics / metrics
## Accuracy
qda_model_2_CM$overall[1]*100     #(The greater the better)
qda_model_2_CM$byClass[3,1]*100   #(The greater the better)
qda_model_2_CM$table[3,1] 



qda_models_compare = resamples(list(QDA = qda_model_1,
                                    stepQDA = qda_model_2))

qda_scales = list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(qda_models_compare, scales = qda_scales)


Method.qda <- c("qda","stepQDA")
Accuraracy.qda <- c(round(100*qda_model_1_CM$overall[1],2),
                    round(100*qda_model_2_CM$overall[1],2))
High.Sensitivity.qda <- c(round(100*qda_model_1_CM$byClass[3,1],2),
                          round(100*qda_model_2_CM$byClass[3,1],2))
Worst.Error.qda <- c(qda_model_1_CM$table[3,1],
                     qda_model_2_CM$table[3,1] )
summary.methods.qda <- data.frame(Method.qda,
                                  Accuraracy.qda,
                                  High.Sensitivity.qda,
                                  Worst.Error.qda,
                                  row.names = NULL)
rm(Method.qda,Accuraracy.qda,High.Sensitivity.qda,Worst.Error.qda)
summary.methods.qda

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## 2.3 NAIVE BAYES
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## 2.3.1 NAIVE BAYES

## a) Creating model
set.seed(42)
nb_model_1 <- train(quality.class ~ ., 
                    method = "nb", 
                    data = wineQuality.Train,
                    preProcess = c("center", "scale"),
                    metric = "Accuracy",
                    trControl = ctrl)

## b) Save / Load the model
train.models[[9]] <- nb_model_1
#nb_model_1 <- train.models[[9]]

## c) Summary / Predictions / CM
nb_model_1_pred.test <-  predict(nb_model_1, wineQuality.Test)
nb_model_1_CM <- confusionMatrix(nb_model_1_pred.test, wineQuality.Test$quality.class)

## d) Statistics / metrics
## Accuracy
nb_model_1_CM$overall[1]*100     #(The greater the better)
nb_model_1_CM$byClass[3,1]*100   #(The greater the better)
nb_model_1_CM$table[3,1] 


Method.lda <- c("nb")
Accuraracy.nb <- c(round(100*nb_model_1_CM$overall[1],2))
High.Sensitivity.nb <- c(round(100*nb_model_1_CM$byClass[3,1],2))
Worst.Error.nb <- c(nb_model_1_CM$table[3,1])
summary.methods.nb <- data.frame(Method.nb,
                                  Accuraracy.nb,
                                  High.Sensitivity.nb,
                                  Worst.Error.nb,
                                  row.names = NULL)
rm(Method.nb,Accuraracy.nb,High.Sensitivity.nb,Worst.Error.nb)
summary.methods.nb




save(train.models, file = "rda/train_models_classification.rda")


## BEST MODELS

models_compare = resamples(list(Penalized.Log.Reg = log_reg_model_3,
                                 sparseLDA = lda_model_2,
                                 stepQDA = qda_model_2,
                                 NaiveBayes = nb_model_1))

models_scales = list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(models_compare, scales = models_scales)


Method<- c("Penalized.Log.Reg","sparseLDA","stepQDA","NaiveBayes")
Accuraracy <- c(round(100*log_reg_model_3_CM$overall[1],2),
                round(100*lda_model_2_CM$overall[1],2),
                round(100*qda_model_2_CM$overall[1],2),
                round(100*nb_model_1_CM$overall[1],2))
High.Sensitivity <- c(round(100*log_reg_model_3_CM$byClass[3,1],2),
                      round(100*lda_model_2_CM$byClass[3,1],2),
                      round(100*qda_model_2_CM$byClass[3,1],2),
                      round(100*nb_model_1_CM$byClass[3,1],2))
Worst.Error <- c(log_reg_model_3_CM$table[3,1],
                 lda_model_2_CM$table[3,1],
                 qda_model_2_CM$table[3,1],
                 nb_model_1_CM$table[3,1])
summary.methods <- data.frame(Method,
                                 Accuraracy,
                                 High.Sensitivity,
                                 Worst.Error,
                                 row.names = NULL)
rm(Method,Accuraracy,High.Sensitivity,Worst.Error)
summary.methods
