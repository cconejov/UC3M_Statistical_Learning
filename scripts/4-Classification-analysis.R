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

wineQuality$quality.class <- factor(ifelse(wineQuality$quality < 6, "Low",
                                           ifelse(wineQuality$quality < 7, "Medium", "High")), 
                                    levels = c("Low", "Medium", "High"))

wineQuality$quality <- NULL

summary(wineQuality)

# Training and testing sets

set.seed(42)
spl <-  createDataPartition(wineQuality$quality.class, p = 0.2, list = FALSE)  # 80% for training
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

## b) Save / Load the model
train.models[[1]] <- log_reg_model_1
#log_reg_model_1 <- train.models[[1]]

## c) Summary / Predictions / CM
summary(log_reg_model_1)

log_reg_model_1_prob.test = predict(log_reg_model_1, 
                                    newdata = wineQuality.Test, 
                                    type = "response")

log_reg_model_1_pred.test <- factor(levels(wineQuality$quality.class)[max.col(log_reg_model_1_prob.test)], levels = c("Low", "Medium", "High"))


log_reg_model_1_CM <-  confusionMatrix(log_reg_model_1_pred.test,
                                       wineQuality.Test$quality.class)


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

log_reg_model_1_CM$byClass[3,1]*100   #(The greater the better)

log_reg_model_1_CM$table[3,1]         #(The less the better, worse error!)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 1.2. Multiple Logistic regression

log_reg_model_2 <- vglm(quality.class ~ ., 
                        family = multinomial(refLevel=1), 
                        data = wineQuality.Train)

train.models[[2]] <- log_reg_model_2
log_reg_model_2 <- train.models[[2]] 



# Save the model
save(train.models, file = "Assigment/train_models.rda")

summary(log_reg_model_2)



log_reg_model_2_prob.test = predict(log_reg_model_2, 
                                    newdata = wineQuality.Test, 
                                    type = "response")
#head(log_reg_model_1_prob.test)

log_reg_model_2_pred.test <- factor(levels(wineQuality$quality.class)[max.col(log_reg_model_2_prob.test)], levels = c("Low", "Medium", "High"))
rm(log_reg_model_2_prob.test)
#head(log_reg_model_1_pred.test)


log_reg_model_2_CM <-  confusionMatrix(log_reg_model_2_pred.test,
                                       wineQuality.Test$quality.class)

log_reg_model_2_CM


# Real values
table(wineQuality.Test$quality.class)
# Predicted value
table(log_reg_model_2_pred.test)

log_reg_model_2_CM$table
log_reg_model_2_CM$byClass
log_reg_model_2_CM$overall


# Summary of metrics
## Accuracy
log_reg_model_2_CM$overall[1]*100     #(The greater the better)

log_reg_model_2_CM$byClass[3,1]*100   #(The greater the better)

log_reg_model_2_CM$table[3,1]         #(The less the better, worse error!)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 1.3. Penalized Logistic regression

# 1.3.a Dummy Version (determine the right lambda)


# Dumy code categorical predictor variables
x <- model.matrix(quality.class~., wineQuality.Train)[,-1]
# Convert the outcome (class) to a numerical variable
y <- ifelse(wineQuality.Train$quality.class == "High", 2, 
            ifelse(wineQuality.Train$quality.class == "Medium",1,0))

library(glmnet)

set.seed(42)
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "multinomial")
plot(cv.lasso)
cv.lasso$lambda.min




# 1.3.b Caret Version
ctrl <- trainControl(method = "repeatedcv", 
                     repeats = 1,
                     number = 2)

set.seed(42)
log_reg_model_3 <- train(quality.class ~ ., 
                         method = "glmnet",
                         family = "multinomial",
                         data = wineQuality.Train,
                         preProcess = c("center", "scale"),
                         #tuneGrid = expand.grid(alpha = seq(0, 1, 0.1), lambda = seq(0, 2, 0.01)),
                         metric = "Accuracy",
                         #tuneLength = 20,
                         trControl = ctrl)

#Save this object
print(log_reg_model_3)

log_reg_model_3_pred.test <- predict(log_reg_model_3, wineQuality.Test)

log_reg_model_3_CM <-  confusionMatrix(log_reg_model_3_pred.test,
                                       wineQuality.Test$quality.class)

log_reg_model_3_CM


# Real values
table(wineQuality.Test$quality.class)
# Predicted value
table(log_reg_model_3_pred.test)

log_reg_model_3_CM$table
log_reg_model_3_CM$byClass
log_reg_model_3_CM$overall


# Summary of metrics
## Accuracy
log_reg_model_3_CM$overall[1]*100     #(The greater the better)

log_reg_model_3_CM$byClass[3,1]*100   #(The greater the better)

log_reg_model_3_CM$table[3,1]         #(The less the better, worse error!)


log_reg_model_3_imp <- varImp(log_reg_model_3, scale = F)
plot(log_reg_model_3_imp, scales = list(y = list(cex = .95)))


models_compare = resamples(list(ADABOOST = log_reg_model_3, RF = log_reg_model_3, XGBDART = log_reg_model_3))


scales = list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(models_compare, scales = scales)



Method <- c("Simple.Log.Reg","Multiple.Log.Reg","Penalized.Log.Reg")
Accuraracy <- c(round(100*log_reg_model_1_CM$overall[1],2),
                round(100*log_reg_model_2_CM$overall[1],2),
                round(100*log_reg_model_3_CM$overall[1],2))
High.Sensitivity <- c(round(100*log_reg_model_3_CM$byClass[3,1],2),
                      round(100*log_reg_model_2_CM$byClass[3,1],2),
                      round(100*log_reg_model_3_CM$byClass[3,1],2))
Worst.Error <- c(log_reg_model_1_CM$table[3,1] ,
                 log_reg_model_2_CM$table[3,1] ,
                 log_reg_model_3_CM$table[3,1] )
summary.methods <- data.frame(Method,
                              Accuraracy,
                              High.Sensitivity,
                              Worst.Error,
                              row.names = NULL)
summary.methods






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

# 2.1 LDA

## 2.1.1 LDA

set.seed(42)
lda_model_1 <- train(quality.class ~ ., 
                     method = "lda", 
                     #method = "PenalizedLDA", 
                     #method = "sparseLDA", 
                     #method = "stepLDA", 
                     data = wineQuality.Train,
                     preProcess = c("center", "scale"),
                     metric = "Accuracy",
                     trControl = ctrl)
lda_model_1_pred.test = predict(lda_model_1, wineQuality.Test)
lda_model_1_CM <- confusionMatrix(lda_model_1_pred.test,wineQuality.Test$quality.class)
lda_model_1_CM

## It will work if we proceed with preprocessing
#lda_model_1_imp <- varImp(lda_model_1, scale = F)
#plot(lr_imp, scales = list(y = list(cex = .95)))

lda_model_1_CM$overall[1]*100     #(The greater the better)

lda_model_1_CM$byClass[3,1]*100   #(The greater the better)

lda_model_1_CM$table[3,1]         #(The less the better, worse error!)


set.seed(42)
lda_model_2 <- train(quality.class ~ ., 
                     #method = "lda", 
                     #method = "PenalizedLDA", 
                     method = "sparseLDA", 
                     #method = "stepLDA", 
                     data = wineQuality.Train,
                     preProcess = c("center", "scale"),
                     metric = "Accuracy",
                     trControl = ctrl)
lda_model_2_pred.test = predict(lda_model_2, wineQuality.Test)
lda_model_2_CM <- confusionMatrix(lda_model_2_pred.test,wineQuality.Test$quality.class)
lda_model_2_CM

set.seed(42)
lda_model_3 <- train(quality.class ~ ., 
                     #method = "lda", 
                     #method = "PenalizedLDA", 
                     #method = "sparseLDA", 
                     method = "stepLDA", 
                     data = wineQuality.Train,
                     preProcess = c("center", "scale"),
                     metric = "Accuracy",
                     trControl = ctrl)
lda_model_3_pred.test = predict(lda_model_3, wineQuality.Test)
lda_model_3_CM <- confusionMatrix(lda_model_2_pred.test,wineQuality.Test$quality.class)
lda_model_3_CM


lda_models_compare = resamples(list(LDA = lda_model_1,
                                    sparseLDA = lda_model_2, 
                                    stepLDA = lda_model_3))

lda_scales = list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(lda_models_compare, scales = lda_scales)


# 2.2 QDA

# 2.2.0 dummmy qda 

qda.model <- qda(quality.class ~ ., data = wineQuality.Train)
qda.model


## 2.2.1 QDA

set.seed(42)
qda_model_1 <- train(quality.class ~ ., 
                     method = "qda", 
                     data = wineQuality.Train,
                     preProcess = c("center", "scale"),
                     metric = "Accuracy",
                     trControl = ctrl)
qda_model_1_pred.test = predict(qda_model_1, wineQuality.Test)
qda_model_1_CM <- confusionMatrix(qda_model_1_pred.test,wineQuality.Test$quality.class)
qda_model_1_CM


## 2.2.2 stepQDA

set.seed(42)
qda_model_2 <- train(quality.class ~ ., 
                     method = "stepQDA", 
                     data = wineQuality.Train,
                     preProcess = c("center", "scale"),
                     metric = "Accuracy",
                     trControl = ctrl)
qda_model_2_pred.test = predict(qda_model_2, wineQuality.Test)
qda_model_2_CM <- confusionMatrix(qda_model_2_pred.test,wineQuality.Test$quality.class)
qda_model_2_CM

qda_models_compare = resamples(list(QDA = qda_model_1,
                                    stepQDA = qda_model_2))

qda_scales = list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(qda_models_compare, scales = qda_scales)


## 2.3 NAIVE BAYES

nb_model_1 <- train(quality.class ~ ., 
                    method = "nb", 
                    data = wineQuality.Train,
                    preProcess = c("center", "scale"),
                    metric = "Accuracy",
                    trControl = ctrl)
nb_model_1_pred.test <-  predict(nb_model_1, wineQuality.Test)
nb_model_1 <- confusionMatrix(nb_model_1_pred.test, wineQuality.Test$quality.class)
