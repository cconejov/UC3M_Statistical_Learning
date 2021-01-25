#################################
# WINE QUALITY: Classification Analysis. 3 Groups ML Tools
#################################

library(tidyverse)
library(caret)
library(MASS)

load('rda/wineQuality.rda')
#load('rda/wineQualityLightVersion.rda')
table(wineQuality$quality)


wineQuality$quality.class <- factor(ifelse(wineQuality$quality < 6, "Low",
                                           ifelse(wineQuality$quality < 7, "Medium", "High")), 
                                    levels = c("Low", "Medium", "High"))

wineQuality$quality <- NULL

set.seed(42)
spl <- createDataPartition(wineQuality$quality.class, p = 0.8, list = FALSE)  # 80% for training
wineQuality.Train <- wineQuality[spl,]
wineQuality.Test  <- wineQuality[-spl,]
rm(spl)


#  Caret Version ctrl function
ctrl <- trainControl(method = "repeatedcv", 
                     repeats = 5,
                     number = 10)


# Create list for saving the models
train_models_3class <- list()

###############################
# BenchMark/ The best model in Midterm
###############################

## a) Creating model
set.seed(42)
lda_model <- train(quality.class ~ ., 
                     method = "sparseLDA", 
                     data = wineQuality.Train,
                     preProcess = c("center", "scale"),
                     metric = "Accuracy",
                     trControl = ctrl)

lda_model$finalModel

## b) Save / Load the model
train_models_3class[[1]] <- lda_model


###############################
# ML Models
###############################

#*****************************
#* KNN
#*****************************

## a) Creating model
set.seed(42)
knn_model <- train(quality.class ~ ., 
                   method = "knn", 
                   data = wineQuality.Train,
                   preProcess = c("center", "scale"),
                   metric = "Accuracy",
                   trControl = ctrl)

knn_model$finalModel

## b) Save / Load the model
train_models_3class[[2]] <- knn_model

#*****************************
#* SVM
#*****************************

## a) Creating model
set.seed(42)
SVM_model <- train(quality.class ~ .,
                   method = "svmRadial",
                   data = wineQuality.Train,
                   preProcess = c("center", "scale"),
                   #tuneGrid = expand.grid(C = c(.25, .5, 1),
                   #                      sigma = c(0.01,.05)), 
                   metric = "Accuracy",
                   trControl = ctrl)

SVM_model$finalModel

## b) Save / Load the model
train_models_3class[[3]] <- SVM_model

#*****************************
#* Decision Tree/Random Forest
#*****************************

## a) Creating model
set.seed(42)
RF_model <- train(quality.class ~ .,
                   method = "rf",
                   data = wineQuality.Train,
                   preProcess = c("center", "scale"),
                   #tuneGrid = expand.grid(C = c(.25, .5, 1),
                   #                      sigma = c(0.01,.05)), 
                   # cutoff = 
                   metric = "Accuracy",
                   trControl = ctrl)

RF_model$finalModel

## b) Save / Load the model
train_models_3class[[4]] <- RF_model

#*****************************
#* Gradient Boosting
#*****************************

#a) Creating the model

# Let's try now xgboost with Caret:
xgb_grid = expand.grid(nrounds = c(500,1000),
                       eta = c(0.01, 0.001), # c(0.01,0.05,0.1)
                       max_depth = c(2, 4, 6),
                       gamma = 1,
                       colsample_bytree = c(0.2, 0.4),
                       min_child_weight = c(1,5),
                       subsample = 1)

set.seed(42)
xgb_model = train(quality.class ~ .,
                  method = "xgbTree",
                  data = wineQuality.Train,
                  trControl = ctrl,
                  metric="Accuracy",
                  #tuneGrid = xgb_grid,
                  preProcess = c("center", "scale")
                  )


xgb_model$finalModel

## b) Save / Load the model
train_models_3class[[5]] <- xgb_model
#*****************************
#* Neural Networks
#*****************************

#a) Creating the model

#ctrl$sampling <- NULL

set.seed(42)
nn_model = train(quality.class ~ .,
                  method = "nnet",
                  data = wineQuality.Train,
                  trControl = ctrl,
                  metric="Accuracy",
                  preProcess = c("center", "scale")
)


nn_model$finalModel

## b) Save / Load the model
train_models_3class[[6]] <- nn_model


#*****************************
#*  Deep Neural Networks
#*****************************

#a) Creating the model

#ctrl$sampling <- NULL

set.seed(42)
dnn_model = train(quality.class ~ .,
                 method = "dnn",
                 data = wineQuality.Train,
                 trControl = ctrl,
                 metric="Accuracy",
                 preProcess = c("center", "scale")
)


dnn_model$finalModel

## b) Save / Load the model
train_models_3class[[7]] <- dnn_model


#####################
# save the list with the trained models
####################
save(train_models_3class, file = 'rda/train_models_3class_ML.rda')
