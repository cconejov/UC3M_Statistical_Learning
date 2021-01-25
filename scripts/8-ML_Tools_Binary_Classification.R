#################################
# WINE QUALITY: Classification Analysis. 2 Groups ML Tools
#################################

library(tidyverse)
library(caret)
library(MASS)

load('rda/wineQuality.rda')

wineQuality$quality.class <- factor(ifelse(wineQuality$quality < 6, "Low",
                                           ifelse(wineQuality$quality < 7, "Medium", "High")), 
                                    levels = c("Low", "Medium", "High"))

wineQuality$quality <- NULL

# New Categories
wineQuality$quality.class <- factor(ifelse(wineQuality$quality.class == "High", "High", "Regular"),
                                    levels = c("Regular","High"))


# Train/test split

set.seed(42)
spl = createDataPartition(wineQuality$quality.class, p = 0.8, list = FALSE)  # 80% for training
wineQuality.Train = wineQuality[spl,]
wineQuality.Test = wineQuality[-spl,]
rm(spl)


#  Caret Version ctrl function
ctrl <- trainControl(method = "repeatedcv",
                     repeats = 5,
                     number = 10,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)


###############################
# BenchMark/ The best model in Midterm
###############################

lda_model_2_class <- train(quality.class ~ ., 
                             method = "sparseLDA", 
                             data = wineQuality.Train,
                             preProcess = c("center", "scale"),
                             metric = "Spec",
                             trControl = ctrl)


train_models_2class <- list()
train_models_2class[[1]] <- lda_model_2_class
save(train_models_2class, file = 'rda/train_models_2class_ML.rda')
rm(lda_model_2_class)
###############################
# ML Models
###############################

#*****************************
#* KNN
#*****************************

## a) Creating model
set.seed(42)
knn_model_2_class <- train(quality.class ~ ., 
                           method = "knn", 
                           data = wineQuality.Train,
                           preProcess = c("center", "scale"),
                           metric = "Spec",
                           trControl = ctrl)



## b) Save / Load the model
train_models_2class[[2]] <- knn_model_2_class
save(train_models_2class, file = 'rda/train_models_2class_ML.rda')
rm(knn_model_2_class)

#*****************************
#* SVM
#*****************************

## a) Creating model
set.seed(42)
SVM_model_2_class <- train(quality.class ~ .,
                          method = "svmRadial",
                          data = wineQuality.Train,
                          preProcess = c("center", "scale"),
                          #tuneGrid = expand.grid(C = c(.25, .5, 1),
                          #                      sigma = c(0.01,.05)), 
                          metric = "Spec",
                          trControl = ctrl)

## b) Save / Load the model
train_models_2class[[3]] <- SVM_model_2_class
save(train_models_2class, file = 'rda/train_models_2class_ML.rda')
rm(SVM_model_2_class)
#*****************************
#* Decision Tree/Random Forest
#*****************************

## a) Creating model
set.seed(42)
RF_model_2_class <- train(quality.class ~ .,
                         method = "rf",
                         data = wineQuality.Train,
                         preProcess = c("center", "scale"),
                         #tuneGrid = expand.grid(C = c(.25, .5, 1),
                         #                      sigma = c(0.01,.05)), 
                         # cutoff = 
                         metric = "Spec",
                         trControl = ctrl)


## b) Save / Load the model
train_models_2class[[4]] <- RF_model_2_class
save(train_models_2class, file = 'rda/train_models_2class_ML.rda')
rm(RF_model_2_class)

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
xgb_model_2_class = train(quality.class ~ .,
                  method = "xgbTree",
                  data = wineQuality.Train,
                  trControl = ctrl,
                  metric = "Spec",
                  #tuneGrid = xgb_grid,
                  preProcess = c("center", "scale")
)


## b) Save / Load the model
train_models_2class[[5]] <- xgb_model_2_class
save(train_models_2class, file = 'rda/train_models_2class_ML.rda')
rm(xgb_grid,xgb_model_2_class)
#*****************************
#* Neural Networks
#*****************************

#a) Creating the model

#ctrl$sampling <- NULL

set.seed(42)
nn_model_2_class = train(quality.class ~ .,
                 method = "nnet",
                 data = wineQuality.Train,
                 trControl = ctrl,
                 metric="Spec",
                 preProcess = c("center", "scale")
)

## b) Save / Load the model
train_models_2class[[6]] <- nn_model_2_class
save(train_models_2class, file = 'rda/train_models_2class_ML.rda')
rm(nn_model_2_class)
#*****************************
#*  Deep Neural Networks
#*****************************

#a) Creating the model

#ctrl$sampling <- NULL

set.seed(42)
dnn_model_2_class = train(quality.class ~ .,
                  method = "dnn",
                  data = wineQuality.Train,
                  trControl = ctrl,
                  metric="Spec",
                  preProcess = c("center", "scale")
)


## b) Save / Load the model
train_models_2class[[7]] <- dnn_model_2_class
save(train_models_2class, file = 'rda/train_models_2class_ML.rda')
rm(dnn_model_2_class)