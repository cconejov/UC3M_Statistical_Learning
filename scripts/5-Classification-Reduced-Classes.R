library(caret)

load('rda/wineQuality.rda')
load('rda/train_models_classification.rda')


wineQuality$quality.class <- factor(ifelse(wineQuality$quality < 6, "Low",
                                           ifelse(wineQuality$quality < 7, "Medium", "High")), 
                                    levels = c("Low", "Medium", "High"))

wineQuality$quality <- NULL

# New Categories
wineQuality$quality.class <- factor(ifelse(wineQuality$quality.class == "High", "High", "No High"),
                                    levels = c("No High","High"))

set.seed(42)
spl = createDataPartition(wineQuality$quality.class, p = 0.8, list = FALSE)  # 80% for training
wineQuality.Train = wineQuality[spl,]
wineQuality.Test = wineQuality[-spl,]
rm(spl)

table(wineQuality.Test$quality.class)

#  Caret Version ctrl function
ctrl <- trainControl(method = "repeatedcv", 
                     repeats = 5,
                     number = 10)

set.seed(42)
lda_model_2_classes <- train(quality.class ~ ., 
                             method = "sparseLDA", 
                             data = wineQuality.Train,
                             preProcess = c("center", "scale"),
                             metric = "Accuracy",
                             trControl = ctrl)

#train.models.reduced.class <- list()
#train.models.reduced.class[[1]] <- lda_model_2_classes
lda_model_2_classes <- train.models[[1]]

lda_model_2_classes_pred.test <-  predict(lda_model_2_classes, wineQuality.Test)
lda_model_2_classes_CM <- confusionMatrix(lda_model_2_classes_pred.test,
                                          wineQuality.Test$quality.class,
                                          positive = "High")
lda_model_2_classes_CM


str(lda_model_2_classes_CM)


round(100*lda_model_2_classes_CM$overall[1],2)
round(100*lda_model_2_classes_CM$byClass[1],2)

# In these case, the accuracy is 80%. However, the sensitivity is of 30%. Even less than the 
# some results given with the previous methods. Increase the threshold

# scenario 1
lda_model_2_classes_prob.test = predict(lda_model_2_classes, wineQuality.Test, type = "prob")
threshold = 0.
#head(lda_model_2_classes_prob.test)

lda_model_2_classes_pred.test_from_prob <- rep("No High", nrow(wineQuality.Test))

lda_model_2_classes_pred.test_from_prob[which(lda_model_2_classes_prob.test[,2] > threshold)] = "High"
lda_model_2_classes_pred.test_from_prob <- factor(lda_model_2_classes_pred.test_from_prob, levels = c("No High","High"))
#head(lda_model_2_classes_pred.test_from_prob)
lda_model_1_CM <- confusionMatrix(lda_model_2_classes_pred.test_from_prob, 
                                  wineQuality.Test$quality.class,
                                  positive = "High")

lda_model_1_CM$table

relative.price <- c(0,-0.5,0.1,1)

CM <- lda_model_1_CM$table

sum(relative.price * CM )

##################
# relative gain
##################



threshold_values <-  seq(0.15,0.75,0.05)
no.iter <- 30

superavit.i = matrix(NA, nrow = no.iter, ncol = length(threshold_values))
# 30 replicates for training/testing sets for each of the 10 values of threshold
superavit.i

j <- 0
ctrl <- trainControl(method = "none")

for (threshold in threshold_values){
  
  j <- j + 1
  cat(j)
  for(i in 1:no.iter){
    
    # partition data intro training (80%) and testing sets (20%)
    set.seed(42 + i)
    d <- createDataPartition(wineQuality.Train$quality.class, p = 0.8, list = FALSE)
    # select training sample
    
    levels(wineQuality.Train)
    
    train<-wineQuality.Train[d,]
    test <-wineQuality.Train[-d,]  
    
    ldaFit <- train(quality.class ~ ., 
                    method = "sparseLDA", 
                    data = train,
                    preProcess = c("center", "scale"),
                    metric = "Accuracy",
                    trControl = ctrl)
    lrProb = predict(ldaFit, test, type="prob")
    
    lrPred = rep("No High", nrow(test))
    lrPred[which(lrProb[,2] > threshold)] = "High"
    lrPred = factor(lrPred, levels = c("No High","High"))
    
    
    CM = confusionMatrix(lrPred, test$quality.class)$table
    
    superavit.i[i,j] <- sum(relative.price*CM) # unitary cost
    
  }
}


boxplot(superavit.i, main = "Hyper-parameter selection",
        ylab = "Superavit",
        xlab = "threshold value",
        names = threshold_values,col="royalblue2")



load('rda/train_models_classification.rda')

#Use this two position for save the new data
train.models[[1]] <- lda_model_2_classes
train.models[[2]] <- superavit.i

save(train.models, file = "rda/train_models_classification.rda")

##


Prediction_And_Reference <- c("No High", "High")
No_High <- c(0,-0.5)
High <- c(0.1,1)
economic <- data.frame(Prediction_And_Reference,
                              No_High,
                              High)
economic
