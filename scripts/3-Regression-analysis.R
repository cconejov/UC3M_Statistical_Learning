#################################
# WINE QUALITY: Regression Analysis
#################################

load('rda/wineQuality.rda')
table(wineQuality$quality)

load('rda/wineQualityLightVersion.rda')
table(wineQuality$quality)


library(tidyverse)
library(caret)
library(corrplot)

set.seed(42)
spl = createDataPartition(wineQuality$quality, p = 0.8, list = FALSE)  # 80% for training
wineQuality.Train = wineQuality[spl,]
wineQuality.Test = wineQuality[-spl,]

## Introduction

correlation <- cor(wineQuality.Train[,-13])
corrplot(correlation[-12,-12], method="color", type ="upper")




corr_quality <- sort(correlation["quality",], decreasing = T)
corr <- data.frame(corr_quality)
ggplot(corr,aes(x = row.names(corr), y = corr_quality)) + 
  geom_bar(stat = "identity", fill = "lightblue") + 
  scale_x_discrete(limits= row.names(corr)) +
  labs(x = "", y = "Quality", title = "Correlations") + 
  theme(plot.title = element_text(hjust = 0, size = rel(1.5)),
        axis.text.x = element_text(angle = 45, hjust = 1)) 

## Simple linear regression

lin.reg.model <- lm(quality ~ alcohol, data = wineQuality.Train)
summary(lin.reg.model)


## Prediction
lin.reg.pred <- predict(lin.reg.model, newdata = wineQuality.Test)

cor(wineQuality.Test$quality, lin.reg.pred)^2

## Visualization
ggplot(data = wineQuality.Train, aes(x=alcohol, y=quality))+
  geom_point() + 
  stat_smooth(method='lm')

# Two categories
ggplot(data = wineQuality.Train, aes(x=alcohol, y=quality, col = Category)) + 
  geom_point() + 
  scale_color_manual(values=c("red3", "steelblue")) +
  stat_smooth(method='lm')

## Multiple Linear regression


mult.reg.model <- lm(quality ~ ., data = wineQuality.Train)
summary(mult.reg.model)

## Prediction
mult.reg.pred <- predict(mult.reg.model, newdata = wineQuality.Test)

cor(wineQuality.Test$quality, mult.reg.pred)^2


# Proportions

wineQuality$quality
nrow(wineQuality)

statDisData <-  t(as.data.frame(round(100*table(wineQuality$quality)/nrow(wineQuality),4)))

tbl <- as.data.frame(statDisData, row.names = FALSE)
report <- tbl[2,]
colnames(report)  <- tbl[1,]