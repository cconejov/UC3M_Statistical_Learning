#################################
# WINE QUALITY: Regression Analysis
#################################

load('rda/wineQuality.rda')
table(wineQuality$quality)

load('rda/wineQualityLightVersion.rda')
table(wineQuality$quality)

library(tidyverse)
library(caret)

set.seed(42)
spl = createDataPartition(wineQuality$quality, p = 0.8, list = FALSE)  # 80% for training

wineQuality.Train = wineQuality[spl,]
wineQuality.Test = wineQuality[-spl,]

## Introduction

corr_quality <- sort(cor(wineQuality.Train[,-13])["quality",], decreasing = T)
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


ggplot(data = wineQuality.Train, aes(x=alcohol, y=quality))+geom_point()+stat_smooth(method='lm')


ggplot(data = wineQuality.Train, aes(x=alcohol, y=quality, col = Category))+geom_point()+stat_smooth(method='lm')
