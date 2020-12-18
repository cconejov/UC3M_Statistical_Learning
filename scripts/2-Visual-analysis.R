#################################
# WINE QUALITY: Visual Analysis
#################################


load('rda/wineQuality.rda')
table(wineQuality$quality)

load('rda/wineQualityLightVersion.rda')
table(wineQuality$quality)

library(tidyverse)
library(caret)


colnames(wineQuality)
summary(wineQuality)


set.seed(42)
spl = createDataPartition(wineQuality$quality, p = 0.8, list = FALSE)  # 80% for training

wineQuality.Train = wineQuality[spl,]
wineQuality.Test = wineQuality[-spl,]

######################
# Visualitization
######################

ggplot(wineQuality.Train, aes(x = quality)) + geom_bar()



# Quality vs alcohol

ggplot(wineQuality.Train, aes(x=alcohol, y= quality)) +
  geom_point(alpha=0.8) + ggtitle("Wine Quality vs alcohol")

# Quality by category

ggplot(wineQuality.Train, aes(x= quality, fill = Category)) + geom_bar()



ggplot(data = wineQuality.Train, aes(x=alcohol))  + 
  xlab("Alcohol") + 
  ylab("Quantity") + 
  facet_grid(quality~Category, scale="free_y") +  
  geom_histogram(binwidth=0.5) + 
  ggtitle("Alcohol level by quality and Category")


ggplot(data = wineQuality.Train, aes(x=alcohol, fill=as.factor(quality)))  + 
  xlab("Alcohol") + 
  ylab("Quantity") +
  facet_wrap(~Category) +   
  geom_density() + 
  ggtitle("Alcohol by category") + 
  scale_fill_discrete(name  ="Quality")