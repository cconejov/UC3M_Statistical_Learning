#################################
# WINE QUALITY: Visual Analysis
#################################


load('rda/wineQuality.rda')
table(wineQuality$quality)

load('rda/wineQualityLightVersion.rda')
table(wineQuality$quality)

library(ggplot2)
#library(tidyverse)
library(caret)
library(GGally)
source("functions/graph_functions.R")


colnames(wineQuality)
summary(wineQuality)


set.seed(42)
spl = createDataPartition(wineQuality$quality, p = 0.8, list = FALSE)  # 80% for training
wineQuality.Train = wineQuality[spl,]
wineQuality.Test = wineQuality[-spl,]
rm(spl)

######################
# Visualitization
######################


# General count
ggplot(wineQuality.Train, aes(x= quality)) + 
  geom_bar(fill="steelblue") + 
  geom_text(stat='count', aes(label=..count..), vjust=-1)+
  theme_minimal()

# Count by category
ggplot(wineQuality.Train,aes(x=factor(quality),fill= Category)) +
  geom_bar(stat="count") +
  labs(title = "Distribution of Wine Quality by Category") +
  scale_fill_manual(values=c("red3", "steelblue")) +
  theme_minimal()

# Quality vs alcohol

ggplot(wineQuality.Train, aes(x=alcohol, y= quality,color= Category)) +
  geom_point(alpha=0.8) + 
  scale_color_manual(values=c("red3", "steelblue")) +
  ggtitle("Wine Quality vs Alcohol") +
  theme_minimal()


# Set of histograms with alcohol levels by quality and category
ggplot(data = wineQuality.Train, aes(x=alcohol))  + 
  xlab("Alcohol") + 
  ylab("Quantity") + 
  facet_grid(quality~Category, scale="free_y") +  
  geom_histogram(binwidth=0.5) + 
  ggtitle("Alcohol level by quality and Category")


# Set of densities with alcohol levels by quality and category
ggplot(data = wineQuality.Train, aes(x=alcohol, fill=as.factor(quality)))  + 
  xlab("Alcohol") + 
  ylab("Quantity") +
  facet_wrap(~Category) +   
  geom_density() + 
  ggtitle("Alcohol by category") + 
  scale_fill_discrete(name  ="Quality")


# Univariate analysis
plothist(col_name = "alcohol", 
         df = wineQuality.Train, 
         ylabtext = "Alcohol",  
         "mediumseagreen", 
         density_plot = FALSE)


# Feature plot: Relation of the other predictor variables with the dependent variable

featurePlot(x = wineQuality.Train[,-c(12,13)],
            y = wineQuality.Train[,12],
            plot = "scatter")