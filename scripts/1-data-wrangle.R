#################################
# WINE QUALITY
#################################

# URL data.set: https://data.world/uci/wine-quality

# 1. Download data:

# Wine quality red
wineRed <- read.csv("https://query.data.world/s/y34cljjfywvmzblkc7h6e7t55idqyk", header=TRUE, stringsAsFactors=FALSE);

# Wine quality white
wineWhite <- read.csv("https://query.data.world/s/cuyknshnx6fv7nq57oflryvhk2bhun", header=TRUE, stringsAsFactors=FALSE);

# 2. Data wrangling (join the tables)

wineRed$Category <- "Red"
wineWhite$Category <- "White"

wineQuality <- as.data.frame(rbind(wineRed,wineWhite))
wineQuality$Category <- as.factor(wineQuality$Category)

save(wineQuality, file = 'rda/wineQuality.rda')


## LIGHT VERSION: FOR CREATE ALL THE PROJECT

load('rda/wineQuality.rda')

library(caret)
set.seed(42)
spl = createDataPartition(wineQuality$quality, p = 0.11, list = FALSE)  # 11% for evaluating
wineQuality <- wineQuality[spl,]
rm(spl)

save(wineQuality, file = 'rda/wineQualityLightVersion.rda')
