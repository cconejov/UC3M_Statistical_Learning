#######################
# Function 1: Draw a Nice confusion Matrix
#######################

# More details
# https://stackoverflow.com/questions/23891140/r-how-to-visualize-confusion-matrix-using-the-caret-package/42940553

draw_confusion_matrix <- function(cm, Class1, Class2, title_def = 'Confusion Matrix') {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title(title_def, cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 445, Class1, cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 445, Class2, cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, Class1, cex=1.2, srt=90)
  text(140, 335, Class2, cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 65, names(cm$overall[1]), cex=1.5, font=2)
  text(10, 35, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(30, 65, names(cm$overall[2]), cex=1.5, font=2)
  text(30, 35, round(as.numeric(cm$overall[2]), 3), cex=1.4)
  text(50, 65, names(cm$byClass[1]), cex=1.2, font=2)
  text(50, 35, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(70, 65, names(cm$byClass[2]), cex=1.2, font=2)
  text(70, 35, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(90, 65, "High misclassif.", cex=1.2, font=2)
  text(90, 35, round(as.numeric(cm$table[1,2]), 0), cex=1.2)
  
  
  # add in the accuracy information 
  #text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  #text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  #text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  #text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
} 