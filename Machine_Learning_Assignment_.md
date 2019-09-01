---
title: "Machine Learning Assignment"
author: "UcepH1"
date: "24/08/2019"
output: 
  html_document: 
    keep_md: yes
---



## Overview

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.

## Getting and Preparing Data

Let's start by getting data. Two files are available :

- *pml-training.csv* : file to build and test the model
- *pml-testing.csv* : file to test the model (course quiz)

```r
## check if files were already downloaded
trainfile = "pml-training.csv"
if (!file.exists(trainfile)){
        # Download
        url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        download.file(url, trainfile)
}
testfile = "pml-testing.csv"
if (!file.exists(testfile)){
        # Download
        url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
        download.file(url, testfile)
}
```

Let's read data files. 

```r
## Reading files training and testing sets 
training = read.csv(trainfile, na.strings = c("NA", "#DIV/0!")) # used to build model
testing = read.csv(testfile, na.strings = c("NA", "#DIV/0!")) # ultimate test set
```

Training file contains 19622 rows and 160 columns while testing file contains 20 rows and 160 columns.

*NA* values are represented by following strings : *NA* and *#DIV/0!*.

Let's clean the data from *NA* values.

```r
## Removing columns with NA values
train_data = training[, colSums(is.na(training)) == 0] #
test_data =  testing[, colSums(is.na(testing)) == 0] # 
```

In addition, the first few columns (up to the 7th) do not seem to provide useful information. Hence, we will remove them.


```r
## Removing columns 1 to 7
train_data = train_data[, -c(1:7)] #
test_data =  test_data[, -c(1:7)] # 
```

We are now down to 53 variables.

## Modelling and prediction

We will split our training data into two sets: 60% to build model, 40% to evaluate out of sample error.

```r
## Split training set into two sets : 
inTrain <- createDataPartition(y=train_data$classe,
                              p=0.6, list=FALSE)

train_set = train_data[inTrain,]
test_set = train_data[-inTrain,]
```

Next, we will look at two models :

1. Classification Trees
2. Random Forest

### Classification trees

Here we use the *rpart* method combined with a *cross validation*.

```r
## Cross validation
fitControl = trainControl(method = 'cv', number = 10, verboseIter = FALSE)
set.seed(2019)
## Build model 1
modelFit = train(classe ~ .,data=train_set, method="rpart", trControl=fitControl)
```

Testing final model on test set (obtained by splitting training file) :


```r
## Predict with Test_set based on model 1
CT_result = predict(modelFit, newdata=test_set)
M_CT = confusionMatrix(data=CT_result, reference=test_set[,53])
M_CT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1984  626  629  595  204
##          B   35  502   43  212  180
##          C  177  390  696  479  363
##          D    0    0    0    0    0
##          E   36    0    0    0  695
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4941         
##                  95% CI : (0.483, 0.5053)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3394         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8889  0.33070  0.50877   0.0000  0.48197
## Specificity            0.6341  0.92573  0.78249   1.0000  0.99438
## Pos Pred Value         0.4913  0.51646  0.33064      NaN  0.95075
## Neg Pred Value         0.9349  0.85220  0.88295   0.8361  0.89501
## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
## Detection Rate         0.2529  0.06398  0.08871   0.0000  0.08858
## Detection Prevalence   0.5147  0.12388  0.26829   0.0000  0.09317
## Balanced Accuracy      0.7615  0.62821  0.64563   0.5000  0.73817
```


```r
## Plot
fancyRpartPlot(modelFit$finalModel)
```

![](Machine_Learning_Assignment__files/figure-html/unnamed-chunk-8-1.png)<!-- -->

### Random Forest

Here we use *random forest* combined with *cross validation*.

```r
## Cross validation
fitControl = trainControl(method = 'cv', number = 2, verboseIter = FALSE)
set.seed(2019)
## Build model 2
modelFit2 = train(classe ~ .,data=train_set, method="rf", trControl=fitControl)
```

Testing final model on test set (obtained by splitting training file) :


```r
## Predict with Test_set based on model 2
RF_result = predict(modelFit2, newdata=test_set)
M_RF = confusionMatrix(data=RF_result, reference=test_set[,53])
M_RF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2223   25    0    0    0
##          B    3 1487    5    1    0
##          C    5    5 1360   24    0
##          D    0    1    3 1261    0
##          E    1    0    0    0 1442
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9907          
##                  95% CI : (0.9883, 0.9927)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9882          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9960   0.9796   0.9942   0.9806   1.0000
## Specificity            0.9955   0.9986   0.9948   0.9994   0.9998
## Pos Pred Value         0.9889   0.9940   0.9756   0.9968   0.9993
## Neg Pred Value         0.9984   0.9951   0.9988   0.9962   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2833   0.1895   0.1733   0.1607   0.1838
## Detection Prevalence   0.2865   0.1907   0.1777   0.1612   0.1839
## Balanced Accuracy      0.9958   0.9891   0.9945   0.9900   0.9999
```

## Conclusion

From previous results, we found that ***out-of-sample error*** on test set is : 

- 0.5059 for *Classification Tree*
- 0.0093 for *Random Forest*

Therefore, the model we will apply for the course quiz is the *Random Forest* as it is the most accurate.


```r
## Predict with Test_set based on model 2
Course_Quiz_result = predict(modelFit2, newdata=test_data)
Course_Quiz_result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
