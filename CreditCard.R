#########################################
#         Classification model          #
#          Machine Learning 1           #
#########################################

# Importing the required packages and libraries

install.packages(c('ROCR','ggplot2','corrplot','caTools','class',
                   'randomForest','pROC','imbalance'))
library(ROCR)
library(ggplot2)
library(corrplot)
library(caTools)
library(class)
library(randomForest)
library(pROC)
library(imbalance)
library(rpart)

# The data is saved as a csv file namely creditcard.csv

# Importing the data

transaction <- read.csv(file.choose())

# About the Dataset:
# The dataset includes 284807 observations and 31 variables.
# Out of these 31 variables, one variable (Class) is the variable of interest.
# The dataset contains transactions made by credit cards in September 2013 by 
# European cardholders over a span of two days.
# Due to confidentiality issues, dataset contains only numerical input variables
# which are the result of a PCA transformation. Only features which have not
# been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the
# seconds elapsed between each transaction and the first transaction in the
# dataset. The feature 'Amount' is the transaction Amount.

# The dataset is made available on the Kaggle platform by a research unit of the
# Université Libre de Bruxelles, Machine Learning Group - ULB.
# The dataset can be found here-
# https://www.kaggle.com/mlg-ulb/creditcardfraud

# Basic Exploratory Data Analysis

# Viewing some initial observations of the dataset

head(transaction)

# Viewing Structure of transactions

str(transaction)

sapply(transaction, FUN=class)

# After the basic data exploration we found that all of the variables including
# the target variable are of numeric data type, due to the PCA.

# Viewing the Frequency distribution of "Class" Variable

table(transaction$Class)
prop.table(table(transaction$Class))
ggplot(data = transaction,aes(x=Class))+geom_bar(col="slateblue3") 

# The dataset is highly imbalanced, the positive class (frauds) account for 
# mere 0.172% of all transactions. This is really good for the credit card
# company but not for this analysis.

# Plotting the Amount - Fraud chart

ggplot(data = transaction,aes(y=Amount,x=Class))+geom_point(col="slateblue3") +
  facet_grid(~Class)

# This plot tells us that all the fraudulent transactions were made for less
# than around 2500, which is considerably less than that of true transactions.

# Exploring Time Column with Regards to the Class variable

ggplot(data = transaction,aes(x=Time),fill=factor(Class))+
  geom_histogram(col='red',bins=48) +  facet_grid(Class ~ ., scales = 'free_y')

# As the unit of measure for Time here is seconds, it makes it harder to 
# understand and get anything useful out of this.

# Converting Time(Seconds) into Hours for better understanding

transaction$hours <- round(transaction$Time/3600)
unique(transaction$hours)

# Plotting Frauds with time (in hours)

ggplot(data = transaction,aes(x=hours,fill=factor(Class)))+
  geom_histogram(col='red',bins=48) +  facet_grid(Class ~ ., scales = 'free_y')

# Replacing the Time column with hours column to keep it simple for analysis

transaction$Time <- transaction$hours
transaction <- subset(transaction,select=-hours)

# Checking correlation between the variables

transaction$Class <- as.numeric(transaction$Class)
corr <- cor(transaction[],method="pearson")
corrplot(corr,method = "circle", type = "lower")

# This shows that there is not much correlation among the variables.
# "V2" has some negative correlation with "Amount". However, we can ignore it.

# Converting "Class" to factor type for Modeling purposes

transaction$Class <- as.factor(transaction$Class)

# Splitting the dataset into the Training dataset and Test dataset
# We are doing 75-25 split..

set.seed(123)         # For reproducability of the random selection

# We are using sample.split and subset functions from caTools Library to split
# the data randomly

split = sample.split(transaction$Class, SplitRatio = 0.75)
training_set = subset(transaction, split == TRUE)
test_set = subset(transaction, split == FALSE)

# Verify if the proportion of Class is still the same

table(training_set$Class)
prop.table(table(training_set$Class))

table(test_set$Class)
prop.table(table(test_set$Class))

# To balance out the minority(fraud) and majority class in dataset, we are using
# the SMOTE technique. We are setting ratio at 0.8 which means we are
# oversampling our minority class to come upto 80% of the majority class. 
# It will maintain the ratio and the machine learning model will now be able to
# easily understand the minority class also.

smoted_training_set<-oversample(dataset= training_set, 
                                method = 'SMOTE',ratio = 0.8)
table(smoted_training_set$Class) 
prop.table(table(smoted_training_set$Class)) 

colnames(smoted_training_set)

# Excluding the Class variable to calculate the scaled values for each columns

smoted_training_set[-31] <- scale(smoted_training_set[-31])
test_set[-31] <- scale(test_set[-31])

# Fitting Logistic Regression to the Training set

classifier <- glm(formula = Class ~ .,
                  family = binomial,
                  data = smoted_training_set)


summary(classifier)

# Summary of this model clearly tells that almost all of the variables are 
# statistically significant to the model. Only V24 is there which is not 
# significant. We can ignore this and keep it with us as the value is slightly
# higher than the 5% level of significance.

# Predicting the Test set results

prob_pred <- predict(classifier, type = 'response', newdata = test_set[-31])

# Setting a condition that even if there are more than 40% chances(>0.4) of 
# getting a Fraud Transaction, model should predict it as Fraud.

y_pred <- ifelse(prob_pred > 0.4, 1, 0)

# Calculating and Plotting the AUC-ROC curve
# AUC(Area under the receiver operating characteristic curve) curve tells that 
# how accurately our model is predicting together with its overall Performance.
# The thumb-rule is that Higher the AUC Better our model is at predicting.

calc_auc <- function (prob_pred,test_y){
  predict_log <- prediction(prob_pred,test_y) # Prediction Probability
  
  #Calculating ROC Curve...
  table(test_y, prob_pred>0.4)
  roc_curve<- performance(predict_log,"tpr","fpr")  
  # plot(roc_curve)
  plot(roc_curve, colorize=T)
  
  #Calculating AUC and printing....
  auc<- performance(predict_log,"auc")
  paste(auc@y.values[[1]])
  
}

# Make confusion matrix

cm <- table(test_set[, 31], y_pred > 0.4)
cm

# Print AUC

print(calc_auc(prob_pred,test_set$Class))

# The value of AUC is 0.97 which is really high. However, we cannot just decide
# if it is the best model for our dataset or not.

# Creating Random Forest Model
# ntree = 100

set.seed(123)
forest.classifier = randomForest(x = smoted_training_set[-31],
                                 y = smoted_training_set$Class,
                                 ntree = 100)

# Predicting Classes using the model

forest.y_pred = predict(forest.classifier, newdata = test_set[-31])

# Predicting Probabilities

forest.y_prob = predict(forest.classifier, newdata = test_set[-31], type='prob')

# Predicting Log for AUC Generation

forest.predict_log <- prediction(as.numeric(y_pred),as.numeric(test_set$Class)) 

# Calculating AUC Curve

forest.auc<- performance(forest.predict_log,"auc")
print((forest.auc@y.values[[1]]))

# The AUC value in this case is 0.52, which means the logistic regression is far
# better than Random forest model for this dataset.

# Making the confusion matrix

cm <- table(test_set[, 31], forest.y_pred)
cm

# Creating AUC ROC Curve

roc_curve<- performance(forest.predict_log,"tpr","fpr")  
plot(roc_curve, colorize=T)

# Decision Tree Classifier

tree.model <- rpart(Class~.,smoted_training_set,method='class')
tree.predict <- predict(tree.model, newdata = test_set[-31],type = 'class')
tree.predict_log <- prediction(as.numeric(tree.predict),
                               as.numeric(test_set$Class)) 

# Calculating AUC Curve

tree.auc<- performance(tree.predict_log,"auc")
print((tree.auc@y.values[[1]]))

# The AUC value comes out to be 0.68 which is significantly lower than that of 
# logistic regression.

# Conclusions:

# Logistic Regression works well for this dataset.
# We are having 0.97 of AUC for Logistic Regression which is a significant 
# performance of the model.
# There can be seen a rise in the Fraud transactions during the morning or early
# morning time period.
# Frauds were not of high values in comparison to the normal transactions.
# Maximum amount of fraud transaction was 2500-3000 in the native currency.