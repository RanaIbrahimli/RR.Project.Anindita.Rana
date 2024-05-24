# RR_FraudDetection

## Project Overview

This project builds upon the initial work on classifying fraudulent credit card transactions. By incorporating advanced data processing techniques, including Exploratory Data Analysis (EDA), feature engineering, and logistic regression, we aim to improve the predictive accuracy and provide deeper insights into the data patterns.

## Objectives

- To enhance the initial fraud detection model with more sophisticated data analysis and machine learning techniques.
- To explore the underlying patterns in the data through EDA.
- To engineer new features that could improve the model's performance.
- To implement logistic regression to compare its effectiveness against other models.

## Methodologies

### Data Preparation and EDA

- **Exploratory Data Analysis**: Conduct a thorough analysis to understand the distributions and relationships of features.
- **Visualization**: Use libraries like `ggplot2` and `corrplot` to visualize complex interactions and correlations.

### Dataset Reference

The data used in this project is hosted on Kaggle. You can access the dataset here: [Link to Kaggle Dataset](https://www.kaggle.com/)

### Installation and Setup

Ensure you have Python installed and execute the following to install required packages:

```r
install.packages(c("ROCR", "ggplot2", "corrplot", "caTools", "class", "randomForest", "pROC"))
