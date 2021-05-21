# 1. Get Data

## 1.1 Make Virtual Environment
[venv: mlproject]
- please refer to requirement.txt to check libraries used in this project

## 1.2 Download Data
- Data
    - gender_submission.csv (submission)
    - test.csv (test data)
    - train.csv (train data)
- Download
    - using Kaggle API


# 2. Data Analyze (overview.ipynb)

## 2.1 Overview Data (overview.ipynb)
- Label
    - Survived
- Column that has null value -> have to process
    - Age
    - Cabin
    - Embarked
- Categorical column -> have to encoding
    - Pclass: already ordinal encoded
    - Sex
    - Embarked
- Correlation of columns
    - direct proportion with label('Survived'):
        - Fare
        - Pclass: In fact, the larger class of Pclass, the lower value it has, therefore correlation label with Pclass is actually direcr proportion / Fare and Pclass are direct proportion
    - inverse proportion
        - Parch: It looks like direct propotion in corr matrix because of outlier, In fact, we can check out that's not true in scatter matrix
        - SibSp
        - Age


# 3. Preprocessing Data (preprocessing.py)

## 3.1 Data Cleaning
- Delete unnecessary column
    - Name: no matter to train good model
    - Ticket: too many text values to encode
    - Cabin: too many null and text values
    - Embarked: I think it isn't related to survival so much
- Process null values
    - Age: fill null with median
- Standardize
- Process categorical column
    - Sex: female:0, male:1 (used OrdinalEncoder)

## 4. Training Model