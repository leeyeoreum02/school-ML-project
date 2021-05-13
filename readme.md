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

## 1.3 Overview Data Structure (overview.ipynb)
- using pandas attribute
    - .head()
    - .info()
    - .value_counts()
    - .describe()
    - .hist(bins=, figsize=)
- using matplotlib.pyplot for visualization
- histogram of straified age column


# 2. Data Search and Visualization (overview.ipynb)

## 2.1 Analyze correlation
- compare correlation survival with others


# 3. Preprocessing Data (preprocessing.py)

## 3.1 Data Cleaning
- split data (rate=0.2)

====================================================

# 5/13 to do list
- overview.ipynb에서 범주형 데이터 처리 파이프라인 만들기
- overview.ipynb에서 titanic.info() 범주형 데이터 처리 전후 비교
- 데이터 처리 후 상관관계 다시 보기
- preprocessing.py에서 핸즈온 머신러닝 99p 이후 진행하기