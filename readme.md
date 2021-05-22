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

## 2.1 Overview Data
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
- Data Shuffle
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


## 4. Training Model (train.py)

- validation strategy: using stratified k-fold to avoid data bias and overfitting

- (model) SGD Classifier
    - Review: Some regularization didn't show better performance, I think that because dataset is pretty simple. non-regularization, rasso SGD Classifier showed overfitting.
    - score example:
        - non-regularization:
            - accuracy(5-fold validation): [0.68, 0.78, 0.79, 0.76, 0.76]
            - confusion matrix: [[429 120]
                                [ 78 264]]
            - precision score: 0.69
            - recall score: 0.77
            - f1 score: 0.72
            - kaggle submission score(accuracy): 0.71
        - rasso regularization:
            - accuracy(5-fold validation): [0.68, 0.78, 0.79, 0.76, 0.77]
            - confusion matrix: [[442 107]
                                [ 95 247]]
            - precision score: 0.70
            - recall score: 0.72
            - f1 score: 0.71
            - kaggle submission score(accuracy): 0.72
        - ridge regularization:
            - accuracy(5-fold validation): [0.75, 0.73, 0.78, 0.79, 0.73]
            - confusion matrix: [[397 152]
                                [ 71 271]]
            - precision score: 0.64
            - recall score: 0.79
            - f1 score: 0.71
            - kaggle submission score(accuracy): 0.76
        - elastic regularization:
            - accuracy(5-fold validation): [0.75, 0.73, 0.78, 0.79, 0.73]
            - confusion matrix: [[486  63]
                                [125 217]]
            - precision score: 0.78
            - recall score: 0.63
            - f1 score: 0.70
            - kaggle submission score(accuracy): 0.76

- (model) Logistic Regression
- Review: Regularization didn't show better performance as well. it showed overfitting little bit.
- score example:
    - non-regularization:
        - accuracy(5-fold validation): [0.79, 0.75, 0.80, 0.81, 0.81]
        - confusion matrix: [[471  78]
                            [101 241]]
        - precision score: 0.76
        - recall score: 0.70
        - f1 score: 0.73
        - kaggle submission score(accuracy): 0.76
    - ridge regularization(default):
        - accuracy(5-fold validation): [0.79, 0.75, 0.80, 0.82, 0.81]
        - confusion matrix: [[474 75]
                            [101 241]]
        - precision score: 0.76
        - recall score: 0.70
        - f1 score: 0.73
        - kaggle submission score(accuracy): 0.76

- (model) Support Vector Classifier
    - Review: This model showed better performance than SGDClassifier, LogisticRegression but there was overfitting.
    - score example:
        - non-regularization
            - accuracy(5-fold validation): [0.82, 0.81, 0.87, 0.80, 0.82]
            - confusion matrix: [[499  50]
                                [ 99 243]]
            - precision score: 0.83
            - recall score: 0.71
            - f1 score: 0.77
            - kaggle submission score(accuracy): 0.78

- (model) Random Forest Classifier
    - Review: First emsemble model, it showed better performance than single model but there was overfitting little bit.
    - score example:
        - accuracy(5-fold validation): [0.78, 0.79, 0.90, 0.80, 0.80]
        - confusion matrix: [[527  22]
                            [ 26 316]]
        - precision score: 0.93
        - recall score: 0.92
        - f1 score: 0.93
        - kaggle submission score(accuracy): 0.77

- (model) Voting Classifier
    - Review: Ensemble model, it used estimators that are model used before. it showed better performance than single model too. there was no critical overfitting.
    - score example:
        - accuracy(5-fold validation): [0.80, 0.77, 0.83, 0.81, 0.83]
        - confusion matrix: [[501  48]
                            [112 230]]
        - precision score: 0.83
        - recall score: 0.67
        - f1 score: 0.74
        - kaggle submission score(accuracy): 0.78

- (model) Bagging Classifier: didn't use because it's known that it didn't show good performance in small dataset.

- (model) AdaBoost Classifier
    - Review: Ensemble model, it used Decision Tree. it showed worse preformance than single model. there was overfitting.
    - score example:
        - accuracy(5-fold validation): [0.85, 0.81, 0.78, 0.81, 0.77]
        - confusion matrix: [[478  71]
                            [ 79 263]]
        - precision score: 0.79
        - recall score: 0.77
        - f1 score: 0.78
        - kaggle submission score(accuracy): 0.75

- Conclusion: Ensemble model showed better performance than single model usually. But among single model, Support Vector Classifier showed good performance like ensemble model