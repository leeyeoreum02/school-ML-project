import pandas as pd
import numpy as np
from load_data import load_titanic_data

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_preprocessed_data(mode):
    titanic = load_titanic_data(mode)
    if mode == 'train':
        shuffled_titanic = titanic.sample(frac=1)
        shuffled_titanic = shuffled_titanic.drop('PassengerId', axis=1)
        # print(shuffled_titanic.head())
        # print(shuffled_titanic.head())
        x_data, t_data = shuffled_titanic.drop('Survived', axis=1), \
                        shuffled_titanic['Survived'].copy()

        x_data_num = x_data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
        num_attribs = list(x_data_num)
        cat_attribs = ['Sex']
        # print('attribs:', num_attribs + cat_attribs)

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler()),
        ])
            
        full_pipeline = ColumnTransformer([
            ('num', num_pipeline, num_attribs),
            ('cat', OrdinalEncoder(), cat_attribs),
        ])

        x_data_prepared = full_pipeline.fit_transform(x_data)
        t_data = np.array(t_data, dtype=np.uint8)
        
        return x_data_prepared, t_data
    elif mode == 'test':
        titanic_num = titanic.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
        num_attribs = list(titanic_num)
        cat_attribs = ['Sex']
        # print(titanic_num.head())

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler()),
        ])
            
        full_pipeline = ColumnTransformer([
            ('num', num_pipeline, num_attribs),
            ('cat', OrdinalEncoder(), cat_attribs),
        ])

        titanic_prepared = full_pipeline.fit_transform(titanic)

        return titanic_prepared


# if __name__ == '__main__':
#     xdata, tdata = load_preprocessed_data('train')
#     print(xdata.shape, tdata.shape)
#     data = load_preprocessed_data('test')
#     print(data.shape)
