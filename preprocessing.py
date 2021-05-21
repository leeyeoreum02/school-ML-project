import pandas as pd
import numpy as np
from load_data import load_titanic_data

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_preprocessed_data():
    titanic = load_titanic_data('train')
    x_data, t_data = titanic.drop('Survived', axis=1), titanic['Survived'].copy()

    x_data_num = x_data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    num_attribs = list(x_data_num)
    cat_attribs = ['Sex']
    print(num_attribs + cat_attribs)

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

if __name__ == '__main__':
    x_data, t_data = load_preprocessed_data()
    print(x_data[:5], t_data[:5])
