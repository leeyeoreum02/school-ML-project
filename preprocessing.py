import pandas as pd
import numpy as np
from load_data import load_titanic_data
from sklearn.impute import SimpleImputer


titanic = load_titanic_data('train')
x_data, t_data = titanic.drop('Survived', axis=1), titanic['Survived'].copy()

imputer = SimpleImputer(strategy='median')
x_data_num = x_data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
imputer.fit(x_data_num)