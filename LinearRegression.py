import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.regression import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix


data = pd.read_csv('wine.data')

data.columns = [  'name'
                 ,'alcohol'
             	,'malicAcid'
             	,'ash'
            	,'ashalcalinity'
             	,'magnesium'
            	,'totalPhenols'
             	,'flavanoids'
             	,'nonFlavanoidPhenols'
             	,'proanthocyanins'
            	,'colorIntensity'
             	,'hue'
             	,'od280_od315'
             	,'proline'
                ]

classes=['Wine 1','Wine 2','Wine 3']

y = data['alcohol' ]
X = data.drop('alcohol' ,axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=94)

# pd.plotting.scatter_matrix(data,
#                             figsize=(10,10), 
#                             diagonal='kde', 
#                             s=40,                            
#                             alpha=0.5,
#                             marker='*');

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lin = LinearRegression()
lin.fit(X_train_scaled, y_train)
lin_pred_train = lin.predict(X_train_scaled)
lin_pred_test = lin.predict(X_test_scaled)

print("MSE for train set: %.4f" % mean_squared_error(y_train, lin_pred_train))
print("MSE for test set: %.4f" % mean_squared_error(y_test, lin_pred_test))
lin_coeff = pd.DataFrame(
    {'coef': lin.coef_, 'coef_abs': np.abs(lin.coef_)}, index=X.columns)
lin_coeff.sort_values(by='coef_abs', ascending=False)


