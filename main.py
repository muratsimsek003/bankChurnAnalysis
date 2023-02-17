import pandas as pd
import warnings
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from preprep import *
from YontemML import YontemML
from modelTuning import OptimizeModel
from handlingImbalancedData import ImbalanceDuzenle

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("data/Churn_Modelling.csv")

df.head(10)

df.isnull().sum()

check_df(df)

df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

df.head()

df["Exited"].value_counts()

cat_cols, num_cols, cat_but_car=grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

for col in num_cols:
    print(col)




df.info()

df.head(10)

for col in num_cols:
    print(col, check_outlier(df, col))



sns.boxplot(x = df.Age)
plt.show()

low_limit, up_limit= outlier_thresholds(df,  "Age")

df_out=df[~((df["Age"] < (low_limit)) | (df["Age"]> (up_limit)))]

sns.boxplot(x = df_out.Age)
plt.show()

df_out.head()

lbe=LabelEncoder()
df_out["Gender"]=lbe.fit_transform(df_out["Gender"])
df_out["Geography"]=lbe.fit_transform(df_out["Geography"])

df_out.head()


df_out["Balance"]=df_out["Balance"].astype(int)
df_out["EstimatedSalary"]=df_out["EstimatedSalary"].astype(int)

df_out.head()


y=df_out["Exited"]
X=df_out.drop(["Exited"],axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


yontemML=YontemML(X,y)

"""
Kullanılan yöntemler

Logistic Regression
SVM
KNN
Decision Tree
Random Forest
GBM
Light GBM
XGBoost
MLPC 

"""

loj_model=yontemML.Logistic()
svm_model=yontemML.Svm()
knn_model=yontemML.Knn()
cart_model=yontemML.Decisiontree()
rf_model=yontemML.Randomforest()
gbm_model=yontemML.Gbm()
lgbm_model=yontemML.LightGBM()
xgbm_model=yontemML.Xgboost()
mlpc_model=yontemML.Mlpc()


ImbalanceDuzenle=ImbalanceDuzenle(X,y)


X_ros,y_ros= ImbalanceDuzenle.ROS()
ROSyontemML=YontemML(X_ros,y_ros)

ROSloj_model=ROSyontemML.Logistic()
ROSsvm_model=ROSyontemML.Svm()
ROSknn_model=ROSyontemML.Knn()
ROScart_model=ROSyontemML.Decisiontree()
ROSrf_model=ROSyontemML.Randomforest()
ROSgbm_model=ROSyontemML.Gbm()
ROSlgbm_model=ROSyontemML.LightGBM()
ROSxgbm_model=ROSyontemML.Xgboost()
ROSmlpc_model=ROSyontemML.Mlpc()

