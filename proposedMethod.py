################################################
# End-to-End Machine Learning Pipeline I
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################################################
# 1. Exploratory Data Analysis
################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

from preprep import *
from YontemML import YontemML
from modelTuning import OptimizeModel
from handlingImbalancedData import ImbalanceDuzenle
df = pd.read_csv("data/Churn_Modelling.csv")

df.head(10)

df.isnull().sum()

check_df(df)

df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

df.head()

#Korelasyon grafiği çizdirelim
plt.figure(figsize=(16,12))
sns.heatmap(df.corr(), cmap="bwr", annot=True)
plt.show()
plt.savefig('correlationmatrix.tiff')



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

df_out.info()

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

ImbalanceDuzenle=ImbalanceDuzenle(X,y)

X_ros,y_ros= ImbalanceDuzenle.ROS()

X_smote,y_smote=ImbalanceDuzenle.Smote()

X_trainROS, X_testROS, y_trainROS, y_testROS = train_test_split(X_ros, y_ros, test_size=0.33, random_state=42)
X_trainSMOTE, X_testSMOTE, y_trainSMOTE, y_testSMOTE = train_test_split(X_smote, y_smote, test_size=0.33, random_state=42)

# group / ensemble of models
estimator = []
estimator.append(('RF',RandomForestClassifier()))
estimator.append(('GBM', GradientBoostingClassifier()))
estimator.append(('LightGBM', LGBMClassifier()))




# Voting Classifier with hard voting
vot_hardROS = VotingClassifier(estimators=estimator, voting='hard').fit(X_trainROS,y_trainROS)
vot_hardSMOTE = VotingClassifier(estimators=estimator, voting='hard').fit(X_trainSMOTE,y_trainSMOTE)

y_predvot_hardROS=y_pred = vot_hardROS.predict(X_testROS)
y_predvot_hardSMOTE=y_pred = vot_hardSMOTE.predict(X_testSMOTE)

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

vot_hardROSScore= accuracy_score(y_predvot_hardROS,y_testROS)
print('vot_hardROS Model Dogrulugu:', vot_hardROSScore)
print("vot_hardROS Classification report")
cf_matrix_knn=confusion_matrix(y_predvot_hardROS,y_test)
sns.heatmap(cf_matrix_knn,annot=True,cbar=False, fmt='g')
plt.title("vot_hardROS Model Doğruluğu:"+str(vot_hardROSScore))
print(classification_report(y_predvot_hardROS,y_test))
plt.show()





# Voting Classifier with soft voting
vot_soft = VotingClassifier(estimators=estimator, voting='soft')
vot_soft.fit(X_train, y_train)
y_pred = vot_soft.predict(X_test)








cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
print(f"F1Score: {cv_results['test_f1'].mean()}")
print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")



######################################################
# 4. Automated Hyperparameter Optimization
######################################################

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}



######################################################
# 6. Prediction for a New Observation
######################################################

X.columns
random_user = X.sample(1, random_state=45)
voting_clf.predict(random_user)

joblib.dump(voting_clf, "voting_clf2.pkl")

new_model = joblib.load("voting_clf2.pkl")
new_model.predict(random_user)











