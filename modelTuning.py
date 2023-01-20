from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing, model_selection
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split, cross_val_score, \
    ShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt


class OptimizeModel():

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33,
                                                                                random_state=42)

    def logresTuned(self):
        # grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
        solvers = ['newton-cg', 'lbfgs', 'liblinear']
        penalty = ["l1", "l2"]
        c_values = [100, 10, 1.0, 0.1, 0.01]
        logreg = LogisticRegression()
        grid = dict(solver=solvers, penalty=penalty, C=c_values)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        logreg_cv = GridSearchCV(logreg, grid, cv=cv).fit(self.X_train, self.y_train)
        logreg_best_param = logreg_cv.best_params_
        logreg_best = logreg_cv.best_estimator_
        logreg_tuned = logreg_best.fit(self.X_train, self.y_train)
        y_pred_tuned = logreg_tuned.predict(self.X_test)
        logreg_skor = accuracy_score(y_pred_tuned, self.y_test)
        print('LogisticTuned Model Dogrulugu:', logreg_skor)
        print("LogisticTuned Classification report")
        cf_matrix_logreg = confusion_matrix(y_pred_tuned, self.y_test)
        sns.heatmap(cf_matrix_logreg, annot=True, cbar=False, fmt='g')
        plt.title("LogisticTuned Model Doğruluğu:" + str(logreg_skor))
        print(classification_report(y_pred_tuned, self.y_test))
        plt.show()
        return logreg_tuned

    def knnTuned(self):
        n_neighbors = range(1, 21, 2)
        weights = ['uniform', 'distance']
        metric = ['euclidean', 'manhattan', 'minkowski']
        knn = KNeighborsClassifier()
        grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        knn_cv = GridSearchCV(knn, grid, cv=cv).fit(self.X_train, self.y_train)
        knn_best_param = knn_cv.best_params_
        knn_best = knn_cv.best_estimator_
        knn_tuned = knn_best.fit(self.X_train, self.y_train)
        y_pred_tuned = knn_tuned.predict(self.X_test)
        knn_skor = accuracy_score(y_pred_tuned, self.y_test)
        print('KNNtuned Model Dogrulugu:', knn_skor)
        print("KNNtuned Classification report")
        cf_matrix_knn = confusion_matrix(y_pred_tuned, self.y_test)
        sns.heatmap(cf_matrix_knn, annot=True, cbar=False, fmt='g')
        plt.title("KNNtuned Model Doğruluğu:" + str(knn_skor))
        print(classification_report(y_pred_tuned, self.y_test))
        plt.show()
        return knn_tuned

    def svmTuned(self):
        kernel = ['poly', 'rbf', 'sigmoid']
        C = [50, 10, 1.0, 0.1, 0.01]
        gamma = ['scale']
        svm = SVC()
        grid = dict(kernel=kernel, C=C, gamma=gamma)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        svm_cv = GridSearchCV(svm, grid, cv=cv).fit(self.X_train, self.y_train)
        svm_best_param = svm_cv.best_params_
        svm_best = svm_cv.best_estimator_
        svm_tuned = svm_best.fit(self.X_train, self.y_train)
        y_pred_tuned = svm_tuned.predict(self.X_test)
        svm_skor = accuracy_score(y_pred_tuned, self.y_test)
        print('svmTuned Model Dogrulugu:', svm_skor)
        print("svmTuned Classification report")
        cf_matrix_svm = confusion_matrix(y_pred_tuned, self.y_test)
        sns.heatmap(cf_matrix_svm, annot=True, cbar=False, fmt='g')
        plt.title("svmTuned Model Doğruluğu:" + str(svm_skor))
        print(classification_report(y_pred_tuned, self.y_test))
        plt.show()
        return svm_tuned

    def randomForestTuned(self):
        n_estimators = [10, 100, 1000]
        max_features = ['sqrt', 'log2']
        rf = RandomForestClassifier()
        grid = dict(n_estimators=n_estimators, max_features=max_features)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        rf_cv = GridSearchCV(rf, grid, cv=cv).fit(self.X_train, self.y_train)
        rf_best_param = rf_cv.best_params_
        rf_best = rf_cv.best_estimator_
        rf_tuned = rf_best.fit(self.X_train, self.y_train)
        y_pred_tuned = rf_tuned.predict(self.X_test)
        rf_skor = accuracy_score(y_pred_tuned, self.y_test)
        print('randomForestTuned Model Dogrulugu:', rf_skor)
        print("randomForestTuned Classification report")
        cf_matrix_rf = confusion_matrix(y_pred_tuned, self.y_test)
        sns.heatmap(cf_matrix_rf, annot=True, cbar=False, fmt='g')
        plt.title("randomForestTuned Model Doğruluğu:" + str(rf_skor))
        print(classification_report(y_pred_tuned, self.y_test))
        plt.show()
        return rf_tuned


    def decisionTreeTuned(self):
        max_depth = [1, 3, 5, 8, 10]
        min_samples_split = [2, 3, 5, 10, 20, 50]
        cart = DecisionTreeClassifier()
        grid = dict(max_depth=max_depth, min_samples_split=min_samples_split)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        cart_cv = GridSearchCV(cart, grid, cv=cv).fit(self.X_train, self.y_train)
        cart_best_param = cart_cv.best_params_
        cart_best = cart_cv.best_estimator_
        cart_tuned = cart_best.fit(self.X_train, self.y_train)
        y_pred_tuned = cart_tuned.predict(self.X_test)
        cart_skor = accuracy_score(y_pred_tuned, self.y_test)
        print('decisionTreeTuned Model Dogrulugu:', cart_skor)
        print("decisionTreeTuned Classification report")
        cf_matrix_cart = confusion_matrix(y_pred_tuned, self.y_test)
        sns.heatmap(cf_matrix_cart, annot=True, cbar=False, fmt='g')
        plt.title("decisionTreeTuned Model Doğruluğu:" + str(cart_skor))
        print(classification_report(y_pred_tuned, self.y_test))
        plt.show()
        return cart_tuned

    def GBMTuned(self):
        n_estimators = [10, 100, 1000]
        learning_rate = [0.001, 0.01, 0.1]
        subsample = [0.5, 0.7, 1.0]
        max_depth = [3, 7, 9]
        gbm = GradientBoostingClassifier()
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        gbm_cv = GridSearchCV(gbm, grid, cv=cv).fit(self.X_train, self.y_train)
        gbm_best_param = gbm_cv.best_params_
        gbm_best = gbm_cv.best_estimator_
        gbm_tuned = gbm_best.fit(self.X_train, self.y_train)
        y_pred_tuned = gbm_tuned.predict(self.X_test)
        gbm_skor = accuracy_score(y_pred_tuned, self.y_test)
        print('GBMTuned Model Dogrulugu:', gbm_skor)
        print("GBMTuned Classification report")
        cf_matrix_gbm = confusion_matrix(y_pred_tuned, self.y_test)
        sns.heatmap(cf_matrix_gbm, annot=True, cbar=False, fmt='g')
        plt.title("GBMTuned Model Doğruluğu:" + str(gbm_skor))
        print(classification_report(y_pred_tuned, self.y_test))
        plt.show()
        return gbm_tuned

    def XGBMTuned(self):
        n_estimators = [100, 500, 1000]
        learning_rate = [0.001, 0.01, 0.1]
        subsample = [0.6, 0.8, 1.0]
        max_depth = [3, 5, 7]
        xgbm = XGBClassifier()
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        xgbm_cv = GridSearchCV(xgbm, grid, cv=cv).fit(self.X_train, self.y_train)
        xgbm_best_param = xgbm_cv.best_params_
        xgbm_best = xgbm_cv.best_estimator_
        xgbm_tuned = xgbm_best.fit(self.X_train, self.y_train)
        y_pred_tuned = xgbm_tuned.predict(self.X_test)
        xgbm_skor = accuracy_score(y_pred_tuned, self.y_test)
        print('XGBMTuned Model Dogrulugu:', xgbm_skor)
        print("XGBMTuned Classification report")
        cf_matrix_xgbm = confusion_matrix(y_pred_tuned, self.y_test)
        sns.heatmap(cf_matrix_xgbm, annot=True, cbar=False, fmt='g')
        plt.title("XGBMTuned Model Doğruluğu:" + str(xgbm_skor))
        print(classification_report(y_pred_tuned, self.y_test))
        plt.show()
        return xgbm_tuned

    def LGBMTuned(self):
        n_estimators = [100, 500, 1000]
        learning_rate = [0.001, 0.01, 0.1]
        subsample = [0.6, 0.8, 1.0]
        max_depth = [3, 5, 7]
        lgbm = LGBMClassifier()
        grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        lgbm_cv = GridSearchCV(lgbm, grid, cv=cv).fit(self.X_train, self.y_train)
        lgbm_best_param = lgbm_cv.best_params_
        lgbm_best = lgbm_cv.best_estimator_
        lgbm_tuned = lgbm_best.fit(self.X_train, self.y_train)
        y_pred_tuned = lgbm_tuned.predict(self.X_test)
        lgbm_skor = accuracy_score(y_pred_tuned, self.y_test)
        print('LGBMTuned Model Dogrulugu:', lgbm_skor)
        print("LGBMTuned Classification report")
        cf_matrix_lgbm = confusion_matrix(y_pred_tuned, self.y_test)
        sns.heatmap(cf_matrix_lgbm, annot=True, cbar=False, fmt='g')
        plt.title("LGBMTuned Model Doğruluğu:" + str(lgbm_skor))
        print(classification_report(y_pred_tuned, self.y_test))
        plt.show()
        return lgbm_tuned


    def MLPCTuned(self):
        alpha = [1, 5, 0.1, 0.01, 0.03, 0.005, 0.0001]
        hidden_layer_sizes = [(10, 10), (100, 100, 100), (100, 100), (3, 5)]
        activation = ["logistic", "relu", "Tanh"]
        learning_rate = ["constant", "invscaling", "adaptive"]
        MLPC = MLPClassifier()
        grid = dict(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                    learning_rate=learning_rate)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        mlpc_cv = GridSearchCV(MLPC, grid, cv=cv).fit(self.X_train, self.y_train)
        mlpc_best_param = mlpc_cv.best_params_
        mlpc_best = mlpc_cv.best_estimator_
        mlpc_tuned = mlpc_best.fit(self.X_train, self.y_train)
        y_pred_tuned = mlpc_tuned.predict(self.X_test)
        mlpc_skor = accuracy_score(y_pred_tuned, self.y_test)
        print('MLPCTuned Model Dogrulugu:', mlpc_skor)
        print("MLPCTuned Classification report")
        cf_matrix_mlpc = confusion_matrix(y_pred_tuned, self.y_test)
        sns.heatmap(cf_matrix_mlpc, annot=True, cbar=False, fmt='g')
        plt.title("MLPCTuned Model Doğruluğu:" + str(mlpc_skor))
        print(classification_report(y_pred_tuned, self.y_test))
        plt.show()
        return mlpc_tuned
