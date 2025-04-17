from joblib import dump
from matplotlib import pyplot as plt
import numpy as np
#from sklearn.base import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder #for plot learning curve
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class __main__():

    def __init__(self,inputpath='C:\\Users\\serpe\\Desktop\\Practica\\Datasets\\data2.csv' ,test_size=0.6,save_raport=True, save_learning_curve=True, save_model=True):
        self.inputpath = inputpath
        self.test_size = test_size
        self.data = pd.read_csv(self.inputpath) #
        self.lst_of_models = [] #
        self.vectorizer = TfidfVectorizer(min_df=5, max_df=0.8) #
        self.save_raport = save_raport # T/F
        self.save_learning_curve = save_learning_curve # T/F
        self.save_model = save_model # T/F
        self.x_train, self.x_test, self.y_train, self.y_test = self.__split_data__() #



    def __load_data__(self):
        self.data = pd.read_csv(self.inputpath)

    def __split_data__(self):
        x = self.data['summary']
        y = self.data['genre']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, stratify=y, random_state=42)

        x_train = self.vectorizer.fit_transform(x_train)
        x_test = self.vectorizer.transform(x_test)

        smote = SMOTE(random_state=42)
        X_train, y_train  = smote.fit_resample(X_train, y_train)
        return x_train, x_test, y_train, y_test or x_train, y_train

    def __preprocess_data__(self):
        numeric_features = self.data.select_dtypes(include=['float64', 'int64']).columns
        categorical_features = self.data.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
                ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
            ])
        
        return preprocessor
    
    def __train_models__(self):
        print('Data split')

        preprocessor = self.__preprocess_data__()
        print('Data preprocess')
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])

        # Hyperparameter tuning
        param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy') 
        grid_search.fit(self.x_train, self.y_train)
        data = self.__load_data__()
        print('Models trained')
        lst_modele_raport_antrenare = []
        for model in grid_search:
            str = ''
            predictions = model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, predictions)
            report = classification_report(self.y_test, predictions, zero_division=0)
            str += 'Train data procentage:'+ str(self.x_train.shape[0] / data.shape[0]) + "\n" 
            str += 'Test data procentage:'+ str(self.x_test.shape[0] / data.shape[0]) + "\n" 
            str += "Best C:" + str(grid_search.best_params_['classifier__C']) + "\n"
            str += "Dataset: " + self.inputpath +  "\n\n" 
       
            str += "TF-IDF\n" 
            str += "Acuratețea modelului: " + str(accuracy) + "\n" 
            str += "\n\nRaport de clasificare:\n"+ report + "\n\n"
            lst_modele_raport_antrenare.append([model, str])
        return lst_modele_raport_antrenare

    def __test_models__(self,lst=None, inputpathForTest='C:\\Users\\serpe\\Desktop\\Practica\\Datasets\\TestForDt2Dt2Sim.csv'):
        if lst == None:
            lst_models_training_report = self.__train_models__()
        else:
            lst_models_training_report = lst
        outputpath = "C:\\Users\\serpe\\Desktop\\Practica\\Raports\\FinalRaports"
        data_test = pd.read_csv("C:\\Users\\serpe\\Desktop\\Practica\\Datasets\\TestForDt2Dt2Sim.csv")
        x = self.vectorizer.transform(data_test['summary'])
        truth = data_test['genre']
        i = 0
        for lst in lst_models_training_report:
            model = lst[0]
            raport = lst[1]
            prediction = model.predict(x)
            cm = confusion_matrix(truth, prediction)
            raport += str(cm) + "\n"
            i += 1
            if self.save_raport:
                self.__save_classification_report__(raport, i)
            if self.save_learning_curve:
                plot_learning_curve(model, "Learning Curve (Logistic Regression)", x_train, y_train, cv=3, output_path="C:\\Users\\serpe\\Desktop\\Practica\\Raports\\FinalRaports\\LR" + str(i))
            if self.save_model:
                self.__save_model__(model, outputpath + "Modele\\LR"+str(i)+".joblib" )
            
            
    def __save_model__(self, model, model_path):
        dump(model, model_path) 

    def __save_classification_report__(self, str, i):
        outputpath = "C:\\Users\\serpe\\Desktop\\Practica\\Raports\\FinalRaports"
        with open(outputpath +"\\model" + str(i) + '.txt', 'w', encoding='utf-8') as f:
                f.write('model '+ str(i) + '\n')
                f.write(str)

    def __save_learning_curve__(self, model, i):
        plot_learning_curve(model, "Logistic Regression" + str(i), self.x_train, self.y_train, cv=3, output_path="C:\\Users\\serpe\\Desktop\\Practica\\Raports\\FinalRaports\\LR" + str(i))

    
    def __all(self):
        lst = self.__train_models__()
        for [model, raport] in lst:
            pass

make = __main__() #make me a confusiom matrix in 2 dimensions with the models


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5), output_path='learning_curve.png'):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if cv is None:
        cv = StratifiedKFold(n_splits=5)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(output_path)
    plt.close()

#lst = make.__train_models__() #toate modele antrenate (ca avem finetuning ca bazeti)
#make.__test_models__() # daca nu apelezi train_models inainte, atunci o face el automat

