import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, learning_curve
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from utils import plot_learning_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import SMOTE

def logistic_regression_BoW(inputpath, outputPath, plotPath, model_path, test_size=0.4, C=1.0):
    # Load data
    data = pd.read_csv(inputpath)  

    # Option 1: Bag of Words
    vectorizer = CountVectorizer()

    
    X = vectorizer.fit_transform(data['summary'])  # Convert text to numerical features
    y = data['genre']


    # Define preprocessing for categorical and numerical columns
    
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Create a pipeline with preprocessing and logistic regression
    model = Pipeline(steps=[
        #('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', C=C))
    ])

    # Fit the model
    model.fit(X_train, y_train)

    dump(model, model_path)

    # Predictions and evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    # Save the classification report
    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Train data procentage:'+ str(X_train.shape[0] / data.shape[0]) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n")
        f.write("C:" + str(C) + "\n")
        f.write("Dataset: " + inputpath +  "\n\n")
       
        f.write("Bag of Words\n")
        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

    # Plot learning curve
    plot_learning_curve(model, "Learning Curve (Logistic Regression)", X_train, y_train, cv=3,train_sizes=np.linspace(0.1, 0.5, 3), output_path=plotPath)

def logistic_regression_TFIDF(inputpath, outputPath, plotPath, model_path, test_size=0.4, C=1.0):
    # Load data
    data = pd.read_csv(inputpath)  
    print('Data loaded')
    # Option 2: TF-IDF
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8)

    X = vectorizer.fit_transform(data['summary'])  # Convert text to numerical features
    y = data['genre']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_train, y_train  = smote.fit_resample(X_train, y_train)
    print('Data split')

    # Create a pipeline with preprocessing and logistic regression
    model = Pipeline(steps=[
        ('classifier', LogisticRegression(
            C=C))
    ])

    # Hyperparameter tuning
    # Add max iter in param_grid
    param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10], 'classifier__max_iter': [1000]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy') 
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print('Model trained')

    dump(best_model, model_path)
    print('Model saved')
    # Predictions and evaluation
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    # Save the classification report
    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Train data procentage:'+ str(X_train.shape[0] / data.shape[0]) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n")
        f.write("Best C:" + str(grid_search.best_params_['classifier__C']) + "\n")
        f.write("Dataset: " + inputpath +  "\n\n")
       
        f.write("TF-IDF\n")
        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")
    print('Report saved')
    # Plot learning curve
    plot_learning_curve(best_model, "Learning Curve (Logistic Regression)", X_train, y_train, cv=3, train_sizes=np.linspace(0.1, 0.5, 3), output_path=plotPath)
    print('Plot saved')

inputpath2 = 'Datasets//data2_simplified.csv'
outputPath4 = 'Raports//data2_simplified//logistic_regression_TFIDF.txt'
plotPath4 = 'Raports//data2_simplified//logistic_regression_TFIDF.png'
modelPath4 = 'Models//data2_simplified//logistic_regression_TFIDF.joblib'
#logistic_regression_TFIDF(inputpath2, outputPath4, plotPath4, modelPath4, test_size=0.2, C=0.01)

inputpath2 = 'Datasets//data2_simplified.csv'
outputPath4 = 'Raports//data2_simplified//logistic_regression_BoW_C001.txt'
plotPath4 = 'Raports//data2_simplified//logistic_regression_BoW_C001.png'
modelPath4 = 'Models//data2_simplified//logistic_regression_BoW_C001.joblib'
#logistic_regression_BoW(inputpath2, outputPath4, plotPath4, modelPath4, test_size=0.2, C=0.01)
outputPath5 = 'Raports//data2_simplified//logistic_regression_BoW_C0001.txt'
plotPath5 = 'Raports//data2_simplified//logistic_regression_BoW_C0001.png'
modelPath5 = 'Models//data2_simplified//logistic_regression_BoW_C0001.joblib'
#logistic_regression_BoW(inputpath2, outputPath5, plotPath5, modelPath5, test_size=0.2, C=0.001)
outputPath6 = 'Raports//data2_simplified//logistic_regression_BoW_C1_2.txt'
plotPath6 = 'Raports//data2_simplified//logistic_regression_BoW_C1_2.png'
modelPath6 = 'Models//data2_simplified//logistic_regression_BoW_C1_2.joblib'
#logistic_regression_BoW(inputpath2, outputPath6, plotPath6, modelPath6, test_size=0.2, C=1.0)
outputPath7 = 'Raports//data2_simplified//logistic_regression_BoW_C5_2.txt'
plotPath7 = 'Raports//data2_simplified//logistic_regression_BoW_C5_2.png'
modelPath7 = 'Models//data2_simplified//logistic_regression_BoW_C5_2.joblib'
#logistic_regression_BoW(inputpath2, outputPath7, plotPath7, modelPath7, test_size=0.2, C=5.0)

data = pd.read_csv(inputpath2)  
print('Data loaded')
# Option 2: TF-IDF
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8)
X = vectorizer.fit_transform(data['summary'])  # Convert text to numerical features
joblib.dump(vectorizer, 'Models//data2_simplified//vectorizerTF_IDF.joblib')

vectorizer = CountVectorizer()    
X = vectorizer.fit_transform(data['summary'])  # Convert text to numerical features
joblib.dump(vectorizer, 'Models//data2_simplified//vectorizerBoW.joblib')