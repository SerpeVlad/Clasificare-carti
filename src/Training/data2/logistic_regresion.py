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


def logistic_regression(inputpath, outputPath, plotPath, model_path, test_size=0.6):
    # Load data
    data = pd.read_csv(inputpath)  

    # Split data
    X = data.drop(columns=['genre'])
    y = data['genre']


    # Define preprocessing for categorical and numerical columns
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Impute missing values and scale numeric features, OneHotEncode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Create a pipeline with preprocessing and logistic regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
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
        f.write("Dataset: " + inputpath +  "\n\n")
        f.write("Bag of Words\n")

        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

    # Plot learning curve
    plot_learning_curve(model, "Learning Curve (Logistic Regression)", X_train, y_train, cv=3,train_sizes=np.linspace(0.1, 0.5, 3), output_path=plotPath)

def logistic_regression_BoW(inputpath, outputPath, plotPath, model_path, test_size=0.4, C=0.1):
    # Load data
    data = pd.read_csv(inputpath)  
    print('Data loaded')
    # Option 2: Bag of Words
    vectorizer = CountVectorizer(min_df=5, max_df=0.8)

    X = data['title']  # Convert text to numerical features
    y = data['genre']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train, y_train  = smote.fit_resample(X_train, y_train)
    print('Data split')

    # Create a pipeline with preprocessing and logistic regression
    model = Pipeline(steps=[
        ('classifier', LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            C=C))
    ])

    # Hyperparameter tuning
    param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]}
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
       
        f.write("Bag of Words\n")
        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")
    print('Report saved')
    # Plot learning curve
    plot_learning_curve(best_model, "Learning Curve (Logistic Regression)", X_train, y_train, cv=3, train_sizes=np.linspace(0.1, 0.5, 3), output_path=plotPath)
    print('Plot saved')

    # Plot learning curve
  
def logistic_regression_TFIDF(inputpath, outputPath, test_size=0.4, C=1.0):
    # Load data
    data = pd.read_csv(inputpath)  
    print('Data loaded')
    # Option 2: TF-IDF
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8)

    X = data['summary']  # Convert text to numerical features
    y = data['genre']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train, y_train  = smote.fit_resample(X_train, y_train)
    print('Data split')

    # Create a pipeline with preprocessing and logistic regression
    model = Pipeline(steps=[
        ('classifier', LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            C=C))
    ])

    # Hyperparameter tuning
    param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy') 
    grid_search.fit(X_train, y_train)
    results = grid_search.cv_results_
    sorted_indices = np.argsort(results['mean_test_score'])
    lst =  [-1, -2, 0] #-1 best model, -2 second best model, 0 worst model
    for i in lst: 
        best_model = sorted_indices[i]
        print('Model trained')
        if i == -1:
            model_path = outputPath + '_best_model.joblib'
        elif i == -2:
            model_path = outputPath + '_second_best_model.joblib'
        else:
            model_path = outputPath + '_worst_model.joblib'
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
            f.write("C:" + str(grid_search.best_params_['classifier__C']) + "\n")
            if i == -1:
                f.write("Best model\n")
            elif i == -2:
                f.write("Second best model\n")
            else:
                f.write("Worst model\n")
            f.write("Dataset: " + inputpath +  "\n\n")
        
            f.write("TF-IDF\n")
            f.write("Acuratețea modelului: " + str(accuracy) + "\n")
            f.write("\n\nRaport de clasificare:\n"+ report + "\n")
        print('Report saved')
        # Plot learning curve
        if i == -1:
            plotPath = outputPath + '_best_model.png'
            plot_learning_curve(best_model, "Best Logistic Regression", X_train, y_train, cv=3, output_path=plotPath)
        elif i == -2:
            plotPath = outputPath + '_second_best_model.png'
            plot_learning_curve(best_model, "Second Best Logistic Regression", X_train, y_train, cv=3, output_path=plotPath)
        else:
            plotPath = outputPath + '_worst_model.png'
            plot_learning_curve(best_model, "Worst Logistic Regression", X_train, y_train, cv=3, output_path=plotPath)
        print('Plot saved')
        print('Done with model ' + str(i))



inputpath2 = 'Datasets//data2.csv'
outputPath6 = 'Raports//FinalRaports//logistic_regression_TF_IDF'
logistic_regression_TFIDF(inputpath2, outputPath6, test_size=0.2)

         