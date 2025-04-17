from joblib import dump
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils import plot_learning_curve

def random_forest_TF_IDF(inputpath, outputPath, plotPath, model_path, test_size=0.4):
    data = pd.read_csv(inputpath)

# 1. Preprocessing the Text Data
# Vectorize the text using Bag of Words or TF-IDF
# Option 1: Bag of Words
# vectorizer = CountVectorizer()

    # Option 2: TF-IDF
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(data['summary'])  # Convert text to numerical features
    y = data['genre']

    # 2. Split Data into Train and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 3. Train a Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    dump(model, model_path)

    # Plot learning curve
    plot_learning_curve(model, "Learning Curve (Random Forest)", X_train, y_train, cv=3,train_sizes=np.linspace(0.1, 0.5, 3), output_path=plotPath)

    # Predictions and evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

   


    # Save the classification report
    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Train data procentage:'+ str(X_train.shape[0] / data.shape[0]) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n")
        f.write("Dataset: " + inputpath +  "\n\n")
        f.write("TF-IDF\n")

        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

def random_forest_BoW(inputpath, outputPath, plotPath, model_path, test_size=0.4):
    data = pd.read_csv(inputpath)

# 1. Preprocessing the Text Data
# Vectorize the text using Bag of Words or TF-IDF
# Option 1: Bag of Words
    vectorizer = CountVectorizer()

    

    X = vectorizer.fit_transform(data['summary'])  # Convert text to numerical features
    y = data['genre']

    # 2. Split Data into Train and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 3. Train a Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    dump(model, model_path)

    # Plot learning curve
    plot_learning_curve(model, "Learning Curve (Random Forest)", X_train, y_train, cv=3,train_sizes=np.linspace(0.1, 0.5, 3), output_path=plotPath)

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

inputpath = 'Datasets//data2.csv'
outputPath = 'Raports//data2//random_forest_TF_IDF.txt'
plotPath = 'Raports//data2//random_forest_TF_IDF.png'
modelPath = 'Models//data2//random_forest_TF_IDF.joblib'
#random_forest_TF_IDF(inputpath, outputPath, plotPath, modelPath, test_size=0.2)
outputPath2 = 'Raports//data2//random_forest_TF_IDF2.txt'
plotPath2 = 'Raports//data2//random_forest_TF_IDF2.png'
modelPath2 = 'Models//data2//random_forest_TF_IDF2.joblib'
#random_forest_TF_IDF(inputpath, outputPath2, plotPath2, modelPath2, test_size=0.4)


outputPath3 = 'Raports//data2//random_forest_BoW.txt'
plotPath3 = 'Raports//data2//random_forest_BoW.png'
modelPath3 = 'Models//data2//random_forest_BoW.joblib'
random_forest_BoW(inputpath, outputPath3, plotPath3, modelPath3, test_size=0.2)
outputPath4 = 'Raports//data2//random_forest_BoW2.txt'
plotPath4 = 'Raports//data2//random_forest_BoW2.png'
modelPath4 = 'Models//data2//random_forest_BoW2.joblib'
#random_forest_BoW(inputpath, outputPath4, plotPath4, modelPath4, test_size=0.4)

