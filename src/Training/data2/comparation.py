import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def comparation(model1Path, model2Path, dataPath1, dataPath2, outputPath='Raports//comparation_dt2//comparation.txt'):
    model1 = joblib.load(model1Path)
    model2 = joblib.load(model2Path)
    data = pd.read_csv(dataPath1)
    data_simplified = pd.read_csv(dataPath2)

    if 'BoW' in model1Path:
        vectorizer = CountVectorizer()
    elif 'TFIDF' in model1Path or 'TF_IDF' in model1Path:
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.8)
    X = vectorizer.fit_transform(data['summary'])
    predictions1 = model1.predict(X)

    if 'BoW' in model2Path:
        vectorizer = CountVectorizer()
    elif 'TFIDF' in model2Path or 'TF_IDF' in model2Path:
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.8)
    X = vectorizer.fit_transform(data_simplified['summary'])  
    predictions2 = model2.predict(X)

    truth1 = data['genre']
    truth2 = data_simplified['genre']
    cc = []
    cf = []
    fc = []
    ff = []
    for i in range(len(predictions1)):
        if predictions2[i] == truth2[i] and predictions1[i] == truth1[i]:
            cc.append(i)
        elif predictions2[i] != truth2[i] and predictions1[i] == truth1[i]:
            cf.append(i)
            print('P2 (F): ' + predictions2[i] + " P1 (T): " + predictions1[i])
        elif predictions2[i] == truth2[i] and predictions1[i] != truth1[i]:
            fc.append(i)
            print('P2 (T): ' + predictions2[i] + " P1 (F): " + predictions1[i])
        else:
            ff.append(i)
            print('P2 (F): ' + predictions2[i] + " P1 (F): " + predictions1[i])

    CF = {}
    for i in cf:
        if CF.get(predictions1[i]+'-'+predictions2[i]) == None:
            CF[predictions1[i]+'-'+predictions2[i]] = 1
        else:
            CF[predictions1[i]+'-'+predictions2[i]] += 1
    FC = {}
    for i in fc:
        if FC.get(predictions1[i]+'-'+predictions2[i]) == None:
            FC[predictions1[i]+'-'+predictions2[i]] = 1
        else:
            FC[predictions1[i]+'-'+predictions2[i]] += 1       
    FF = {}
    for i in ff:
        if FF.get(predictions1[i]+'-'+predictions2[i]) == None:
            FF[predictions1[i]+'-'+predictions2[i]] = 1
        else:
            FF[predictions1[i]+'-'+predictions2[i]] += 1

    for key in list(CF.keys()):
        if CF[key] < 5:
            del CF[key]
    for key in list(FC.keys()):
        if FC[key] < 5:
            del FC[key]
    for key in list(FF.keys()):
        if FF[key] < 5:
            del FF[key]

    CF = dict(sorted(CF.items(), key=lambda item: item[1], reverse=True))
    FC = dict(sorted(FC.items(), key=lambda item: item[1], reverse=True))
    FF = dict(sorted(FF.items(), key=lambda item: item[1], reverse=True))

    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Model 1: ' + model1Path + "\n")
        f.write('Model 2: ' + model2Path + "\n")
        f.write('Model 1 Corect predictions: ' + str((len(cc) + len(cf)) / len(predictions1)) + "\n")
        f.write('Model 2 Corect predictions: ' + str((len(cc) + len(fc)) / len(predictions1) ) + "\n")
        
        f.write("\n\n")

        f.write('Corect-Corect: ' + str(len(cc)) + "                              "+str(len(cc) / len(predictions1)) + "%\n")
        f.write('Corect-Gresit: ' + str(len(cf)) + "                              "+str(len(cf) / len(predictions1)) + "%\n")
        f.write('Gresit-Corect: ' + str(len(fc)) + "                              "+str(len(fc) / len(predictions1)) + "%\n")
        f.write('Gresit-Gresit: ' + str(len(ff)) + "                              "+str(len(ff) / len(predictions1)) + "%\n")
        f.write('Total: ' + str(len(predictions1)) + "\n")
        
        f.write("\n\nCorect-Gresit: " + str(CF) + '\n\n')

        f.write("\n\nGresit-Corect" + str(FC) + '\n\n')

        f.write("\n\nGresit-Gresit" + str(FF) + '\n\n')

        f.write("\n\n")
        lst_key = list(CF.keys())
        for key in list(FC.keys()):
            if key not in lst_key:
                lst_key.append(key)
        for key in lst_key:
            a = 0
            b = 0
            c = 0
            d = 0
            kety2 = key.split('-')
            key2 = kety2[1] + '-' + kety2[0]
            if key in CF.keys():
                f.write('C-G: '+key + ' : ' + str(CF[key]) + '\n')
                a += CF[key]
                if key2 in CF.keys():
                    c += CF[key2]
                    f.write('C-G: '+key2 + ' : ' + str(CF[key2]) + '\n')
                    try:
                        lst_key.remove(key2)
                    except:
                        pass
                else:
                    f.write('C-G: '+key2 + ' : ' + '0' + '\n')
            else:
                f.write('C-G: '+key + ' : ' + '0' + '\n')
                if key2 in CF.keys():
                    c += CF[key2]
                    f.write('C-G: '+key2 + ' : ' + str(CF[key2]) + '\n')
                    try:
                        lst_key.remove(key2)
                    except:
                        pass                
                else:
                    f.write('C-G: '+key2 + ' : ' + '0' + '\n')

            if key in FC.keys():
                f.write('G-C: '+key + ' : ' + str(FC[key]) + '\n')
                b += FC[key]
                if key2 in FC.keys():
                    d += FC[key2]
                    f.write('G-C: '+key2 + ' : ' + str(FC[key2]) + '\n')
                    try:
                        lst_key.remove(key2)
                    except:
                        pass
                else:
                    f.write('G-C: '+key2 + ' : ' + '0' + '\n')
            else:
                f.write('G-C: '+key + ' : ' + '0' + '\n')   
                if key2 in FC.keys():
                    d += FC[key2]
                    f.write('G-C: '+key2 + ' : ' + str(FC[key2]) + '\n')
                    try:
                        lst_key.remove(key2)
                    except:
                        pass
                else:
                    f.write('G-C: '+key2 + ' : ' + '0' + '\n')
            if key in FF.keys():
                f.write('G-G: '+key + ' : ' + str(FF[key]) + '\n')
            else:
                f.write('G-G: '+key + ' : ' + '0' + '\n')
            if key2 in FF.keys():
                f.write('G-G: '+key2 + ' : ' + str(FF[key2]) + '\n')
            else:
                f.write('G-G: '+key2 + ' : ' + '0' + '\n')
            if a > b:
                f.write(key + ': M1 > M2' +'\n')
            elif a < b:
                f.write(key + ': M2 > M1' +'\n')
            
            if c > d:
                f.write(key2 + ': M1 > M2' +'\n')
            elif c < d:
                f.write(key2 + ': M2 > M1' +'\n')
            f.write('\n\n')
            

def comparation_testData(model1Path, model2Path, outputPath='Raports//comparation_dt2//comparation_test_data.txt'):
    model1 = joblib.load(model1Path)
    model2 = joblib.load(model2Path)
    data = pd.read_csv('Datasets\\TestForDt2Dt2Sim.csv')

    if 'BoW' in model1Path:
        vectorizer = joblib.load('Models\\data2_simplified\\vectorizerBoW.joblib')
    elif 'TFIDF' in model1Path or 'TF_IDF' in model1Path:
        vectorizer = joblib.load('Models\\data2_simplified\\vectorizerTF_IDF.joblib')
    X = vectorizer.transform(data['summary'])
    predictions1 = model1.predict(X)

    if 'BoW' in model2Path:
        vectorizer = joblib.load('Models\\data2_simplified\\vectorizerBoW.joblib')
    elif 'TFIDF' in model2Path or 'TF_IDF' in model2Path:
        vectorizer = joblib.load('Models\\data2_simplified\\vectorizerTF_IDF.joblib')
    X = vectorizer.transform(data['summary'])  

    predictions2 = model2.predict(X)

    truth1 = data['genre']
    truth2 = data['genre']
    cc = []
    cf = []
    fc = []
    ff = []
    for i in range(len(predictions1)):
        if predictions2[i] == truth2[i] and predictions1[i] == truth1[i]:
            cc.append(i)
        elif predictions2[i] != truth2[i] and predictions1[i] == truth1[i]:
            cf.append(i)
            print('P2 (F): ' + predictions2[i] + " P1 (T): " + predictions1[i])
        elif predictions2[i] == truth2[i] and predictions1[i] != truth1[i]:
            fc.append(i)
            print('P2 (T): ' + predictions2[i] + " P1 (F): " + predictions1[i])
        else:
            ff.append(i)
            print('P2 (F): ' + predictions2[i] + " P1 (F): " + predictions1[i])

    CF = {}
    for i in cf:
        if CF.get(predictions1[i]+'-'+predictions2[i]) == None:
            CF[predictions1[i]+'-'+predictions2[i]] = 1
        else:
            CF[predictions1[i]+'-'+predictions2[i]] += 1
    FC = {}
    for i in fc:
        if FC.get(predictions1[i]+'-'+predictions2[i]) == None:
            FC[predictions1[i]+'-'+predictions2[i]] = 1
        else:
            FC[predictions1[i]+'-'+predictions2[i]] += 1       
    FF = {}
    for i in ff:
        if FF.get(predictions1[i]+'-'+predictions2[i]) == None:
            FF[predictions1[i]+'-'+predictions2[i]] = 1
        else:
            FF[predictions1[i]+'-'+predictions2[i]] += 1

    for key in list(CF.keys()):
        if CF[key] < 5:
            del CF[key]
    for key in list(FC.keys()):
        if FC[key] < 5:
            del FC[key]
    for key in list(FF.keys()):
        if FF[key] < 5:
            del FF[key]

    CF = dict(sorted(CF.items(), key=lambda item: item[1], reverse=True))
    FC = dict(sorted(FC.items(), key=lambda item: item[1], reverse=True))
    FF = dict(sorted(FF.items(), key=lambda item: item[1], reverse=True))

    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Model 1: ' + model1Path + "\n")
        f.write('Model 2: ' + model2Path + "\n")
        f.write('Model 1 Corect predictions: ' + str((len(cc) + len(cf)) / len(predictions1)) + "\n")
        f.write('Model 2 Corect predictions: ' + str((len(cc) + len(fc)) / len(predictions1) ) + "\n")
        
        f.write("\n\n")

        f.write('Corect-Corect: ' + str(len(cc)) + "                              "+str(len(cc) / len(predictions1)) + "%\n")
        f.write('Corect-Gresit: ' + str(len(cf)) + "                              "+str(len(cf) / len(predictions1)) + "%\n")
        f.write('Gresit-Corect: ' + str(len(fc)) + "                              "+str(len(fc) / len(predictions1)) + "%\n")
        f.write('Gresit-Gresit: ' + str(len(ff)) + "                              "+str(len(ff) / len(predictions1)) + "%\n")
        f.write('Total: ' + str(len(predictions1)) + "\n")
        
        f.write("\n\nCorect-Gresit: " + str(CF) + '\n\n')

        f.write("\n\nGresit-Corect" + str(FC) + '\n\n')

        f.write("\n\nGresit-Gresit" + str(FF) + '\n\n')

        f.write("\n\n")
        lst_key = list(CF.keys())
        for key in list(FC.keys()):
            if key not in lst_key:
                lst_key.append(key)
        for key in lst_key:
            a = 0
            b = 0
            c = 0
            d = 0
            kety2 = key.split('-')
            key2 = kety2[1] + '-' + kety2[0]
            if key in CF.keys():
                f.write('C-G: '+key + ' : ' + str(CF[key]) + '\n')
                a += CF[key]
                if key2 in CF.keys():
                    c += CF[key2]
                    f.write('C-G: '+key2 + ' : ' + str(CF[key2]) + '\n')
                    try:
                        lst_key.remove(key2)
                    except:
                        pass
                else:
                    f.write('C-G: '+key2 + ' : ' + '0' + '\n')
            else:
                f.write('C-G: '+key + ' : ' + '0' + '\n')
                if key2 in CF.keys():
                    c += CF[key2]
                    f.write('C-G: '+key2 + ' : ' + str(CF[key2]) + '\n')
                    try:
                        lst_key.remove(key2)
                    except:
                        pass                
                else:
                    f.write('C-G: '+key2 + ' : ' + '0' + '\n')

            if key in FC.keys():
                f.write('G-C: '+key + ' : ' + str(FC[key]) + '\n')
                b += FC[key]
                if key2 in FC.keys():
                    d += FC[key2]
                    f.write('G-C: '+key2 + ' : ' + str(FC[key2]) + '\n')
                    try:
                        lst_key.remove(key2)
                    except:
                        pass
                else:
                    f.write('G-C: '+key2 + ' : ' + '0' + '\n')
            else:
                f.write('G-C: '+key + ' : ' + '0' + '\n')   
                if key2 in FC.keys():
                    d += FC[key2]
                    f.write('G-C: '+key2 + ' : ' + str(FC[key2]) + '\n')
                    try:
                        lst_key.remove(key2)
                    except:
                        pass
                else:
                    f.write('G-C: '+key2 + ' : ' + '0' + '\n')
            if key in FF.keys():
                f.write('G-G: '+key + ' : ' + str(FF[key]) + '\n')
            else:
                f.write('G-G: '+key + ' : ' + '0' + '\n')
            if key2 in FF.keys():
                f.write('G-G: '+key2 + ' : ' + str(FF[key2]) + '\n')
            else:
                f.write('G-G: '+key2 + ' : ' + '0' + '\n')

                
            if a > b:
                f.write(key + ': M1 > M2' +'\n')
            elif a < b:
                f.write(key + ': M2 > M1' +'\n')
            
            if c > d:
                f.write(key2 + ': M1 > M2' +'\n')
            elif c < d:
                f.write(key2 + ': M2 > M1' +'\n')
            f.write('\n\n')
            
    


model_path1 = "Models\\data2\\logistic_regression_BoW.joblib"
model_path2 = "Models\\data2\\logistic_regression_TF_IDF3.joblib"
outh_path = 'Raports\\comparation_dt2\\comparation_test_data.txt'
comparation_testData(model_path1, model_path2, outputPath=outh_path)
        
        