import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def comparation(model1Path, model2Path, dataPath1, dataPath2, outputPath='Raports//comparation_dt2_dt2_sim//comparation.txt'):
    model1 = joblib.load(model1Path)
    model2 = joblib.load(model2Path)
    data = pd.read_csv(dataPath1)
    data_simplified = pd.read_csv(dataPath2)

    if 'BoW' in model1Path:
        vectorizer = CountVectorizer()
    elif 'TF_IDF' in model1Path or 'TFIDF' in model1Path:
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.8)
    X = vectorizer.fit_transform(data['summary'])
    predictions1 = model1.predict(X)

    if 'BoW' in model2Path:
        vectorizer = CountVectorizer()
    elif 'TF_IDF' in model2Path or 'TFIDF' in model2Path:
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
    CFP2 = {}
    for i in cf:
        if CF.get(predictions1[i]+'-'+predictions2[i]) == None:
            CF[predictions1[i]+'-'+predictions2[i]] = 1
        else:
            CF[predictions1[i]+'-'+predictions2[i]] += 1
        if CFP2.get(predictions2[i]) == None:
            CFP2[predictions2[i]] = 1
        else:
            CFP2[predictions2[i]] += 1
    FC = {}
    FCP2 = {}
    for i in fc:
        if FC.get(predictions1[i]+'-'+predictions2[i]) == None:
            FC[predictions1[i]+'-'+predictions2[i]] = 1
        else:
            FC[predictions1[i]+'-'+predictions2[i]] += 1   
        if FCP2.get(predictions2[i]) == None:
            FCP2[predictions2[i]] = 1
        else:
            FCP2[predictions2[i]] += 1     
    FF = {}
    FFP2 = {}
    for i in ff:
        if FF.get(predictions1[i]+'-'+predictions2[i]) == None:
            FF[predictions1[i]+'-'+predictions2[i]] = 1
        else:
            FF[predictions1[i]+'-'+predictions2[i]] += 1
        if FFP2.get(predictions2[i]) == None:
            FFP2[predictions2[i]] = 1
        else:
            FFP2[predictions2[i]] += 1
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
    CFP2 = dict(sorted(CFP2.items(), key=lambda item: item[1], reverse=True))
    FCP2 = dict(sorted(FCP2.items(), key=lambda item: item[1], reverse=True))
    FFP2 = dict(sorted(FFP2.items(), key=lambda item: item[1], reverse=True))
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
        f.write("Pred Simplified(Gresit): " + str(CFP2) + '\n')

        f.write("\n\nGresit-Corect" + str(FC) + '\n\n')
        f.write("Pred Simplified(Corect): " + str(FCP2) + '\n')

        f.write("\n\nGresit-Gresit" + str(FF) + '\n\n')
        f.write("Pred Simplified(Gresit): " + str(FFP2) + '\n')

        f.write("\n\n")

        lst_key = list(CF.keys())
        for key in list(FC.keys()):
            if key not in lst_key:
                lst_key.append(key)
        for key in lst_key:
            a = 0
            b = 0
            if key in CF.keys():
                f.write('C-G: '+key + ' : ' + str(CF[key]) + '\n')
                a += CF[key]
            else:
                f.write('C-G: '+key + ' : ' + '0' + '\n')

            if key in FC.keys():
                f.write('G-C: '+key + ' : ' + str(FC[key]) + '\n')
                b += FC[key]
            else:
                f.write('G-C: '+key + ' : ' + '0' + '\n')   
                
            if key in FF.keys():
                f.write('G-G: '+key + ' : ' + str(FF[key]) + '\n')
            else:
                f.write('G-G: '+key + ' : ' + '0' + '\n')
            

                
            if a > b:
                f.write(key + ': M1 > M2' +'\n')
            elif a < b:
                f.write(key + ': M2 > M1' +'\n')
            
            f.write('\n\n')

        


model_path1 = "Models\\data2\\logistic_regression_BoW.joblib"
model_path2 = "Models\\data2_simplified\\logistic_regression_BoW.joblib"
model_path3 = "Models\\data2_simplified\\logistic_regression_TFIDF.joblib"
data_path1 = "Datasets\\data2.csv"
data_path2 = "Datasets\\data2_simplified.csv"
outh_path = 'Raports\\comparation_dt2_dt2_sim\\comparation.txt'
outh_path2 = 'Raports\\comparation_dt2_dt2_sim\\comparation3.txt'
comparation(model_path1, model_path2, data_path1, data_path2,outputPath=outh_path)
comparation(model_path1, model_path3, data_path1, data_path2,outputPath=outh_path2)
        