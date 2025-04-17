import pandas as pd

def abouthDataset(inputpath, outputPath):
    data = pd.read_csv(inputpath)
    string = '     - Colmuns: '

    for columns in data.columns:
        string += columns + ', '
    string += '\n'
    string += '     - genres: '
    genre_counts = data['genre'].value_counts(normalize=True) * 100
    i = 0
    for genre, percentage in genre_counts.items():
        i += 1
        if i % 5 == 0:
            string += "\n"
        string += f"{genre}: {percentage:.2f}%, "
    string += '\n'
    string +='     - Number of rows: ' + str(len(data)) + '\n'
    string +='     - Number of Classes: ' + str(len(data['genre'].unique())) + '\n'
    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Dataset: ' + inputpath +  "\n")
        f.write(string)

    print(string)
    data2 = data.copy()
    data2 = data2.dropna(subset=['summary'])
    print(len(data2), len(data))

inputpath1 = 'Datasets//data1.csv'
outputPath1 = 'Datasets//About//data1.txt'
#abouthDataset(inputpath1, outputPath1)
inputpath2 = 'Datasets//data2.csv'
outputPath2 = 'Datasets//About//data2.txt'
#abouthDataset(inputpath2, outputPath2)
inputpath3 = 'Datasets//data2_simplified.csv'
outputPath3 = 'Datasets//About//data2_simplified.txt'
#abouthDataset(inputpath3, outputPath3)
inputpath4 = 'Datasets//TestForDt2Dt2Sim.csv'
outputPath4 = 'Datasets//About//TestForDt2Dt2Sim.txt'
abouthDataset(inputpath4, outputPath4)
