import pandas as pd


def proccess_data(inputpath):
    # Load data
    data = pd.read_csv(inputpath)
    #add columns summary_simplified
    data['genre_simplified'] = data['genre']
    for s in data['genre']:
        if s in ['fantasy', 'thriller']:
            data['genre'] = data['genre'].replace(s, 'fiction')
            data['genre_simplified'] = data['genre_simplified'].replace(s, 'fiction')
        elif s in ['history', 'science', 'psychology']:
            data['genre_simplified'] = data['genre_simplified'].replace(s, 'education')
        elif s in ['sports']:
            data['genre'] = data['genre'].replace(s, 'sports & recreation')
            data['genre_simplified'] = data['genre_simplified'].replace(s, 'sports & recreation')
        elif s in ['travel']:
            data['genre_simplified'] = data['genre_simplified'].replace(s, 'travel')
        elif s in ['horror', 'crime', 'romance']:
            data = data[data['genre'] != s]
    data.to_csv('Datasets\\TestForDt2Dt2Sim.csv', index=False)

inputpath = 'Datasets\\data1.csv'
proccess_data(inputpath)