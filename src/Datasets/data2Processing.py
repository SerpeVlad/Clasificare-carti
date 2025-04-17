import pandas as pd


def processdata1():
    data = pd.read_csv("Datasets\\data2_raw.csv")

    # rename the columns so it maches the data1.csv
    data.columns = [col.lower() for col in data.columns]
    data.rename(columns={'category': 'genre'}, inplace=True)
    data.rename(columns={'description': 'summary'}, inplace=True)    
    data = data.dropna(subset=['summary'])
    data = data.dropna(subset=['genre'])

    data['genre'] = data['genre'].str.lower()
    data['genre'] = data['genre'].str.split(',').str[0].str.strip() # From "history , general" to "history"

    #only one exemple so we remove it
    data = data[data['genre'] != "children's music"]
    data = data[data['genre'] != "television: national geographic"]


    data.to_csv("Datasets\\data2.csv", index=False)

def processdata2():
    data = pd.read_csv("Datasets\\data2_raw.csv")

    # rename the columns so it maches the data1.csv
    data.columns = [col.lower() for col in data.columns]
    data.rename(columns={'category': 'genre'}, inplace=True)
    data.rename(columns={'description': 'summary'}, inplace=True)    
    data = data.dropna(subset=['summary'])
    data = data.dropna(subset=['genre'])

    data['genre'] = data['genre'].str.lower()
    data['genre'] = data['genre'].str.split(',').str[0].str.strip() # From "history , general" to "history"

    #only one exemple so we remove it
    data = data[data['genre'] != "children's music"]
    data = data[data['genre'] != "television: national geographic"]

    for genre in data['genre'].unique():
        if data['genre'].value_counts()[genre] < 10:
            data = data[data['genre'] != genre]

    data2 = data.copy()


    #rename genres to be more general
    for genre in data['genre'].unique():
        if 'fiction' in genre:
            if 'nonfiction' not in genre:
                data['genre'] = data['genre'].replace(genre, "fiction")
            elif 'nonfiction' in genre: 
                data['genre'] = data['genre'].replace(genre, "nonfiction") # I also added true crime
        elif genre in ['literary collections', 'literary criticism']:
            data['genre'] = data['genre'].replace(genre, "literature")
        elif 'art' in genre or genre in ['poetry', 'drama', 'music', 'comics & graphic novels', 'photography']:
            data['genre'] = data['genre'].replace(genre, "art")
        elif genre in ['games', 'games & activities']:
            data['genre'] = data['genre'].replace(genre, "games")
        elif genre in ['religion', 'bibles']:
            data['genre'] = data['genre'].replace(genre, "religion")
        elif genre in ['medical', 'health & fitness', 'body']:
            data['genre'] = data['genre'].replace(genre, "health")
        elif genre in ['nature', 'gardening']:
            data['genre'] = data['genre'].replace(genre, "nature")
        elif genre in ['business & economics', 'history', 'political science', 
                       'social science', 'psychology', 'philosophy',
                       'mathematics', 'science', 'technology & engineering',
                       'law', 'architecture', 'foreign language study', 'self-help']:
            data['genre'] = data['genre'].replace(genre, "education")
        elif genre in ['true crime']:
            data['genre'] = data['genre'].replace(genre, "nonfiction")
    genre_counts = data['genre'].value_counts(normalize=True) * 100
    str = ''
    i = 0
    for genre, percentage in genre_counts.items():
        if i % 4 == 0:
            str += "\n"
        str += f"{genre}: {percentage:.2f}%, "
        i += 1

    print(str)
    print(len(data['genre'].unique()), len(data2['genre'].unique())) 
    data.to_csv("Datasets\\data2_simplified.csv", index=False)

processdata2()