from transformers import BertTokenizer
import pandas as pd

def tokenize_data(inputpath):
    data = pd.read_csv(inputpath)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', )
    data['summary'] = data['summary'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512))
    data['title'] = data['title'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512))
    outputPath = inputpath.replace('.csv', '_tokenized.csv')
    data.to_csv(outputPath, index=False)

tokenize_data("Datasets\\data1.csv")
#tokenize_data("Datasets\\data2.csv")