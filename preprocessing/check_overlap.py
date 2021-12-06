import pandas as pd
import os
import json

def get_test_data(data_dir):
    df = pd.read_csv(data_dir + 'final_dank.csv')
    df = df[['id', 'media']]
    df.id =df.id.astype(str)
    test = [filename[:-4] for filename in os.listdir(data_dir + 'test') if '(1)' not in filename]
    
    test = pd.DataFrame({'id': test})
    test.id = test.id.astype(str)

    test = test.merge(df, on='id')
    test.media = test.media.str.extract(r'([^\/]+)$')

    return test

def read_data(data_dir):
    instances = []
    text_data = []

    data_dir = os.path.expanduser(data_dir)
    for root, _, fnames in os.walk(data_dir, followlinks=True):
        for fname in fnames:
            if fname.lower().endswith('json'):
                id = fname[:-5]
                instances.append(id)
                with open(data_dir + fname) as f:
                    d = {}
                    data = json.load(f)
                    if 'text_data' in data:
                        text_data.append(entry['text'] for entry in data['text_data'])
                    
    return pd.DataFrame({'media': instances, 'text_data': text_data}) 


test_set_data = get_test_data('/data/edward/dank_data/original_data/')
pretrain_set_data = read_data('/data/edward/pretrain/')

df = test_set_data.merge(pretrain_set_data, on='media')

print(len(df))