import os
import sys

import pandas as pd
import pinecone
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def prints(s):
    print(f'[X] {s}')


def main():
    df = pd.read_parquet('embeddings.parquet')
    prints('embeddings parquet loaded')
    # connect to pinecone environment
    pinecone.init(
        api_key=os.environ.get('API_KEY'),
        environment="us-east-1-aws"  # find next to API key in console
    )
    prints('pinecone initalized')
    index_name = "image-search"

    # pinecone.delete_index(index_name)
    # prints('pinecone old index deleted')

    # check if the image-search index exists
    if index_name not in pinecone.list_indexes():
        # create the index if it does not exist
        pinecone.create_index(
            index_name,
            dimension=2048,
            metric="cosine"
        )
    prints('pinecone index created')

    # connect to audio-search index we created
    index = pinecone.Index(index_name)
    prints('pinecone index loaded')
    BATCH_SIZE = 64
    df['embeddings'] = df['embeddings'].map(lambda x: x.tolist())
    df['file']=df['file'].astype('str')
    print('[*] starting persistence to pinecone')
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        i_end = min(i + BATCH_SIZE, len(df))
        batch = df.iloc[i:i_end]
        embs = list(batch['embeddings'])
        filenames = list(batch['file'])
        data = list(zip(filenames, embs))
        print(len(data))
        # break

        index.upsert(vectors=data)

    prints('persistence completed')


if __name__ == '__main__':
    main()
