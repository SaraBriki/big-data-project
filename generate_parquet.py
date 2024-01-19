import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# import uuid
# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import ResNet50
# from pyspark.sql.functions import col, pandas_udf, PandasUDFType
# from pyspark.context import SparkContext

# def p(*args):
#     print('#'*20)
#     print(*args)

def main():
    if len(sys.argv) < 3:
        print('pass the local_dataset_path file_name in the argument')
        sys.exit()
    local_path = Path(sys.argv[1])
    file_name = Path(sys.argv[2])
    files = []
    for folder in os.listdir(local_path):
        for file in os.listdir(local_path / folder):
            files.append(local_path / folder / file)
    print(f'file count{len(files)}')

    image_data = []
    for file in tqdm(files):
        img = Image.open(file)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        data = np.array(img, dtype="float32").reshape([224 * 224 * 3])

        # file=str(file).split('/')

        idx = int(str(file.name)[:-4])
        # data = np.append(data, idx)
        image_data.append({"data": data, "filename": idx})

    print(data.shape)
    pandas_df = pd.DataFrame(image_data, columns=['data', 'filename'])
    pandas_df.to_parquet(file_name)


if __name__ == '__main__':
    main()
