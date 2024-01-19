import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType
from tensorflow.keras.applications.resnet50 import ResNet50


def prints(s):
    print('#' * 40)
    print(s)
    print('#' * 40)


if len(sys.argv) < 2:
    print('pass the local_dataset_path file_name in the argument')
    sys.exit()
local_path = Path(sys.argv[1])

spark = (SparkSession
         .builder
         .appName("image_search")
         .getOrCreate())

# Loads the parquet dataframe
df = spark.read.parquet(str(local_path))
prints(df.count())
# Decrease the batch size of the Arrorw reader to avoid OOM errors on smaller instance types
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "16")
spark.conf.set("spark.sql.parquet.columnarReaderBatchSize", "16")
df.show(10)

# This line will fail if the vectorized reader runs out of memory
assert len(df.head()) > 0, "`df` should not be empty"

# Get the model and broadcast its weights to all workers in the cluster
model = ResNet50(include_top=False)
sc = spark.sparkContext
bc_model_weights = sc.broadcast(model.get_weights())


def parse_image(image_data):
    # image = tf.image.convert_image_dtype(image_data, dtype=tf.float32) * (2. / 255) - 1
    image = tf.reshape(image_data, [224, 224, 3])
    final_input = tf.keras.applications.resnet50.preprocess_input(image)
    return final_input


# @pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR_ITER)
def predict_batch_udf(iterator):
    batch_size = 8
    avg_pool = tf.keras.layers.GlobalAvgPool2D()
    model = ResNet50(weights=None, include_top=False)
    model.set_weights(bc_model_weights.value)
    for batch in iterator:
        image_batch=batch['data']
        filenames=batch['filename']
        # Transform dataframe object to numpy array
        images = np.vstack(image_batch)
        # Transform numpy array to tensorflow dataset
        dataset = tf.data.Dataset.from_tensor_slices(images)

        dataset = dataset.map(parse_image, num_parallel_calls=2).prefetch(200).batch(batch_size)
        preds = avg_pool(model.predict(dataset))
        preds = preds.numpy()
        # prints(preds.shape)
        yield pd.DataFrame({"embeddings":list(preds),"file":filenames})


# predictions_df = df.select(predict_batch_udf(col("data")).alias("prediction"))
predictions_df = df.mapInPandas(predict_batch_udf,"embeddings ARRAY<DOUBLE>, file INT")
predictions_df.write.mode("overwrite").parquet('embeddings.parquet')



spark.stop()
