import argparse

from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F


class RegressionModel:
    def __init__(self, args):
        spark_config = SparkConf()
        spark_config.set("spark.app.name", "mle_hw3")
        spark_config.set("spark.master", "local")
        spark_config.set("spark.executor.cores", "16")
        spark_config.set("spark.executor.instances", "1")
        spark_config.set("spark.executor.memory", "16g")
        spark_config.set("spark.locality.wait", "0")
        spark_config.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        spark_config.set("spark.kryoserializer.buffer.max", "2000")
        spark_config.set("spark.executor.heartbeatInterval", "6000s")
        spark_config.set("spark.network.timeout", "10000000s")
        spark_config.set("spark.shuffle.spill", "true")
        spark_config.set("spark.driver.memory", "8g")
        spark_config.set("spark.driver.maxResultSize", "8g")
        spark_config.set("spark.jars", "C:\\postgresql-42.5.0.jar")  # To load psql

        self.sc = SparkContext.getOrCreate(conf=spark_config)

        self.postgres = SQLContext(self.sc)
        self.properties = {
            "driver": "org.postgresql.Driver",
            "user": "postgres",
            "password": args.p
        }
        self.url = args.url
        self.train_data = self.postgres.read.jdbc(url=self.url, table="train", properties=self.properties)
        self.test_data = self.postgres.read.jdbc(url=self.url, table="test", properties=self.properties)
        self.vector_assembler = VectorAssembler(inputCols=['calcium_100g', 'fat_100g', 'proteins_100g', 'energy_100g'],
                                                outputCol="features")
        self.model = LogisticRegression(labelCol='cluster', featuresCol='features', maxIter=args.max_iter,
                                        family='multinomial')
        self.pipeline = Pipeline(stages=[self.vector_assembler, self.model])

    def train(self):
        model = self.pipeline.fit(self.train_data)
        test_prediction = model.transform(self.test_data)
        accuracy = MulticlassClassificationEvaluator(labelCol='cluster', predictionCol='prediction',
                                                     metricName='accuracy')
        f1 = MulticlassClassificationEvaluator(labelCol='cluster', predictionCol='prediction',
                                               metricName='f1')
        print(f'Accuracy on test set = {accuracy.evaluate(test_prediction)}')
        print(f'F1 on test set = {f1.evaluate(test_prediction)}')
        import matplotlib.pyplot as plt
        plt.plot(model.stages[1].summary.objectiveHistory)
        plt.title('Loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
        cm = MulticlassMetrics(test_prediction.select(['prediction', 'cluster']).withColumn('cluster',
                                                                                            F.col('cluster').cast(
                                                                                                FloatType())).orderBy(
            'prediction').select(['prediction', 'cluster']).rdd.map(tuple))
        print('Confusion matrix:')
        print(cm.confusionMatrix().toArray())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('-u', default='postgres')
    parser.add_argument('-p', type=str, required=True)
    parser.add_argument('--url', type=str, default='jdbc:postgresql://localhost:5432/mle_hw3')
    reg_model = RegressionModel(parser.parse_args())
    reg_model.train()
