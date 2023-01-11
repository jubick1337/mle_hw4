from pyspark import SparkContext, SparkConf, SQLContext


class Splitter:
    def __init__(self):
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
            "password": "admin"
        }
        self.url = 'jdbc:postgresql://localhost:5432/mle_hw3'
        self.dataset = self.postgres.read.jdbc(url=self.url, table="clustering_results", properties=self.properties)

    def split(self, sizes):
        assert len(sizes) == 2, 'Please provide two sizes to split (train and test)'
        train, test = self.dataset.randomSplit(sizes, seed=1337)
        train.write.jdbc(url=self.url, table="train", mode="overwrite", properties=self.properties)
        test.write.jdbc(url=self.url, table="test", mode="overwrite", properties=self.properties)


if __name__ == '__main__':
    s = Splitter()
    s.split([0.8, 0.2])
