from __future__ import print_function
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.ml import Pipeline, PipelineModel
from pyspark.streaming.kafka import KafkaUtils
from pyspark.sql import Row, SparkSession

if __name__ == "__main__":
    sc = SparkContext(appName="PythonStreaming")
    ssc = StreamingContext(sc, 10)
    kvs = KafkaUtils.createStream(ssc, "localhost:2181", "spark-streamingconsumer", {"inclass": 1})
    lines = kvs.map(lambda x: x[1])


    def process(t, rdd):
        if rdd.isEmpty():
            return


    spark = SparkSession.builder.getOrCreate()

    import pyspark.sql.functions as F

    rawRdd = rdd.map(lambda x: x.split(",")).map(lambda d:
                                                 Row(NAME_CONTRACT_TYPE=d[0], CODE_GENDER=d[1],
                                                     FLAG_OWN_CAR=d[2]
                                                     , FLAG_OWN_REALTY=d[3], CNT_CHILDREN=int(d[4]),
                                                     AMT_INCOME_TOTAL=Double(d[5]), AMT_CREDIT=double(d[6]),
                                                     AMT_ANNUITY=double(d[7]
                                                                        ), NAME_INCOME_TYPE=d[8],
                                                     NAME_EDUCATION_TYPE=d[9], NAME_FAMILY_STATUS=d[10],
                                                     NAME_HOUSING_TYPE=d[11], DAYS_BIRTH=int(d[12]),
                                                     DAYS_EMPLOYED=int(d[13]), FLAG_MOBIL=int(d[14]),
                                                     FLAG_EMP_PHONE=int(d[15]), FLAG_WORK_PHONE=int(d[16]),
                                                     FLAG_CONT_MOBILE=int(d[17
                                                                          ]), FLAG_PHONE=int(d[18]),
                                                     CNT_FAM_MEMBERS=double(d[19]),
                                                     REGION_RATING_CLIENT=int(d[20]),
                                                     REGION_RATING_CLIENT_W_CITY=int(d[21
                                                                                     ]),
                                                     REG_REGION_NOT_LIVE_REGION=int(d[22]),
                                                     REG_REGION_NOT_WORK_REGION=int(d[23]), ORGANIZATION_TYPE=d[23],
                                                     FLAG_DOCUMENT_2=int(
                                                         d[24]), FLAG_DOCUMENT_3=int(d[25]), FLAG_DOCUMENT_4=int(d[26]),
                                                     FLAG_DOCUMENT_5=int(d[27]), FLAG_DOCUMENT_6=int(d[28]),
                                                     FLAG_DOCUMENT_7=int(d[29]), FLAG_DOCUMENT_8=int(d[30]),
                                                     FLAG_DOCUMENT_9=int(d[31]), FLAG_DOCUMENT_10=int(d[32]),
                                                     FLAG_DOCUMENT_11=int(
                                                         d[33]), FLAG_DOCUMENT_12=int(d[34])))
    raw = spark.createDataFrame(rawRdd)
    if not rdd.isEmpty():

        df = rdd.map(lambda r: Row(message=r))
        df = df.withColumn('CREDIT_INCOME_PERCENT', F.col('AMT_CREDIT') / F.col('AMT_INCOME_TOTAL'))
        df = df.withColumn("AGE", F.col("DAYS_BIRTH") / -365)
        df = df.withColumn('DAYS_EMPLOYED_ANOM', F.col("DAYS_EMPLOYED") == 365243)
        df = df.withColumn('DAYS_EMPLOYED',
                           F.when(F.col('DAYS_EMPLOYED') == 365243, 0).otherwise(F.col('DAYS_EMPLOYED')))
        df = df.withColumn('CREDIT_INCOME_PERCENT', F.col('AMT_CREDIT') / F.col('AMT_INCOME_TOTAL'))
        df = df.withColumn('ANNUITY_INCOME_PERCENT', F.col('AMT_ANNUITY') / F.col('AMT_INCOME_TOTAL'))
        df = df.withColumn('CREDIT_TERM', F.col('AMT_ANNUITY') / F.col('AMT_CREDIT'))
        df = df.withColumn('DAYS_EMPLOYED_PERCENT', F.col('DAYS_EMPLOYED') / F.col('DAYS_BIRTH'))
        pipeline = PipelineModel.load("in_class/pymodel.model")
        predictions = pipeline.transform(df)
        lines.foreachRDD(process)
    ssc.start()
    ssc.awaitTermination()
