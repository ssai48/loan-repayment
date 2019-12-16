from pyspark.sql import SparkSession
import configparser as cp
from pyspark.sql.types import Row
import sys

properties = cp.RawConfigParser()
properties.read('/src/main/resources/application.properties')
environment = sys.argv[1]
properties.get(environment, "execution.mode")

spark = SparkSession \
    .builder \
    .appName('Loan Risk Analyzer') \
    .master() \
    .getOrCreate()
import pyspark.sql.functions as F
from collections import Counter

inputbasedir = properties.get(environment, "input.base.dir")
outputbasedir = properties.get(environment, "output.base.dir")

raw = spark.read.text(environment + "/data.csv")
df = raw.rdd.map(lambda r: r[0].split(",")).map(lambda arr: Row(
    TARGET=int(arr[0]),
    NAME_CONTRACT_TYPE=arr[1],
    CODE_GENDER=arr[2],
    FLAG_OWN_CAR=arr[3],
    FLAG_OWN_REALTY=arr[4],
    CNT_CHILDREN=int(arr[5]),
    AMT_INCOME_TOTAL=float(arr[6]),
    AMT_CREDIT=float(arr[7]),
    AMT_ANNUITY=float(arr[8]),
    NAME_EDUCATION_TYPE=arr[9]
)).toDF()

columns = [
    'TARGET',
    'NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'CNT_CHILDREN',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'FLAG_MOBIL',
    'FLAG_EMP_PHONE',
    'FLAG_WORK_PHONE',
    'FLAG_CONT_MOBILE',
    'FLAG_PHONE',
    'CNT_FAM_MEMBERS',
    'REGION_RATING_CLIENT',
    'REGION_RATING_CLIENT_W_CITY',
    'REG_REGION_NOT_LIVE_REGION',
    'REG_REGION_NOT_WORK_REGION',
    'ORGANIZATION_TYPE',
    'FLAG_DOCUMENT_2',
    'FLAG_DOCUMENT_3',
    'FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5',
    'FLAG_DOCUMENT_6',
    'FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8',
    'FLAG_DOCUMENT_9',
    'FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_11',
    'FLAG_DOCUMENT_12'
]

data = spark.read.option("inferSchema", True).csv(inputbasedir + "*").limit(1000)
data = data.toDF(*columns)

data.cache()

# number of loans falling into each target with percentage
data.groupBy("TARGET").count().withColumn("Percentage", F.col("count") * 100 / data.count()).show()

# number of missing values in each column
counts = [(x, data.filter(F.col(x).isNull()).count()) for x in data.columns]
counts.sort()

# number of columns in  each datatype
[(name, x, round(x * 100.0 / data.count(), 2)) for name, x in counts if x > 0]
print(Counter((x[1] for x in data.dtypes)))

# view unique values in all string columns
str_col_names = [x.name for x in data.schema.fields if x.dataType ==
                 F.StringType()]
unique_df = data.agg(*((F.countDistinct(F.col(c))).alias(c) for c in
                       str_col_names))
unique_df.show()

# describe days employed
data.select('DAYS_EMPLOYED').describe().show()

# describe days birth column
data = data.withColumn("AGE", F.col("DAYS_BIRTH") / -365)
data.select("DAYS_BIRTH", "AGE").describe().show()

# dig deep on days employed

anom = data.filter(F.col('DAYS_EMPLOYED') == 365243)
non_anom = data.filter(F.col('DAYS_EMPLOYED') != 365243)

print('The non-anomalies default on %0.2f%% of loans' % (100 *
                                                         non_anom.select(F.avg(non_anom.TARGET)).first()[0]))
print('The anomalies default on %0.2f%% of loans' % (100 *
                                                     anom.select(F.avg(non_anom.TARGET)).first()[0]))
print('There are %d anomalous days of employment' % anom.count())

# create anamoly flag column
data = data.withColumn('DAYS_EMPLOYED_ANOM', F.col("DAYS_EMPLOYED") == 365243)
data = data.withColumn('DAYS_EMPLOYED', F.when(F.col('DAYS_EMPLOYED') == 365243, 0).otherwise(F.col('DAYS_EMPLOYED')))

# effect of age on repayment by binning the column and generating the pivot table
data.select("AGE").describe().show()
from pyspark.ml.feature import Bucketizer

splits = [0, 25.0, 35.0, 55.0, 100.0]
bucketizer = Bucketizer(splits=splits, inputCol="AGE",
                        outputCol="bucketedData")

bucketedData = bucketizer.transform(encoded)
print("Bucketizer output with %d buckets" % (len(bucketizer.getSplits()) - 1))

bucketedData.groupBy("bucketedData").pivot("TARGET").count().show()

# create new variables

bucketedData = bucketedData.withColumn('CREDIT_INCOME_PERCENT', F.col('AMT_CREDIT'
                                                                      ) / F.col('AMT_INCOME_TOTAL'))

bucketedData = bucketedData.withColumn('ANNUITY_INCOME_PERCENT', F.col('AMT_ANNUITY') / F.col('AMT_INCOME_TOTAL'))

bucketedData = bucketedData.withColumn('CREDIT_TERM', F.col('AMT_ANNUITY') / F.col('AMT_CREDIT'))

bucketedData = bucketedData.withColumn('DAYS_EMPLOYED_PERCENT', F.col('DAYS_EMPLOYED') / F.col('DAYS_BIRTH'))

# convert string column with only 2 unique values to a column  of label indeces

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

indexers = [StringIndexer(inputCol=column, outputCol=column + "_Index")
            for column in
            ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]

pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(bucketedData).transform(bucketedData)

# convert string column with values >2 to onehotencoder
from pyspark.ml.feature import OneHotEncoder

indexers = [StringIndexer(inputCol=column, outputCol=column + "_Index")
            for column in ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'ORGANIZATION_TYPE']]
encoder = [OneHotEncoder().setInputCol(column + "_Index").setOutputCol(column + "_Vec")
           for column in ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'ORGANIZATION_TYPE']]

pipeline = Pipeline(stages=indexers + encoder)
encoded = pipeline.fit(df_r).transform(df_r)

# generate feture columns
feature_cols = [
    'CNT_CHILDREN',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'DAYS_EMPLOYED',
    'FLAG_MOBIL',
    'FLAG_EMP_PHONE',
    'FLAG_WORK_PHONE',
    'FLAG_CONT_MOBILE',
    'FLAG_PHONE',
    'CNT_FAM_MEMBERS',
    'REGION_RATING_CLIENT',
    'REGION_RATING_CLIENT_W_CITY',
    'REG_REGION_NOT_LIVE_REGION',
    'REG_REGION_NOT_WORK_REGION',
    'FLAG_DOCUMENT_2',
    'FLAG_DOCUMENT_3',
    'FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5',
    'FLAG_DOCUMENT_6',
    'FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8',
    'FLAG_DOCUMENT_9',
    'FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_11',
    'FLAG_DOCUMENT_12',
    'NAME_CONTRACT_TYPE_Index',
    'CODE_GENDER_Index',
    'FLAG_OWN_CAR_Index',
    'FLAG_OWN_REALTY_Index',
    'NAME_INCOME_TYPE_Vec',
    'NAME_EDUCATION_TYPE_Vec',
    'ORGANIZATION_TYPE_Vec',
    'AGE',
    'DAYS_EMPLOYED_ANOM',
    'bucketedData',
    'CREDIT_INCOME_PERCENT',
    'ANNUITY_INCOME_PERCENT',
    'CREDIT_TERM',
    'DAYS_EMPLOYED_PERCENT']

# assemble features
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
assembler = VectorAssembler().setInputCols(feature_cols).setOutputCol("features")
output = assembler.transform(encoded).withColumn("label",  F.col("TARGET"))

# train logistic regression model
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(output)
print("Coefficients: " + str(lrModel.coefficients))

# get model accuracy

from pyspark.mllib.evaluation import MulticlassMetrics

transformed = lrModel.transform(output)
results = transformed.select(['prediction', 'label'])
results = results.withColumn("label", F.col("label").cast(DoubleType()))
predictionAndLabels=results.rdd
metrics = MulticlassMetrics(predictionAndLabels)
cm=metrics.confusionMatrix().toArray()
accuracy=(cm[0][0]+cm[1][1])/cm.sum()
print("RandomForestClassifier: accuracy",accuracy)