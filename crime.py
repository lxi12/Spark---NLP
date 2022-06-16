## Python and pyspark modules required

import sys
 
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row, DataFrame, Window, functions as F
from pyspark.sql.types import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, NGram, CountVectorizer, OneHotEncoder, StringIndexer, VectorAssembler, HashingTF, IDF
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

#from pretty import SparkPretty  # download pretty.py from LEARN
#pretty = SparkPretty(limit=5)

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
 

## Load
data = (
  spark.read.format('com.databricks.spark.csv')
  .options(header='true', inferSchema='true')        
  .load('hdfs:////data/crime/train.csv')
)
data.show()
# +-------------------+--------------+--------------------+---------+----------+--------------+--------------------+-------------------+------------------+
# |              Dates|      Category|            Descript|DayOfWeek|PdDistrict|    Resolution|             Address|                  X|                 Y|
# +-------------------+--------------+--------------------+---------+----------+--------------+--------------------+-------------------+------------------+
# |2015-05-13 23:53:00|      WARRANTS|      WARRANT ARREST|Wednesday|  NORTHERN|ARREST, BOOKED|  OAK ST / LAGUNA ST|  -122.425891675136|  37.7745985956747|
# |2015-05-13 23:53:00|OTHER OFFENSES|TRAFFIC VIOLATION...|Wednesday|  NORTHERN|ARREST, BOOKED|  OAK ST / LAGUNA ST|  -122.425891675136|  37.7745985956747|
# |2015-05-13 23:33:00|OTHER OFFENSES|TRAFFIC VIOLATION...|Wednesday|  NORTHERN|ARREST, BOOKED|VANNESS AV / GREE...|   -122.42436302145|  37.8004143219856|
# |2015-05-13 23:30:00| LARCENY/THEFT|GRAND THEFT FROM ...|Wednesday|  NORTHERN|          NONE|1500 Block of LOM...|-122.42699532676599| 37.80087263276921|
# |2015-05-13 23:30:00| LARCENY/THEFT|GRAND THEFT FROM ...|Wednesday|      PARK|          NONE|100 Block of BROD...|  -122.438737622757|37.771541172057795|
# |2015-05-13 23:30:00| LARCENY/THEFT|GRAND THEFT FROM ...|Wednesday| INGLESIDE|          NONE| 0 Block of TEDDY AV|-122.40325236121201|   37.713430704116|
# |2015-05-13 23:30:00| VEHICLE THEFT|   STOLEN AUTOMOBILE|Wednesday| INGLESIDE|          NONE| AVALON AV / PERU AV|  -122.423326976668|  37.7251380403778|
# |2015-05-13 23:30:00| VEHICLE THEFT|   STOLEN AUTOMOBILE|Wednesday|   BAYVIEW|          NONE|KIRKWOOD AV / DON...|  -122.371274317441|  37.7275640719518|
# |2015-05-13 23:00:00| LARCENY/THEFT|GRAND THEFT FROM ...|Wednesday|  RICHMOND|          NONE|600 Block of 47TH AV|  -122.508194031117|37.776601260681204|
# |2015-05-13 23:00:00| LARCENY/THEFT|GRAND THEFT FROM ...|Wednesday|   CENTRAL|          NONE|JEFFERSON ST / LE...|  -122.419087676747|  37.8078015516515|
# |2015-05-13 22:58:00| LARCENY/THEFT|PETTY THEFT FROM ...|Wednesday|   CENTRAL|          NONE|JEFFERSON ST / LE...|  -122.419087676747|  37.8078015516515|
# |2015-05-13 22:30:00|OTHER OFFENSES|MISCELLANEOUS INV...|Wednesday|   TARAVAL|          NONE|0 Block of ESCOLT...|  -122.487983072777|37.737666654332706|
# |2015-05-13 22:30:00|     VANDALISM|MALICIOUS MISCHIE...|Wednesday|TENDERLOIN|          NONE|  TURK ST / JONES ST|-122.41241426358101|  37.7830037964534|
# |2015-05-13 22:06:00| LARCENY/THEFT|GRAND THEFT FROM ...|Wednesday|  NORTHERN|          NONE|FILLMORE ST / GEA...|  -122.432914603494|  37.7843533426568|
# |2015-05-13 22:00:00|  NON-CRIMINAL|      FOUND PROPERTY|Wednesday|   BAYVIEW|          NONE|200 Block of WILL...|  -122.397744427103|  37.7299346936044|
# |2015-05-13 22:00:00|  NON-CRIMINAL|      FOUND PROPERTY|Wednesday|   BAYVIEW|          NONE|0 Block of MENDEL...|-122.38369150395901|  37.7431890419965|
# |2015-05-13 22:00:00|       ROBBERY|ROBBERY, ARMED WI...|Wednesday|TENDERLOIN|          NONE|  EDDY ST / JONES ST|  -122.412597377187|37.783932027727296|
# |2015-05-13 21:55:00|       ASSAULT|AGGRAVATED ASSAUL...|Wednesday| INGLESIDE|          NONE|GODEUS ST / MISSI...|  -122.421681531572|  37.7428222004845|
# |2015-05-13 21:40:00|OTHER OFFENSES|   TRAFFIC VIOLATION|Wednesday|   BAYVIEW|ARREST, BOOKED|MENDELL ST / HUDS...|-122.38640086995301|   37.738983491072|
# |2015-05-13 21:30:00|  NON-CRIMINAL|      FOUND PROPERTY|Wednesday|TENDERLOIN|          NONE|100 Block of JONE...|  -122.412249767634|   37.782556330202|
# +-------------------+--------------+--------------------+---------+----------+--------------+--------------------+-------------------+------------------+

# remove cols not need
drop_list = ['Dates', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
data = data.select([column for column in data.columns if column not in drop_list])
data.show(5)
# +--------------+--------------------+
# |      Category|            Descript|
# +--------------+--------------------+
# |      WARRANTS|      WARRANT ARREST|
# |OTHER OFFENSES|TRAFFIC VIOLATION...|
# |OTHER OFFENSES|TRAFFIC VIOLATION...|
# | LARCENY/THEFT|GRAND THEFT FROM ...|
# | LARCENY/THEFT|GRAND THEFT FROM ...|
# +--------------+--------------------+

data.printSchema()

# Top 20 crime categories
data.groupBy("Category") \
    .count() \
    .orderBy(F.col("count").desc()) \
    .show()     
# +--------------------+------+
# |            Category| count|
# +--------------------+------+
# |       LARCENY/THEFT|174900|
# |      OTHER OFFENSES|126182|
# |        NON-CRIMINAL| 92304|
# |             ASSAULT| 76876|
# |       DRUG/NARCOTIC| 53971|
# |       VEHICLE THEFT| 53781|
# |           VANDALISM| 44725|
# |            WARRANTS| 42214|
# |            BURGLARY| 36755|
# |      SUSPICIOUS OCC| 31414|
# |      MISSING PERSON| 25989|
# |             ROBBERY| 23000|
# |               FRAUD| 16679|
# |FORGERY/COUNTERFE...| 10609|
# |     SECONDARY CODES|  9985|
# |         WEAPON LAWS|  8555|
# |        PROSTITUTION|  7484|
# |            TRESPASS|  7326|
# |     STOLEN PROPERTY|  4540|
# |SEX OFFENSES FORC...|  4388|
# +--------------------+------+


## Data processing / Model Pipeline
# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="words", pattern="\\W")
# stop words
add_stopwords = ["http","https","amp","rt","t","c","the"] 
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
# bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
# encodes label col
label_stringIdx = StringIndexer(inputCol = "Category", outputCol = "label")

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])

# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.show(5)
# +--------------+--------------------+--------------------+--------------------+--------------------+-----+
# |      Category|            Descript|               words|            filtered|            features|label|
# +--------------+--------------------+--------------------+--------------------+--------------------+-----+
# |      WARRANTS|      WARRANT ARREST|   [warrant, arrest]|   [warrant, arrest]|(809,[17,32],[1.0...|  7.0|
# |OTHER OFFENSES|TRAFFIC VIOLATION...|[traffic, violati...|[traffic, violati...|(809,[11,17,35],[...|  1.0|
# |OTHER OFFENSES|TRAFFIC VIOLATION...|[traffic, violati...|[traffic, violati...|(809,[11,17,35],[...|  1.0|
# | LARCENY/THEFT|GRAND THEFT FROM ...|[grand, theft, fr...|[grand, theft, fr...|(809,[0,2,3,4,6],...|  0.0|
# | LARCENY/THEFT|GRAND THEFT FROM ...|[grand, theft, fr...|[grand, theft, fr...|(809,[0,2,3,4,6],...|  0.0|
# +--------------+--------------------+--------------------+--------------------+--------------------+-----+


## Splitting 
# set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))          #Training Dataset Count: 614691
print("Test Dataset Count: " + str(testData.count()))                  #Test Dataset Count: 263358


## Model Training
## ---------- Logistic Regression ----------##
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)

predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
# +------------------------------+-------------+------------------------------+-----+----------+
# |                      Descript|     Category|                   probability|label|prediction|
# +------------------------------+-------------+------------------------------+-----+----------+
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8730172912256584,0.02054...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8730172912256584,0.02054...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8730172912256584,0.02054...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8730172912256584,0.02054...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8730172912256584,0.02054...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8730172912256584,0.02054...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8730172912256584,0.02054...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8730172912256584,0.02054...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8730172912256584,0.02054...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8730172912256584,0.02054...|  0.0|       0.0|
# +------------------------------+-------------+------------------------------+-----+----------+
 

## Evaluation
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
LR = evaluator.evaluate(predictions)         
LR   # 0.9721281687008145, the accuracy is excellent 


## ---------- Logistic Regression using TF-IDF Features ----------##
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)                  #minDocFreq: remove sparse terms

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx])
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.show(5)
# +--------------+--------------------+--------------------+--------------------+--------------------+--------------------+-----+
# |      Category|            Descript|               words|            filtered|         rawFeatures|            features|label|
# +--------------+--------------------+--------------------+--------------------+--------------------+--------------------+-----+
# |      WARRANTS|      WARRANT ARREST|   [warrant, arrest]|   [warrant, arrest]|(10000,[6760,8730...|(10000,[6760,8730...|  7.0|
# |OTHER OFFENSES|TRAFFIC VIOLATION...|[traffic, violati...|[traffic, violati...|(10000,[6,5133,87...|(10000,[6,5133,87...|  1.0|
# |OTHER OFFENSES|TRAFFIC VIOLATION...|[traffic, violati...|[traffic, violati...|(10000,[6,5133,87...|(10000,[6,5133,87...|  1.0|
# | LARCENY/THEFT|GRAND THEFT FROM ...|[grand, theft, fr...|[grand, theft, fr...|(10000,[182,3590,...|(10000,[182,3590,...|  0.0|
# | LARCENY/THEFT|GRAND THEFT FROM ...|[grand, theft, fr...|[grand, theft, fr...|(10000,[182,3590,...|(10000,[182,3590,...|  0.0|
# +--------------+--------------------+--------------------+--------------------+--------------------+--------------------+-----+

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)   #As model pipeline changed, need to rerun split

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)

predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
# +------------------------------+-------------+------------------------------+-----+----------+
# |                      Descript|     Category|                   probability|label|prediction|
# +------------------------------+-------------+------------------------------+-----+----------+
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8846868443885018,0.01886...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8846868443885018,0.01886...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8846868443885018,0.01886...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8846868443885018,0.01886...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8846868443885018,0.01886...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8846868443885018,0.01886...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8846868443885018,0.01886...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8846868443885018,0.01886...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8846868443885018,0.01886...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.8846868443885018,0.01886...|  0.0|       0.0|
# +------------------------------+-------------+------------------------------+-----+----------+        

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
TFIDF = evaluator.evaluate(predictions)              
TFIDF   # 0.9721507088140049, similar to before LR one


## ---------- Cross Validation, tune the count vectors LR ----------##
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.3, 0.5])        # regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
#            .addGrid(model.maxIter, [10, 20, 50])         # Number of iterations
#            .addGrid(idf.numFeatures, [10, 100, 1000])    # Number of features
             .build())
             
# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)

cvModel = cv.fit(trainingData)

predictions = cvModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
# +------------------------------+-------------+------------------------------+-----+----------+
# |                      Descript|     Category|                   probability|label|prediction|
# +------------------------------+-------------+------------------------------+-----+----------+
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.9488448720024348,0.00899...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.9488448720024348,0.00899...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.9488448720024348,0.00899...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.9488448720024348,0.00899...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.9488448720024348,0.00899...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.9488448720024348,0.00899...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.9488448720024348,0.00899...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.9488448720024348,0.00899...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.9488448720024348,0.00899...|  0.0|       0.0|
# |THEFT, BICYCLE, <$50, NO SE...|LARCENY/THEFT|[0.9488448720024348,0.00899...|  0.0|       0.0|
# +------------------------------+-------------+------------------------------+-----+----------+

# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
CV = evaluator.evaluate(predictions)              
CV   # 0.9918960725584134, the performance improved


## ---------- Naive Bayes ----------##
nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData)

predictions = model.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
# +----------------------------+-------------+------------------------------+-----+----------+
# |                    Descript|     Category|                   probability|label|prediction|
# +----------------------------+-------------+------------------------------+-----+----------+
# |PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.999999999986501,1.521776...|  0.0|       0.0|
# |PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.999999999986501,1.521776...|  0.0|       0.0|
# |PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.999999999986501,1.521776...|  0.0|       0.0|
# |PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.999999999986501,1.521776...|  0.0|       0.0|
# |PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.999999999986501,1.521776...|  0.0|       0.0|
# |PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.999999999986501,1.521776...|  0.0|       0.0|
# |PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.999999999986501,1.521776...|  0.0|       0.0|
# |PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.999999999986501,1.521776...|  0.0|       0.0|
# |PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.999999999986501,1.521776...|  0.0|       0.0|
# |PETTY THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.999999999986501,1.521776...|  0.0|       0.0|
# +----------------------------+-------------+------------------------------+-----+----------+
    
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
NB = evaluator.evaluate(predictions)
NB   # 0.9934900857765636, similar to CV one


## ---------- Random Forest ----------##
rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)

# Train model with Training Data
rfModel = rf.fit(trainingData)

predictions = rfModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
# +----------------------------+-------------+------------------------------+-----+----------+
# |                    Descript|     Category|                   probability|label|prediction|
# +----------------------------+-------------+------------------------------+-----+----------+
# |GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.5363087322090729,0.07639...|  0.0|       0.0|
# |GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.5363087322090729,0.07639...|  0.0|       0.0|
# |GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.5363087322090729,0.07639...|  0.0|       0.0|
# |GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.5363087322090729,0.07639...|  0.0|       0.0|
# |GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.5363087322090729,0.07639...|  0.0|       0.0|
# |GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.5363087322090729,0.07639...|  0.0|       0.0|
# |GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.5363087322090729,0.07639...|  0.0|       0.0|
# |GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.5363087322090729,0.07639...|  0.0|       0.0|
# |GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.5363087322090729,0.07639...|  0.0|       0.0|
# |GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT|[0.5363087322090729,0.07639...|  0.0|       0.0|
# +----------------------------+-------------+------------------------------+-----+----------+

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
RF = evaluator.evaluate(predictions)
RF   # 0.6630333225012168


## ---------- Comparing Models ----------##

schema= StructType([
  StructField('Model', StringType(), True),
  StructField('Accuracy', DoubleType(), True)
])

models = spark.createDataFrame(
  sc.parallelize(
    [
      Row('Logistic Regression', LR),
      Row('LR using TF-IDF', TFIDF),
      Row('LR with CV', CV),
      Row('Naive Bayes', NB),
      Row('Random Forest', RF)
    ]
  ), schema=schema)
  
models.orderBy('Accuracy', ascending=False).show()
# +-------------------+------------------+
# |              Model|          Accuracy|
# +-------------------+------------------+
# |        Naive Bayes|0.9934900857765636|
# |         LR with CV|0.9918960725584134|
# |    LR using TF-IDF|0.9721507088140049|
# |Logistic Regression|0.9721281687008145|
# |      Random Forest|0.6630333225012168|
# +-------------------+------------------+

# Random Forest performs poor at this high-dimensional sparse data.
# Navie Bayes and Logistic Regression with Cross Validation performances are the best among these models, very close to 1. 
# As LR with CV takes more time to train than NB, we would choose Naive Bayes in this experiment.