import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
import pickle


#import the data to train and test the model
wine_train = spark.read.csv('/home/hadoop/TrainingDataset.csv', inferSchema = True, header = True)
wine_valid = spark.read.csv('/home/hadoop/ValidationDataset.csv', inferSchema = True, header = True)

#convert to vector representation for MLlib									 
assembler = vectorAssembler(inputCols = ['fixed acidity',
                                     'volatile acidity',
                                     'citric acid',
                                     'residual sugar',
                                     'chlorides',
                                     'free sulfur dioxide',
                                     'total sulfur dioxide',
                                     'density',
                                     'pH',
                                     'sulphates'], 
                                     outputCol = 'features')
wine_train = assembler.transform(wine_train).select('features', 'quality')
wine_valid = assembler.transform(wine_valid).select('features', 'quality')
									 
#initialize linear regression model
lr = LinearRegression(featuresCol = 'features', labelCol='quality', maxIter = 10, regParam = 0.1, elasticNetParam = 0.5)

#fitting model
model = lr.fit(wine_train)
wine_prediction = model.transform(wine_valid)

#calculate results
r = wine_prediction.stat.corr('features', 'quality')
print("R-Squared: " + str(r ** 2))

crossval = CrossValidator(estimator=LinearRegression(labelCol = "quality"),  
                         estimatorParamMaps=ParamGridBuilder().addGrid(
                           LinearRegression.elasticNetParam, [0, 0.5, 1.0]).build(),
                         evaluator=RegressionEvaluator(
                           labelCol = "quality", metricName = "r2"),
                         numFolds=10)

#cross validate the model and choose the best fit
cvModel = crossval.fit(wine_train)
model = cvModel.bestModel

#calculate with improved model
wine_prediction = model.transform(wine_valid)
r = wine_prediction.stat.corr('features', 'quality')
print("R-squared: " + (str ** 2))

#exports model to be used 
Pkl_Filename = 'TrainedModel.pkl'
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)