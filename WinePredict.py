import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
import pickle

#import the trained model from the pickle file
with open('TrainedModel.pkl', 'rb') as file:  
    lr = pickle.load(file)

#import test dataset from home directory
wine_test = spark.read.csv('/home/hadoop/TestDataset.csv', inferSchema = True, header = True)

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
wine_test = assembler.transform(wine_test).select('features', 'quality')

#calculate with test dataset
wine_prediction = model.transform(wine_test)
r = wine_prediction.stat.corr('features', 'quality')
print("R-squared: " + (str ** 2))