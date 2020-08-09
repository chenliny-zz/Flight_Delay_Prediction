# Databricks notebook source
# MAGIC %md ## Spark Playground - Flight Delay Status Classification
# MAGIC ##### Machine Learning At Scale (Spark, Spark ML)
# MAGIC Chenlin Ye, Hongsuk Nam, Swati Akella

# COMMAND ----------

# MAGIC %md ### Project Formulation
# MAGIC 
# MAGIC Flight delays create problems in scheduling for airlines and airports, leading to passenger inconvenience, and huge economic losses. As a result, there is growing interest in predicting flight delays beforehand in order to optimize operations and improve customer satisfaction. The objective of this playground project is to predict flight departure delays two hours ahead of departure at scale. A delay is defined as 15-minute delay or greater with respect to the planned time of departure. Given that the target variable is a label of value 0 (no-delay) or 1 (delay), we framed this as a classification problem. In order to justify an accurately performing model suitable for the business purpose, false positive rate can be an effective model evaluation metrics. Throughout the project, we explore a series of data transformation and ML pipelines in Spark and conclude with challenges faced and key lessons learned.  
# MAGIC 
# MAGIC Datasets used in the project include the following:
# MAGIC - flight dataset from the [US Department of Transportation](https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time) containing flight information from 2015 to 2019 
# MAGIC <br>**(31,746,841 x 109 dataframe)**
# MAGIC - weather dataset from the [National Oceanic and Atmospheric Administration repository](https://www.ncdc.noaa.gov/orders/qclcd/) containing weather information from 2015 to 2019 
# MAGIC <br>**(630,904,436 x 177 dataframe)**
# MAGIC - airport dataset from the [US Department of Transportation](https://www.transtats.bts.gov/DL_SelectFields.asp)
# MAGIC <br>**(18,097 x 10 dataframe)**

# COMMAND ----------

# MAGIC %md ### Environment Set-up

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, Imputer
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql import SQLContext
import pyspark.ml.feature as ftr
import pyspark.ml as ml

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import PipelineModel
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import PCA

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline

sqlContext = SQLContext(sc)

# COMMAND ----------

# MAGIC %md ### Data Extraction

# COMMAND ----------

# MAGIC %md ##### Data directory

# COMMAND ----------

# Data directory
DATA_PATH = "dbfs:/mnt/mids-w261/data/datasets_final_project/"

# Create file path
# dbutils.fs.mkdirs('dbfs:/mnt/w261/team22')
FILE_PATH = 'dbfs:/mnt/w261/team22/'
display(dbutils.fs.ls(DATA_PATH))

# COMMAND ----------

# Weather data 
display(dbutils.fs.ls(DATA_PATH+"weather_data"))

# COMMAND ----------

# MAGIC %md ##### Load flight data

# COMMAND ----------

# Airline data
airlines = spark.read.option("header", "true").parquet(DATA_PATH+"parquet_airlines_data/*.parquet")

# Check the number of records loaded
f'{airlines.count():,}'

# COMMAND ----------

display(airlines)

# COMMAND ----------

# MAGIC %md ##### Load weather data

# COMMAND ----------

# Weather data
weather = spark.read.option("header", "true").parquet(DATA_PATH+"weather_data/*.parquet")

# # Check the number of records loaded
f'{weather.count():,}'

# COMMAND ----------

display(weather)

# COMMAND ----------

# MAGIC %md ##### Load airport data

# COMMAND ----------

file_location = "/FileStore/tables/193498910_T_MASTER_CORD.csv"
file_type = "csv"

infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

airport = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location)

display(airport)

# COMMAND ----------

# MAGIC %md ### Data Transformation

# COMMAND ----------

# MAGIC %md ##### Airport Data

# COMMAND ----------

# MAGIC %md Filter down to US airports only

# COMMAND ----------

# function to filter down to US airport only 
# remove duplicates (original dataset uses polygon to represent some airports; one point each airport is sufficient for us)
def airport_transform(dataframe):
  return (
    dataframe
    .filter("AIRPORT_COUNTRY_CODE_ISO = 'US'")
    .dropDuplicates(['AIRPORT'])
    )

# COMMAND ----------

# tranform
airport = airport_transform(airport)
display(airport)

# COMMAND ----------

# MAGIC %md Keep airports that co-exist in the airline dataset

# COMMAND ----------

# spark sql
airlines.createOrReplaceTempView('airlines')
airport.createOrReplaceTempView('airport')

airport = spark.sql(
  """
  SELECT * 
  FROM
    airport 
  WHERE 
    AIRPORT IN (
      SELECT DISTINCT ORIGIN FROM airlines
      UNION
      SELECT DISTINCT DEST FROM airlines
      )
  """
)


# COMMAND ----------

# MAGIC %md ##### Check point - Transformed airport data
# MAGIC - This parquet check point contains airports that co-exist in the airlines dataset

# COMMAND ----------

# Write to parquet
#dbutils.fs.rm(FILE_PATH + "airport_cleaned.parquet", recurse=True)
# airport.write.parquet(FILE_PATH + "airport_cleaned.parquet")

# COMMAND ----------

# Read from parquet
airport = spark.read.option("header", "true").parquet(FILE_PATH+"airport_cleaned.parquet")
airport.createOrReplaceTempView('airport')
f'{airport.count():,}'

# COMMAND ----------

# MAGIC %md ##### Transforming airline Data
# MAGIC - The below columns from the airlines dataset were determinted to impact flight delay.

# COMMAND ----------

# Airlines transformation function

def airlines_transform(dataframe):
  
  # Selected Columns
  selected_col = [
  "YEAR",
  "QUARTER",
  "MONTH",
  "DAY_OF_MONTH",
  "DAY_OF_WEEK",
  "FL_DATE",
  "OP_UNIQUE_CARRIER",
  "TAIL_NUM",
  "OP_CARRIER_FL_NUM",
  "ORIGIN",
  "ORIGIN_CITY_NAME",
  "ORIGIN_STATE_ABR",
  "DEST",
  "DEST_CITY_NAME",
  "DEST_STATE_ABR",
  "CRS_DEP_TIME",
  "CRS_DEP_TIME_HOUR",
  "DEP_TIME_HOUR",
  "DEP_DELAY_NEW",
  "DEP_TIME_BLK",
  "CRS_ARR_TIME",
  "CRS_ARR_TIME_HOUR",
  "ARR_TIME_HOUR",
  "ARR_DELAY_NEW",
  "ARR_TIME_BLK",
  "DISTANCE",
  "DISTANCE_GROUP",
  "DEP_DEL15",
  "ARR_DEL15",
  "ORIGIN_AIRPORT_ID",
  "DEST_AIRPORT_ID",
  "CRS_DEP_TIMESTAMP",
  "CRS_ARR_TIMESTAMP",
  "PR_ARR_DEL15"]
  
  # Creating a window partition to extract prior arrival delay for each flight
  windowSpec = Window.partitionBy("TAIL_NUM").orderBy("CRS_DEP_TIMESTAMP")
  
  return (
    dataframe
    .filter("CANCELLED != 1 AND DIVERTED != 1")
    .withColumn("FL_DATE", f.col("FL_DATE").cast("date"))
    .withColumn("OP_CARRIER_FL_NUM", f.col("OP_CARRIER_FL_NUM").cast("string"))
    .withColumn("DEP_TIME_HOUR", dataframe.DEP_TIME_BLK.substr(1, 2).cast("int"))
    .withColumn("ARR_TIME_HOUR", dataframe.ARR_TIME_BLK.substr(1, 2).cast("int"))
    .withColumn("CRS_DEP_TIME_HOUR", f.round((f.col("CRS_DEP_TIME")/100)).cast("int"))
    .withColumn("CRS_ARR_TIME_HOUR", f.round((f.col("CRS_ARR_TIME")/100)).cast("int"))
    .withColumn("DISTANCE_GROUP", f.col("DISTANCE_GROUP").cast("string"))
    .withColumn("OP_CARRIER_FL_NUM", f.concat(f.col("OP_CARRIER"),f.lit("_"),f.col("OP_CARRIER_FL_NUM")))
    .withColumn("DEP_DEL15", f.col("DEP_DEL15").cast("string"))
    .withColumn("ARR_DEL15", f.col("ARR_DEL15").cast("string"))
    .withColumn("FL_DATE_string", f.col("FL_DATE").cast("string"))
    .withColumn("YEAR", f.col("YEAR").cast("string"))
    .withColumn("QUARTER", f.col("QUARTER").cast("string"))
    .withColumn("MONTH", f.col("MONTH").cast("string"))
    .withColumn("DAY_OF_MONTH", f.col("DAY_OF_MONTH").cast("string"))
    .withColumn("DAY_OF_WEEK", f.col("DAY_OF_WEEK").cast("string"))
    .withColumn("CRS_DEP_TIME_string", f.col("CRS_DEP_TIME").cast("string"))
    .withColumn("CRS_ARR_TIME_string", f.col("CRS_ARR_TIME").cast("string"))
    .withColumn("CRS_DEP_TIME_HOUR_string", f.col("CRS_DEP_TIME_HOUR").cast("string"))
    .withColumn("CRS_ARR_TIME_HOUR_string", f.col("CRS_ARR_TIME_HOUR").cast("string"))
    .withColumn("CRS_DEP_TIME_HH", f.lpad("CRS_DEP_TIME_string", 4, '0').substr(1,2))
    .withColumn("CRS_DEP_TIME_MM", f.lpad("CRS_DEP_TIME_string", 4, '0').substr(3,2))
    .withColumn("CRS_ARR_TIME_HH", f.lpad("CRS_ARR_TIME_string", 4, '0').substr(1,2))
    .withColumn("CRS_ARR_TIME_MM", f.lpad("CRS_ARR_TIME_string", 4, '0').substr(3,2))
    .withColumn("CRS_DEP_TIMESTAMP", f.concat(f.col("FL_DATE_string"),f.lit(" "),f.col("CRS_DEP_TIME_HH"), f.lit(":"),f.col("CRS_DEP_TIME_MM")).cast("timestamp"))
    .withColumn("CRS_ARR_TIMESTAMP", f.concat(f.col("FL_DATE_string"),f.lit(" "),f.col("CRS_ARR_TIME_HH"), f.lit(":"),f.col("CRS_ARR_TIME_MM")).cast("timestamp"))
    .withColumn("CRS_ELAPSED_TIME", f.round((f.col("CRS_ELAPSED_TIME")/60)).cast("int"))
    .withColumn("PR_ARR_DEL15", f.lag(f.col("ARR_DEL15"), 1).over(windowSpec).cast("string"))
    .select(selected_col)
    )

# COMMAND ----------

# Transform
airlines = airlines_transform(airlines)

# COMMAND ----------

# MAGIC %md ##### Check point - transformed airlines data

# COMMAND ----------

# write to parquet
# dbutils.fs.rm(FILE_PATH + "airlines_cleaned.parquet", recurse=True)
# airlines.write.parquet(FILE_PATH + "airlines_cleaned.parquet")

# COMMAND ----------

# Read from parquet
airlines = spark.read.option("header", "true").parquet(FILE_PATH+"airlines_cleaned.parquet")
airlines.createOrReplaceTempView('airlines')
f'{airlines.count():,}'

# COMMAND ----------

display(airlines)

# COMMAND ----------

# MAGIC %md ##### Transforming Weather Data
# MAGIC The weather dataset contains columns and rows that do not pertain to aviation. We narrowed down the dataset using five transformation steps.

# COMMAND ----------

# MAGIC %md ##### Transforming weather data - part 1
# MAGIC - Filter weather dataset to US and Report Type "FM-15"

# COMMAND ----------

# Filter out US only
# Keep weather records for FM15 report type

def weather_transformation_reduction(dataframe, shorlisted_weather_cols):
  return (
    dataframe
      .withColumn("COUNTRY", f.substring(f.col("NAME"), -2, 2))
      .filter("COUNTRY = 'US'")
      .filter("REPORT_TYPE LIKE '%FM-15%'")
      .select(shorlisted_weather_cols)
  )

shorlisted_weather_cols = ["STATION", "DATE", "LATITUDE", 'LONGITUDE', 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'AA1', 'AJ1', 'AT1', 'GA1', 'IA1', 'MA1']

# COMMAND ----------

# MAGIC %md ##### Check point - transformed weather data 
# MAGIC - US and FM-15 report type

# COMMAND ----------

# Writing to Parquet
# dbutils.fs.rm(FILE_PATH + "weather_us.parquet", recurse=True)
# weather_transformation_reduction(weather, shorlisted_weather_cols).write.parquet(FILE_PATH + "weather_us.parquet")

# COMMAND ----------

weather = spark.read.option("header", "true").parquet(FILE_PATH+"weather_us.parquet")
weather.createOrReplaceTempView('weather')

# COMMAND ----------

# MAGIC %md ##### Transforming weather data - part 2
# MAGIC - Keep station records that co-exist in the airport dataset

# COMMAND ----------

airport = spark.read.option("header", "true").parquet(FILE_PATH+"airport_cleaned.parquet")
# create sql view
airport.createOrReplaceTempView('airport')

# COMMAND ----------

# MAGIC %md Weather station to airport spatial join - so each airport can have a closest weather station id attached

# COMMAND ----------

# weather stations table with coordinates

weather_coordinates = spark.sql(
"""
SELECT 
  DISTINCT STATION, CALL_SIGN, LATITUDE, LONGITUDE
FROM 
  weather
"""
)


# COMMAND ----------

weather_coordinates = spark.read.option("header", "true").parquet(FILE_PATH + "weather_stations.parquet")
weather_coordinates_pdf = weather_coordinates.toPandas()
#weather_coordinates_pdf

# COMMAND ----------

airport_pdf = airport.toPandas()
airport_pdf

# COMMAND ----------

# Eclidean Distance calculation
X_coordinates = airport_pdf[['LATITUDE', 'LONGITUDE']]
Y_coordinates = weather_coordinates_pdf[['LATITUDE', 'LONGITUDE']]

weather_station_idx = metrics.pairwise_distances_argmin_min(X_coordinates, Y_coordinates, metric='euclidean')[0]
weather_station_idx

# COMMAND ----------

station_id = [weather_coordinates_pdf.iloc[i]['STATION'] for i in weather_station_idx]
station_id_weather_filter = spark.createDataFrame(station_id,StringType())
station_id_weather_filter.createOrReplaceTempView('station_id_weather_filter')

# COMMAND ----------

# MAGIC %md ##### Check point - transformed weather data
# MAGIC - weather data with stations co-exist in the airport dataset

# COMMAND ----------

dbutils.fs.rm(FILE_PATH + "weather_us_stations.parquet", recurse=True)
weather = weather.where(f.col("STATION").isin(set(station_id)))
# Write to parquet
weather.write.parquet(FILE_PATH + "weather_us_stations.parquet")
# Create SQL View
weather.createOrReplaceTempView('weather')

# COMMAND ----------

weather = spark.read.option("header", "true").parquet(FILE_PATH+"weather_us_stations.parquet")
weather.createOrReplaceTempView('weather')

# COMMAND ----------

display(weather)

# COMMAND ----------

# MAGIC %md ##### Transforming weather data - part 3
# MAGIC - Weather feature extraction and tarnsformation
# MAGIC   - Extract relevant features that would affect to the airline delay.
# MAGIC   - Fill missing and erroneous values with Null.
# MAGIC   - Parsed out substrings into new columns.
# MAGIC   - Assign erroneous data which codes are 3 and 7 to "999" which indicates missing values.
# MAGIC   - Drop unnecessary columns.

# COMMAND ----------

def weather_transformation(dataframe):  
  return (
    dataframe
       # Mandatory data section - WND - Create substring columns to parse out values delimited by ","
      .withColumn("WND_temp", f.substring_index("WND", ",", -2))\
      .withColumn("WND_SPEED", f.substring_index("WND_temp", ",", 1))\
      .withColumn("WND_SPEED_QUALITY", f.substring_index("WND_temp", ",", -1))\
      # Filter out erroneous data
      .withColumn("WND_SPEED_QUALITY", f.when((f.col("WND_SPEED_QUALITY") == "3") | (f.col("WND_SPEED_QUALITY") == "7") , "999").otherwise(f.col("WND_SPEED_QUALITY")))\
      # Fill/Change missing values to Null
      .withColumn("WND_SPEED", f.when((f.col("WND_SPEED") == "") | (f.col("WND_SPEED") == "9999") | (f.col("WND_SPEED_QUALITY") == "999"), None).otherwise(f.col("WND_SPEED")))\
      # Drop unnecessary columns
      .drop("WND_temp","WND", "WND_SPEED_QUALITY")\
      # Mandatory data section - CIG - Create substring columns to parse out multiple values delimited by ","
      .withColumn("CIG_CAVOC", f.substring_index("CIG", ",", -1))\
      # Change missing values to Null
      .withColumn("CIG_CAVOC", f.when((f.col("CIG_CAVOC") == "") | (f.col("CIG_CAVOC") == "9"), None).otherwise(f.col("CIG_CAVOC")))\
      # Drop unnecessary columns
      .drop("CIG")\
      # Mandatory data section - VIS - Create substring columns to parse out multiple values delimited by ","
      .withColumn("VIS_temp", f.substring_index("VIS", ",", 2))\
      .withColumn("VIS_DISTANCE", f.substring_index("VIS_temp", ",", 1))\
      .withColumn("VIS_DISTANCE_QUALITY", f.substring_index("VIS_temp", ",", -1))\
      # Filter out erroneous data
      .withColumn("VIS_DISTANCE_QUALITY", f.when((f.col("VIS_DISTANCE_QUALITY") == "3") | (f.col("VIS_DISTANCE_QUALITY") == "7"), "999").otherwise(f.col("VIS_DISTANCE_QUALITY")))\
      # Fill/Change the missing values to Null
      .withColumn("VIS_DISTANCE", f.when((f.col("VIS_DISTANCE") == "") | (f.col("VIS_DISTANCE") == "999999") | (f.col("VIS_DISTANCE_QUALITY") == "999"), None).otherwise(f.col("VIS_DISTANCE")))\
      # Drop unnecessary columns
      .drop("VIS_temp", "VIS_DISTANCE_QUALITY", "VIS")\
      # Mandatory data section - TMP - Create substring columns to parse out multiple values delimited by ","
      .withColumn("TMP_TEMP", f.substring_index("TMP", ",", 1))\
      .withColumn("TMP_TEMP_QUALITY", f.substring_index("TMP", ",", -1))\
      # Filter out erroneous data
      .withColumn("TMP_TEMP_QUALITY", f.when((f.col("TMP_TEMP_QUALITY") == "3") | (f.col("TMP_TEMP_QUALITY") == "7"), "999").otherwise(f.col("TMP_TEMP_QUALITY")))\
      # Fill/Change the missing values to Null
      .withColumn("TMP_TEMP", f.when((f.col("TMP_TEMP") == "") | (f.col("TMP_TEMP") == "+9999") | (f.col("TMP_TEMP_QUALITY") == "999"), None).otherwise(f.col("TMP_TEMP")))\
      # Drop unnecessary columns
      .drop("TMP_TEMP_QUALITY", "TMP")\
      # Mandatory data section - DEW - Create substring columns to parse out multiple values delimited by ","
      .withColumn("DEW_TEMP", f.substring_index("DEW", ",", 1))\
      .withColumn("DEW_TEMP_QUALITY", f.substring_index("DEW", ",", -1))\
      # Filter out erroneous data
      .withColumn("DEW_TEMP_QUALITY", f.when((f.col("DEW_TEMP_QUALITY") == "3") | (f.col("DEW_TEMP_QUALITY") == "7"), "999").otherwise(f.col("DEW_TEMP_QUALITY")))\
      # Fill/Change missing values to Null
      .withColumn("DEW_TEMP", f.when((f.col("DEW_TEMP") == "") | (f.col("DEW_TEMP") == "+9999") | (f.col("DEW_TEMP_QUALITY") == "999"), None).otherwise(f.col("DEW_TEMP")))\
      # Drop unnecessary columns
      .drop("DEW_TEMP_QUALITY", "DEW")\
      # Mandatory data section - SLP - Create substring columns to parse out multiple values delimited by ","
      .withColumn("SLP_PRESSURE", f.substring_index("SLP", ",", 1))\
      .withColumn("SLP_PRESSURE_QUALITY", f.substring_index("SLP", ",", -1))\
      # Filter out erroneous data
      .withColumn("SLP_PRESSURE_QUALITY", f.when((f.col("SLP_PRESSURE_QUALITY") == "3") | (f.col("SLP_PRESSURE_QUALITY") == "7"), "999").otherwise(f.col("SLP_PRESSURE_QUALITY")))\
      # Fill/Change missing values to Null
      .withColumn("SLP_PRESSURE", f.when((f.col("SLP_PRESSURE") == "") | (f.col("SLP_PRESSURE") == "99999") | (f.col("SLP_PRESSURE_QUALITY") == "999"), None).otherwise(f.col("SLP_PRESSURE")))\
      # Drop unnecessary columns
      .drop("SLP_PRESSURE_QUALITY", "SLP" )\
      # Additional data section - AA1 - Create substring columns to parse out multiple values delimited by ","
      .withColumn("AA1_temp", f.substring_index("AA1", ",", -3))\
      .withColumn("PRECIPITATION", f.substring_index("AA1_temp", ",", 1))\
      .withColumn("PRECIPITATION_QUALITY", f.substring_index("AA1_temp", ",", -1))\
      # Filter out erroneous data
      .withColumn("PRECIPITATION_QUALITY", f.when((f.col("PRECIPITATION_QUALITY") == "3") | (f.col("PRECIPITATION_QUALITY") == "7"), "999").otherwise(f.col("PRECIPITATION_QUALITY")))\
      # Fill/Change missing values to Null
      .withColumn("PRECIPITATION", f.when((f.col("PRECIPITATION") == "") | (f.col("PRECIPITATION") == "9999") | (f.col("PRECIPITATION_QUALITY") == "999"), None).otherwise(f.col("PRECIPITATION")))\
      # Drop unnecessary columns
      .drop("AA1_temp", "AA1", "PRECIPITATION_QUALITY")\
      # Additional data section - AJ1 - Create substring columns to parse out multiple values delimited by ","
      .withColumn("AJ1_temp", f.substring_index("AJ1", ",", 3))\
      .withColumn("SNOW", f.substring_index("AJ1_temp", ",", 1))\
      .withColumn("SNOW_QUALITY", f.substring_index("AJ1_temp", ",", -1))\
      # Filter out erroneous data
      .withColumn("SNOW_QUALITY", f.when((f.col("SNOW_QUALITY") == "3") | (f.col("SNOW_QUALITY") == "7"), "999").otherwise(f.col("SNOW_QUALITY")))\
      # Fill/Change missing values to Null
      .withColumn("SNOW", f.when((f.col("SNOW") == "") | (f.col("SNOW") == "9999") | (f.col("SNOW_QUALITY") == "999"), None).otherwise(f.col("SNOW")))\
      # Drop unnecessary columns
      .drop("AJ1_temp", "AJ1", "SNOW_QUALITY")\
      # Additional data section - AT1 - Create substring columns to parse out multiple values delimited by ","
      .withColumn("AT1_temp", f.substring_index("AT1", ",", -3))\
      .withColumn("WEATHER_OBSERVATION", f.substring_index("AT1_temp", ",", 1))\
      .withColumn("WEATHER_OBSERVATION_QUALITY", f.substring_index("AT1_temp", ",", -1))\
      # Filter out erroneous data
      .withColumn("WEATHER_OBSERVATION_QUALITY", f.when((f.col("WEATHER_OBSERVATION_QUALITY") == "3") | (f.col("WEATHER_OBSERVATION_QUALITY") == "7"), "999").otherwise(f.col("WEATHER_OBSERVATION_QUALITY")))\
      # Fill/Change missing values to Null
      .withColumn("WEATHER_OBSERVATION", f.when((f.col("WEATHER_OBSERVATION") == "") | (f.col("WEATHER_OBSERVATION_QUALITY") == "999"), None).otherwise(f.col("WEATHER_OBSERVATION")))\
      # Drop unnecessary columns
      .drop("AT1", "AT1_temp", "WEATHER_OBSERVATION_QUALITY")\
      # Additional data section - GA1 - Create substring columns to parse out multiple values delimited by ","
      .withColumn("GA1_temp", f.substring_index("GA1", ",", 4))\
      .withColumn("GA1_temp2", f.substring_index("GA1_temp", ",", 2))\
      .withColumn("GA1_temp3", f.substring_index("GA1_temp", ",", -2))\
      .withColumn("CLOUD_COVERAGE", f.substring_index("GA1_temp2", ",", 1))\
      .withColumn("CLOUD_COVERAGE_QUALITY", f.substring_index("GA1_temp2", ",", -1))\
      # Filter out erroneous data
      .withColumn("CLOUD_COVERAGE_QUALITY", f.when((f.col("CLOUD_COVERAGE_QUALITY") == "3") | (f.col("CLOUD_COVERAGE_QUALITY") == "7"), "999").otherwise(f.col("CLOUD_COVERAGE_QUALITY")))\
      # Fill/Change missing values to Null
      .withColumn("CLOUD_COVERAGE", f.when((f.col("CLOUD_COVERAGE") == "") | (f.col("CLOUD_COVERAGE") == "99") | (f.col("CLOUD_COVERAGE") == "9") | (f.col("CLOUD_COVERAGE") == "10") | (f.col("CLOUD_COVERAGE_QUALITY") == "999"), None).otherwise(f.col("CLOUD_COVERAGE")))\
      # Drop unnecessary columns
      .drop("GA1", "GA1_temp", "GA1_temp2", "CLOUD_COVERAGE_QUALITY")\
      # Additional data section - GA1 - Create substring columns to parse out multiple values delimited by ","
      .withColumn("CLOUD_BASE_HEIGHT", f.substring_index("GA1_temp3", ",", 1))\
      .withColumn("CLOUD_BASE_HEIGHT_QUALITY", f.substring_index("GA1_temp3", ",", -1))\
      # Filter out erroneous data
      .withColumn("CLOUD_BASE_HEIGHT_QUALITY", f.when((f.col("CLOUD_BASE_HEIGHT_QUALITY") == "3") | (f.col("CLOUD_BASE_HEIGHT_QUALITY") == "7"), "999").otherwise(f.col("CLOUD_BASE_HEIGHT_QUALITY")))\
      # Fill/Change missing values to Null
      .withColumn("CLOUD_BASE_HEIGHT", f.when((f.col("CLOUD_BASE_HEIGHT") == "") | (f.col("CLOUD_BASE_HEIGHT") == "+99999") | (f.col("CLOUD_BASE_HEIGHT_QUALITY") == "999"), None).otherwise(f.col("CLOUD_BASE_HEIGHT")))\
      # Drop unnecessary columns
      .drop("GA1_temp3", "CLOUD_BASE_HEIGHT_QUALITY")\
      # Additional data section - IA1 - Create substring columns to parse out multiple values delimited by ","
      .withColumn("GROUND_SURFACE", f.substring_index("IA1", ",", 1))\
      .withColumn("GROUND_SURFACE_QUALITY", f.substring_index("IA1", ",", -1))\
      # Filter out erroneous data
      .withColumn("GROUND_SURFACE_QUALITY", f.when(f.col("GROUND_SURFACE_QUALITY") == "3", "999").otherwise(f.col("GROUND_SURFACE_QUALITY")))\
      # Fill/Change missing values to Null
      .withColumn("GROUND_SURFACE", f.when((f.col("GROUND_SURFACE") == "") | (f.col("GROUND_SURFACE") == "99") | (f.col("GROUND_SURFACE_QUALITY") == "999"), None).otherwise(f.col("GROUND_SURFACE")))\
      # Drop unnecessary columns
      .drop("IA1", "GROUND_SURFACE_QUALITY" )\
      # Additional data section - MA1 - Create substring columns to parse out multiple values delimited by ","
      .withColumn("MA1_temp", f.substring_index("MA1", ",", 2))\
      .withColumn("ALTIMETER_SET", f.substring_index("MA1_temp", ",", 1))\
      .withColumn("ALTIMETER_SET_QUALITY", f.substring_index("MA1_temp", ",", -1))\
      # Filter out erroneous data
      .withColumn("ALTIMETER_SET_QUALITY", f.when((f.col("ALTIMETER_SET_QUALITY") == "3") | (f.col("ALTIMETER_SET_QUALITY") == "7"), "999").otherwise(f.col("ALTIMETER_SET_QUALITY")))\
      # Fill/Change missing values to Null
      .withColumn("ALTIMETER_SET", f.when((f.col("ALTIMETER_SET") == "") | (f.col("ALTIMETER_SET") == "99999") | (f.col("ALTIMETER_SET_QUALITY") == "999"), None).otherwise(f.col("ALTIMETER_SET")))\
      # Drop unnecessary columns
      .drop("MA1", "MA1_temp", "ALTIMETER_SET_QUALITY")
  )

# COMMAND ----------

# MAGIC %md ##### Check point - weather with transformed features

# COMMAND ----------

# transform
weather = weather_transformation(weather)

# save to parquet
dbutils.fs.rm(FILE_PATH + "weather_column_transform.parquet", recurse=True)
weather.write.parquet(FILE_PATH + "weather_column_transform.parquet")

# COMMAND ----------

# read from parquet
weather = spark.read.option("header", "true").parquet(FILE_PATH+"weather_column_transform.parquet")
weather.createOrReplaceTempView('weather')

# COMMAND ----------

# MAGIC %md ##### Transforming weather data - part 4
# MAGIC - Weather data UTC time to local time conversion to match with airline dataset which uses local time zone

# COMMAND ----------

# Function to obtain timezone IDs from coordinates
def get_timezone(longitude, latitude):
    from timezonefinder import TimezoneFinder
    tzf = TimezoneFinder()
    return tzf.timezone_at(lng=longitude, lat=latitude)
  
# Initiate udf function
udf_timezone = f.udf(get_timezone, StringType()) 

# transform
weather = weather.withColumn("LOCAL", udf_timezone(weather.LONGITUDE, weather.LATITUDE))

# COMMAND ----------

# Converting UTC timezone to Local timezone
weather = weather.withColumn("LOCAL_DATE", f.from_utc_timestamp(f.col("DATE"), weather.LOCAL))

# COMMAND ----------

# Create new columns in weather data, parsing out timestamps
weather = weather.withColumn("DATE_PART", f.to_date(f.col("LOCAL_DATE")))\
         .withColumn("HOUR_PART", f.hour(f.col("LOCAL_DATE")).cast("int"))\
         .withColumn("MINUTE_PART", f.minute(f.col("LOCAL_DATE")).cast("int"))\
         .withColumn("CALL_SIGN", f.trim(f.col("CALL_SIGN")))\
         .drop("LOCAL", "LOCAL_DATE")

# COMMAND ----------

# MAGIC %md ##### Check point - weather with converted timestamps 

# COMMAND ----------

#write to parquet
dbutils.fs.rm(FILE_PATH + "weather_timestamp.parquet", recurse=True)
weather.write.parquet(FILE_PATH + "weather_timestamp.parquet")

# COMMAND ----------

# read from parquet
weather = spark.read.option("header", "true").parquet(FILE_PATH+"weather_timestamp.parquet")
weather.createOrReplaceTempView('weather')

# COMMAND ----------

# MAGIC %md ##### Transforming weather data - part 5
# MAGIC - Weather data type casting

# COMMAND ----------

# data type casting
cast_columns = ["WND_SPEED", "VIS_DISTANCE", "TMP_TEMP", "DEW_TEMP", "SLP_PRESSURE", "PRECIPITATION", "SNOW", "CLOUD_COVERAGE", "CLOUD_BASE_HEIGHT", "ALTIMETER_SET"]

for c in cast_columns:
  weather = weather.withColumn(c, weather[c].cast("int"))

# COMMAND ----------

display(weather)

# COMMAND ----------

# MAGIC %md ##### Check point - Weather with converted timestamps and appropriate data type

# COMMAND ----------

# write to parquet
dbutils.fs.rm(FILE_PATH + "weather_timestamp_wCasting.parquet", recurse=True)
weather.write.parquet(FILE_PATH + "weather_timestamp_wCasting.parquet")

# read from parquet
weather = spark.read.option("header", "true").parquet(FILE_PATH+"weather_timestamp_wCasting.parquet")
weather.createOrReplaceTempView('weather')

# COMMAND ----------

# MAGIC %md ### Table join 
# MAGIC In order to make logical connections between the weather dataset and the airlines dataset, we would need to join the two datasets together with the airport dataset as the helper table. 
# MAGIC - Identify the nearest (in Euclidean distance) weather station to each airport so that each airport record has a corresponding weather station ID
# MAGIC - Join the airport dataset to the airlines dataset (on airport code) so that each departing flight has a corresponding weather station ID
# MAGIC - Bring scheduled departure timestamp for each flight to 2 hours prior so that we can obtain weather information at least 2 hours prior to departure (weather timestamp in UTC already converted to local timestamps in the previous section)
# MAGIC - Join the weather dataset to the airlines dataset on the following conditions:
# MAGIC   - matching weather station ID
# MAGIC   - for each tiemstamp (2-hour prior to scheduled departure) in the airlines dataset, join by the nearest timestamp in the weather dataset 

# COMMAND ----------

# MAGIC %md ##### Weather station to airport join

# COMMAND ----------

# Create station id list based on the statation id index generated from the Euclidean distance calculation
station_id = [weather_coordinates_pdf.iloc[i]['STATION'] for i in weather_station_idx]

# Create weather_station_id column in the airport dataset with corresponding station id
airport_pdf['weather_station_id'] = station_id

# Convert airport dataframe from pandas to Spark
airport = spark.createDataFrame(airport_pdf)

# create sql view
airport.createOrReplaceTempView('airport')

display(airport)

# COMMAND ----------

# MAGIC %md ##### Airport to airline join

# COMMAND ----------

# spark sql - airport to airlines join
airlines = spark.sql(
"""
SELECT 
  airline.*,
  airport_origin.LATITUDE AS ORIGIN_LATITUDE,
  airport_origin.LONGITUDE AS ORIGIN_LONGITUDE,
  airport_origin.weather_station_id AS weather_station,
  airport_destination.LATITUDE AS DESTINATION_LATITUDE,
  airport_destination.LONGITUDE AS DESTINATION_LONGITUDE
FROM
  airlines airline
  LEFT JOIN airport airport_origin
    ON airline.ORIGIN = airport_origin.AIRPORT
  LEFT JOIN airport airport_destination
    ON airline.DEST = airport_destination.AIRPORT
"""
)

display(airlines)

# COMMAND ----------

# MAGIC %md ##### Check point - airlines with airport information

# COMMAND ----------

# write to parquet
dbutils.fs.rm(FILE_PATH + "airlines_wairport.parquet", recurse=True)
airlines.write.parquet(FILE_PATH + "airlines_wairport.parquet")

# COMMAND ----------

# Read from parquet
airlines = spark.read.option("header", "true").parquet(FILE_PATH+"airlines_wairport.parquet")
airlines.createOrReplaceTempView('airlines')
f'{airlines.count():,}'

# COMMAND ----------

# MAGIC %md ##### Weather to airline join

# COMMAND ----------

# MAGIC %md Create two-hour prior timestamps for airlines based on scheduled departure times

# COMMAND ----------

# Bring scheduled departure time to 2 hours prior
airlines = airlines.withColumn('TWO_HOUR', airlines.CRS_DEP_TIMESTAMP + f.expr('INTERVAL -2 HOURS'))
display(airlines)

# COMMAND ----------

# MAGIC %md Create timestamp usind date/hour/minute part

# COMMAND ----------

weather = weather.withColumn("AL_JOIN_DATE", f.col("DATE_PART").cast("string"))\
                        .withColumn("AL_JOIN_HOUR", f.col("HOUR_PART").cast("string"))\
                        .withColumn("AL_JOIN_MINUTE", f.col("MINUTE_PART").cast("string"))\
                        .withColumn("AL_JOIN_TIMESTAMP", f.concat(f.col("AL_JOIN_DATE"),f.lit(" "),f.col("AL_JOIN_HOUR"), f.lit(":"),f.col("AL_JOIN_MINUTE")).cast("timestamp"))

# COMMAND ----------

# Create helper timestamp column for the airlines to weather join
# Convert timestamp to unix for better performance
# Reduce ThisTimeStamp by a second to avoid overlapping ranges during join

windowSpecJoin = Window.partitionBy("STATION").orderBy("AL_JOIN_TIMESTAMP")
weather_test = weather.withColumn("ThisTimeStamp", f.unix_timestamp(f.col("AL_JOIN_TIMESTAMP"))).withColumn("NextTimeStamp", f.lead(f.col("ThisTimeStamp"), 1).over(windowSpecJoin) -1)

# COMMAND ----------

# Convert airlines timestamp to unix for airlines to weather join.
airlines_temp_join = airlines.withColumn("AL_ThisTimeStamp", f.unix_timestamp(f.col("TWO_HOUR")))

# COMMAND ----------

display(airlines_temp_join)

# COMMAND ----------

# MAGIC %md Weather to airlines left join
# MAGIC   - matching Station ID
# MAGIC   - nearest timestamp partition by station ID for each tiemstamp in airlines

# COMMAND ----------

airlines_temp_join.repartition(363, "weather_station").createOrReplaceTempView('airlines_temp_join')
weather_test.repartition(363, "STATION").createOrReplaceTempView('weather_test')

airlines_weather_leftjoin = spark.sql("""
SELECT a.*, w.*
FROM
  airlines_temp_join a
LEFT JOIN weather_test w
ON
  a.weather_station = w.STATION
  AND a.AL_ThisTimeStamp BETWEEN w.ThisTimeStamp AND w.NextTimeStamp
  """
)

# COMMAND ----------

display(airlines_weather_leftjoin)

# COMMAND ----------

# MAGIC %md ##### Check point - Joined airlines and weather data (Final check point for joined airlines & weather data)

# COMMAND ----------

# write to parquet
dbutils.fs.rm(FILE_PATH + "airlines_weather_leftjoin.parquet", recurse=True)
airlines_weather_leftjoin.repartition(10).write.parquet(FILE_PATH + "airlines_weather_leftjoin.parquet")

# COMMAND ----------

# Read from parquet
airlines_weather = spark.read.option("header", "true").parquet(FILE_PATH+"airlines_weather_leftjoin.parquet")
airlines_weather.createOrReplaceTempView('airlines_weather')
f'{airlines_weather.count():,}'

# COMMAND ----------

# MAGIC %md ### Sample EDA with pyspark and SparkSQL
# MAGIC 
# MAGIC On a dataset level, the airlines dataset has a fairly even distribution across the years. However, the dataset has over four times of on-time records than delayed ones (based on the ```DEP_DEL15``` label). This would introduce a class imbalance issue during the model training phase. 
# MAGIC 
# MAGIC Secondly, on a feature level, the distribution of continuous features would inform us the optimal imputation methods during the ML pipeline. Normally distributed features (i.e. Altimeter Set, SLP pressure, Dew temperature) are more suitable for average value imputation. Similarly, skewed features (i.e. Snow level, wind speed, cloud base height) are more suitable for median imputation. 
# MAGIC 
# MAGIC Through looking at the percentage of delayed flights grouped by various feature groupings, we can see that: 
# MAGIC - Not all carriers are created equal: some suffers more delays than others 
# MAGIC - Not all airports are created equal: some suffers more delays than others 
# MAGIC - Seasonally trends exist: more delays during summer months and holiday seasons
# MAGIC - Temporal trends exist: more delays during the afternoons and evenings 
# MAGIC - Arrival delays and departure delays track very closely to each other (potential for creating a new feature that tracks each flight's previous flight's arrival delay status)

# COMMAND ----------

airlines_weather.printSchema()

# COMMAND ----------

# Checking the number of records under each year
display(airlines_weather.select("YEAR", "DEP_DEL15")
        .groupBy("YEAR")
        .agg({"DEP_DEL15": "COUNT"})
        .withColumn("Count Of Records", f.col("count(DEP_DEL15)"))
        .orderBy("YEAR")
        .drop("count(DEP_DEL15)"))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check for class imbalance
# MAGIC SELECT 
# MAGIC   DEP_DEL15, 
# MAGIC   COUNT(DEP_DEL15) AS Count
# MAGIC FROM 
# MAGIC   airlines_weather
# MAGIC GROUP BY 
# MAGIC   DEP_DEL15
# MAGIC ORDER BY
# MAGIC   DEP_DEL15 DESC

# COMMAND ----------

# MAGIC %md ##### Feature histogram

# COMMAND ----------

numericCols = [feature for (feature, dataType) in airlines_weather.dtypes if ((dataType == "int") & (feature != "DEP_DEL15"))]

feature_hist_df = airlines_weather.select(numericCols).toPandas()
feature_hist_df.hist(figsize=(20, 20), bins=30)
plt.show()

# COMMAND ----------


target_features = [
  "CRS_DEP_TIME_HOUR",
  "DEP_DELAY_NEW",
  "CRS_ARR_TIME_HOUR",
  "ARR_DELAY_NEW",
  "DISTANCE",
  "WND_SPEED",
  "VIS_DISTANCE",
  "TMP_TEMP",
  "DEW_TEMP",
  "SLP_PRESSURE",
  "PRECIPITATION",
  "SNOW",
  "CLOUD_COVERAGE",
  "CLOUD_BASE_HEIGHT",
  "ALTIMETER_SET"  
]

corr = airlines_weather.select(target_features).toPandas().corr()

# COMMAND ----------

# MAGIC %md ##### Correlation matrix

# COMMAND ----------

# Plotting the correlation matrix
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(240, 10, as_cmap=True)
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=.5)

# COMMAND ----------

# MAGIC %md ##### Arrival and departure delay

# COMMAND ----------

# MAGIC %md Percent delay by Carrier
# MAGIC * Not all carriers are created equally
# MAGIC * Arrival delays and departure delays track each other closely 
# MAGIC * % departure delays tend to be higher than arrival delays when grouped by most other features, but some carriers tend to have higher arrival delays

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- percent delay by carrier
# MAGIC SELECT
# MAGIC   OP_UNIQUE_CARRIER AS Carrier,
# MAGIC   AVG(DEP_DEL15) AS PercentDepartureDelay,
# MAGIC   AVG(ARR_DEL15) AS PercentArrivalDelay
# MAGIC FROM
# MAGIC   airlines_weather
# MAGIC GROUP BY
# MAGIC   OP_UNIQUE_CARRIER
# MAGIC ORDER BY 
# MAGIC   2 DESC

# COMMAND ----------

# MAGIC %md Percent delay of day of week
# MAGIC * Mid-week and Saturdays tend to have least amount of delays; Mondays tend to be the worst.

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- percent delay by day of week
# MAGIC SELECT
# MAGIC   DAY_OF_WEEK AS DOW,
# MAGIC   AVG(DEP_DEL15) AS PercentDepartureDelay,
# MAGIC   AVG(ARR_DEL15) AS PercentArrivalDelay
# MAGIC FROM
# MAGIC   airlines_weather
# MAGIC GROUP BY
# MAGIC   1
# MAGIC ORDER BY 
# MAGIC   1

# COMMAND ----------

# MAGIC %md Percent delay by month 
# MAGIC * When we have the entire airlines dataset, it'd interesting to see if seasonality has any influence on delays 

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT
# MAGIC -- percent delay by month
# MAGIC   MONTH AS Month,
# MAGIC   AVG(DEP_DEL15) AS PercentDepartureDelay,
# MAGIC   AVG(ARR_DEL15) AS PercentArrivalDelay
# MAGIC FROM
# MAGIC   airlines_weather
# MAGIC GROUP BY
# MAGIC   1
# MAGIC ORDER BY 
# MAGIC   MONTH

# COMMAND ----------

# MAGIC %md Percent delay by actual departure/arrival time

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- percent delay by actual departure/arrival time block
# MAGIC WITH departure AS (
# MAGIC   SELECT
# MAGIC     DEP_TIME_BLK AS TimeBlock,
# MAGIC     AVG(DEP_DEL15) AS PercentDepartureDelay
# MAGIC   FROM
# MAGIC     airlines_weather
# MAGIC   GROUP BY
# MAGIC     1
# MAGIC   ORDER BY 
# MAGIC     1
# MAGIC   ), arrival AS (
# MAGIC   SELECT
# MAGIC   ARR_TIME_BLK AS TimeBlock,
# MAGIC   AVG(ARR_DEL15) AS PercentArrivalDelay
# MAGIC   FROM
# MAGIC     airlines_weather
# MAGIC   GROUP BY
# MAGIC     1
# MAGIC   ORDER BY 
# MAGIC     1
# MAGIC   )
# MAGIC SELECT
# MAGIC   a.TimeBlock,
# MAGIC   a.PercentArrivalDelay,
# MAGIC   d.PercentDepartureDelay
# MAGIC FROM
# MAGIC   arrival a JOIN departure d
# MAGIC   ON a.TimeBlock = d.TimeBlock
# MAGIC ORDER BY 
# MAGIC   1

# COMMAND ----------

# MAGIC %md Percent delay by scheduled departure/arrival time

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- percent delay by scheduled departure/arrival hour
# MAGIC WITH departure AS (
# MAGIC   SELECT
# MAGIC     CRS_DEP_TIME_HOUR AS Hour,
# MAGIC     AVG(DEP_DEL15) AS PercentDepartureDelay
# MAGIC   FROM
# MAGIC     airlines_weather
# MAGIC   GROUP BY
# MAGIC     1
# MAGIC   ORDER BY 
# MAGIC     1
# MAGIC   ), arrival AS (
# MAGIC   SELECT
# MAGIC   CRS_ARR_TIME_HOUR AS Hour,
# MAGIC   AVG(ARR_DEL15) AS PercentArrivalDelay
# MAGIC   FROM
# MAGIC     airlines_weather
# MAGIC   GROUP BY
# MAGIC     1
# MAGIC   ORDER BY 
# MAGIC     1
# MAGIC   )
# MAGIC SELECT
# MAGIC   a.Hour,
# MAGIC   a.PercentArrivalDelay,
# MAGIC   d.PercentDepartureDelay
# MAGIC FROM
# MAGIC   arrival a JOIN departure d
# MAGIC   ON a.Hour = d.Hour
# MAGIC ORDER BY 
# MAGIC   1

# COMMAND ----------

# MAGIC %md Percent delay by origin airport

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- departure delay
# MAGIC SELECT
# MAGIC   ORIGIN AS Carrier,
# MAGIC   AVG(DEP_DEL15) AS PercentDepartureDelay
# MAGIC FROM
# MAGIC   airlines_weather
# MAGIC GROUP BY
# MAGIC   ORIGIN
# MAGIC ORDER BY 
# MAGIC   2 DESC

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- arrival delay
# MAGIC SELECT
# MAGIC   ORIGIN AS Carrier,
# MAGIC   AVG(ARR_DEL15) AS PercentArrivalDelay
# MAGIC FROM
# MAGIC   airlines_weather
# MAGIC GROUP BY
# MAGIC   ORIGIN
# MAGIC ORDER BY 
# MAGIC   2 DESC

# COMMAND ----------

# MAGIC %md ### Features

# COMMAND ----------

# MAGIC %md ##### Imputation

# COMMAND ----------

# Create imputation candidates by type 
cols_mean = ["ALTIMETER_SET", "DEW_TEMP", "SLP_PRESSURE", "TMP_TEMP"]
cols_median = ["CLOUD_BASE_HEIGHT", "CLOUD_COVERAGE", "PRECIPITATION", "SNOW", "VIS_DISTANCE", "WND_SPEED"]

# Replace null values with mean value
for c in cols_mean:
  c_mean = airlines_weather.agg({c: 'mean'}).collect()[0][0]
  airlines_weather = airlines_weather.na.fill({c: c_mean})

# Replace null values with median value
for c in cols_median:
  c_median = airlines_weather.approxQuantile(c, [0.5],0.1)[0]
  airlines_weather = airlines_weather.na.fill({c: c_median})

# COMMAND ----------

# display the number of null values
null_check = ["ALTIMETER_SET", "DEW_TEMP", "SLP_PRESSURE", "TMP_TEMP", "CLOUD_BASE_HEIGHT", "CLOUD_COVERAGE", "PRECIPITATION", "SNOW", "VIS_DISTANCE", "WND_SPEED"] 

display(airlines_weather.select([f.count(f.when(f.isnan(c) | f.col(c).isNull(), c)).alias(c) for c in null_check]))

# COMMAND ----------

# MAGIC %md ##### Feature Selection

# COMMAND ----------

drop_features = [
 'FL_DATE',
 'ORIGIN_CITY_NAME',
 'ORIGIN_STATE_ABR',
 'DEST_CITY_NAME',
 'DEST_STATE_ABR',
 'CRS_DEP_TIME',
 'DEP_TIME_HOUR',
 'DEP_DELAY_NEW',
 'DEP_TIME_BLK',
 'CRS_ARR_TIME',
 'ARR_TIME_HOUR',
 'ARR_DELAY_NEW',
 'ARR_TIME_BLK',
 'DISTANCE_GROUP',
 'ARR_DEL15',
 'ORIGIN_AIRPORT_ID',
 'DEST_AIRPORT_ID',
 'CRS_DEP_TIMESTAMP',
 'CRS_ARR_TIMESTAMP',
 'ORIGIN_LATITUDE',
 'ORIGIN_LONGITUDE',
 'weather_station',
 'DESTINATION_LATITUDE',
 'DESTINATION_LONGITUDE',
 'TWO_HOUR',
 'AL_ThisTimeStamp',
 'STATION',
 'DATE',
 'LATITUDE',
 'LONGITUDE',
 'NAME',
 'REPORT_TYPE',
 'CALL_SIGN',
 'GROUND_SURFACE',
 'WEATHER_OBSERVATION',
 'CIG_CAVOC',
 'WND',
 'DATE_PART',
 'HOUR_PART',
 'MINUTE_PART',
 'AL_JOIN_DATE',
 'AL_JOIN_HOUR',
 'AL_JOIN_MINUTE',
 'AL_JOIN_TIMESTAMP',
 'ThisTimeStamp',
 'NextTimeStamp']

# resulting features given the drop list
features_model = list(set(airlines_weather.columns).difference(drop_features))
data_model = airlines_weather.select(features_model)
data_model.createOrReplaceTempView("data_model")
features_model

# COMMAND ----------

# MAGIC %md ##### Drop Outcome Variables with NULL Values

# COMMAND ----------

data_model = spark.sql(
"""
SELECT
  *
FROM
  data_model
WHERE
  DEP_DEL15 IS NOT NULL
"""
)

# COMMAND ----------

categoricals = [
 'YEAR',
 'DEP_DEL15',
 'DAY_OF_WEEK',
 'ORIGIN',
 'DEST',
 'CRS_ARR_TIME_HOUR',
 'CRS_DEP_TIME_HOUR',
 'OP_UNIQUE_CARRIER',
 'MONTH',
 'DAY_OF_MONTH',
 'OP_CARRIER_FL_NUM',
 'QUARTER',
 'PR_ARR_DEL15']

Y = 'DEP_DEL15'

numerics = list(set(features_model).difference(categoricals))
numerics

# COMMAND ----------

data_model = data_model.select([f.col(feature).cast("int") for feature in numerics] + [f.col(feature).cast("string") for feature in categoricals])

# COMMAND ----------

# MAGIC %md ##### Split dataset into train, validation & test sets

# COMMAND ----------

data_model.createOrReplaceTempView('data_model')

# COMMAND ----------

# Cache data_model
data_model.cache()

# Filter out 2019 data (2019 data reserved as test set)
preTrainRDD = spark.sql(
"""
SELECT
  *
FROM
  data_model
WHERE
  YEAR != "2019"
"""
)

# COMMAND ----------

# Verify that 2019 data is not in dataset
display(preTrainRDD.select("YEAR").distinct())

# COMMAND ----------

# Keep 2019 data aside for testing 
testRDD = spark.sql(
"""
SELECT
  *
FROM
  data_model
WHERE
  YEAR == "2019"
""")

# COMMAND ----------

# Verify that only 2019 data is in the test dataset
display(testRDD.select("YEAR").distinct())

# COMMAND ----------

# Split the 2015-2018 data into train and validation
trainRDD, validationRDD = preTrainRDD.randomSplit([0.8,0.2], seed = 42)

# Check the number of records for each dataset
print(f"... train dataset has {trainRDD.count()} records for evaluation")
print(f"... validation dataset has {validationRDD.count()} records for evaluation")
print(f"... test dataset has {testRDD.count()} records for evaluation")

# COMMAND ----------

display(trainRDD)

# COMMAND ----------

display(validationRDD)

# COMMAND ----------

display(testRDD)

# COMMAND ----------

# MAGIC %md ##### Check point - trainRDD, validationRDD, testRDD

# COMMAND ----------

# write to parquet

# trainRDD
dbutils.fs.rm(FILE_PATH + "trainRDD2.parquet", recurse=True)
trainRDD.repartition(10).write.parquet(FILE_PATH + "trainRDD2.parquet")

# validationRDD
dbutils.fs.rm(FILE_PATH + "validationRDD2.parquet", recurse=True)
validationRDD.repartition(10).write.parquet(FILE_PATH + "validationRDD2.parquet")

# testRDD
dbutils.fs.rm(FILE_PATH + "testRDD2.parquet", recurse=True)
testRDD.repartition(10).write.parquet(FILE_PATH + "testRDD2.parquet")

# COMMAND ----------

# Read from parquet
trainRDD = spark.read.option("header", "true").parquet(FILE_PATH+"trainRDD2.parquet")
validationRDD = spark.read.option("header", "true").parquet(FILE_PATH+"validationRDD2.parquet")
testRDD = spark.read.option("header", "true").parquet(FILE_PATH+"testRDD2.parquet")


# Checking the number of records for each dataset
print(f"... train dataset has {trainRDD.count()} records for evaluation")
print(f"... validation dataset has {validationRDD.count()} records for evaluation")
print(f"... test dataset has {testRDD.count()} records for evaluation")

# COMMAND ----------

trainRDD.printSchema()

# COMMAND ----------

# MAGIC %md #####PCA demo on numeric features

# COMMAND ----------

numerics = ['CLOUD_BASE_HEIGHT',
 'ALTIMETER_SET',
 'PRECIPITATION',
 'SNOW',
 'SLP_PRESSURE',
 'TMP_TEMP',
 'DEW_TEMP',
 'DISTANCE',
 'CLOUD_COVERAGE',
 'VIS_DISTANCE',
 'WND_SPEED']

vector_assembler = VectorAssembler(inputCols = numerics,
outputCol = 'features')
vector_trainRDD = vector_assembler.transform(trainRDD).select("features")

standardizer = StandardScaler(withMean=True, withStd=True,
inputCol='features', outputCol='std_features').fit(vector_trainRDD)

scaled_trainRDD = standardizer.transform(vector_trainRDD)

pca = PCA(k=11, inputCol='std_features', outputCol="pca_features").fit(scaled_trainRDD)

pca_trainRDD = pca.transform(scaled_trainRDD)
explained_var = pca.explainedVariance.toArray()

# COMMAND ----------

list_var = list(explained_var)
plot_var=[sum(list_var[0:x+1]) for x in range(0,len(list_var))]
plot_var

# COMMAND ----------

# plot principal components and explained variance
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(plot_var)
ax.set_title("Number of Components vs Explained Variance", fontsize=14)
ax.set_xlabel("Number of Components", fontsize=10)
ax.set_ylabel("Explained Variance", fontsize=10)

# COMMAND ----------

# MAGIC %md ### ML Pipeline Demo
# MAGIC - The ML pipeline organizes a series of transformers and estimators into a single model. While pipelines themselves are estimators, the output of ```pipeline.fit()``` will be a pipeline model, which is a transformer. 
# MAGIC - Pipelines of Decision Tree, Random Forest, and Gradient Boosting Trees are demonstrated below

# COMMAND ----------

# Drop TAIL_NUMBER
trainRDD_tree = trainRDD.drop('TAIL_NUM')
testRDD_tree = testRDD.drop('TAIL_NUM')
validationRDD_tree = validationRDD.drop('TAIL_NUM')

# sql view
trainRDD_tree.createOrReplaceTempView('trainRDD_tree')
testRDD_tree.createOrReplaceTempView('testRDD_tree')
validationRDD_tree.createOrReplaceTempView('validationRDD_tree')

# COMMAND ----------

# Write to parquet
#trainRDD
#dbutils.fs.rm(TEAM_PATH + "trainRDD.parquet", recurse=True)
trainRDD_tree.repartition(10).write.parquet(FILE_PATH + "trainRDD3.parquet")

#validationRDD
#dbutils.fs.rm(TEAM_PATH + "validationRDD.parquet", recurse=True)
validationRDD_tree.repartition(10).write.parquet(FILE_PATH + "validationRDD3.parquet")

#testRDD
#dbutils.fs.rm(TEAM_PATH + "testRDD.parquet", recurse=True)
testRDD_tree.repartition(10).write.parquet(FILE_PATH + "testRDD3.parquet")

# COMMAND ----------

# Read from parquet 
trainRDD = spark.read.option("header", "true").parquet(FILE_PATH+"trainRDD3.parquet")
validationRDD = spark.read.option("header", "true").parquet(FILE_PATH+"validationRDD3.parquet")
testRDD = spark.read.option("header", "true").parquet(FILE_PATH+"testRDD3.parquet")

# COMMAND ----------

# MAGIC %md ##### Decision Tree Classifier 
# MAGIC - single tree with ```maxDepth``` of 10
# MAGIC - ```maxBins``` (number of bins used) transforms continuous features into discrete features. The value of ```maxBins``` need to be >= the max number of categories for any categorical feature 
# MAGIC - ```VectorAssembler``` is used to transform all input features to be contained within a single vector in the dataframe, which is needed by many algorithms in Spark. ```VectorAssembler``` takes a list of input columns and creates a new column (named "features" in this case) that combines all input columns into a single vector.
# MAGIC - ```StringIndexer``` is used to encode columns of categories into a columns of indices, with ordering represents the basis of popularity.

# COMMAND ----------

# Initiate Decision Tree classifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth=10)

# obtain numeric features 
numericCols = [feature for (feature, dataType) in trainRDD.dtypes if ((dataType == "double") | (dataType == "int")) & (feature != "DEP_DEL15")]

# obtain categorical features 
categoricalCols = [feature for (feature, dataType) in trainRDD.dtypes if (dataType == "string") & (feature != "DEP_DEL15")]

# create indexer and OHE output columns
  # - no one-hot-encoding needed for DT
indexOutputCols = [x + "Index" for x in categoricalCols]
# oheOutputCols = [x + "OHE" for x in categoricalCols]

# create column indexers for categorical features
  # - no one-hot-encoding needed for DT
  # - categorical features will be transformed to indexOutputCols
stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="keep")
label_stringIndexer = StringIndexer(inputCol='DEP_DEL15', outputCol='label')
# oheEncoder = OneHotEncoder(inputCols=indexOutputCols, outputCols=oheOutputCols)

# create vector assembler so that all features are in one single vector 
  # - indexOutputCols: indexed categorical features
  # - numericCols: original numeric features
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# create stages for pipeline
stages = [stringIndexer, label_stringIndexer, vecAssembler, dt]

# ML pipeline
pipeline = Pipeline(stages=stages)

# the value of max bins needs to be >= max number of categories for any categorical feature
dt.setMaxBins(36184)
pipelineModel_dt = pipeline.fit(trainRDD)

# COMMAND ----------

# MAGIC %md Model Evaluation

# COMMAND ----------

# Metrics - part 1
predictions = pipelineModel_dt.transform(validationRDD)
evaluator = BinaryClassificationEvaluator()

# Metrics - part 2
tp = predictions[(predictions.DEP_DEL15 == 1) & (predictions.prediction == 1)].count()
tn = predictions[(predictions.DEP_DEL15 == 0) & (predictions.prediction == 0)].count()
fp = predictions[(predictions.DEP_DEL15 == 0) & (predictions.prediction == 1)].count()
fn = predictions[(predictions.DEP_DEL15 == 1) & (predictions.prediction == 0)].count()
total = predictions.count()
recall = float(tp)/(tp + fn)

# Metrics - part 3
data = {'Actual: delay': [tp, fn], 'Actual: on-time': [fp, tn]}
confusion_matrix = pd.DataFrame.from_dict(data, orient='index', 
                                          columns=['Prediction: delay', 'Prediction: on-time'])

print("Test Area Under ROC: ", "{:.2f}".format(evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderROC'})))
print("Test Area Under Precision-Recall Curve: ", "{:.2f}".format(evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderPR'})))

print("True positive rate: {:.2%}".format(tp/(tp + fn)))
print("True negative rate: {:.2%}".format(tn/(tn + fp)))
print("False positive rate: {:.2%}".format(fp/(fp + tn)))
print("False negative rate: {:.2%}".format(fn/(tp + fn)))
print("Recall: {:.2%}".format(recall))

print("########### Confusion Matrix ###########")
print(confusion_matrix)

# COMMAND ----------

# MAGIC %md Feature importance

# COMMAND ----------

featureImportance = pipelineModel_dt.stages[-1].featureImportances
va = pipelineModel_dt.stages[-2]

importabnce_df = pd.DataFrame(list(zip(va.getInputCols(), featureImportance)), columns=["feature", "importance"])
importabnce_df.sort_values(by="importance", ascending=False)

# COMMAND ----------

# MAGIC %md ##### Selected features

# COMMAND ----------

selected_ftr = ['PR_ARR_DEL15', 'CRS_DEP_TIME_HOUR', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'CRS_ARR_TIME_HOUR', 
                'DEST', 'DISTANCE', 'DEW_TEMP', 'PRECIPITATION', 'MONTH', 'TMP_TEMP', 'DAY_OF_WEEK', 'DEP_DEL15']

trainRDD_tree_ftr = trainRDD.select(selected_ftr).cache()
validationRDD_tree_ftr = validationRDD.select(selected_ftr).cache()
testRDD_tree_ftr = testRDD.select(selected_ftr).cache()

# COMMAND ----------

# MAGIC %md ##### Decision Tree with selected features 
# MAGIC - 3-fold cross Validation
# MAGIC   - ```maxDepth```: [2, 4, 6, 10, 12]
# MAGIC   - ```maxBins```: [400, 800, 1200]
# MAGIC - ```maxBins``` (number of bins used) transforms continuous features into discrete features. The value of ```maxBins``` need to be >= the max number of categories for any categorical feature 
# MAGIC - ```VectorAssembler``` is used to transform all input features to be contained within a single vector in the dataframe, which is needed by many algorithms in Spark. ```VectorAssembler``` takes a list of input columns and creates a new column (named "features" in this case) that combines all input columns into a single vector.
# MAGIC - ```StringIndexer``` is used to encode columns of categories into a columns of indices, with ordering represents the basis of popularity.

# COMMAND ----------

# Decision Tree Classifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label')

# obtain numeric features 
numericCols = [feature for (feature, dataType) in trainRDD_tree_ftr.dtypes if ((dataType == "double") | (dataType == "int")) & (feature != "DEP_DEL15")]

# obtain categorical features 
categoricalCols = [feature for (feature, dataType) in trainRDD_tree_ftr.dtypes if (dataType == "string") & (feature != "DEP_DEL15")]

# create indexer outputs
indexOutputCols = [x + "Index" for x in categoricalCols]

# create column indexers for categorical features
  # - categorical features will be transformed to indexOutputCols
stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="keep")
label_stringIndexer = StringIndexer(inputCol='DEP_DEL15', outputCol='label')

# create vector assembler so that all features are in one single vector 
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# create stages for pipeline
stages = [stringIndexer, label_stringIndexer, vecAssembler, dt]

# ML pipeline
pipeline = Pipeline(stages=stages)

# construct paramGrid
paramGrid = (ParamGridBuilder()
  .addGrid(dt.maxDepth, [2, 4, 6, 10, 12])
  .addGrid(dt.maxBins, [400, 800, 1200])
  .build())

# define evaluation metrics
evaluator = BinaryClassificationEvaluator().setMetricName('areaUnderROC').setRawPredictionCol('prediction').setLabelCol('label')

# cross-validator
cv = CrossValidator(estimator=pipeline,
                    evaluator=evaluator, 
                    estimatorParamMaps=paramGrid, 
                    numFolds=3, 
                    parallelism =3,
                    seed=42)

cvModel_dt = cv.fit(trainRDD_tree_ftr)
dt_bestModel = cvModel_dt.bestModel

# inspect results
# list(zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics))

# COMMAND ----------

# MAGIC %md Model Parameters

# COMMAND ----------

param_dict = dt_bestModel.stages[-1].extractParamMap()

sane_dict = {}
for k,v in param_dict.items():
  sane_dict[k.name] = v
  
best_maxDepth = sane_dict["maxDepth"]
best_maxBins = sane_dict["maxBins"]

print("Best maxDepth: ", best_maxDepth)
print("Best maxBins: ", best_maxBins)

# COMMAND ----------

# MAGIC %md Model Evaluation

# COMMAND ----------

# Metrics - part 1
predictions = dt_bestModel.transform(testRDD_tree_ftr)
evaluator = BinaryClassificationEvaluator()

# Metrics - part 2
tp = predictions[(predictions.DEP_DEL15 == 1) & (predictions.prediction == 1)].count()
tn = predictions[(predictions.DEP_DEL15 == 0) & (predictions.prediction == 0)].count()
fp = predictions[(predictions.DEP_DEL15 == 0) & (predictions.prediction == 1)].count()
fn = predictions[(predictions.DEP_DEL15 == 1) & (predictions.prediction == 0)].count()
total = predictions.count()
recall = float(tp)/(tp + fn)

# Metrics - part 3
data = {'Actual: delay': [tp, fn], 'Actual: on-time': [fp, tn]}
confusion_matrix = pd.DataFrame.from_dict(data, orient='index', 
                                          columns=['Prediction: delay', 'Prediction: on-time'])

print("Test Area Under ROC: ", "{:.2f}".format(evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderROC'})))
print("Test Area Under Precision-Recall Curve: ", "{:.2f}".format(evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderPR'})))

print("True positive rate: {:.2%}".format(tp/(tp + fn)))
print("True negative rate: {:.2%}".format(tn/(tn + fp)))
print("False positive rate: {:.2%}".format(fp/(fp + tn)))
print("False negative rate: {:.2%}".format(fn/(tp + fn)))
print("Recall: {:.2%}".format(recall))

print("########### Confusion Matrix ###########")
print(confusion_matrix)

# COMMAND ----------

# MAGIC %md ##### Random Forest with optimized maxBins and narrowed range of maximum depth
# MAGIC - 3-fold cross Validation with ```parallelim``` = 3
# MAGIC   - ```maxDepth```: [4, 6, 10]
# MAGIC   - ```numTrees```: [10, 50, 100]
# MAGIC - ```maxBins``` (number of bins used) transforms continuous features into discrete features. The value of ```maxBins``` need to be >= the max number of categories for any categorical feature 
# MAGIC - ```VectorAssembler``` is used to transform all input features to be contained within a single vector in the dataframe, which is needed by many algorithms in Spark. ```VectorAssembler``` takes a list of input columns and creates a new column (named "features" in this case) that combines all input columns into a single vector.
# MAGIC - ```StringIndexer``` is used to encode columns of categories into a columns of indices, with ordering represents the basis of popularity.

# COMMAND ----------

selected_ftr = ['PR_ARR_DEL15', 'CRS_DEP_TIME_HOUR', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'CRS_ARR_TIME_HOUR', 
                'DEST', 'DISTANCE', 'DEW_TEMP', 'PRECIPITATION', 'MONTH', 'TMP_TEMP', 'DAY_OF_WEEK', 'DEP_DEL15']

trainRDD_tree_ftr = trainRDD.select(selected_ftr).cache()
validationRDD_tree_ftr = validationRDD.select(selected_ftr).cache()
testRDD_tree_ftr = testRDD.select(selected_ftr).cache()

# COMMAND ----------

# Random Forest Classifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', maxBins=800)

# obtain numeric features 
numericCols = [feature for (feature, dataType) in trainRDD_tree_ftr.dtypes if ((dataType == "double") | (dataType == "int")) & (feature != "DEP_DEL15")]

# obtain categorical features 
categoricalCols = [feature for (feature, dataType) in trainRDD_tree_ftr.dtypes if (dataType == "string") & (feature != "DEP_DEL15")]

# create indexer and OHE output columns
  # - no one-hot-encoding needed for DT
indexOutputCols = [x + "Index" for x in categoricalCols]
# oheOutputCols = [x + "OHE" for x in categoricalCols]

# create column indexers for categorical features
  # - no one-hot-encoding needed for DT
  # - categorical features will be transformed to indexOutputCols
stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="keep")
label_stringIndexer = StringIndexer(inputCol='DEP_DEL15', outputCol='label')
# oheEncoder = OneHotEncoder(inputCols=indexOutputCols, outputCols=oheOutputCols)

# create vector assembler so that all features are in one single vector 
  # - indexOutputCols: indexed categorical features
  # - numericCols: original numeric features
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# ML pipeline
pipeline = Pipeline(stages = [stringIndexer, label_stringIndexer, vecAssembler, rf])

# construct paramGrid
paramGrid = (ParamGridBuilder()
  .addGrid(rf.maxDepth, [4, 6, 10])
  .addGrid(rf.numTrees, [10, 50, 100])
  .build())

# define evaluation metrics
evaluator = BinaryClassificationEvaluator().setMetricName('areaUnderROC').setRawPredictionCol('prediction').setLabelCol('label')

cv = CrossValidator(estimator=pipeline,
                    evaluator=evaluator, 
                    estimatorParamMaps=paramGrid, 
                    numFolds=3, 
                    parallelism=3,
                    seed=42)

cvModel_rf2 = cv.fit(trainRDD_tree_ftr)
rf_bestModel_cv = cvModel_rf2.bestModel

# inspect results
# list(zip(tvsModel.getEstimatorParamMaps(), tvsModel.avgMetrics))

# COMMAND ----------

# Metrics - part 1
predictions = rf_bestModel_cv.transform(testRDD_tree_ftr)
evaluator = BinaryClassificationEvaluator()

# Metrics - part 2
tp = predictions[(predictions.DEP_DEL15 == 1) & (predictions.prediction == 1)].count()
tn = predictions[(predictions.DEP_DEL15 == 0) & (predictions.prediction == 0)].count()
fp = predictions[(predictions.DEP_DEL15 == 0) & (predictions.prediction == 1)].count()
fn = predictions[(predictions.DEP_DEL15 == 1) & (predictions.prediction == 0)].count()
total = predictions.count()
recall = float(tp)/(tp + fn)

# Metrics - part 3
data = {'Actual: delay': [tp, fn], 'Actual: on-time': [fp, tn]}
confusion_matrix = pd.DataFrame.from_dict(data, orient='index', 
                                          columns=['Prediction: delay', 'Prediction: on-time'])

print("Test Area Under ROC: ", "{:.2f}".format(evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderROC'})))
print("Test Area Under Precision-Recall Curve: ", "{:.2f}".format(evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderPR'})))

print("True positive rate: {:.2%}".format(tp/(tp + fn)))
print("True negative rate: {:.2%}".format(tn/(tn + fp)))
print("False positive rate: {:.2%}".format(fp/(fp + tn)))
print("False negative rate: {:.2%}".format(fn/(tp + fn)))
print("Recall: {:.2%}".format(recall))

print("########### Confusion Matrix ###########")
print(confusion_matrix)

# COMMAND ----------

# MAGIC %md ##### Gradient Boosting Tree

# COMMAND ----------

# obtain numeric features 
numerics_gbt = [feature for (feature, dataType) in trainRDD_tree_ftr.dtypes if ((dataType == "double") | (dataType == "int")) & (feature != "DEP_DEL15")]

# obtain categorical features 
categoricals_gbt = [feature for (feature, dataType) in trainRDD_tree_ftr.dtypes if (dataType == "string") & (feature != "DEP_DEL15")]

# Establish stages for our GBT model
inputCol_gbt = [x + "Index" for x in categoricals_gbt]
indexers_gbt = StringIndexer(inputCols=categoricals_gbt, outputCols=inputCol_gbt, handleInvalid="keep")
label_indexers_gbt = StringIndexer(inputCol="DEP_DEL15", outputCol="label")

featureCols_gbt = inputCol_gbt + numerics_gbt

# Define vector assemblers
vector_gbt = VectorAssembler(inputCols=featureCols_gbt, outputCol="features")

# Define a GBT model.
gbt = GBTClassifier(featuresCol="features",
                    labelCol="label",
                    lossType = "logistic",
                    maxIter = 50,
                    maxDepth = 10,
                    maxBins = 800)

# Chain indexer and GBT in a Pipeline
stages_gbt = [indexers_gbt, label_indexers_gbt, vector_gbt, gbt]
pipeline_gbt = Pipeline(stages=stages_gbt)

# Train the tuned model and establish our best model
gbt_model = pipeline_gbt.fit(trainRDD_tree_ftr)

# COMMAND ----------

prediction = gbt_model.transform(testRDD_tree_ftr)
evaluator = BinaryClassificationEvaluator()

# Metrics - part 2
tp = prediction[(prediction.DEP_DEL15 == 1) & (prediction.prediction == 1)].count()
tn = prediction[(prediction.DEP_DEL15 == 0) & (prediction.prediction == 0)].count()
fp = prediction[(prediction.DEP_DEL15 == 0) & (prediction.prediction == 1)].count()
fn = prediction[(prediction.DEP_DEL15 == 1) & (prediction.prediction == 0)].count()
total = prediction.count()
recall = float(tp)/(tp + fn)

# Metrics - part 3
data = {'Actual: delay': [tp, fn], 'Actual: on-time': [fp, tn]}
confusion_matrix = pd.DataFrame.from_dict(data, orient='index', 
                                          columns=['Prediction: delay', 'Prediction: on-time'])

print("Test Area Under ROC: ", "{:.2f}".format(evaluator.evaluate(prediction, {evaluator.metricName: 'areaUnderROC'})))
print("Test Area Under Precision-Recall Curve: ", "{:.2f}".format(evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderPR'})))

print("True positive rate: {:.2%}".format(tp/(tp + fn)))
print("True negative rate: {:.2%}".format(tn/(tn + fp)))
print("False positive rate: {:.2%}".format(fp/(fp + tn)))
print("False negative rate: {:.2%}".format(fn/(tp + fn)))
print("Recall: {:.2%}".format(recall))

print("########### Confusion Matrix ###########")
print(confusion_matrix)

# COMMAND ----------

# MAGIC %md 
# MAGIC The area under the curve metrics tells us how much the model is capable of distinguishing classes. The higher the AUC, the better the model is at predicting 0s as 0s and 1s as 1s. However, given the business problem in hand, major consequences will result if an airline announces a potential flight delay, yet the flight ends up being on-time. Therefore, false positive rates are highlighted as it is the optimal metrics for model evaluation in this context.  
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC | Model                  | **FP %**  | FN %  | TP %   | TN %   | AUC  | Precision | Recall | Area Under PR |
# MAGIC |------------------------|-------|-------|--------|--------|------|-----------|--------|---------------|
# MAGIC | Decision Trees         | **4.94%** |52.01% | 47.99% | 95.06% | 0.29 | 0.69      | 0.48   | 0.13          |
# MAGIC | Random Forest          | **4.59%** |52.68% | 47.32% | 95.41% | 0.79 | 0.70      | 0.47   | 0.59          |
# MAGIC | Gradient Boosted Trees | **3.65%** | 55.61%| 44.39% | 96.35% | 0.81 | 0.73      | 0.44   | 0.59          |

# COMMAND ----------

# MAGIC %md ### Lessons learned
# MAGIC **-------------- What worked ? --------------** <br>
# MAGIC Data transformtion and pre-processing
# MAGIC - Strategic placement of parquet check points throughout the workflow improves performance since Spark transformations are lazily evaluated. Check points avoid unnecessary build-up of transformation steps. 
# MAGIC - Transformation sequence matters - data not pertinent for downstream workflows should be filtered out at each transformation stage so that resources won't be spent on processing and transforming unnecessary data.
# MAGIC - Cache the dataset (triggered by an action call in Spark) before running any feature transformation or modeling steps (especially cross-validation) benefits the algorithm performance
# MAGIC - The number of partitions of the dataset can be increased to match the number of distinct values in the matching column (i.e. weather stations) to improve join performance
# MAGIC 
# MAGIC ML workflow
# MAGIC - The ```parallelism``` parameter inside the ```CrossValidator``` sets the number of threads to use when running parallel algorithms. Increasing it 
# MAGIC - Use the ML model itself instead of the pipeline inside the ```CrossValidator``` speficication speeds up cross-validation, especially when only the model is being tuned. Otherwise, the entire pipeline will be executed for each parameter combination and fold.
# MAGIC 
# MAGIC Infrastructure
# MAGIC - It is important to keep an eye on the **cluster health**, especially when performance is slow. When the driver is overloaded, Garbage Collection slows everything to a grinding halt.
# MAGIC 
# MAGIC <br> 
# MAGIC **-------------- Challenges and next steps --------------** <br>
# MAGIC - Class imbalances can make the model favor negative classification. If the imbalance were to be higher than the existing distribution, oversampling (random oversampling or ideally, SMOTE) techniques should be in place to adjust the class distribution. 
# MAGIC - Although hyperparameter tuning improves classification results, engineering more pertinent features (i.e. aggregated departure demand by hour, average minutes of delay by carrier, etc) holds the key for major performance enhancement (always remember: garbage in, garbage out). 
# MAGIC - Workflow needs to be further optimized for better training performance and scalability

# COMMAND ----------


