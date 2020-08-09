## [Spark Playground - Flight Delay Prediction](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/4292307494740474/1474730470433170/6190017855571830/latest.html)
##### Machine Learning At Scale (Spark, Spark SQL, Spark ML)
![image](/images/sparkjobs.png)

Flight delays create problems in scheduling for airlines and airports, leading to passenger inconvenience, and huge economic losses. As a result, there is growing interest in predicting flight delays beforehand in order to optimize operations and improve customer satisfaction. The objective of this playground project is to predict flight departure delays two hours ahead of departure at scale. The project includes an exploration of a series of data transformation and ML pipelines in **Apache Spark** (using Databricks). It concludes with some challenges faced along the journey and some key lessons learned.

The Databricks notebook is connected with AWS where it can create and manage compute and VPC resources. Data access in the notebook was through a mounted S3 bucket on AWS.

Datasets used in the project include the following:
- flight dataset from the [US Department of Transportation](https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time) containing flight information from 2015 to 2019
<br>**(31,746,841 x 109 dataframe)**
- weather dataset from the [National Oceanic and Atmospheric Administration repository](https://www.ncdc.noaa.gov/orders/qclcd/) containing weather information from 2015 to 2019
<br>**(630,904,436 x 177 dataframe)**
- airport dataset from the [US Department of Transportation](https://www.transtats.bts.gov/DL_SelectFields.asp)
<br>**(18,097 x 10 dataframe)**

**The project can be directly accessed via [Spark Playground - Flight Delay Prediction](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/4292307494740474/1474730470433170/6190017855571830/latest.html)**. This repository also contains the .dbc and .py versions of the Databricks notebook.
