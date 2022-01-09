# Problem Statement

To build a classification methodology to predict the type of Thyroid based on the given training data. 


## Architecture :
![architecture](https://user-images.githubusercontent.com/72031522/148693556-7bc02307-4e1e-4e0b-a7dc-afa5e649b46c.jpg)

# Data Description

The client will send data in multiple sets of files in batches at a given location. Data will contain different classes of thyroid and 30 columns of different values.

"Class" column will have four unique values “negative, compensated_hypothyroid,
primary_hypothyroid, secondary_hypothyroid”.

Apart from training files, we also require a "schema" file from the client, which contains all the relevant information about the training files such as:

Name of the files, Length of Date value in FileName, Length of Time value in FileName, Number of Columns, Name of the Columns, and their datatype.


-------

# Data Validation 
In this step, we perform different sets of validation on the given set of training files.  

1.	 Name Validation- We validate the name of the files based on the given name in the schema file. We have created a regex pattern as per the name given in the schema file to use for validation. After validating the pattern in the name, we check for the length of date in the file name as well as the length of time in the file name. If all the values are as per requirement, we move such files to "Good_Data_Folder" else we move such files to "Bad_Data_Folder."

2.	 Number of Columns - We validate the number of columns present in the files, and if it doesn't match with the value given in the schema file, then the file is moved to "Bad_Data_Folder."


3.	 Name of Columns - The name of the columns is validated and should be the same as given in the schema file. If not, then the file is moved to "Bad_Data_Folder".

4.	 The datatype of columns - The datatype of columns is given in the schema file. This is validated when we insert the files into Database. If the datatype is wrong, then the file is moved to "Bad_Data_Folder".


5.	Null values in columns - If any of the columns in a file have all the values as NULL or missing, we discard such a file and move it to "Bad_Data_Folder".


--------------

# Data Insertion in Database
 
1) Database Creation and connection - Create a database with the given name passed. If the database is already created, open the connection to the database. 

2) Table creation in the database - Table with name - "Good_Data", is created in the database for inserting the files in the "Good_Data_Folder" based on given column names and datatype in the schema file. If the table is already present, then the new table is not created and new files are inserted in the already present table as we want training to be done on new as well as old training files.     

3) Insertion of files in the table - All the files in the "Good_Data_Folder" are inserted in the above-created table. If any file has invalid data type in any of the columns, the file is not loaded in the table and is moved to "Bad_Data_Folder".


--------

# Model Training 

1) Data Export from Db - The data in a stored database is exported as a CSV file to be used for model training.

2) Data Preprocessing   
   
       a) Drop columns not useful for training the model. Such columns were selected while doing the EDA.
   
       b) Replace the invalid values with numpy “nan” so we can use imputer on such values.
   
       c) Encode the categorical values
   
       d) Check for null values in the columns. If present, impute the null values using the KNN imputer.
   
       e)  After imputing, handle the imbalanced dataset by using RandomOverSampler.

3) Clustering - KMeans algorithm is used to create clusters in the preprocessed data. The optimum number of clusters is selected by plotting the elbow plot, and for the dynamic selection of the number of clusters, we are using "KneeLocator" function. The idea behind clustering is to implement different algorithms
   To train data in different clusters. The Kmeans model is trained over preprocessed data and the model is saved for further use in prediction.

4) Model Selection - After clusters are created, we find the best model for each cluster. We are using two algorithms, "Random Forest" and "KNN". For each cluster, both the algorithms are passed with the best parameters derived from GridSearch. We calculate the AUC scores for both models and select the model with the best score. Similarly, the model is selected for each cluster. All the models for every cluster are saved for use in prediction. 


----------

# Prediction Data Description
 
Client will send the data in multiple set of files in batches at a given location. Data will contain Wafer names and 590 columns of different sensor values for each wafer. 
Apart from prediction files, we also require a "schema" file from client which contains all the relevant information about the training files such as:
Name of the files, Length of Date value in FileName, Length of Time value in FileName, Number of Columns, Name of the Columns and their datatype.


-------------

# Prediction 
 
1) Data Export from Db - The data in the stored database is exported as a CSV file to be used for prediction.

2) Data Preprocessing   
       
       a) Drop columns not useful for training the model. Such columns were selected while doing the EDA.
       
       b) Replace the invalid values with numpy “nan” so we can use imputer on such values.
       
       c) Encode the categorical values
       
       d) Check for null values in the columns. If present, impute the null values using the KNN imputer.

3) Clustering - KMeans model created during training is loaded, and clusters for the preprocessed prediction data is predicted.

4) Prediction - Based on the cluster number, the respective model is loaded and is used to predict the data for that cluster.


------------

## Deployment



I had deployed the model to the Pivotal Web Services Platform. 

This is a workflow diagram for the prediction of using the trained model.                  
                                                      

Now let’s see the Thyroid  Detection project folder structure.


![folder_structure](https://user-images.githubusercontent.com/72031522/148693561-0c9c7cee-5f65-463d-a69a-a861038e56f1.png)


requirements.txt file consists of all the packages that you need to deploy the app in the cloud.


----


![main](https://user-images.githubusercontent.com/72031522/148693562-8c1b4845-4aef-40f3-84ad-8a290a3eb379.png)

main.py is the entry point of our application, where the flask server starts. 



----

![predict](https://user-images.githubusercontent.com/72031522/148693565-14af3327-b556-41b2-ac96-8c814a8bbab1.png)

This is the predictionFromModel.py file where the predictions take place based on the data we are giving input to the model.

---------

![yml](https://user-images.githubusercontent.com/72031522/148693709-5d410db3-c35b-4bbd-bcdb-b41e9b8a6868.png)



manifest.yml:- This file contains the instance configuration, app name, and build pack language.

------
![proc](https://user-images.githubusercontent.com/72031522/148693567-b039a26c-b10e-4de8-a65b-5ef0137772dc.png)

Procfile :- It contains the entry point of the app.

-------

![runtime](https://user-images.githubusercontent.com/72031522/148693568-808456e5-801c-4586-9bd2-70187598b548.png)

runtime.txt:- It contains the Python version number.

------

# Now, I Deployed this project on pivptal web services and with the app link which is created we can run the app and predict the data through postman api testing


---

![postman](https://user-images.githubusercontent.com/72031522/148693563-10791381-7089-4208-8642-0db3cde0f56d.png)


# Now it will read all batch files at the path provided by the client and export all the data in sql database and after that
# data from databese extracted in single csv file and clustering is done on data.for each cluster prediciton is done from best suited model.
# and finally all prediction are saves at 'Prediction_Output_File' in csv format

![predoutput](https://user-images.githubusercontent.com/72031522/148697530-4c8ac65d-abce-4536-93de-021fe85e3981.png)




============================================================================================================

## Deployed Model Link

https://thyroid-detection646.herokuapp.com/


