import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import pickle
from imblearn.over_sampling import RandomOverSampler



class Preprocessor:
    """
            This class shall  be used to clean and transform the data before training.

    """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object


    def remove_columns(self,data,columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception

        """
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data = data
        self.columns = columns

        try:
            self.useful_data = self.data.drop(labels = self.columns,axis =1)
            self.logger_object.log(self.file_object,'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self,data,lebel_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X = data.drop(labels = lebel_column_name,axis = 1)
            self.Y = data[lebel_column_name]
            self.logger_object.log(self.file_object,'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')

            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def dropUnnecessaryColumns(self,data,columnNameList):
        """
                                Description: This method drops the unwanted columns as discussed in EDA section.

        """
        data = data.drop(columnNameList,axis=1)

        return data

    def replaceInvalidValuesWithNull(self,data):
        """
               Description: This method replaces invalid values i.e. '?' with null, as discussed in EDA.
        """
        data = data
        for column in data.columns:
            count = data[column][data[column]=='?'].count()
            if count !=0:
                data[column] = data[column].replace('?',np.nan)
        return data

    def is_null_present(self,data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
                                On Failure: Raise Exception

        """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False

        try:
            self.null_counts = data.isna().sum()
            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break
            if (self.null_present):
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv('preprocessing_data/null_values.csv')
            self.logger_object.log(self.file_object,'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')

            return  self.null_present
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def encodeCategoricalValues(self,data):
        """
                                        Method Name: encodeCategoricalValues
                                        Description: This method encodes all the categorical values in the training set.
                                        Output: A Dataframe which has all the categorical values encoded.
                                        On Failure: Raise Exception

        """
        #we can map the catagorical values like below:

        data['sex'] = data['sex'].map({'F': 0, 'M': 1})
        # except for 'Sex' column all the other columns with two categorical data have same value 'f' and 't'.
        # so instead of mapping indvidually, let's do a smarter work

        for column in data.columns:
            if len(data[column].unique()) == 2:
                data[column] = data[column].map({'f': 0, 't': 1})

        # this will map all the rest of the columns as we require. Now there are handful of column left with more than 2 categories.
        # we will use get_dummies with that.

        data = pd.get_dummies(data,columns=['referral_source'])

        encode = LabelEncoder().fit(data['Class'])
        data['Class'] = encode.transform(data['Class'])

        # we will save the encoder as pickle to use when we do the prediction. We will need to decode the predcited values
        # back to original
        with open('EncoderPickle/enc.pickle','wb') as file:
            pickle.dump(encode,file)

        return data

    def encodeCategoricalValuesPrediction(self, data):
        """
                                               Method Name: encodeCategoricalValuesPrediction
                                               Description: This method encodes all the categorical values in the prediction set.
                                               Output: A Dataframe which has all the categorical values encoded.
                                               On Failure: Raise Exception


                            """

        # We can map the categorical values like below:
        data['sex'] = data['sex'].map({'F': 0, 'M': 1})
        cat_data = data.drop(['age', 'T3', 'TT4', 'T4U', 'FTI', 'sex'],
                             axis=1)  # we do not want to encode values with int or float type
        # except for 'Sex' column all the other columns with two categorical data have same value 'f' and 't'.
        # so instead of mapping indvidually, let's do a smarter work
        for column in cat_data.columns:
            if (data[column].nunique()) == 1:
                if data[column].unique()[0] == 'f' or data[column].unique()[
                    0] == 'F':  # map the variables same as we did in training i.e. if only 'f' comes map as 0 as done in training
                    data[column] = data[column].map({data[column].unique()[0]: 0})
                else:
                    data[column] = data[column].map({data[column].unique()[0]: 1})
            elif (data[column].nunique()) == 2: \
                    data[column] = data[column].map({'f': 0, 't': 1})

        # we will use get dummies for 'referral_source'
        data = pd.get_dummies(data, columns=['referral_source'])

        return data

    def handleImbalanceDataset(self,X,Y):
        """
                                                      Method Name: handleImbalanceDataset
                                                      Description: This method handles the imbalance in the dataset by oversampling.
                                                      Output: A Dataframe which is balanced now.
                                                      On Failure: Raise Exception


        """

        rdsmple= RandomOverSampler()
        x_sampled, y_sampled = rdsmple.fit_sample(X, Y)

        return  x_sampled,y_sampled

    def impute_missing_values(self,data):
        """
                                        Method Name: impute_missing_values
                                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values imputed.
                                        On Failure: Raise Exception



        """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')

        self.data = data

        try:
            imputer = KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)

            self.new_array = imputer.fit_transform(self.data)

            self.new_data = pd.DataFrame(self.new_array,columns= self.data.columns)
            self.logger_object.log(self.file_object,'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')

            return self.new_data

        except Exception as  e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def get_columns_with_zero_std_deviation(self,data):
        """
                                                Method Name: get_columns_with_zero_std_deviation
                                                Description: This method finds out the columns which have a standard deviation of zero.
                                                Output: List of the columns with standard deviation of zero
                                                On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns = data.columns
        self.data_n = data.describe()
        self.col_to_drop =[]
        try:
            for x in self.columns:
                if (self.data_n[x]['std'] == 0):
                    self.col_to_drop.append(x)
                self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return  self.col_to_drop
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()



























