#from datetime import datetime
from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from DataTransform_Training.DataTransformation import dataTransform
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from application_logging import logger


class train_validation:
    def __init__(self,path):
        self.raw_data = Raw_Data_validation(path)
        self.dataTransform = dataTransform()
        self.dBOperation=dBOperation()
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()


    def train_validation(self):
        try:
            self.log_writer.log(self.file_object, 'Start of Validation on files for prediction!!')

            #extracting values from training file schema
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.valuesFromSchema()

            regex = self.raw_data.manualRegexCreation()

            # validate filename of training files
            self.raw_data.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)

            #validate column length in the file
            self.raw_data.validateColumnLength(noofcolumns)

            #Validate if any column has all values missing

            self.raw_data.validateMissingValuesInWholeColumn()

            self.log_writer.log(self.file_object,"raw data validation complete!")

            self.log_writer.log(self.file_object,"Starting Data transformation!")
            self.dataTransform.addQuotesToStringValuesInColumn()
            self.log_writer.log(self.file_object,"Data transformation completed!")

            self.log_writer.log(self.file_object,"Creating training database and tables on basis of given schema")

            self.dBOperation.createTableDb('Training',column_names)
            self.log_writer.log(self.file_object,"Table creation completed!")

            #insert csv files into the table
            self.dBOperation.insertIntoTableGoodData('Training')
            self.log_writer.log(self.file_object,"Insertion in table completed!")

            self.log_writer.log(self.file_object,"Deleting Good data folder")
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.file_object,"Good data folder deleted")

            self.log_writer.log(self.file_object,"moving bad data into archived bad data")
            self.raw_data.moveBadFilesToArchiveBad()

            self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.log(self.file_object, "Validation Operation completed!!")
            self.log_writer.log(self.file_object, "Extracting csv file from table")

            self.dBOperation.selectingDatafromtableintocsv('Training')
            self.file_object.close()
        except Exception as e:
            raise e

        












