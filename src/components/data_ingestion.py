import os
import sys # for custom exception

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass    #from python 3.9 for using class variables in-short
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

'''
when performing data ingestion(DI) it requires input for DI i/p can be like where i have to save train data and raw data 
and those kinds of i/p's basically save in another class and this class we mention as DataIngestionConfig.the i/p's requires for data
tranformation DataTranformationConfig and the o/p can be anything for DI like numpy,file saved in some folder.to use DI we use decorator
inside class to define class variable we use __init_ right,but if we use DataClasss here so that we can able to directly define the class
 variable
 '''

@dataclass#inside class to define variables we use __init_ right,but if we use DataClasss,u will be directly define class variable
class DataIngestionConfig: 
    '''
    below are class variables
    #using artifacts folder,so that we can able to see the o/p and the path is that giving to DI and DI o/p components 
    # will save in this path'''
    train_data_path:str=os.path.join('artifacts',"train.csv")#----this is the i/p that i'm giving and later on train.csvfile will save in this particular apth
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"data.csv")
    '''now go and start our class , if we just use variables then we use dataclass but if we have some other functions in class i would suggest
    to go ahed with __init__ constructor'''
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the Dataset as a dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #artifacts is a folder and lets go ahead and create folders
            #with the help of train,test,raw dath paths for that we use os.makedirs and inside directries also  i have to combine directries path w.r.t specific path
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


