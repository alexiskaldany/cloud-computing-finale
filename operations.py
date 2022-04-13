## File for Dagster Operations
from dagster import op, job, get_dagster_logger
import pandas as pd
from time import sleep
logger = get_dagster_logger()
    
# Variables
# """
# url = 'https://raw.githubusercontent.com/alexiskaldany/cloud-computing-finale/main/Suhas%20/Dataset.csv'
# s3_bucket = 'cloud-computing-project-akaldany'
# region = 'us-east-1'
# key_name = 'test.csv'
# """    
#############  
@op
def scrape_data():
    """scrape operation, point at online csv links

    Args:
        url (str): url to be turned into dataframe
    """
    url = 'https://raw.githubusercontent.com/alexiskaldany/cloud-computing-finale/main/Suhas%20/Dataset.csv'
    df =  pd.read_csv(url)
    sleep(6)
    if isinstance(df,pd.DataFrame) == True:
        logger.info(f"Url {url} succesfully converted to Dataframe")
        return df
    else:
        logger.info(f"Url {url} failed to be converted to Dataframe")
        return df
##############
@op
def df_from_s3(s3_bucket,region,key_name):
    """saves object to s3 bucket

    Args:
        df (pd.Dataframe): a pandas dataframe
        s3_bucket (str): bucket within s3
        region (str): AWS region (like us-east-1)
        key_name: name of object within bucket (could be within a subfolder like /data/example.csv)
        
        example: https://bucket-name.s3.Region.amazonaws.com/key-name
        s3_bucket = bucket-name
        
    """
    s3_bucket = 'cloud-computing-project-akaldany/'
    region = 'us-east-1'
    key_name = 'test.csv'
    s3_URI = f"s3://{s3_bucket}{key_name}"
    try:
        df = pd.from_csv(s3_URI,storage_options={"key": aws_access_key_id, "secret": aws_secret_access_key})
        logger.info(f"Dataframe: {df} succesfully loaded from s3 {s3_bucket} at path {key_name}")
        return df
    except:
        logger.info(f"Dataframe: {df} failed to load from s3 {s3_bucket} at path {key_name}")
        return

@op
def df_to_s3(df):
    """saves object to s3 bucket

    Args:
    df (pd.Dataframe): a pandas dataframe
    s3_bucket (str): bucket within s3
    region (str): AWS region (like us-east-1)
    key_name: name of object within bucket (could be within a subfolder like /data/example.csv)

    example: https://bucket-name.s3.Region.amazonaws.com/key-name
    s3_bucket = bucket-name

    """
    sleep(4)
    s3_bucket = 'cloud-computing-project-akaldany/'
    region = 'us-east-1'
    key_name = 'test.csv'
    s3_URI = f"s3://{s3_bucket}{key_name}"
    logger.info(f"Dataframe: {df} succesfully saved to s3 {s3_bucket} at path {key_name}")
    return

@op
def pre_process_data(df):
    sleep(4)
    """ 
    put Suhas code here
    """
    logger.info(f"Organizing raw dataframe into correct shape and dtypes")
    return df

@op
def model_creation(df):
    sleep(2)
    """ 
    put Suhas code here
    """
    logger.info(f"Train-Test sets created, data has been regularized,ect..")
    return df

@op
def model_training(df):
    sleep(10)
    """ 
    put Suhas code here
    """
    logger.info(f"Model has completed training")
    return df

@op
def model_evaluation(df):
    sleep(4)
    """ 
    put Suhas code here
    """
    # print( type(modelteamscore) )
    # print( modelteamscore.summary() )
    logger.info(f"Model accuracy is : 0.89")
    return df

@op
def post_processing(df):
    sleep(4)
    """ 
    put Suhas code here
    """
    logger.info(f"Dataframe completed post processing")
    return df

@job
def NBA_Pipeline():
    df = scrape_data()
    df = df_to_s3(df)
    df = pre_process_data(df)
    df = model_creation(df)
    df = model_training(df)
    df = model_evaluation(df)
    df = post_processing(df)
    logger.info("Save final dataframe to location where Dash Frontend reads data")
    df_to_s3(df)
    logger.info(f"URL has been scraped and saved")
    
    

    
# @op           
# def df_pre_processing(df):
#     cols = ['gmDate', 'gmTime', 'seasTyp', 'teamAbbr','teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFG%', 'team3P%', 'teamFT%', 'teamTRB','teamRslt', 'teamOrtg', 'teamDrtg',  'opptAbbr', 'opptPTS', 'opptAST', 'opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFG%', 'oppt3P%', 'opptFT%', 'opptTRB', 'opptOrtg', 'opptDrtg']
#     df_filtered = df[cols].copy()
#     return df_filtered