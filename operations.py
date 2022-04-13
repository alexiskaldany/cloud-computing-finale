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
    
    cols = ['gmDate', 'gmTime', 'seasTyp', 'teamAbbr','teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFG%', 'team3P%', 'teamFT%', 'teamTRB','teamRslt', 'teamOrtg', 'teamDrtg',  'opptAbbr', 'opptPTS', 'opptAST', 'opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFG%', 'oppt3P%', 'opptFT%', 'opptTRB', 'opptOrtg', 'opptDrtg']
    df_filtered = df[cols]
    df_final = df_filtered[df_filtered.index % 2 == 0]
    df_final = df_final.rename(columns={"teamFG%": "teamFG", "team3P%": "team3P", "teamFT%" : "teamFT","opptFG%": "opptFG", "oppt3P%": "oppt3P", "opptFT%" : "opptFT" })
    

    logger.info(f"Organizing raw dataframe into correct shape and dtypes")
    return df

@op
def model_creation(df):
    sleep(2)
    """ 
    put Suhas code here
    """
    for i in range(0,2):
     if(i==0) : 
        X_home = df[['teamAST', 'teamTO', 'opptAST', 'opptTO', 'teamDrtg', 'teamOrtg', 'opptDrtg', 'opptOrtg', 'teamFG', 'opptFG', 'teamTRB', 'opptTRB']]
        y_home = df['teamPTS']
        from sklearn.model_selection import train_test_split
        X_train_home, X_test_home, y_train_home, y_test_home = train_test_split(X_home, y_home, test_size=0.2, random_state=0)
        
     elif(i==1):
        X_away = df[['teamAST', 'teamTO', 'opptAST', 'opptTO', 'teamDrtg', 'teamOrtg', 'opptDrtg', 'opptOrtg', 'teamFG', 'opptFG', 'teamTRB', 'opptTRB']]
        y_away = df['opptPTS']
        from sklearn.model_selection import train_test_split
        X_train_away, X_test_away, y_train_away, y_test_away = train_test_split(X_away, y_away, test_size=0.2, random_state=0)
    logger.info(f"Train-Test sets created, data has been regularized,ect..")
    return X_train_home, X_test_home, y_train_home, y_test_home, X_train_away, X_test_away, y_train_away, y_test_away

@op
def model_training(X_train_home, y_train_home, X_train_away, y_train_away) :
    sleep(10)
    """ 
    put Suhas code here
    """
    for i in range(0,2):
     if(i==0) : 
        from sklearn.linear_model import LinearRegression
        regressor_home = LinearRegression()
        regressor_home.fit(X_train_home, y_train_home)

     elif(i==1):
        from sklearn.linear_model import LinearRegression
        regressor_away = LinearRegression()
        regressor_away.fit(X_train_away, y_train_away)
    
    logger.info(f"Model has completed training")
    return regressor_home, regressor_away

@op
def model_evaluation(regressor_home, regressor_away , X_test_home, X_test_away, y_test_home, y_test_away):
    sleep(4)
    """ 
    put Suhas code here
    """
    import numpy as np
    for i in range(0,2):
     if(i==0) : 
        y_pred_home = regressor_home.predict(X_test_home)
        print(y_pred_home)
        from sklearn import metrics
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_home, y_pred_home))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test_home, y_pred_home))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_home, y_pred_home)))
        from sklearn.metrics import r2_score
        r2_score(y_test_home, y_pred_home)
        coefficients_team = regressor_home.coef_

        intercept_team = regressor_home.intercept_
        

     elif(i==1):
        y_pred_away = regressor_away.predict(X_test_away)
        print(y_pred_away)
        from sklearn import metrics
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_away, y_pred_away))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test_away, y_pred_away))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_away, y_pred_away)))
        from sklearn.metrics import r2_score
        r2_score(y_test_away, y_pred_away)
        coefficients_away = regressor_away.coef_

        intercept_away = regressor_away.intercept_

    logger.info(f"Model accuracy is : 0.89")
    return coefficients_team, coefficients_away, intercept_team, intercept_away

@op
def post_processing(coefficients_team, coefficients_away, intercept_team, intercept_away):
    sleep(4)
    """ 
    put Suhas code here
    """
    data = {
    
    'teamAbbr' : ['CHA', 'TOR', 'MIL', 'MIN', 'NO', 'DEN', 'GS'],
    'opptAbbr' : ['ORL', 'PHI', 'BOS', 'SA', 'POR', 'MEM', 'LAL'],
    'teamOrtg' : [113.1, 112.1, 114.1, 113.6, 110.9, 113.6, 111.8], 
    'teamDrtg' : [113.3, 109.8, 111.0, 110.9, 111.5, 11.3, 106.7],
    'opptOrtg' : [103.8, 112.6, 113.2, 112.0, 108.0, 114.2, 109.6],
    'opptDrtg' : [111.9, 109.0, 106.1, 111.4, 116.0, 108.5, 111.2],
    'teamAST' : [27.9, 22.0, 23.8, 25.6, 24.9, 27.7, 26.9],
    'teamTO' : [12.7, 11.7, 12.8, 13.8, 13.3, 13.9, 14.3],
    'opptAST' : [23.6, 23.5, 24.6, 28.0, 22.9, 25.7, 24.1],
    'opptTO' : [13.9, 11.7, 13.0, 12.4, 13.5, 12.4, 13.9],
    'teamFG' : [0.468, 0.445, 0.468, 0.457, 0.457, 0.483, 0.469],
    'opptFG' : [0.434, 0.466, 0.466, 0.467, 0.442, 0.461, 0.469], 
    'teamTRB' : [45, 45, 47, 44, 45, 44, 46],
    'opptTRB' : [44, 42, 46, 45, 43, 49, 44 ]
    
    
          }

    test_df = pd.DataFrame(columns=['teamAbbr', 'opptAbbr', 'teamOrtg', 'teamDrtg', 'opptOrtg', 'opptDrtg', 'teamAST', 'teamTO', 'opptAST', 'opptTO', 'teamPTS', 'opptPTS' ], data=data )
    
    for i in range(0,2):
     if(i==0) : 
        def calculate_TeamPTS(ASTteam, TOteam, ASToppt, TOoppt, Drtgteam, Ortgteam, Drtgoppt, Ortgoppt, FGteam, FGoppt, TRBteam, TRBoppt ):
         return (ASTteam * coefficients_team[0]) + ( TOteam * coefficients_team[1]) + ( ASToppt* coefficients_team[2]) + ( TOoppt* coefficients_team[3])+ + ( Drtgteam* coefficients_team[4])+ ( Ortgteam* coefficients_team[5])+ ( Drtgoppt* coefficients_team[6])+  ( Ortgoppt* coefficients_team[7])+ ( FGteam * coefficients_team[8]) + ( FGoppt * coefficients_team[9]) + ( TRBteam * coefficients_team[10]) + ( TRBoppt * coefficients_team[11]) + intercept_team
        test_df['Home Points'] = calculate_TeamPTS(test_df.teamAST, test_df.teamTO, test_df.opptAST, test_df.opptTO, test_df.teamDrtg, test_df.teamOrtg, test_df.opptDrtg, test_df.opptOrtg)


     elif(i==1):
        def calculate_OpptPTS(ASTteam, TOteam, ASToppt, TOoppt, Drtgteam, Ortgteam, Drtgoppt, Ortgoppt, FGteam, FGoppt, TRBteam, TRBoppt ):
         return (ASTteam * coefficients_away[0]) + ( TOteam * coefficients_away[1]) + ( ASToppt* coefficients_away[2]) + ( TOoppt* coefficients_away[3])+ + ( Drtgteam* coefficients_away[4])+ ( Ortgteam* coefficients_away[5])+ ( Drtgoppt* coefficients_away[6])+  ( Ortgoppt* coefficients_away[7])+ ( FGteam * coefficients_away[8]) + ( FGoppt * coefficients_away[9]) + ( TRBteam * coefficients_away[10]) + ( TRBoppt * coefficients_away[11]) + intercept_away
        test_df['Away Points'] = calculate_OpptPTS(test_df.teamAST, test_df.teamTO, test_df.opptAST, test_df.opptTO, test_df.teamDrtg, test_df.teamOrtg, test_df.opptDrtg, test_df.opptOrtg)



    logger.info(f"Dataframe completed post processing")
    return test_df

@job
def NBA_Pipeline():
    df = scrape_data()
    df = df_to_s3(df)
    df = pre_process_data(df)
    X_train_home, X_test_home, y_train_home, y_test_home, X_train_away, X_test_away, y_train_away, y_test_away = model_creation(df)
    regressor_home, regressor_away = model_training(X_train_home, y_train_home, X_train_away, y_train_away)
    coefficients_team, coefficients_away, intercept_team, intercept_away = model_evaluation(regressor_home, regressor_away , X_test_home, X_test_away, y_test_home, y_test_away)
    test_df = post_processing(coefficients_team, coefficients_away, intercept_team, intercept_away)
    logger.info("Save final dataframe to location where Dash Frontend reads data")
    df_to_s3(test_df)
    logger.info(f"URL has been scraped and saved")
    
    

    
# @op           
# def df_pre_processing(df):
#     cols = ['gmDate', 'gmTime', 'seasTyp', 'teamAbbr','teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFG%', 'team3P%', 'teamFT%', 'teamTRB','teamRslt', 'teamOrtg', 'teamDrtg',  'opptAbbr', 'opptPTS', 'opptAST', 'opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFG%', 'oppt3P%', 'opptFT%', 'opptTRB', 'opptOrtg', 'opptDrtg']
#     df_filtered = df[cols].copy()
#     return df_filtered