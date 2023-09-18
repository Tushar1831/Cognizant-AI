# ------- BEFORE STARTING - SOME BASIC TIPS
# You can add a comment within a Python file by using a hashtag '#'
# Anything that comes after the hashtag on the same line, will be considered
# a comment and won't be executed as code by the Python interpreter.

# --- 1) IMPORTING PACKAGES
# The first thing you should always do in a Python file is to import any
# packages that you will need within the file. This should always go at the top
# of the file
from directory import pathname
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# --- 2) DEFINE GLOBAL CONSTANTS
# Constants are variables that should remain the same througout the entire running
# of the module. You should define these after the imports at the top of the file.
# You should give global constants a name and ensure that they are in all upper
# case, such as: UPPER_CASE

N = 10

SPLIT = 0.2

# --- 3) ALGORITHM CODE
# Next, we should write our code that will be executed when a model needs to be 
# trained. There are many ways to structure this code and it is your choice 
# how you wish to do this. The code in the 'module_helper.py' file will break
# the code down into independent functions, which is 1 option. 
# Include your algorithm code in this section below:

def load_data(filename: str):
    """_summary_

    Args:
        filename (str): _description_

    Returns:
        _type_: _description_
    """
    
    path = pathname(filename)
    df = pd.read_csv(path)
    
    return df
 
 def create_target_variable(data: pd.DataFrame, target: str):
     """_summary_

     Args:
         data (pd.DataFrame): _description_
         target (str): _description_
     """
     
     X = data.drop(columns = target)
     y = data[target]
     
     return X, y
 
 def train_model(X: pd.DataFrame, y: pd.Series):
     """_summary_

     Args:
         X (pd.DataFrame): _description_
         y (pd.Series): _description_
     """
     
     accuracy = []
     
     for i in range(0, N):
        model = RandomForestRegressor()
        scaler = StandardScaler()
        #Creating training and testing splits
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = SPLIT, random_state = 42)
        
        #Scaling X data
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        #Train Model
        training_model = model.fit(X_train, y_train)
        
        #predictions
        y_pred = trained_model.predict(X_test)
        
        #Accuracy
        mae = mean_absolute_error(y_true = y_test, y_pred = y_pred)
        accuracy.append(mae)
        print(f"Fold {i + 1}: MAE = {mae:.3f}")
    
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.3f}")
        
        
# --- 4) MAIN FUNCTION
# Your algorithm code should contain modular code that can be run independently.
# You may want to include a final function that ties everything together, to allow
# the entire pipeline of loading the data and training the algorithm to be run all
# at once

def main():
    df = load_data()
    
    X, y = create_target_variable(df)
    
    #Train algorithm
    train_model(X, y)    


if __name__ == '__main__':
    main()