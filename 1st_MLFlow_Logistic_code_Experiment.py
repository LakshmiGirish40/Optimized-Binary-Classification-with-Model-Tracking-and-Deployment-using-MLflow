import numpy as np
from sklearn.datasets import make_classification
 # Correct function name
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')



# Step 1: Create an imbalanced binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights=[0.9, 0.1], flip_y=0, random_state=42)
np.unique(y, return_counts=True)

#-------------------------------------------------------------------------------------
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

#------------------------------------------------------------------------------------
# Train the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

#-----------------------------------------
#LogisticRegression Model Predictions
# Predict on the test set
y_pred = lr.predict(X_test)

#Classification report
report = classification_report(y_test, y_pred)
print(report)

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_dict

#1. Importing MLflow
import mlflow

#2. Setting the Experiment
#If an experiment named "my_first_Experiment" exists in the MLflow tracking server, MLflow will log data to it.
mlflow.set_experiment("my_first_Experiment")

#3. Setting the Tracking URI
#http://127.0.0.1:5000/ and http://localhost:5000 both point to a local MLflow server running on your machine at port 5000.
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
#mlflow.set_tracking_uri("http://localhost:5000")
#---------------------------------------------------------------

#4. Starting an MLflow Run
with mlflow.start_run():
    mlflow.log_params(params)   #Logs the hyperparameters or configurations of the model, stored in the dictionary params.
    
    #Logging Metrics
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
    mlflow.sklearn.log_model(lr, "Logistic Regression")    #Logging the Model