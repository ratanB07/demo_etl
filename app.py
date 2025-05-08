# app.py
from flask import Flask, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import concurrent.futures
import traceback
import logging
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLRunner:
    def __init__(self):
        self.results = []
        self.problem_type = None
        self.target_column = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def detect_problem_type(self, y):
        """Determine if the problem is classification or regression"""
        unique_values = len(np.unique(y))
        if unique_values < 10 or isinstance(y[0], str):
            return "classification"
        return "regression"
    
    def preprocess_data(self, df, target_column):
        """Prepare data for ML training"""
        try:
            # Handle missing values
            df = df.dropna()
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Detect problem type
            self.problem_type = self.detect_problem_type(y)
            
            # Encode target if classification
            if self.problem_type == "classification":
                y = self.label_encoder.fit_transform(y)
            
            # Convert categorical features to numerical
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X[col] = LabelEncoder().fit_transform(X[col])
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            return True
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return False
    
    def get_classification_models(self):
        """Return classification models with pipelines"""
        return [
            ("Logistic Regression", Pipeline([
                ('model', LogisticRegression(max_iter=1000))
            ])),
            ("Decision Tree", Pipeline([
                ('model', DecisionTreeClassifier())
            ])),
            ("Random Forest", Pipeline([
                ('model', RandomForestClassifier())
            ])),
            ("Gradient Boosting", Pipeline([
                ('model', GradientBoostingClassifier())
            ])),
            ("SVM", Pipeline([
                ('model', SVC(probability=True))
            ])),
            ("KNN", Pipeline([
                ('model', KNeighborsClassifier())
            ])),
            ("Naive Bayes", Pipeline([
                ('model', GaussianNB())
            ])),
            ("XGBoost", Pipeline([
                ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
            ])),
            ("LightGBM", Pipeline([
                ('model', LGBMClassifier())
            ])),
            ("Neural Network", Pipeline([
                ('model', MLPClassifier(max_iter=1000))
            ]))
        ]
    
    def get_regression_models(self):
        """Return regression models with pipelines"""
        return [
            ("Linear Regression", Pipeline([
                ('model', LinearRegression())
            ])),
            ("Ridge Regression", Pipeline([
                ('model', Ridge())
            ])),
            ("Lasso Regression", Pipeline([
                ('model', Lasso())
            ])),
            ("Decision Tree", Pipeline([
                ('model', DecisionTreeRegressor())
            ])),
            ("Random Forest", Pipeline([
                ('model', RandomForestRegressor())
            ])),
            ("Gradient Boosting", Pipeline([
                ('model', GradientBoostingRegressor())
            ])),
            ("SVR", Pipeline([
                ('model', SVR())
            ])),
            ("KNN", Pipeline([
                ('model', KNeighborsRegressor())
            ])),
            ("XGBoost", Pipeline([
                ('model', XGBRegressor())
            ])),
            ("LightGBM", Pipeline([
                ('model', LGBMRegressor())
            ]))
        ]
    
    def train_and_evaluate(self, model_name, model):
        """Train and evaluate a single model"""
        try:
            logger.info(f"Training {model_name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            if self.problem_type == "classification":
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                return {
                    "model": model_name,
                    "accuracy": round(accuracy, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1_score": round(f1, 4),
                    "status": "success"
                }
            else:  # regression
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                return {
                    "model": model_name,
                    "mse": round(mse, 4),
                    "rmse": round(rmse, 4),
                    "mae": round(mae, 4),
                    "r2": round(r2, 4),
                    "status": "success"
                }
        except Exception as e:
            logger.error(f"Error in {model_name}: {str(e)}")
            traceback.print_exc()
            return {
                "model": model_name,
                "error": str(e),
                "status": "error"
            }
    
    def run_all_models(self):
        """Run all appropriate models in parallel"""
        models = self.get_classification_models() if self.problem_type == "classification" else self.get_regression_models()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for name, model in models:
                futures.append(executor.submit(self.train_and_evaluate, name, model))
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                self.results.append(result)
                logger.info(f"Completed: {result['model']}")
    
    def get_results(self):
        """Return formatted results"""
        return {
            "problem_type": self.problem_type,
            "target_column": self.target_column,
            "results": self.results,
            "top_models": sorted(
                self.results, 
                key=lambda x: x["accuracy"] if self.problem_type == "classification" else x["r2"], 
                reverse=True
            )[:3]
        }

ml_runner = MLRunner()

@app.route('/ml/run', methods=['POST'])
def run_ml_analysis():
    """Endpoint to run ML analysis"""
    try:
        data = request.json
        df = pd.DataFrame(data['data'])
        target_column = data['target_column']
        
        # Initialize ML runner
        ml_runner.target_column = target_column
        
        # Preprocess data
        if not ml_runner.preprocess_data(df, target_column):
            return jsonify({"status": "error", "message": "Data preprocessing failed"}), 400
        
        # Run models
        ml_runner.run_all_models()
        
        # Get results
        results = ml_runner.get_results()
        
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        logger.error(f"ML analysis error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

from flask import Flask
from waitress import serve

app = Flask(__name__)
# routes here...

if __name__ == '__main__':
    serve(app, port=5001)