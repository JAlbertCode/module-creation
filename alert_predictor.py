"""Predictive alerting system using historical data"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

class AlertPredictor:
    """Predict potential alerts based on historical patterns"""
    
    def __init__(self, db_path: str = 'alerts.db'):
        self.db_path = db_path
        self.model_path = 'alert_predictor_model.joblib'
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'hour', 'day_of_week', 'cpu_usage',
            'memory_usage', 'inference_time', 'request_rate'
        ]
        
    def train_model(self, lookback_days: int = 90):
        """Train prediction model on historical data"""
        # Get historical data
        data = self._get_training_data(lookback_days)
        
        if len(data) < 100:  # Need minimum amount of data
            raise ValueError("Insufficient historical data for training")
        
        # Prepare features
        X = self._prepare_features(data)
        y = self.label_encoder.fit_transform(data['alert_type'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, self.model_path)
        
        # Return model performance
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': dict(zip(
                self.feature_columns,
                model.feature_importances_
            ))
        }
    
    def predict_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Predict potential alerts based on current metrics"""
        try:
            model = joblib.load(self.model_path)
        except:
            raise ValueError("No trained model found. Run train_model first.")
        
        # Prepare features
        features = self._format_features(metrics)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features.reshape(1, -1))[0]
        
        # Get predictions above threshold
        predictions = []
        for i, prob in enumerate(probabilities):
            if prob > 0.7:  # Confidence threshold
                alert_type = self.label_encoder.inverse_transform([i])[0]
                predictions.append({
                    'alert_type': alert_type,
                    'probability': float(prob),
                    'contributing_factors': self._get_contributing_factors(
                        model, features, i
                    )
                })
        
        return predictions
    
    def _get_training_data(self, lookback_days: int) -> pd.DataFrame:
        """Get historical alert and metric data"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                a.timestamp,
                a.level as alert_type,
                m.cpu_usage,
                m.memory_usage,
                m.inference_time,
                m.request_rate,
                m.model_id
            FROM alerts a
            JOIN metrics m ON a.model_id = m.model_id
                AND a.timestamp = m.timestamp
            WHERE a.timestamp >= datetime('now', ?)
        '''
        
        df = pd.read_sql_query(
            query,
            conn,
            params=[f'-{lookback_days} days'],
            parse_dates=['timestamp']
        )
        
        conn.close()
        return df
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from DataFrame"""
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Return feature matrix
        return df[self.feature_columns].values
    
    def _format_features(self, metrics: Dict[str, float]) -> np.ndarray:
        """Format current metrics into feature vector"""
        current_time = datetime.now()
        features = [
            current_time.hour,
            current_time.weekday(),
            metrics.get('cpu_usage', 0),
            metrics.get('memory_usage', 0),
            metrics.get('inference_time', 0),
            metrics.get('request_rate', 0)
        ]
        return np.array(features)
    
    def _get_contributing_factors(self,
                                model: RandomForestClassifier,
                                features: np.ndarray,
                                class_idx: int) -> List[Dict[str, Any]]:
        """Get factors contributing to prediction"""
        # Get feature importance for specific class
        tree_feature_imp = []
        for tree in model.estimators_:
            if tree.predict(features.reshape(1, -1))[0] == class_idx:
                tree_feature_imp.append(tree.feature_importances_)
        
        if not tree_feature_imp:
            return []
            
        # Average feature importance across relevant trees
        avg_importance = np.mean(tree_feature_imp, axis=0)
        
        # Get top contributing factors
        factors = []
        for feature, importance in zip(self.feature_columns, avg_importance):
            if importance > 0.1:  # Importance threshold
                factors.append({
                    'feature': feature,
                    'importance': float(importance)
                })
                
        return sorted(factors, key=lambda x: x['importance'], reverse=True)
    
    def get_alert_suggestions(self, 
                            prediction: Dict[str, Any]) -> List[str]:
        """Get mitigation suggestions for predicted alerts"""
        suggestions = {
            'high_cpu': [
                "Consider scaling up CPU resources",
                "Check for resource-intensive background processes",
                "Optimize batch size settings"
            ],
            'high_memory': [
                "Increase memory allocation",
                "Check for memory leaks",
                "Implement memory-efficient processing"
            ],
            'slow_inference': [
                "Optimize model for inference",
                "Check GPU utilization",
                "Consider model quantization"
            ],
            'error_rate': [
                "Review error logs",
                "Check input data quality",
                "Validate model configuration"
            ]
        }
        
        alert_type = prediction['alert_type'].lower()
        for key in suggestions:
            if key in alert_type:
                return suggestions[key]
        
        return ["Monitor system metrics", "Review alert patterns"]

class RealTimePredictor:
    """Real-time alert prediction and monitoring"""
    
    def __init__(self):
        self.predictor = AlertPredictor()
        self.current_predictions = {}
        
    def update(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Update predictions with new metrics"""
        # Get new predictions
        new_predictions = self.predictor.predict_alerts(metrics)
        
        # Compare with current predictions
        alerts = []
        for pred in new_predictions:
            key = pred['alert_type']
            if key not in self.current_predictions:
                # New prediction
                alerts.append(self._format_prediction(pred))
            elif pred['probability'] > self.current_predictions[key]['probability'] + 0.1:
                # Significantly increased probability
                alerts.append(self._format_prediction(
                    pred, 
                    increased=True,
                    previous_prob=self.current_predictions[key]['probability']
                ))
        
        # Update current predictions
        self.current_predictions = {p['alert_type']: p for p in new_predictions}
        
        return alerts
    
    def _format_prediction(self,
                         prediction: Dict[str, Any],
                         increased: bool = False,
                         previous_prob: float = None) -> Dict[str, Any]:
        """Format prediction for alerting"""
        alert = {
            'type': 'prediction',
            'alert_type': prediction['alert_type'],
            'probability': prediction['probability'],
            'contributing_factors': prediction['contributing_factors'],
            'suggestions': self.predictor.get_alert_suggestions(prediction)
        }
        
        if increased:
            alert['previous_probability'] = previous_prob
            
        return alert

def main():
    """Test prediction system"""
    predictor = AlertPredictor()
    
    # Train model
    performance = predictor.train_model()
    print("Model Performance:", performance)
    
    # Test predictions
    test_metrics = {
        'cpu_usage': 85,
        'memory_usage': 75,
        'inference_time': 0.8,
        'request_rate': 100
    }
    
    predictions = predictor.predict_alerts(test_metrics)
    print("\nPredictions:", predictions)

if __name__ == '__main__':
    main()