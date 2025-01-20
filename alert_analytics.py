"""Analytics system for alert analysis and reporting"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

class AlertAnalytics:
    """Analyze alert patterns and generate insights"""
    
    def __init__(self, db_path: str = 'alerts.db'):
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database for alert analytics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create alerts table
        c.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level TEXT,
                message TEXT,
                model_id TEXT,
                metrics TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_time DATETIME,
                resolution_note TEXT
            )
        ''')
        
        # Create alert patterns table
        c.execute('''
            CREATE TABLE IF NOT EXISTS alert_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                description TEXT,
                frequency INTEGER,
                last_seen DATETIME,
                model_ids TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_alert(self, alert: Dict[str, Any]):
        """Record an alert in the database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO alerts 
            (level, message, model_id, metrics)
            VALUES (?, ?, ?, ?)
        ''', (
            alert['level'],
            alert['message'],
            alert['model_id'],
            json.dumps(alert['metrics'])
        ))
        
        conn.commit()
        conn.close()
    
    def resolve_alert(self, alert_id: int, resolution_note: str):
        """Mark an alert as resolved"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            UPDATE alerts
            SET resolved = TRUE,
                resolution_time = CURRENT_TIMESTAMP,
                resolution_note = ?
            WHERE id = ?
        ''', (resolution_note, alert_id))
        
        conn.commit()
        conn.close()
    
    def analyze_patterns(self, lookback_days: int = 30) -> List[Dict[str, Any]]:
        """Analyze alert patterns"""
        conn = sqlite3.connect(self.db_path)
        
        # Get alerts for the specified period
        query = '''
            SELECT level, message, model_id, timestamp
            FROM alerts
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(
            query, 
            conn, 
            params=[f'-{lookback_days} days'],
            parse_dates=['timestamp']
        )
        
        patterns = []
        
        # Analyze frequency patterns
        frequency_patterns = self._analyze_frequency_patterns(df)
        patterns.extend(frequency_patterns)
        
        # Analyze time-based patterns
        time_patterns = self._analyze_time_patterns(df)
        patterns.extend(time_patterns)
        
        # Analyze model correlations
        correlation_patterns = self._analyze_correlations(df)
        patterns.extend(correlation_patterns)
        
        # Save patterns to database
        self._save_patterns(patterns)
        
        conn.close()
        return patterns
    
    def _analyze_frequency_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze alert frequency patterns"""
        patterns = []
        
        # Group by model and alert type
        grouped = df.groupby(['model_id', 'level']).size().reset_index(name='count')
        
        for _, row in grouped.iterrows():
            if row['count'] > 10:  # Threshold for frequent alerts
                patterns.append({
                    'pattern_type': 'frequency',
                    'description': f"Frequent {row['level']} alerts for {row['model_id']}",
                    'frequency': row['count'],
                    'model_ids': [row['model_id']]
                })
                
        return patterns
    
    def _analyze_time_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze time-based patterns"""
        patterns = []
        
        # Add hour of day
        df['hour'] = df['timestamp'].dt.hour
        
        # Check for time-of-day patterns
        hourly_counts = df.groupby(['model_id', 'hour']).size().reset_index(name='count')
        
        for model_id in hourly_counts['model_id'].unique():
            model_data = hourly_counts[hourly_counts['model_id'] == model_id]
            peak_hour = model_data.loc[model_data['count'].idxmax()]
            
            if peak_hour['count'] > 5:  # Threshold for time pattern
                patterns.append({
                    'pattern_type': 'time',
                    'description': f"Peak alerts at hour {peak_hour['hour']} for {model_id}",
                    'frequency': peak_hour['count'],
                    'model_ids': [model_id]
                })
                
        return patterns
    
    def _analyze_correlations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze correlations between model alerts"""
        patterns = []
        
        # Create time windows
        df['time_window'] = df['timestamp'].dt.floor('1H')
        
        # Find models with correlated alerts
        pivoted = pd.pivot_table(
            df,
            index='time_window',
            columns='model_id',
            values='level',
            aggfunc='count',
            fill_value=0
        )
        
        if pivoted.shape[1] > 1:  # Need at least 2 models
            corr = pivoted.corr()
            
            # Find highly correlated pairs
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if corr.iloc[i,j] > 0.7:  # Correlation threshold
                        patterns.append({
                            'pattern_type': 'correlation',
                            'description': f"Correlated alerts between {corr.columns[i]} and {corr.columns[j]}",
                            'frequency': None,
                            'model_ids': [corr.columns[i], corr.columns[j]]
                        })
                        
        return patterns
    
    def _save_patterns(self, patterns: List[Dict[str, Any]]):
        """Save discovered patterns to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for pattern in patterns:
            c.execute('''
                INSERT INTO alert_patterns
                (pattern_type, description, frequency, last_seen, model_ids)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
            ''', (
                pattern['pattern_type'],
                pattern['description'],
                pattern['frequency'],
                json.dumps(pattern['model_ids'])
            ))
            
        conn.commit()
        conn.close()
    
    def generate_report(self, 
                       lookback_days: int = 30,
                       output_format: str = 'html') -> str:
        """Generate alert analysis report"""
        conn = sqlite3.connect(self.db_path)
        
        # Get alert data
        alerts_df = pd.read_sql_query(
            'SELECT * FROM alerts WHERE timestamp >= datetime("now", ?)',
            conn,
            params=[f'-{lookback_days} days'],
            parse_dates=['timestamp']
        )
        
        # Get patterns
        patterns_df = pd.read_sql_query(
            'SELECT * FROM alert_patterns',
            conn
        )
        
        # Create visualizations
        figs = self._create_report_visualizations(alerts_df)
        
        # Generate report content
        if output_format == 'html':
            report = self._generate_html_report(alerts_df, patterns_df, figs)
        else:
            report = self._generate_markdown_report(alerts_df, patterns_df, figs)
            
        conn.close()
        return report
    
    def _create_report_visualizations(self, 
                                    df: pd.DataFrame) -> List[go.Figure]:
        """Create visualizations for the report"""
        figs = []
        
        # Alert frequency over time
        fig1 = go.Figure()
        for level in df['level'].unique():
            level_data = df[df['level'] == level]
            fig1.add_trace(go.Scatter(
                x=level_data['timestamp'],
                y=level_data.groupby('timestamp').size(),
                name=level,
                mode='lines'
            ))
        fig1.update_layout(title='Alert Frequency Over Time')
        figs.append(fig1)
        
        # Alert distribution by model
        fig2 = go.Figure(data=[
            go.Bar(
                x=df['model_id'].value_counts().index,
                y=df['model_id'].value_counts().values
            )
        ])
        fig2.update_layout(title='Alerts by Model')
        figs.append(fig2)
        
        return figs
    
    def _generate_html_report(self,
                            alerts_df: pd.DataFrame,
                            patterns_df: pd.DataFrame,
                            figs: List[go.Figure]) -> str:
        """Generate HTML report"""
        html = [
            "<html><head>",
            "<link href='https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css' rel='stylesheet'>",
            "</head><body class='p-8'>",
            "<h1 class='text-2xl font-bold mb-6'>Alert Analysis Report</h1>"
        ]
        
        # Summary statistics
        html.extend([
            "<div class='bg-white rounded-lg shadow-md p-6 mb-6'>",
            "<h2 class='text-xl font-semibold mb-4'>Summary</h2>",
            f"<p>Total Alerts: {len(alerts_df)}</p>",
            f"<p>Models Affected: {alerts_df['model_id'].nunique()}</p>",
            f"<p>Unresolved Alerts: {len(alerts_df[~alerts_df['resolved']])}</p>",
            "</div>"
        ])
        
        # Patterns
        html.extend([
            "<div class='bg-white rounded-lg shadow-md p-6 mb-6'>",
            "<h2 class='text-xl font-semibold mb-4'>Detected Patterns</h2>",
            "<ul class='list-disc pl-5'>"
        ])
        
        for _, pattern in patterns_df.iterrows():
            html.append(f"<li>{pattern['description']}</li>")
            
        html.append("</ul></div>")
        
        # Visualizations
        for fig in figs:
            html.append(fig.to_html(full_html=False))
        
        html.append("</body></html>")
        return "\n".join(html)
    
    def _generate_markdown_report(self,
                                alerts_df: pd.DataFrame,
                                patterns_df: pd.DataFrame,
                                figs: List[go.Figure]) -> str:
        """Generate Markdown report"""
        md = [
            "# Alert Analysis Report\n",
            "## Summary",
            f"- Total Alerts: {len(alerts_df)}",
            f"- Models Affected: {alerts_df['model_id'].nunique()}",
            f"- Unresolved Alerts: {len(alerts_df[~alerts_df['resolved']])}\n",
            "## Detected Patterns"
        ]
        
        for _, pattern in patterns_df.iterrows():
            md.append(f"- {pattern['description']}")
            
        return "\n".join(md)

def main():
    """Test alert analytics"""
    analytics = AlertAnalytics()
    
    # Generate and print report
    report = analytics.generate_report(lookback_days=7)
    print(report)

if __name__ == '__main__':
    main()