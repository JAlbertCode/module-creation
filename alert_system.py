"""Alert system for model monitoring"""

import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

@dataclass
class Alert:
    """Alert information container"""
    level: str  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message: str
    metrics: Dict[str, Any]
    timestamp: datetime
    model_id: str

class AlertSystem:
    """Handle alerting through various channels"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.logger = self.setup_logging()
        
    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load alerting configuration"""
        default_config = {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'recipients': []
            },
            'slack': {
                'enabled': False,
                'webhook_url': '',
                'channel': '#alerts'
            },
            'telegram': {
                'enabled': False,
                'bot_token': '',
                'chat_id': ''
            },
            'thresholds': {
                'cpu_usage': 80,  # percentage
                'memory_usage': 80,  # percentage
                'inference_time': 1.0,  # seconds
                'error_rate': 1.0  # percentage
            },
            'alert_cooldown': 300  # seconds between repeated alerts
        }
        
        if config_path:
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}")
                
        return default_config
    
    def setup_logging(self) -> logging.Logger:
        """Set up logging for alerts"""
        logger = logging.getLogger('alert_system')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('alerts.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def check_metrics(self, metrics: Dict[str, Any], model_id: str) -> List[Alert]:
        """Check metrics against thresholds"""
        alerts = []
        thresholds = self.config['thresholds']
        
        # Check CPU usage
        if metrics.get('cpu_usage', 0) > thresholds['cpu_usage']:
            alerts.append(Alert(
                level='WARNING',
                message=f"High CPU usage: {metrics['cpu_usage']}%",
                metrics=metrics,
                timestamp=datetime.now(),
                model_id=model_id
            ))
            
        # Check memory usage
        if metrics.get('memory_usage', 0) > thresholds['memory_usage']:
            alerts.append(Alert(
                level='WARNING',
                message=f"High memory usage: {metrics['memory_usage']}%",
                metrics=metrics,
                timestamp=datetime.now(),
                model_id=model_id
            ))
            
        # Check inference time
        if metrics.get('inference_time', 0) > thresholds['inference_time']:
            alerts.append(Alert(
                level='WARNING',
                message=f"Slow inference: {metrics['inference_time']}s",
                metrics=metrics,
                timestamp=datetime.now(),
                model_id=model_id
            ))
            
        # Check error rate
        if metrics.get('error_rate', 0) > thresholds['error_rate']:
            alerts.append(Alert(
                level='ERROR',
                message=f"High error rate: {metrics['error_rate']}%",
                metrics=metrics,
                timestamp=datetime.now(),
                model_id=model_id
            ))
            
        return alerts
    
    def send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        self.logger.log(
            getattr(logging, alert.level),
            f"[{alert.model_id}] {alert.message}"
        )
        
        # Send email if configured
        if self.config['email']['enabled']:
            self._send_email_alert(alert)
            
        # Send Slack message if configured
        if self.config['slack']['enabled']:
            self._send_slack_alert(alert)
            
        # Send Telegram message if configured
        if self.config['telegram']['enabled']:
            self._send_telegram_alert(alert)
    
    def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        try:
            config = self.config['email']
            
            msg = MIMEMultipart()
            msg['Subject'] = f"[{alert.level}] Model Alert: {alert.model_id}"
            msg['From'] = config['username']
            msg['To'] = ', '.join(config['recipients'])
            
            body = f"""
            Alert Level: {alert.level}
            Model: {alert.model_id}
            Message: {alert.message}
            Time: {alert.timestamp}
            
            Metrics:
            {json.dumps(alert.metrics, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
    
    def _send_slack_alert(self, alert: Alert):
        """Send alert via Slack"""
        try:
            config = self.config['slack']
            
            message = {
                'channel': config['channel'],
                'attachments': [{
                    'color': self._get_alert_color(alert.level),
                    'title': f"Model Alert: {alert.model_id}",
                    'text': alert.message,
                    'fields': [
                        {
                            'title': 'Level',
                            'value': alert.level,
                            'short': True
                        },
                        {
                            'title': 'Time',
                            'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'short': True
                        }
                    ],
                    'footer': 'Lilypad Monitoring'
                }]
            }
            
            requests.post(config['webhook_url'], json=message)
            
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {e}")
    
    def _send_telegram_alert(self, alert: Alert):
        """Send alert via Telegram"""
        try:
            config = self.config['telegram']
            
            message = f"""
            🚨 *Model Alert*
            *Level:* {alert.level}
            *Model:* {alert.model_id}
            *Message:* {alert.message}
            *Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
            data = {
                'chat_id': config['chat_id'],
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            requests.post(url, json=data)
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram alert: {e}")
    
    def _get_alert_color(self, level: str) -> str:
        """Get color for alert level"""
        colors = {
            'INFO': '#36a64f',  # green
            'WARNING': '#fcba03',  # yellow
            'ERROR': '#fc0303',  # red
            'CRITICAL': '#4a0303'  # dark red
        }
        return colors.get(level, '#000000')

def main():
    """Test alert system"""
    alert_system = AlertSystem()
    
    # Test metrics
    test_metrics = {
        'cpu_usage': 90,
        'memory_usage': 85,
        'inference_time': 1.5,
        'error_rate': 2.0
    }
    
    # Check metrics and send alerts
    alerts = alert_system.check_metrics(test_metrics, 'test-model')
    for alert in alerts:
        alert_system.send_alert(alert)

if __name__ == '__main__':
    main()