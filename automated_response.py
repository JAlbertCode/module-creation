"""Automated response system for handling alerts"""

import logging
import docker
import json
from typing import Dict, Any, List
import requests
from datetime import datetime
import time
import subprocess

class AutomatedResponse:
    """Handle automated responses to alerts"""
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.logger = self.setup_logger()
        self.docker_client = docker.from_env()
        self.actions_taken = []
        
    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load response configuration"""
        default_config = {
            'actions': {
                'high_cpu': ['scale_resources', 'optimize_batch_size'],
                'high_memory': ['clear_cache', 'scale_resources'],
                'slow_inference': ['optimize_model', 'check_gpu'],
                'error_spike': ['restart_service', 'check_logs']
            },
            'thresholds': {
                'cpu_usage': 80,
                'memory_usage': 80,
                'error_rate': 5
            },
            'cooldown': 300  # seconds between same actions
        }
        
        if config_path:
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                
        return default_config
    
    def setup_logger(self) -> logging.Logger:
        """Set up logging for automated responses"""
        logger = logging.getLogger('automated_response')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('automated_responses.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def handle_alert(self, alert: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle an incoming alert"""
        responses = []
        alert_type = self._categorize_alert(alert)
        
        if not self._check_cooldown(alert_type):
            self.logger.info(f"Skipping {alert_type} - in cooldown period")
            return responses
            
        actions = self.config['actions'].get(alert_type, [])
        for action in actions:
            try:
                result = getattr(self, f'_action_{action}')(alert)
                if result:
                    responses.append({
                        'action': action,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    })
                    self._record_action(action, alert_type)
            except Exception as e:
                self.logger.error(f"Error executing {action}: {e}")
                
        return responses
    
    def _categorize_alert(self, alert: Dict[str, Any]) -> str:
        """Categorize alert for appropriate response"""
        metrics = alert.get('metrics', {})
        
        if metrics.get('cpu_usage', 0) > self.config['thresholds']['cpu_usage']:
            return 'high_cpu'
        elif metrics.get('memory_usage', 0) > self.config['thresholds']['memory_usage']:
            return 'high_memory'
        elif metrics.get('error_rate', 0) > self.config['thresholds']['error_rate']:
            return 'error_spike'
        elif metrics.get('inference_time', 0) > 1.0:  # More than 1 second
            return 'slow_inference'
            
        return 'unknown'
    
    def _check_cooldown(self, action_type: str) -> bool:
        """Check if action is in cooldown period"""
        now = time.time()
        
        # Check recent actions
        for action in self.actions_taken:
            if (action['type'] == action_type and 
                now - action['timestamp'] < self.config['cooldown']):
                return False
                
        return True
    
    def _record_action(self, action: str, alert_type: str):
        """Record an action taken"""
        self.actions_taken.append({
            'action': action,
            'type': alert_type,
            'timestamp': time.time()
        })
    
    def _action_scale_resources(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Scale up resources for a container"""
        try:
            container_name = alert['container_id']
            container = self.docker_client.containers.get(container_name)
            
            # Get current resource limits
            current_config = container.attrs['HostConfig']
            
            # Calculate new limits
            new_memory = int(current_config['Memory'] * 1.5)  # Increase by 50%
            new_cpus = int(current_config['NanoCpus'] / 1e9) + 1  # Add 1 CPU
            
            # Update container
            container.update(
                mem_limit=new_memory,
                nano_cpus=new_cpus * 1e9
            )
            
            return {
                'status': 'success',
                'new_memory': new_memory,
                'new_cpus': new_cpus
            }
            
        except Exception as e:
            self.logger.error(f"Error scaling resources: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _action_optimize_batch_size(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust batch size based on performance"""
        try:
            # Get current batch size
            current_batch = int(alert.get('metrics', {}).get('batch_size', 1))
            
            if alert['metrics'].get('cpu_usage', 0) > 80:
                # Decrease batch size
                new_batch = max(1, current_batch // 2)
            else:
                # Try increasing batch size
                new_batch = current_batch * 2
            
            # Update configuration
            self._update_model_config(alert['model_id'], {'batch_size': new_batch})
            
            return {
                'status': 'success',
                'old_batch_size': current_batch,
                'new_batch_size': new_batch
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing batch size: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _action_clear_cache(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Clear system caches"""
        try:
            # Drop system caches
            subprocess.run(['sync'])
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3')
            
            # Clear model cache if available
            if hasattr(self, 'model_cache'):
                self.model_cache.clear()
            
            return {
                'status': 'success',
                'message': 'Caches cleared'
            }
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _action_optimize_model(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model for better performance"""
        try:
            # Get current configuration
            model_id = alert['model_id']
            
            # Try optimization techniques
            optimizations = {
                'half_precision': True,  # Use FP16
                'optimize_memory': True,
                'enable_fusion': True
            }
            
            # Update model configuration
            self._update_model_config(model_id, optimizations)
            
            return {
                'status': 'success',
                'optimizations': list(optimizations.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing model: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _action_check_gpu(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Check GPU status and optimize if needed"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {
                    'status': 'error',
                    'message': 'No GPU available'
                }
            
            # Get GPU utilization
            gpu_util = torch.cuda.utilization()
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            
            # Take action based on utilization
            actions_taken = []
            
            if gpu_util < 30:  # Low utilization
                actions_taken.append('increase_batch_size')
            elif gpu_util > 90:  # High utilization
                actions_taken.append('decrease_batch_size')
                
            if memory_allocated / memory_reserved > 0.9:  # High memory usage
                actions_taken.append('clear_gpu_cache')
                torch.cuda.empty_cache()
            
            return {
                'status': 'success',
                'gpu_utilization': gpu_util,
                'memory_usage': memory_allocated / memory_reserved,
                'actions_taken': actions_taken
            }
            
        except Exception as e:
            self.logger.error(f"Error checking GPU: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _action_restart_service(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Restart the service experiencing issues"""
        try:
            container_name = alert['container_id']
            container = self.docker_client.containers.get(container_name)
            
            # Restart container
            container.restart()
            
            return {
                'status': 'success',
                'message': f'Container {container_name} restarted'
            }
            
        except Exception as e:
            self.logger.error(f"Error restarting service: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _action_check_logs(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Check service logs for errors"""
        try:
            container_name = alert['container_id']
            container = self.docker_client.containers.get(container_name)
            
            # Get recent logs
            logs = container.logs(
                tail=100,  # Last 100 lines
                timestamps=True
            ).decode('utf-8')
            
            # Parse logs for errors
            error_lines = [
                line for line in logs.split('\n')
                if 'error' in line.lower() or 
                   'exception' in line.lower() or
                   'failed' in line.lower()
            ]
            
            return {
                'status': 'success',
                'error_count': len(error_lines),
                'recent_errors': error_lines[-5:]  # Last 5 errors
            }
            
        except Exception as e:
            self.logger.error(f"Error checking logs: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _update_model_config(self, model_id: str, updates: Dict[str, Any]):
        """Update model configuration"""
        config_path = f'configs/{model_id}.json'
        
        try:
            # Load current config
            with open(config_path) as f:
                config = json.load(f)
            
            # Update configuration
            config.update(updates)
            
            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
            raise

def main():
    """Test automated response system"""
    responder = AutomatedResponse()
    
    # Test alert
    test_alert = {
        'level': 'WARNING',
        'message': 'High CPU usage detected',
        'metrics': {
            'cpu_usage': 85,
            'memory_usage': 60,
            'batch_size': 32
        },
        'model_id': 'test-model',
        'container_id': 'test-container'
    }
    
    # Handle alert
    responses = responder.handle_alert(test_alert)
    print("Automated Responses:", json.dumps(responses, indent=2))

if __name__ == '__main__':
    main()