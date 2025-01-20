"""Configuration templates and management"""

import json
import os
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConfigTemplate:
    name: str
    description: str
    config: Dict[str, Any]
    model_type: str
    created_at: datetime
    last_used: datetime

class TemplateManager:
    def __init__(self, templates_dir: str = 'config_templates'):
        self.templates_dir = templates_dir
        os.makedirs(templates_dir, exist_ok=True)
        
    def save_template(self, name: str, description: str, config: Dict[str, Any], model_type: str) -> ConfigTemplate:
        """Save a configuration template"""
        template = ConfigTemplate(
            name=name,
            description=description,
            config=config,
            model_type=model_type,
            created_at=datetime.now(),
            last_used=datetime.now()
        )
        
        # Save to file
        template_path = os.path.join(self.templates_dir, f"{name}.json")
        with open(template_path, 'w') as f:
            json.dump({
                'name': template.name,
                'description': template.description,
                'config': template.config,
                'model_type': template.model_type,
                'created_at': template.created_at.isoformat(),
                'last_used': template.last_used.isoformat()
            }, f, indent=2)
            
        return template
    
    def load_template(self, name: str) -> ConfigTemplate:
        """Load a configuration template"""
        template_path = os.path.join(self.templates_dir, f"{name}.json")
        with open(template_path, 'r') as f:
            data = json.load(f)
            template = ConfigTemplate(
                name=data['name'],
                description=data['description'],
                config=data['config'],
                model_type=data['model_type'],
                created_at=datetime.fromisoformat(data['created_at']),
                last_used=datetime.fromisoformat(data['last_used'])
            )
            
        # Update last used time
        template.last_used = datetime.now()
        self.save_template(
            template.name,
            template.description,
            template.config,
            template.model_type
        )
        
        return template
    
    def list_templates(self, model_type: str = None) -> List[Dict[str, Any]]:
        """List available templates, optionally filtered by model type"""
        templates = []
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.templates_dir, filename), 'r') as f:
                    data = json.load(f)
                    if not model_type or data['model_type'] == model_type:
                        templates.append({
                            'name': data['name'],
                            'description': data['description'],
                            'model_type': data['model_type'],
                            'last_used': data['last_used']
                        })
        
        # Sort by last used time
        templates.sort(key=lambda x: x['last_used'], reverse=True)
        return templates
    
    def delete_template(self, name: str):
        """Delete a configuration template"""
        template_path = os.path.join(self.templates_dir, f"{name}.json")
        if os.path.exists(template_path):
            os.remove(template_path)
    
    def export_template(self, name: str, export_path: str):
        """Export a template to a file"""
        template = self.load_template(name)
        with open(export_path, 'w') as f:
            json.dump({
                'name': template.name,
                'description': template.description,
                'config': template.config,
                'model_type': template.model_type
            }, f, indent=2)
    
    def import_template(self, import_path: str) -> ConfigTemplate:
        """Import a template from a file"""
        with open(import_path, 'r') as f:
            data = json.load(f)
            return self.save_template(
                data['name'],
                data['description'],
                data['config'],
                data['model_type']
            )