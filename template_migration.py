"""Template migration and version management"""

from typing import Dict, Any
import json
from template_validator import validate_template

class TemplateMigration:
    """Handle template version migrations"""
    
    def __init__(self):
        self.current_version = 2  # Current template version
        self.migrations = {
            1: self._migrate_v1_to_v2
        }
    
    def migrate_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate a template to the current version"""
        template_version = template.get('version', 1)
        
        while template_version < self.current_version:
            if template_version not in self.migrations:
                raise ValueError(f"No migration path from version {template_version}")
                
            template = self.migrations[template_version](template)
            template_version += 1
            template['version'] = template_version
            
        return template
    
    def _migrate_v1_to_v2(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1 to version 2"""
        # Copy template to avoid modifying original
        new_template = template.copy()
        
        # Add new version 2 fields
        if 'config' in new_template:
            config = new_template['config']
            
            # Add monitoring section if not present
            if 'monitoring' not in config:
                config['monitoring'] = {
                    'enable_metrics': True,
                    'log_level': 'INFO'
                }
                
            # Update optimization section
            if 'optimization' in config:
                opt = config['optimization']
                if 'enable_tensorrt' in opt and opt['enable_tensorrt']:
                    opt['gpu_optimization'] = {
                        'type': 'tensorrt',
                        'precision': 'fp16',
                        'dynamic_shapes': True
                    }
                    
            # Update resource specifications
            if 'resources' in config:
                res = config['resources']
                if 'memory' in res and not res['memory'].endswith('Gi'):
                    # Convert to Gi format
                    try:
                        mem_value = int(res['memory'].rstrip('GMK'))
                        if res['memory'].endswith('G'):
                            res['memory'] = f"{mem_value}Gi"
                        elif res['memory'].endswith('M'):
                            res['memory'] = f"{mem_value // 1024}Gi"
                        elif res['memory'].endswith('K'):
                            res['memory'] = f"{mem_value // (1024 * 1024)}Gi"
                    except ValueError:
                        res['memory'] = '4Gi'  # Default if conversion fails
        
        return new_template
    
    def validate_migration(self, template: Dict[str, Any]) -> bool:
        """Validate a migrated template"""
        is_valid, errors, warnings = validate_template(template)
        if not is_valid:
            raise ValueError(f"Migration resulted in invalid template: {errors}")
        return True

def migrate_template(template: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to migrate a template"""
    migrator = TemplateMigration()
    migrated = migrator.migrate_template(template)
    migrator.validate_migration(migrated)
    return migrated

def migrate_template_file(input_path: str, output_path: str):
    """Migrate a template file"""
    with open(input_path, 'r') as f:
        template = json.load(f)
    
    migrated = migrate_template(template)
    
    with open(output_path, 'w') as f:
        json.dump(migrated, f, indent=2)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python template_migration.py input_template.json output_template.json")
        sys.exit(1)
        
    migrate_template_file(sys.argv[1], sys.argv[2])