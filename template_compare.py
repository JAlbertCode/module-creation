"""Compare and analyze different templates"""

from typing import Dict, Any, List, Tuple
import json
from deepdiff import DeepDiff
import pandas as pd
from template_tester import TemplateTester

class TemplateComparator:
    """Compare configuration templates and analyze differences"""
    
    def __init__(self):
        self.tester = TemplateTester()
    
    def compare_templates(self, template1: Dict[str, Any], 
                         template2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two templates and return differences"""
        
        # Use DeepDiff for detailed comparison
        diff = DeepDiff(template1, template2, ignore_order=True)
        
        # Format the differences
        differences = {
            'added': diff.get('dictionary_item_added', []),
            'removed': diff.get('dictionary_item_removed', []),
            'changed': diff.get('values_changed', {}),
            'type_changes': diff.get('type_changes', {})
        }
        
        return differences
    
    def analyze_differences(self, differences: Dict[str, Any]) -> List[str]:
        """Analyze and explain template differences"""
        analysis = []
        
        # Analyze additions
        if differences['added']:
            analysis.append("New configurations added:")
            for item in differences['added']:
                analysis.append(f"- Added: {item}")
        
        # Analyze removals
        if differences['removed']:
            analysis.append("\nRemoved configurations:")
            for item in differences['removed']:
                analysis.append(f"- Removed: {item}")
        
        # Analyze changes
        if differences['changed']:
            analysis.append("\nChanged values:")
            for path, change in differences['changed'].items():
                analysis.append(
                    f"- {path}: {change['old_value']} → {change['new_value']}"
                )
        
        # Analyze type changes
        if differences['type_changes']:
            analysis.append("\nType changes:")
            for path, change in differences['type_changes'].items():
                analysis.append(
                    f"- {path}: {change['old_type']} → {change['new_type']}"
                )
        
        return analysis
    
    def compare_performance(self, template1: Dict[str, Any], 
                          template2: Dict[str, Any],
                          model_id: str) -> pd.DataFrame:
        """Compare performance metrics between templates"""
        
        # Test both templates
        result1 = self.tester.test_template(template1, model_id)
        result2 = self.tester.test_template(template2, model_id)
        
        # Create comparison DataFrame
        metrics = pd.DataFrame({
            'Template 1': result1.metrics,
            'Template 2': result2.metrics
        }).fillna('-')
        
        # Add relative difference
        metrics['Difference (%)'] = pd.Series({
            k: ((result2.metrics[k] - result1.metrics[k]) / result1.metrics[k] * 100)
            for k in result1.metrics
            if isinstance(result1.metrics[k], (int, float)) and 
               k in result2.metrics and 
               isinstance(result2.metrics[k], (int, float))
        })
        
        return metrics
    
    def generate_report(self, template1: Dict[str, Any], 
                       template2: Dict[str, Any],
                       model_id: str = None) -> str:
        """Generate a comprehensive comparison report"""
        report = ["# Template Comparison Report\n"]
        
        # Compare configurations
        differences = self.compare_templates(template1, template2)
        analysis = self.analyze_differences(differences)
        
        report.append("## Configuration Differences")
        report.extend(analysis)
        report.append("")
        
        # Compare performance if model_id provided
        if model_id:
            report.append("## Performance Comparison")
            metrics = self.compare_performance(template1, template2, model_id)
            
            # Convert DataFrame to markdown
            report.append(metrics.to_markdown())
            report.append("")
            
            # Add recommendations
            report.append("## Recommendations")
            recommendations = self._generate_recommendations(metrics, differences)
            report.extend(recommendations)
        
        return "\n".join(report)
    
    def _generate_recommendations(self, metrics: pd.DataFrame, 
                                differences: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison"""
        recommendations = []
        
        # Check significant performance differences
        for metric, row in metrics.iterrows():
            if 'Difference (%)' in row and isinstance(row['Difference (%)'], float):
                diff = row['Difference (%)']
                if abs(diff) > 10:  # More than 10% difference
                    better = "better" if diff < 0 else "worse"
                    recommendations.append(
                        f"- {metric} is {abs(diff):.1f}% {better} in Template 2"
                    )
        
        # Resource utilization recommendations
        resource_changes = [d for d in differences['changed'] 
                          if 'resources' in d]
        if resource_changes:
            recommendations.append("\nResource Configuration:")
            for change in resource_changes:
                recommendations.append(f"- Consider the impact of {change}")
        
        return recommendations

def compare_template_files(template1_path: str, template2_path: str, 
                         model_id: str = None) -> str:
    """Convenience function to compare template files"""
    with open(template1_path) as f1, open(template2_path) as f2:
        template1 = json.load(f1)
        template2 = json.load(f2)
    
    comparator = TemplateComparator()
    return comparator.generate_report(template1, template2, model_id)

if __name__ == '__main__':
    import sys
    if len(sys.argv) not in [3, 4]:
        print("Usage: python template_compare.py template1.json template2.json [model_id]")
        sys.exit(1)
    
    model_id = sys.argv[3] if len(sys.argv) == 4 else None
    print(compare_template_files(sys.argv[1], sys.argv[2], model_id))