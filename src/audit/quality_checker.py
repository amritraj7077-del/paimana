"""
Data Quality Audit Module
Pre-extraction validation and quality scoring for PAIMANA reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Data quality audit report"""
    completeness_score: float
    anomalies: List[Dict]
    missing_fields: List[str]
    reliability_score: float
    total_records: int
    valid_records: int


class DataQualityAuditor:
    """
    Audit data quality of extracted PAIMANA project data
    Identifies missing fields, anomalies, and calculates quality scores
    """
    
    # Required fields for a complete project record
    REQUIRED_FIELDS = [
        'project_id',
        'project_name',
        'district',
        'category',
        'sanctioned_cost',
        'expenditure_to_date',
        'physical_progress_percent'
    ]
    
    def __init__(self):
        """Initialize quality auditor"""
        self.audit_history = []
    
    def check_completeness(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """
        Check completeness of required fields
        
        Args:
            df: DataFrame with project data
            
        Returns:
            Tuple of (completeness_score, list of missing fields)
        """
        present_fields = []
        missing_fields = []
        
        for field in self.REQUIRED_FIELDS:
            if field in df.columns:
                # Check how many non-null values
                non_null_count = df[field].notna().sum()
                if non_null_count > 0:
                    present_fields.append(field)
                else:
                    missing_fields.append(field)
            else:
                missing_fields.append(field)
        
        completeness_score = (len(present_fields) / len(self.REQUIRED_FIELDS)) * 100
        
        return completeness_score, missing_fields
    
    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect statistical anomalies in project data
        
        Args:
            df: DataFrame with project data
            
        Returns:
            List of anomaly dictionaries
        """
        anomalies = []
        
        # Anomaly 1: Physical progress > 100%
        if 'physical_progress_percent' in df.columns:
            over_complete = df[df['physical_progress_percent'] > 100]
            for idx, row in over_complete.iterrows():
                anomalies.append({
                    'type': 'OVER_100_PROGRESS',
                    'severity': 'HIGH',
                    'project_id': row.get('project_id', idx),
                    'value': row['physical_progress_percent'],
                    'message': f"Physical progress {row['physical_progress_percent']}% exceeds 100%"
                })
        
        # Anomaly 2: Expenditure exceeds sanctioned cost by >20%
        if 'expenditure_to_date' in df.columns and 'sanctioned_cost' in df.columns:
            df['overrun_percent'] = ((df['expenditure_to_date'] - df['sanctioned_cost']) / 
                                     df['sanctioned_cost'] * 100)
            
            high_overrun = df[df['overrun_percent'] > 20]
            for idx, row in high_overrun.iterrows():
                anomalies.append({
                    'type': 'COST_OVERRUN',
                    'severity': 'MEDIUM',
                    'project_id': row.get('project_id', idx),
                    'value': row['overrun_percent'],
                    'message': f"Cost overrun of {row['overrun_percent']:.1f}%"
                })
        
        # Anomaly 3: Zero or negative costs
        for cost_field in ['sanctioned_cost', 'expenditure_to_date']:
            if cost_field in df.columns:
                invalid_costs = df[(df[cost_field] <= 0) | (df[cost_field].isna())]
                for idx, row in invalid_costs.iterrows():
                    anomalies.append({
                        'type': 'INVALID_COST',
                        'severity': 'HIGH',
                        'project_id': row.get('project_id', idx),
                        'field': cost_field,
                        'message': f"Invalid {cost_field}: {row[cost_field]}"
                    })
        
        # Anomaly 4: Progress > 80% but expenditure < 50%
        if all(field in df.columns for field in ['physical_progress_percent', 'expenditure_to_date', 'sanctioned_cost']):
            df['exp_percent'] = (df['expenditure_to_date'] / df['sanctioned_cost'] * 100)
            mismatch = df[(df['physical_progress_percent'] > 80) & (df['exp_percent'] < 50)]
            
            for idx, row in mismatch.iterrows():
                anomalies.append({
                    'type': 'PROGRESS_EXPENDITURE_MISMATCH',
                    'severity': 'MEDIUM',
                    'project_id': row.get('project_id', idx),
                    'message': f"Progress {row['physical_progress_percent']}% but only {row['exp_percent']:.1f}% spent"
                })
        
        return anomalies
    
    def calculate_reliability_score(self, completeness: float, anomaly_count: int, 
                                    total_records: int) -> float:
        """
        Calculate overall reliability score
        
        Args:
            completeness: Completeness score (0-100)
            anomaly_count: Number of anomalies detected
            total_records: Total number of records
            
        Returns:
            Reliability score (0-100)
        """
        # Start with completeness
        score = completeness
        
        # Penalize for anomalies (each anomaly reduces score)
        if total_records > 0:
            anomaly_penalty = (anomaly_count / total_records) * 30  # Max 30 point penalty
            score -= min(anomaly_penalty, 30)
        
        return max(0, score)
    
    def audit(self, df: pd.DataFrame) -> QualityReport:
        """
        Perform comprehensive quality audit on project data
        
        Args:
            df: DataFrame with extracted project data
            
        Returns:
            QualityReport object with audit results
        """
        logger.info(f"Starting quality audit on {len(df)} records")
        
        # Check completeness
        completeness_score, missing_fields = self.check_completeness(df)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(df)
        
        # Count valid records (no critical anomalies)
        critical_anomalies = [a for a in anomalies if a['severity'] == 'HIGH']
        critical_project_ids = set([a['project_id'] for a in critical_anomalies])
        valid_records = len(df) - len(critical_project_ids)
        
        # Calculate reliability score
        reliability_score = self.calculate_reliability_score(
            completeness_score, 
            len(anomalies), 
            len(df)
        )
        
        report = QualityReport(
            completeness_score=completeness_score,
            anomalies=anomalies,
            missing_fields=missing_fields,
            reliability_score=reliability_score,
            total_records=len(df),
            valid_records=valid_records
        )
        
        self.audit_history.append(report)
        
        logger.info(f"Audit complete: Completeness={completeness_score:.1f}%, "
                   f"Anomalies={len(anomalies)}, Reliability={reliability_score:.1f}%")
        
        return report
    
    def generate_audit_summary(self, report: QualityReport) -> dict:
        """Generate human-readable audit summary"""
        high_severity = len([a for a in report.anomalies if a['severity'] == 'HIGH'])
        medium_severity = len([a for a in report.anomalies if a['severity'] == 'MEDIUM'])
        
        return {
            'overall_grade': self._get_grade(report.reliability_score),
            'completeness_score': f"{report.completeness_score:.1f}%",
            'reliability_score': f"{report.reliability_score:.1f}%",
            'total_records': report.total_records,
            'valid_records': report.valid_records,
            'anomalies': {
                'total': len(report.anomalies),
                'high_severity': high_severity,
                'medium_severity': medium_severity
            },
            'missing_fields': report.missing_fields,
            'recommendation': self._get_recommendation(report)
        }
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90: return 'A - Excellent'
        elif score >= 75: return 'B - Good'
        elif score >= 60: return 'C - Fair'
        elif score >= 40: return 'D - Poor'
        else: return 'F - Unreliable'
    
    def _get_recommendation(self, report: QualityReport) -> str:
        """Generate recommendation based on audit"""
        if report.reliability_score >= 80:
            return "Data quality is excellent. Safe to use for analysis."
        elif report.reliability_score >= 60:
            return "Data quality is acceptable. Review anomalies before critical analysis."
        else:
            return "Data quality is poor. Manual verification recommended before use."


def main():
    """Demo usage"""
    # Create sample data with some anomalies
    sample_data = pd.DataFrame({
        'project_id': [f'P{i:03d}' for i in range(10)],
        'project_name': [f'Project {i}' for i in range(10)],
        'district': ['Mumbai', 'Pune'] * 5,
        'category': ['Roads'] * 10,
        'sanctioned_cost': [100000, 200000, None, 150000, 180000, 220000, 500000, 300000, 250000, 400000],
        'expenditure_to_date': [80000, 250000, 140000, 120000, 150000, 180000, 600000, 200000, 180000, 350000],
        'physical_progress_percent': [75, 95, 85, 70, 110, 80, 90, 65, 72, 88]
    })
    
    auditor = DataQualityAuditor()
    report = auditor.audit(sample_data)
    
    print("\n=== Data Quality Audit Report ===")
    summary = auditor.generate_audit_summary(report)
    
    print(f"\nOverall Grade: {summary['overall_grade']}")
    print(f"Completeness: {summary['completeness_score']}")
    print(f"Reliability: {summary['reliability_score']}")
    print(f"Valid Records: {summary['valid_records']}/{summary['total_records']}")
    print(f"\nAnomalies Found: {summary['anomalies']['total']}")
    print(f"  - High Severity: {summary['anomalies']['high_severity']}")
    print(f"  - Medium Severity: {summary['anomalies']['medium_severity']}")
    print(f"\nMissing Fields: {summary['missing_fields']}")
    print(f"\nRecommendation: {summary['recommendation']}")
    
    print("\n=== Anomaly Details ===")
    for i, anomaly in enumerate(report.anomalies[:5], 1):  # Show first 5
        print(f"{i}. [{anomaly['severity']}] {anomaly['type']}: {anomaly['message']}")


if __name__ == '__main__':
    main()
