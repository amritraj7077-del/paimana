"""
Delay Detection & Analytics Module
Identifies delayed projects and calculates cost overruns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DelayAnalyzer:
    """
    Analyze infrastructure project delays and cost overruns
    """
    
    def __init__(self, delay_threshold_percent: float = 20.0):
        """
        Initialize delay analyzer
        
        Args:
            delay_threshold_percent: Threshold for flagging significant delays (default 20%)
        """
        self.delay_threshold = delay_threshold_percent
    
    def calculate_delay_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate delay in days for each project
        
        Args:
            df: DataFrame with 'planned_completion_date' column
            
        Returns:
            DataFrame with added 'delay_days' column
        """
        if 'planned_completion_date' not in df.columns:
            logger.warning("No 'planned_completion_date' column found")
            return df
        
        # Convert to datetime
        df['planned_completion_date'] = pd.to_datetime(df['planned_completion_date'], errors='coerce')
        current_date = datetime.now()
        
        # Calculate delay: positive = delayed, negative = ahead of schedule
        df['delay_days'] = (current_date - df['planned_completion_date']).dt.days
        
        # Only count as delay if project is not yet complete
        if 'physical_progress_percent' in df.columns:
            # If project is 100% complete, delay is 0
            df.loc[df['physical_progress_percent'] >= 100, 'delay_days'] = 0
        
        return df
    
    def calculate_cost_overrun(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cost overrun percentage using dataset columns
        """
        if 'Cost_Overrun_Ratio' in df.columns:
            df['cost_overrun_percent'] = df['Cost_Overrun_Ratio'] * 100
        elif 'Revised Cost (Rs. Crore)' in df.columns and 'Original Cost (Rs. Crore)' in df.columns:
            orig = df['Original Cost (Rs. Crore)']
            rev = df['Revised Cost (Rs. Crore)']
            df['cost_overrun_percent'] = np.where(orig > 0, (rev - orig) / orig * 100, 0)
        elif all(col in df.columns for col in ['sanctioned_cost', 'expenditure_to_date']):
            df['cost_overrun_percent'] = (
                (df['expenditure_to_date'] - df['sanctioned_cost']) / df['sanctioned_cost'] * 100
            )
        else:
            df['cost_overrun_percent'] = 0.0
        
        return df
    
    def find_delayed_projects(self, df: pd.DataFrame, threshold_percent: float = None) -> pd.DataFrame:
        """
        Find projects that are delayed
        """
        # Calculate delays if not already present
        if 'delay_days' not in df.columns:
            df = self.calculate_delay_days(df)
        
        # Filter delayed projects
        delayed = df[df['delay_days'] > 0].copy()
        if len(delayed) == 0:
            return pd.DataFrame(columns=df.columns)

        delayed['delay_percent'] = (delayed['delay_days'] / 730) * 100
        
        if threshold_percent is not None:
            filtered = delayed[delayed['delay_percent'] >= threshold_percent]
            if len(filtered) > 0:
                delayed = filtered
        
        logger.info(f"Found {len(delayed)} delayed projects")
        return delayed.sort_values('delay_days', ascending=False)
    
    def find_cost_overruns(self, df: pd.DataFrame, threshold_percent: float = 0.0) -> pd.DataFrame:
        """
        Find projects with cost overruns
        """
        if 'cost_overrun_percent' not in df.columns:
            df = self.calculate_cost_overrun(df)
        
        overruns = df[df['cost_overrun_percent'] > threshold_percent].copy()
        if len(overruns) == 0 and threshold_percent > 0:
            overruns = df[df['cost_overrun_percent'] > 0].copy()

        logger.info(f"Found {len(overruns)} projects with >{threshold_percent}% cost overrun")
        return overruns.sort_values('cost_overrun_percent', ascending=False)
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for projects
        
        Args:
            df: DataFrame with project data
            
        Returns:
            Dictionary with summary statistics
        """
        # Ensure calculations are done
        if 'delay_days' not in df.columns:
            df = self.calculate_delay_days(df)
        if 'cost_overrun_percent' not in df.columns:
            df = self.calculate_cost_overrun(df)
        
        stats = {
            'total_projects': len(df),
            'delayed_projects': len(df[df['delay_days'] > 0]),
            'on_time_projects': len(df[df['delay_days'] <= 0]),
            'average_delay_days': df['delay_days'].mean(),
            'max_delay_days': df['delay_days'].max(),
            'projects_with_cost_overrun': len(df[df['cost_overrun_percent'] > 0]),
            'average_cost_overrun_percent': df[df['cost_overrun_percent'] > 0]['cost_overrun_percent'].mean(),
            'total_sanctioned_cost': df['sanctioned_cost'].sum() if 'sanctioned_cost' in df.columns else 0,
            'total_expenditure': df['expenditure_to_date'].sum() if 'expenditure_to_date' in df.columns else 0,
            'average_progress_percent': df['physical_progress_percent'].mean() if 'physical_progress_percent' in df.columns else 0
        }
        
        return stats
    
    def generate_analytics_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive analytics report
        
        Args:
            df: DataFrame with project data
            
        Returns:
            Dictionary with full analytics report
        """
        logger.info("Generating analytics report")
        
        # Ensure all calculations are done
        df = self.calculate_delay_days(df)
        df = self.calculate_cost_overrun(df)
        
        # Get summary stats
        stats = self.generate_summary_stats(df)
        
        # Top delayed projects
        delayed = self.find_delayed_projects(df, threshold_percent=20)
        top_delayed = delayed.nlargest(10, 'delay_days')[
            ['project_id', 'project_name', 'district', 'delay_days', 'physical_progress_percent']
        ].to_dict('records') if len(delayed) > 0 else []
        
        # Top cost overruns
        overruns = self.find_cost_overruns(df, threshold_percent=10)
        top_overruns = overruns.nlargest(10, 'cost_overrun_percent')[
            ['project_id', 'project_name', 'cost_overrun_percent', 'sanctioned_cost', 'expenditure_to_date']
        ].to_dict('records') if len(overruns) > 0 else []
        
        # Category-wise analysis
        category_stats = {}
        if 'category' in df.columns:
            for category in df['category'].unique():
                cat_df = df[df['category'] == category]
                category_stats[category] = {
                    'total_projects': len(cat_df),
                    'delayed_projects': len(cat_df[cat_df['delay_days'] > 0]),
                    'average_delay': cat_df['delay_days'].mean(),
                    'average_progress': cat_df['physical_progress_percent'].mean() if 'physical_progress_percent' in cat_df.columns else 0
                }
        
        report = {
            'summary_statistics': stats,
            'top_delayed_projects': top_delayed,
            'top_cost_overruns': top_overruns,
            'category_analysis': category_stats,
            'generated_at': datetime.now().isoformat()
        }
        
        return report


def main():
    """Demo usage"""
    # Create sample data
    from src.scrapers.paimana_scraper import PAIMANAScraper
    
    print("Generating sample project data...")
    scraper = PAIMANAScraper(state='maharashtra')
    projects_df = scraper.extract_projects()
    
    print(f"\nAnalyzing {len(projects_df)} projects...")
    analyzer = DelayAnalyzer(delay_threshold_percent=20)
    
    # Generate analytics report
    report = analyzer.generate_analytics_report(projects_df)
    
    print("\n=== ANALYTICS REPORT ===\n")
    print("SUMMARY STATISTICS:")
    for key, value in report['summary_statistics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTOP DELAYED PROJECTS: {len(report['top_delayed_projects'])}")
    for i, proj in enumerate(report['top_delayed_projects'][:5], 1):
        print(f"  {i}. {proj['project_id']}: {proj['delay_days']} days delay ({proj['physical_progress_percent']:.1f}% complete)")
    
    print(f"\nTOP COST OVERRUNS: {len(report['top_cost_overruns'])}")
    for i, proj in enumerate(report['top_cost_overruns'][:5], 1):
        print(f"  {i}. {proj['project_id']}: {proj['cost_overrun_percent']:.1f}% overrun")
    
    print("\nCATEGORY ANALYSIS:")
    for category, stats in report['category_analysis'].items():
        print(f"  {category}: {stats['delayed_projects']}/{stats['total_projects']} delayed (avg: {stats['average_delay']:.1f} days)")


if __name__ == '__main__':
    main()
