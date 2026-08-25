"""
Demo Script: End-to-End Pipeline
Demonstrates the complete PAIMANA Intelligence Platform workflow
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.scrapers.paimana_scraper import PAIMANAScraper
from src.audit.quality_checker import DataQualityAuditor
from src.analytics.delay_detector import DelayAnalyzer
import pandas as pd


def main():
    print("="*70)
    print("PAIMANA Intelligence Platform - Complete Pipeline Demo")
    print("="*70)
    
    # Step 1: Scrape PAIMANA portal data
    print("\n[Step 1/4] Extracting project data from PAIMANA portal...")
    print("-" * 70)
    scraper = PAIMANAScraper(state='maharashtra')
    projects_df = scraper.extract_projects()
    print(f"✓ Extracted {len(projects_df)} projects from Maharashtra")
    print(f"\nSample projects:")
    print(projects_df[['project_id', 'project_name', 'district', 'category']].head())
    
    # Step 2: Data Quality Audit
    print("\n[Step 2/4] Running data quality audit...")
    print("-" * 70)
    auditor = DataQualityAuditor()
    quality_report = auditor.audit(projects_df)
    summary = auditor.generate_audit_summary(quality_report)
    
    print(f"✓ Quality Audit Complete")
    print(f"  Overall Grade: {summary['overall_grade']}")
    print(f"  Reliability Score: {summary['reliability_score']}")
    print(f"  Valid Records: {summary['valid_records']}/{summary['total_records']}")
    print(f"  Anomalies Found: {summary['anomalies']['total']} ({summary['anomalies']['high_severity']} high severity)")
    print(f"  Recommendation: {summary['recommendation']}")
    
    # Step 3: Analytics - Delay Detection & Cost Overruns
    print("\n[Step 3/4] Analyzing delays and cost overruns...")
    print("-" * 70)
    analyzer = DelayAnalyzer()
    analytics_report = analyzer.generate_analytics_report(projects_df)
    
    stats = analytics_report['summary_statistics']
    print(f"✓ Analytics Complete")
    print(f"  Total Projects: {stats['total_projects']}")
    print(f"  Delayed Projects: {stats['delayed_projects']}")
    print(f"  Average Delay: {stats['average_delay_days']:.1f} days")
    print(f"  Projects with Cost Overrun: {stats['projects_with_cost_overrun']}")
    print(f"  Average Progress: {stats['average_progress_percent']:.1f}%")
    
    # Top delayed projects
    print(f"\n  Top 5 Delayed Projects:")
    for i, proj in enumerate(analytics_report['top_delayed_projects'][:5], 1):
        print(f"    {i}. {proj['project_id']}: {proj['delay_days']} days ({proj['physical_progress_percent']:.1f}% complete)")
    
    # Top cost overruns
    if analytics_report['top_cost_overruns']:
        print(f"\n  Top 5 Cost Overruns:")
        for i, proj in enumerate(analytics_report['top_cost_overruns'][:5], 1):
            print(f"    {i}. {proj['project_id']}: {proj['cost_overrun_percent']:.1f}% overrun")
    
    # Step 4: Export Data
    print("\n[Step 4/4] Exporting results...")
    print("-" * 70)
    
    # Export processed data
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    projects_df.to_csv(output_dir / 'maharashtra_projects_analyzed.csv', index=False)
    print(f"✓ Exported analyzed projects to: {output_dir / 'maharashtra_projects_analyzed.csv'}")
    
    # Export analytics report
    import json
    with open(output_dir / 'analytics_report.json', 'w') as f:
        json.dump(analytics_report, f, indent=2, default=str)
    print(f"✓ Exported analytics report to: {output_dir / 'analytics_report.json'}")
    
    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE - Key Takeaways")
    print("="*70)
    print(f"""
✅ Successfully extracted and analyzed {len(projects_df)} infrastructure projects
✅ Identified {stats['delayed_projects']} delayed projects for immediate attention
✅ Detected {len(quality_report.anomalies)} data quality anomalies
✅ Generated actionable insights for policymakers and citizens
✅ Exported structured datasets for further analysis

Next Steps:
- Run the dashboard: python -m src.dashboard.app
- Access API endpoints at http://localhost:5000/api/*
- Extend to other states by modifying scraper configuration
- Integrate with AIKosh for public data sharing
    """)


if __name__ == '__main__':
    main()
