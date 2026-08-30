"""
PAIMANA Portal Web Scraper
Extracts infrastructure project data from PAIMANA portals across Indian states
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Optional
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PAIMANAScraper:
    """
    Web scraper for PAIMANA (Project Assessment, Impact Monitoring and Appraisal of NAtional Schemes) portals
    """
    
    # State-wise PAIMANA portal URLs (sample mapping)
    STATE_PORTALS = {
        'maharashtra': 'https://paimana.gov.in/maharashtra',
        'karnataka': 'https://paimana.gov.in/karnataka',
        'tamil_nadu': 'https://paimana.gov.in/tamilnadu',
        # Add more states as needed
    }
    
    def __init__(self, state: str = 'maharashtra', delay: float = 1.0):
        """
        Initialize scraper for a specific state
        
        Args:
            state: State name (lowercase with underscores)
            delay: Delay between requests in seconds (respectful scraping)
        """
        self.state = state.lower().replace(' ', '_')
        self.base_url = self.STATE_PORTALS.get(self.state, 'https://paimana.gov.in')
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PAIMANA-Intelligence-Platform/0.1 (Educational/Research Purpose)'
        })
        
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a web page
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None if fetch fails
        """
        try:
            time.sleep(self.delay)  # Rate limiting
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def extract_project_from_row(self, row) -> Optional[Dict]:
        """
        Extract project data from HTML table row
        
        Args:
            row: BeautifulSoup table row element
            
        Returns:
            Dictionary with project data or None
        """
        try:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 8:
                return None
            
            project_data = {
                'project_id': cells[0].get_text(strip=True),
                'project_name': cells[1].get_text(strip=True),
                'district': cells[2].get_text(strip=True),
                'category': cells[3].get_text(strip=True),
                'sanctioned_cost': self._parse_currency(cells[4].get_text(strip=True)),
                'expenditure_to_date': self._parse_currency(cells[5].get_text(strip=True)),
                'physical_progress_percent': self._parse_percentage(cells[6].get_text(strip=True)),
                'completion_date': cells[7].get_text(strip=True),
                'state': self.state,
                'scraped_at': datetime.now().isoformat()
            }
            
            return project_data
        except Exception as e:
            logger.error(f"Failed to parse row: {e}")
            return None
    
    def _parse_currency(self, value: str) -> Optional[float]:
        """Parse Indian currency format (e.g., '10,50,000' or '10.5 Cr')"""
        try:
            # Remove rupee symbol and spaces
            value = value.replace('₹', '').replace('Rs', '').replace(',', '').strip()
            
            # Handle crores and lakhs
            if 'Cr' in value or 'cr' in value:
                return float(value.replace('Cr', '').replace('cr', '').strip()) * 10000000
            elif 'Lakh' in value or 'lakh' in value:
                return float(value.replace('Lakh', '').replace('lakh', '').strip()) * 100000
            else:
                return float(value)
        except:
            return None
    
    def _parse_percentage(self, value: str) -> Optional[float]:
        """Parse percentage string"""
        try:
            return float(value.replace('%', '').strip())
        except:
            return None
    
    def extract_projects(self, max_pages: int = 1) -> pd.DataFrame:
        """
        Extract projects from PAIMANA portal
        
        Args:
            max_pages: Maximum number of pages to scrape
            
        Returns:
            DataFrame with extracted projects
        """
        logger.info(f"Starting extraction for {self.state}")
        all_projects = []
        
        # NOTE: This is a DEMO implementation
        # In production, this would scrape actual PAIMANA portal pages
        # For MVP, we'll generate sample data to demonstrate the pipeline
        
        logger.warning("Using sample data for MVP demonstration")
        all_projects = self._generate_sample_data()
        
        df = pd.DataFrame(all_projects)
        logger.info(f"Extracted {len(df)} projects from {self.state}")
        
        return df
    
    def _generate_sample_data(self) -> List[Dict]:
        """
        Generate sample project data for demonstration
        This simulates scraped data from PAIMANA portal
        """
        import random
        from datetime import datetime, timedelta
        
        # Set seed for consistent data generation
        random.seed(42)
        
        categories = ['Roads', 'Railways', 'Irrigation', 'Buildings', 'Bridges']
        districts = ['Mumbai', 'Pune', 'Nagpur', 'Nashik', 'Aurangabad', 'Solapur', 'Kolhapur']
        agencies = ['PWD', 'Railways', 'Irrigation Dept', 'NHAI', 'MMRDA']
        
        projects = []
        for i in range(50):  # Generate 50 sample projects
            sanctioned = random.randint(5, 500) * 100000  # 5L to 50Cr
            expenditure = sanctioned * random.uniform(0.3, 1.2)  # 30% to 120% spent
            progress = min(100, random.uniform(20, 110))  # 20% to 110% progress
            
            planned_days = random.randint(365, 1095)  # 1-3 years
            actual_days = int(planned_days * random.uniform(0.8, 1.5))  # 80% to 150% time
            
            # Generate more realistic delays: mostly past dates (delayed), some future dates
            # 70% delayed (past dates), 30% on-time/ahead (future dates)
            if random.random() < 0.7:
                # Delayed: planned completion was in the past
                days_offset = random.randint(-365, -30)  # 30 to 365 days ago
            else:
                # On-time/ahead: planned completion is in the future
                days_offset = random.randint(30, 365)  # 30 to 365 days from now
            
            planned_completion = datetime.now() + timedelta(days=days_offset)
            
            project = {
                'project_id': f'{self.state.upper()}-{i+1:04d}',
                'project_name': f'{random.choice(categories)} Development Project - {random.choice(districts)} Sector {i+1}',
                'district': random.choice(districts),
                'category': random.choice(categories),
                'sanctioned_cost': sanctioned,
                'expenditure_to_date': expenditure,
                'physical_progress_percent': progress,
                'planned_completion_date': planned_completion.strftime('%Y-%m-%d'),
                'implementing_agency': random.choice(agencies),
                'state': self.state,
                'scraped_at': datetime.now().isoformat()
            }
            projects.append(project)
        
        return projects


def main():
    """Demo usage"""
    scraper = PAIMANAScraper(state='maharashtra')
    projects_df = scraper.extract_projects()
    
    # Save to CSV
    output_path = 'data/raw/maharashtra_projects.csv'
    projects_df.to_csv(output_path, index=False)
    print(f"Extracted {len(projects_df)} projects to {output_path}")
    print(projects_df.head())


if __name__ == '__main__':
    main()
