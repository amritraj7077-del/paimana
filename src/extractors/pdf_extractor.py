"""
PDF Extraction Module
Extracts structured data from PAIMANA progress report PDFs using NLP and pattern matching
"""

import fitz  # PyMuPDF
import re
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extract infrastructure project data from PAIMANA PDF reports
    """
    
    # Common patterns in PAIMANA reports (regex)
    PATTERNS = {
        'project_name': r'Project\s+Name\s*:?\s*(.+?)(?:\n|$)',
        'district': r'District\s*:?\s*([A-Za-z\s]+)',
        'sanctioned_cost': r'Sanctioned\s+Cost\s*:?\s*Rs\.?\s*([\d,\.]+)\s*(Cr|Lakh)?',
        'expenditure': r'Expenditure\s*:?\s*Rs\.?\s*([\d,\.]+)\s*(Cr|Lakh)?',
        'progress': r'Physical\s+Progress\s*:?\s*([\d\.]+)\s*%',
        'completion_date': r'Completion\s+Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
    }
    
    def __init__(self):
        """Initialize PDF extractor"""
        self.extracted_count = 0
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""
    
    def extract_project_data(self, text: str) -> Dict:
        """
        Extract structured project data from text using pattern matching
        
        Args:
            text: PDF text content
            
        Returns:
            Dictionary with extracted fields
        """
        project_data = {}
        
        for field, pattern in self.PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if field in ['sanctioned_cost', 'expenditure']:
                    # Parse currency with unit
                    value = match.group(1).replace(',', '')
                    unit = match.group(2) if len(match.groups()) > 1 else ''
                    multiplier = 10000000 if 'Cr' in unit else (100000 if 'Lakh' in unit else 1)
                    project_data[field] = float(value) * multiplier
                elif field == 'progress':
                    project_data[field] = float(match.group(1))
                else:
                    project_data[field] = match.group(1).strip()
            else:
                project_data[field] = None
        
        return project_data
    
    def extract_from_pdf(self, pdf_path: str) -> Optional[Dict]:
        """
        Main extraction method: PDF -> structured data
        
        Args:
            pdf_path: Path to PAIMANA PDF report
            
        Returns:
            Extracted project data dictionary
        """
        logger.info(f"Extracting data from {pdf_path}")
        
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        
        project_data = self.extract_project_data(text)
        project_data['source_file'] = Path(pdf_path).name
        
        self.extracted_count += 1
        return project_data
    
    def extract_from_directory(self, directory_path: str) -> pd.DataFrame:
        """
        Extract data from all PDFs in a directory
        
        Args:
            directory_path: Path to directory containing PDF reports
            
        Returns:
            DataFrame with all extracted projects
        """
        directory = Path(directory_path)
        pdf_files = list(directory.glob('*.pdf'))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        all_projects = []
        for pdf_file in pdf_files:
            project_data = self.extract_from_pdf(str(pdf_file))
            if project_data:
                all_projects.append(project_data)
        
        df = pd.DataFrame(all_projects)
        logger.info(f"Successfully extracted {len(df)} projects from PDFs")
        
        return df


def main():
    """Demo usage"""
    extractor = PDFExtractor()
    
    # For MVP, demonstrate with sample data since we don't have real PDFs
    print("PDF Extractor initialized successfully")
    print("In production, use: extractor.extract_from_directory('path/to/pdfs')")
    
    # Demo sample extraction
    sample_text = """
    Project Name: NH-48 Widening Project
    District: Pune
    Sanctioned Cost: Rs. 150 Cr
    Expenditure: Rs. 120 Cr
    Physical Progress: 75.5%
    Completion Date: 31-12-2025
    """
    
    result = extractor.extract_project_data(sample_text)
    print("\nSample extraction result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
