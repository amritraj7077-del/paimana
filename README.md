# PAIMANA Intelligence Platform
## Infrastructure Transparency Through AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An AI-powered platform that transforms India's fragmented infrastructure project data into actionable insights through automated extraction, quality auditing, and predictive analytics.

## Problem Statement

India invests over Rs.10 lakh crores annually in infrastructure, yet tracking project progress remains opaque. The PAIMANA portal publishes monthly progress reports, but data is locked in PDFs and inconsistent formats, making accountability nearly impossible.

## Solution

PAIMANA Intelligence Platform combines three powerful capabilities:

1. **Data Quality Audit** - Pre-extraction validation ensuring reliable downstream analytics
2. **Intelligent Extraction** - NLP-powered extraction from PDFs and web portals with schema normalization
3. **Geo-Intelligence Analytics** - Delay detection, cost overrun analysis, and ML-based completion prediction

## Key Features

- **Automated Web Scraping**: Extract data from PAIMANA portals across all states
- **PDF Intelligence**: Parse unstructured reports using NLP and custom extraction rules
- **Quality Assurance**: Audit data completeness, detect anomalies, score reliability
- **Geocoding**: Map projects to GPS coordinates for spatial analysis
- **Delay Detection**: Identify projects >20% behind schedule automatically
- **Cost Analytics**: Track expenditure vs. sanctioned budgets
- **Predictive Models**: ML-based forecasting of project completion dates
- **Interactive Dashboard**: Map-based visualization with filters and exports
- **Open Data**: All outputs available as CSV/JSON for research and integration

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/[username]/paimana-intelligence-platform.git
cd paimana-intelligence-platform

# Install dependencies
pip install -r requirements.txt

# Run data quality audit on sample reports
python -m src.audit.quality_checker --input data/sample_reports

# Extract data from PAIMANA portal
python -m src.scrapers.paimana_scraper --state maharashtra --output data/extracted

# Run analytics pipeline
python -m src.analytics.delay_detector --input data/extracted/projects.csv

# Launch dashboard
python -m src.dashboard.app
```

## Project Structure

```
paimana-intelligence-platform/
├── src/
│   ├── scrapers/          # Web scraping modules
│   ├── extractors/        # PDF extraction and NLP
│   ├── audit/             # Data quality validation
│   ├── analytics/         # Delay detection, predictions
│   ├── geocoding/         # Location normalization
│   ├── dashboard/         # Web interface
│   └── utils/             # Shared utilities
├── data/
│   ├── raw/               # Raw scraped data
│   ├── processed/         # Cleaned datasets
│   ├── sample_reports/    # Demo data
│   └── schemas/           # Data schemas
├── notebooks/             # Jupyter analysis notebooks
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md              # This file
```

## Usage Examples

### 1. Audit Data Quality

```python
from src.audit.quality_checker import DataQualityAuditor

auditor = DataQualityAuditor()
report = auditor.audit_paimana_report('path/to/report.pdf')

print(f"Completeness Score: {report.completeness_score}")
print(f"Anomalies Found: {len(report.anomalies)}")
print(f"Missing Fields: {report.missing_fields}")
```

### 2. Extract Project Data

```python
from src.scrapers.paimana_scraper import PAIMANAScraper

scraper = PAIMANAScraper(state='maharashtra')
projects = scraper.extract_projects()

# Save to CSV
projects.to_csv('maharashtra_projects.csv', index=False)
```

### 3. Detect Delays

```python
from src.analytics.delay_detector import DelayAnalyzer

analyzer = DelayAnalyzer()
delayed_projects = analyzer.find_delayed_projects(
    'data/processed/projects.csv',
    threshold_percent=20
)

print(f"Found {len(delayed_projects)} delayed projects")
```

## Data Sources

- **PAIMANA Portal**: https://paimana.gov.in (Open Government Data License - India)
- **India Administrative Boundaries**: DataMeet India Maps (CC-BY 4.0)
- **Census District Codes**: Census of India 2011 (Public Domain)

## Output Data Schema

All extracted projects follow this standardized schema:

```csv
project_id,project_name,state,district,category,sanctioned_cost,total_expenditure,
physical_progress_percent,planned_completion_date,current_expected_completion_date,
implementing_agency,last_updated,latitude,longitude,delay_days,cost_overrun_percent
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Roadmap

### Phase 1 (Current - Hackathon MVP)
- [x] Core extraction pipeline
- [x] Data quality audit module
- [x] Basic analytics (delay detection, cost overrun)
- [x] Sample dataset extraction
- [x] Interactive dashboard prototype (Plotly maps, ML predictions, Excel export)

### Phase 2 (Mentorship Phase)
- [ ] Scale to all 28 states
- [ ] ML-based delay prediction model
- [ ] Natural language search
- [ ] Mobile-responsive PWA
- [ ] AIKosh integration

### Phase 3 (Production)
- [ ] Real-time monitoring
- [ ] WhatsApp/Telegram alerts
- [ ] Crowdsourced validation
- [ ] Multi-portal integration (GeM, CPPP)

## Limitations & Current Scope

### MVP Constraints

This is a **hackathon MVP** built to demonstrate core functionality. The following limitations apply to the current version:

#### 1. **Sample Data Generation**
- **Current State**: The scraper uses `_generate_sample_data()` to create 50 synthetic projects instead of scraping live PAIMANA portal data
- **Reason**: Real portal requires authentication, has varying HTML structures across states, and rate-limiting constraints
- **Impact**: Demonstrates the complete pipeline (extraction → audit → analytics → visualization) without requiring live portal access
- **Future**: Will be adapted to actual PAIMANA portal URLs and HTML structures in production

#### 2. **Basic ML Model**
- **Current State**: Uses `scikit-learn`'s `LinearRegression` for delay prediction
- **Limitations**: 
  - Trained on synthetic data, not historical real-world projects
  - Simple feature set (progress, expenditure, cost, category)
  - No time-series or seasonal factors
- **Future Improvements**:
  - Random Forest / Gradient Boosting models
  - Training on 2+ years of historical project data
  - Advanced features (agency performance, budget allocation timelines, monsoon seasons)

#### 3. **Geocoding Approach**
- **Current State**: Uses Nominatim (OpenStreetMap) API with fallbacks to pre-defined Maharashtra district coordinates
- **Reason**: Nominatim has rate limits (1 request/second); not all districts return accurate results
- **Impact**: Ensures reliable demo with accurate map visualization
- **Future**: Implement caching, bulk geocoding, and premium geocoding services (Google Maps API)

#### 4. **Dashboard Architecture**
- **Current State**: Flask-based backend with in-memory data caching
- **Limitations**:
  - Data reloads on server restart (no persistence)
  - Single-threaded, not production-ready for high traffic
  - No user authentication or multi-tenancy
- **Future**: Migrate to PostgreSQL/MongoDB for persistence, add Redis caching, implement user accounts

#### 5. **Data Volume**
- **Current Scope**: 50 sample projects for demonstration
- **Reason**: Manageable for hackathon demo, ensures fast dashboard loading
- **Scalability**: Can easily increase to 500+ by modifying `num_projects` parameter
- **Production Target**: 10,000+ real projects across all states

#### 6. **PDF Extraction**
- **Current State**: Regex-based pattern matching with PyMuPDF
- **Limitations**: Requires PDF patterns to match expected formats; brittle to layout changes
- **Future**: Implement ML-based table extraction (e.g., Camelot, LayoutLM) and OCR for scanned PDFs

#### 7. **Real-time Updates**
- **Current State**: Static dataset loaded at startup
- **Future**: Automated monthly scraping pipeline, webhook-based update notifications

#### 8. **Browser Compatibility**
- **Tested On**: Modern Chrome/Edge
- **Known Issues**: Plotly.js maps may render slowly on older browsers
- **Future**: Progressive enhancement, fallback to static images

### What Works Reliably in MVP

✅ **Fully Functional Features**:
- Data quality audit with 4 anomaly types and A-F grading
- Delay detection and cost overrun analytics
- ML-based delay predictions (on sample data)
- Interactive Plotly map with color-coded project status
- Category comparison chart
- Excel export (projects + predictions)
- RESTful API endpoints (`/api/projects`, `/api/analytics`, `/api/quality-report`, `/api/ml-predictions`)
- Modular, well-documented codebase

### Known Issues

- **No automated tests**: MVP lacks comprehensive unit tests (planned for Phase 2)
- **Hardcoded thresholds**: Delay threshold (20%) and quality scoring are hardcoded, not configurable
- **Limited error handling**: Some edge cases in PDF extraction may not gracefully fail

## Impact

**Primary Beneficiaries:**
- 200M+ citizens in project-affected areas
- Investigative journalists seeking accountability data
- RTI activists and civil society organizations
- Government auditors and policymakers
- Infrastructure researchers and academics

**Measurable Outcomes:**
- 10,000+ projects made queryable (vs. 0 currently)
- 80% reduction in research time for journalists/activists
- Evidence base for infrastructure policy reforms

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this data or code in your research, please cite:

```bibtex
@software{paimana_intelligence_2026,
  title = {PAIMANA Intelligence Platform: Infrastructure Transparency Through AI},
  author = {[Your Name/Team]},
  year = {2026},
  url = {https://github.com/[username]/paimana-intelligence-platform}
}
```

## Acknowledgments

- **AI for All Hackathon** by Factly and Meta
- **AIKosh** for open data infrastructure
- **DataMeet** for India administrative boundary datasets
- All contributors to open government data in India

## Contact

- GitHub Issues: For bug reports and feature requests
- Email: [your-email@example.com]
- Project Website: [Coming Soon]

## Support the Project

If this project helps your work, please:
- ⭐ Star the repository
- 🔗 Share with others working on civic tech
- 📝 Cite in your publications
- 🤝 Contribute improvements

---

**Built with ❤️ for transparent governance in India**
