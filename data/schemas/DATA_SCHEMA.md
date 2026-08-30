# Data Schema Documentation

## Project Data Schema

All extracted infrastructure projects follow this standardized schema:

### Core Fields

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `project_id` | string | Unique project identifier | "MH-0001" |
| `project_name` | string | Full project name | "NH-48 Widening Project" |
| `state` | string | State name (lowercase) | "maharashtra" |
| `district` | string | District name | "Pune" |
| `category` | string | Project category | "Roads", "Railways", "Irrigation" |
| `sanctioned_cost` | float | Approved project budget (INR) | 150000000.0 (1.5 Cr) |
| `expenditure_to_date` | float | Amount spent so far (INR) | 120000000.0 (1.2 Cr) |
| `physical_progress_percent` | float | Completion percentage | 75.5 |
| `planned_completion_date` | string (ISO 8601) | Original completion date | "2025-12-31" |
| `implementing_agency` | string | Agency executing project | "PWD", "NHAI", "Railways" |
| `scraped_at` | string (ISO 8601) | Data extraction timestamp | "2026-01-22T23:00:00" |

### Calculated Fields (Added by Analytics)

| Field Name | Type | Description | Calculation |
|------------|------|-------------|-------------|
| `delay_days` | integer | Days delayed from planned completion | `current_date - planned_completion_date` |
| `cost_overrun_percent` | float | Budget overrun percentage | `((expenditure - sanctioned) / sanctioned) * 100` |
| `latitude` | float | GPS latitude (for mapping) | Geocoded from district |
| `longitude` | float | GPS longitude (for mapping) | Geocoded from district |

### Data Quality Fields

| Field Name | Type | Description |
|------------|------|-------------|
| `source_file` | string | Origin file/portal if applicable |
| `quality_score` | float | Reliability score (0-100) |

## Data Formats

### CSV Export Format
```csv
project_id,project_name,state,district,category,sanctioned_cost,expenditure_to_date,physical_progress_percent,planned_completion_date,implementing_agency,delay_days,cost_overrun_percent
MH-0001,NH-48 Widening,maharashtra,Pune,Roads,150000000,120000000,75.5,2025-12-31,NHAI,45,15.2
```

### JSON Export Format
```json
{
  "project_id": "MH-0001",
  "project_name": "NH-48 Widening",
  "state": "maharashtra",
  "district": "Pune",
  "category": "Roads",
  "sanctioned_cost": 150000000,
  "expenditure_to_date": 120000000,
  "physical_progress_percent": 75.5,
  "planned_completion_date": "2025-12-31",
  "implementing_agency": "NHAI",
  "delay_days": 45,
  "cost_overrun_percent": 15.2,
  "latitude": 18.5204,
  "longitude": 73.8567
}
```

## Data Quality Indicators

### Completeness Score
- **100%**: All required fields present and valid
- **75-99%**: Minor missing fields (e.g., implementing_agency)
- **50-74%**: Moderate data gaps
- **<50%**: Unreliable, manual verification needed

### Anomaly Types

| Anomaly Code | Severity | Description |
|--------------|----------|-------------|
| `OVER_100_PROGRESS` | HIGH | Physical progress exceeds 100% |
| `COST_OVERRUN` | MEDIUM | Expenditure exceeds sanctioned cost by >20% |
| `INVALID_COST` | HIGH | Zero, negative, or missing cost values |
| `PROGRESS_EXPENDITURE_MISMATCH` | MEDIUM | High progress but low expenditure (or vice versa) |

## Usage Notes

- All currency values are in Indian Rupees (INR)
- Dates follow ISO 8601 format (YYYY-MM-DD)
- GPS coordinates use WGS84 datum
- Categories are standardized across states
- District names follow Census 2011 conventions
