# Railway Deployment Guide

## Architecture

The PAIMANA Intelligence Platform is a **monolithic Flask application** - the frontend (HTML/JavaScript) is embedded directly in the Flask backend. This means:

- Frontend and backend are served from the same origin
- No separate frontend framework (React, Vue, etc.)
- API calls use relative paths (e.g., `/api/analytics`)
- No need to configure separate API URLs

## Deployment Steps

### 1. Railway Configuration

Create a new Railway project and connect your GitHub repository.

### 2. Environment Variables

Set the following environment variable in Railway:

**Optional - Only needed if deploying a separate frontend:**
```
FRONTEND_URL=https://your-frontend-domain.com
```

If not set, CORS allows all origins (suitable for monolithic deployment).

### 3. Build Settings

Railway will automatically detect the Python project. Ensure:

- **Python Version**: 3.8 or later
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn src.dashboard.app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

The `Procfile` in the repository already specifies the correct start command.

### 4. Dependencies

The `requirements.txt` includes:
- scikit-learn==1.7.2 (required for ML model compatibility)
- Flask, Flask-CORS
- pandas, numpy
- plotly (for charts)
- gunicorn (production server)

### 5. Data Files

The application requires:
- `data/processed/df_reference.csv` (2,246 project records)
- `data/project_intelligence_models.pkl` (ML models)

These files are included in the repository and will be deployed automatically.

## Verification

After deployment, test the following endpoints:

1. **Health Check**: `https://your-app.railway.app/health`
   - Should return: `{"service": "paimana-intelligence-platform", "status": "healthy"}`

2. **Analytics**: `https://your-app.railway.app/api/analytics`
   - Should return JSON with 2,246 projects

3. **Map Data**: `https://your-app.railway.app/api/map-data`
   - Should return geographic coordinates

4. **Dashboard**: `https://your-app.railway.app/`
   - Should load the full dashboard with statistics, charts, and tables

## Troubleshooting

### Issue: Blank dashboard or "Loading stats..."

**Check browser console for errors:**
1. Open browser DevTools (F12)
2. Check Console tab for JavaScript errors
3. Check Network tab for failed API requests

**Common causes:**
- Data loading failed (check server logs)
- API endpoints returning errors (check Network tab)
- CORS issues (check Console for CORS errors)

### Issue: API endpoints return 500 errors

**Check Railway logs:**
1. Go to Railway dashboard
2. Select your project
3. View deployment logs
4. Look for Python traceback errors

**Common causes:**
- Missing dependencies (check requirements.txt)
- Data file not found (check file paths)
- ML model incompatibility (scikit-learn version)

### Issue: CORS errors

**Solution:**
- Set `FRONTEND_URL` environment variable to your frontend domain
- Or leave unset to allow all origins (development)

### Issue: Map not displaying

**Check:**
- Geographic coordinates available in dataset
- Map data API returns valid coordinates
- Plotly library loads correctly

## Monitoring

Monitor your Railway deployment:
- Check deployment logs regularly
- Monitor response times
- Track error rates
- Set up Railway alerts for failures

## Scaling

For higher traffic:
- Increase Gunicorn workers: `--workers 2` or `--workers 4`
- Add Railway scaling (horizontal scaling)
- Consider caching for analytics data

## Security

- Railway provides HTTPS automatically
- No sensitive data in environment variables
- CORS configured appropriately
- Rate limiting can be added if needed
