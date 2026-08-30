# Netlify Deployment Guide

This guide will help you deploy the PAIMANA Intelligence Platform to Netlify without errors.

## Prerequisites

- Netlify account (free tier works)
- Git repository (GitHub, GitLab, or Bitbucket)
- Netlify CLI (optional, for local testing)

## Deployment Steps

### Option 1: Deploy via Netlify Dashboard (Recommended)

1. **Push your code to a Git repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Create a new site on Netlify**
   - Go to https://app.netlify.com
   - Click "Add new site" → "Import an existing project"
   - Select your Git provider and repository
   - Configure build settings:
     - **Build command**: `pip install -r requirements.txt`
     - **Publish directory**: Leave empty (we're using functions)
   - Click "Deploy site"

3. **Configure environment variables** (if needed)
   - Go to Site settings → Environment variables
   - Add any required environment variables

4. **Verify deployment**
   - Your site will be available at `https://your-site-name.netlify.app`
   - The API will be accessible at `https://your-site-name.netlify.app/.netlify/functions/api`

### Option 2: Deploy via Netlify CLI

1. **Install Netlify CLI**
   ```bash
   npm install -g netlify-cli
   ```

2. **Login to Netlify**
   ```bash
   netlify login
   ```

3. **Initialize Netlify**
   ```bash
   netlify init
   ```
   - Follow the prompts to create a new site or link to an existing one

4. **Deploy**
   ```bash
   netlify deploy --prod
   ```

## Troubleshooting Common Issues

### Issue: Build fails with Python version error

**Solution**: Ensure your `.python-version` file specifies a supported version (3.8, 3.9, or 3.10).

### Issue: Function timeout errors

**Solution**: The Flask app generates data on first load which may take time. Add this to your `netlify.toml`:
```toml
[functions]
  timeout = 30
```

### Issue: Missing dependencies

**Solution**: Ensure all dependencies are in `requirements.txt` and versions are compatible with Netlify's Python runtime.

### Issue: Import errors in serverless function

**Solution**: The function handler automatically adds the project root to the Python path. If you still have issues, check that the file structure matches:
```
netlify/
  functions/
    api.py
src/
  dashboard/
    app.py
```

## Testing Locally

To test your Netlify functions locally before deploying:

1. **Install Netlify CLI** (if not already installed)
   ```bash
   npm install -g netlify-cli
   ```

2. **Start local development server**
   ```bash
   netlify dev
   ```

3. **Access your app**
   - Dashboard: http://localhost:8888
   - API: http://localhost:8888/.netlify/functions/api

## Post-Deployment Checklist

- [ ] Visit your site URL and verify the dashboard loads
- [ ] Test API endpoints (`/api/projects`, `/api/analytics`, etc.)
- [ ] Check that the map visualization renders correctly
- [ ] Verify ML predictions are working
- [ ] Test CSV/Excel download functionality

## Monitoring

- Check Netlify dashboard for function logs
- Monitor build logs for any errors
- Set up error notifications in Netlify settings

## Cost

- Netlify Functions free tier: 125,000 requests/month
- Build minutes: 300 minutes/month (free tier)
- This should be sufficient for development and moderate usage

## Support

If you encounter issues:
1. Check Netlify's documentation: https://docs.netlify.com
2. Review build logs in Netlify dashboard
3. Ensure all dependencies are compatible with Netlify's Python runtime
