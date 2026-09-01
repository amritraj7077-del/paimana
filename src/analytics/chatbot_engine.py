"""
PAIMANA Intelligence Chatbot Engine
NLP-aware intent classification, fuzzy project matching, conversation context,
dynamic filtering, ML prediction queries — all grounded in real PAIMANA data.
"""

import os
import re
import json
import urllib.request
import pandas as pd
import numpy as np
from difflib import SequenceMatcher


class PAIMANAChatbotEngine:
    """Dataset-grounded PAIMANA project intelligence assistant."""

    # ── Intent keywords (scored, not exact-match) ──────────────────────────
    INTENT_KEYWORDS = {
        "project_lookup": [
            "tell me about", "details of", "info on", "information about",
            "show project", "project details", "what is project", "find project",
            "look up", "lookup", "describe project",
        ],
        "project_search": [
            "projects in", "show me projects", "list projects", "search projects",
            "which projects", "find projects", "projects from", "projects near",
        ],
        "delay_analysis": [
            "delay", "delayed", "most delayed", "highest delay", "late",
            "behind schedule", "overdue", "slow", "sluggish", "stalled",
            "why delayed", "delay reason", "average delay", "total delay",
        ],
        "cost_analysis": [
            "cost", "overrun", "over budget", "budget", "expenditure",
            "spending", "spent", "sanctioned", "revised cost", "actual cost",
            "planned cost", "cost overrun", "money", "expensive", "cheapest",
        ],
        "progress_analysis": [
            "progress", "completion", "completed", "ongoing", "running",
            "under construction", "how far", "percent complete", "almost done",
            "nearly complete", "below 50", "above 80", "stalled progress",
        ],
        "category_analysis": [
            "category", "sector", "compare categories", "compare sectors",
            "which category", "which sector", "best category", "worst category",
            "road", "highway", "railway", "power", "bridge", "water",
            "telecom", "petroleum", "coal", "health",
        ],
        "location_analysis": [
            "state", "district", "location", "located", "where",
            "geographic", "geography", "region", "map", "maharashtra",
            "karnataka", "tamil nadu", "gujarat", "rajasthan", "uttar pradesh",
            "bihar", "west bengal", "odisha", "jharkhand", "delhi",
        ],
        "risk_analysis": [
            "risk", "risky", "high risk", "critical", "danger",
            "attention", "priority", "urgent", "problematic", "trouble",
            "worst", "need attention", "management", "prioritize",
        ],
        "ml_prediction": [
            "predict", "prediction", "predicted", "forecast", "estimated",
            "ml", "machine learning", "model", "ai prediction",
            "will it be delayed", "expected delay", "future delay",
        ],
        "portfolio_summary": [
            "summary", "overview", "portfolio", "dashboard", "overall",
            "total", "how many", "count", "statistics", "stats", "aggregate",
            "trend", "biggest problem", "top 10", "top 5",
        ],
        "comparison": [
            "compare", "comparison", "versus", "vs", "difference between",
            "better", "worse",
        ],
        "ranking": [
            "top", "bottom", "highest", "lowest", "biggest", "smallest",
            "most", "least", "rank", "best", "worst",
        ],
        "filtering": [
            "greater than", "less than", "more than", "above", "below",
            "between", "filter", "only", "at least",
        ],
        "help": [
            "help", "what can you", "how to use", "capabilities", "features",
        ],
        "platform_info": [
            "what is paimana", "about paimana", "platform", "how does",
            "algorithm", "methodology",
        ],
    }

    # ── State name normalization ───────────────────────────────────────────
    STATE_ALIASES = {
        "ap": "Andhra Pradesh", "ts": "Telangana", "tn": "Tamil Nadu",
        "ka": "Karnataka", "mh": "Maharashtra", "gj": "Gujarat",
        "rj": "Rajasthan", "up": "Uttar Pradesh", "mp": "Madhya Pradesh",
        "wb": "West Bengal", "br": "Bihar", "jh": "Jharkhand",
        "od": "Odisha", "or": "Odisha", "pb": "Punjab", "hr": "Haryana",
        "dl": "Delhi", "hp": "Himachal Pradesh", "uk": "Uttarakhand",
        "goa": "Goa", "jk": "Jammu & Kashmir",
    }

    def __init__(self):
        self._last_project_code = None  # For follow-up context

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════

    def process_chat(self, message: str, history: list, df: pd.DataFrame,
                     analytics: dict = None, predictions_df: pd.DataFrame = None) -> dict:
        """Main entry point.  Returns {"answer": str, "sources": list}."""
        if not message or not isinstance(message, str):
            return self._reply("I couldn't understand the question. Could you rephrase?")

        msg = message.strip()
        hist = self._normalize_history(history)

        # Resolve follow-up pronouns ("it", "its", "this project", "that one")
        msg = self._resolve_pronouns(msg, hist)

        # Classify intent
        intent, score = self._classify_intent(msg)

        # Build answer
        try:
            answer = self._route(intent, msg, df, analytics, predictions_df)
        except Exception as e:
            answer = f"I encountered an error while processing your question: {e}"

        return self._reply(answer)

    # ═══════════════════════════════════════════════════════════════════════
    # INTENT CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════════════

    def _classify_intent(self, msg: str) -> tuple:
        """Score each intent category; return (best_intent, score)."""
        lower = msg.lower()
        scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            s = sum(1 for kw in keywords if kw in lower)
            scores[intent] = s

        # Check for specific project lookup (P123456 or project name)
        if re.search(r'\bP?\d{5,7}\b', msg, re.IGNORECASE):
            scores["project_lookup"] = max(scores.get("project_lookup", 0), 5)

        best = max(scores, key=scores.get)
        return (best, scores[best]) if scores[best] > 0 else ("portfolio_summary", 0)

    # ═══════════════════════════════════════════════════════════════════════
    # ROUTING
    # ═══════════════════════════════════════════════════════════════════════

    def _route(self, intent, msg, df, analytics, predictions_df):
        lower = msg.lower()

        # Priority: if a project code is explicitly mentioned, do lookup
        code = self._extract_project_code(msg)
        if code is not None:
            return self._project_lookup(code, msg, df, predictions_df)

        # Named project fuzzy match
        proj_match = self._fuzzy_find_project(msg, df)
        if proj_match is not None and intent not in ("portfolio_summary", "help", "platform_info"):
            return self._project_lookup(proj_match, msg, df, predictions_df)

        handlers = {
            "project_lookup": lambda: self._handle_project_lookup(msg, df, predictions_df),
            "project_search": lambda: self._handle_project_search(msg, df),
            "delay_analysis": lambda: self._handle_delay(msg, df, predictions_df),
            "cost_analysis": lambda: self._handle_cost(msg, df),
            "progress_analysis": lambda: self._handle_progress(msg, df),
            "category_analysis": lambda: self._handle_category(msg, df),
            "location_analysis": lambda: self._handle_location(msg, df),
            "risk_analysis": lambda: self._handle_risk(msg, df, predictions_df),
            "ml_prediction": lambda: self._handle_ml(msg, df, predictions_df),
            "portfolio_summary": lambda: self._handle_summary(msg, df, predictions_df),
            "comparison": lambda: self._handle_comparison(msg, df, predictions_df),
            "ranking": lambda: self._handle_ranking(msg, df, predictions_df),
            "filtering": lambda: self._handle_filtering(msg, df),
            "help": lambda: self._handle_help(),
            "platform_info": lambda: self._handle_platform_info(),
        }
        handler = handlers.get(intent, lambda: self._handle_summary(msg, df, predictions_df))
        return handler()

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Project Lookup
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_project_lookup(self, msg, df, preds):
        code = self._extract_project_code(msg)
        if code is None:
            code = self._fuzzy_find_project(msg, df)
        if code is None:
            return "Could you specify the project ID or name? For example: *'Tell me about project P108841'*."
        return self._project_lookup(code, msg, df, preds)

    def _project_lookup(self, code, msg, df, preds):
        row = self._find_project(code, df)
        if row is None:
            return f"I couldn't find project **{code}** in the PAIMANA dataset."

        self._last_project_code = str(row.get('Project Code', ''))
        lower = msg.lower()

        # Detect sub-topic
        if any(w in lower for w in ["cost", "budget", "expenditure", "spent", "overrun", "money"]):
            return self._project_cost_detail(row)
        if any(w in lower for w in ["delay", "late", "behind", "overdue"]):
            return self._project_delay_detail(row, preds)
        if any(w in lower for w in ["predict", "prediction", "ml", "forecast", "will it"]):
            return self._project_prediction_detail(row, preds)
        if any(w in lower for w in ["progress", "completion", "complete", "status"]):
            return self._project_progress_detail(row)
        if any(w in lower for w in ["location", "where", "located", "state", "district"]):
            return self._project_location_detail(row)

        # Full detail card
        return self._project_full_detail(row, preds)

    def _project_full_detail(self, r, preds):
        code = r.get('Project Code', 'N/A')
        name = r.get('Project Name', 'Unknown')
        state = r.get('State', 'N/A')
        sector = r.get('Sector', 'N/A')
        ministry = r.get('Ministry', 'N/A')
        progress = r.get('Physical Progress (%)', 0)
        orig_cost = r.get('Original Cost (Rs. Crore)', 0)
        rev_cost = r.get('Revised Cost (Rs. Crore)', 0)
        cum_exp = r.get('Cumulative Expenditure (Rs. Crore)', 0)
        delay_m = r.get('Actual_Delay_Months', 0)
        delay_d = round(max(delay_m, 0) * 30)
        risk = r.get('Risk_Level', 'N/A')
        overrun = r.get('Cost_Overrun_Ratio', 0) * 100

        a = f"### 📌 Project P{code}\n\n"
        a += f"**{name}**\n\n"
        a += f"| Field | Value |\n|---|---|\n"
        a += f"| State | {state} |\n"
        a += f"| Sector | {sector} |\n"
        a += f"| Ministry | {ministry} |\n"
        a += f"| Physical Progress | {progress:.1f}% |\n"
        a += f"| Original Cost | ₹{orig_cost:,.2f} Cr |\n"
        if rev_cost and rev_cost != orig_cost:
            a += f"| Revised Cost | ₹{rev_cost:,.2f} Cr |\n"
        a += f"| Expenditure | ₹{cum_exp:,.2f} Cr |\n"
        a += f"| Cost Overrun | {overrun:+.1f}% |\n"
        a += f"| Actual Delay | {delay_d} days ({delay_m:.1f} months) |\n"
        a += f"| Risk Level | `{risk}` |\n"

        # Add ML prediction if available
        pred_row = self._find_prediction(code, preds)
        if pred_row is not None:
            ml_delay = pred_row.get('ML_Predicted_Delay_Days', 0)
            ml_risk = pred_row.get('ML_Risk_Level', 'N/A')
            ml_conf = pred_row.get('ML_Risk_Confidence_%', 0)
            a += f"\n**ML Prediction:** {int(ml_delay)} days predicted delay | Risk: `{ml_risk}` ({ml_conf:.0f}% confidence)\n"

        return a

    def _project_cost_detail(self, r):
        code = r.get('Project Code', 'N/A')
        name = r.get('Project Name', 'Unknown')
        orig = r.get('Original Cost (Rs. Crore)', 0)
        rev = r.get('Revised Cost (Rs. Crore)', 0)
        exp = r.get('Cumulative Expenditure (Rs. Crore)', 0)
        overrun = r.get('Cost_Overrun_Ratio', 0) * 100
        util = r.get('Budget_Utilization', 0) * 100 if r.get('Budget_Utilization') else (exp / orig * 100 if orig > 0 else 0)

        a = f"**Cost details for P{code}** ({name}):\n\n"
        a += f"- **Original Cost:** ₹{orig:,.2f} Crore\n"
        if rev and rev != orig:
            a += f"- **Revised Cost:** ₹{rev:,.2f} Crore\n"
        a += f"- **Cumulative Expenditure:** ₹{exp:,.2f} Crore\n"
        a += f"- **Cost Overrun:** {overrun:+.1f}%\n"
        a += f"- **Budget Utilization:** {util:.1f}%\n"
        return a

    def _project_delay_detail(self, r, preds):
        code = r.get('Project Code', 'N/A')
        name = r.get('Project Name', 'Unknown')
        delay_m = r.get('Actual_Delay_Months', 0)
        delay_d = round(max(delay_m, 0) * 30)
        progress = r.get('Physical Progress (%)', 0)

        a = f"**Delay details for P{code}** ({name}):\n\n"
        if delay_m <= 0:
            a += f"This project is **on schedule** (no delay recorded).\n"
        else:
            a += f"- **Current Delay:** {delay_d} days ({delay_m:.1f} months)\n"
        a += f"- **Physical Progress:** {progress:.1f}%\n"

        pred_row = self._find_prediction(code, preds)
        if pred_row is not None:
            ml_delay = pred_row.get('ML_Predicted_Delay_Days', 0)
            a += f"- **ML Predicted Delay:** {int(ml_delay)} days\n"
            a += f"- **ML Risk Level:** `{pred_row.get('ML_Risk_Level', 'N/A')}`\n"
        return a

    def _project_prediction_detail(self, r, preds):
        code = r.get('Project Code', 'N/A')
        name = r.get('Project Name', 'Unknown')
        pred_row = self._find_prediction(code, preds)
        if pred_row is None:
            return f"No ML prediction data available for project P{code}."

        ml_delay = pred_row.get('ML_Predicted_Delay_Days', 0)
        ml_cost = pred_row.get('ML_Predicted_Cost_Overrun_%', 0)
        ml_risk = pred_row.get('ML_Risk_Level', 'N/A')
        ml_conf = pred_row.get('ML_Risk_Confidence_%', 0)
        actual_d = round(max(r.get('Actual_Delay_Months', 0), 0) * 30)

        a = f"**ML Prediction for P{code}** ({name}):\n\n"
        a += f"| Metric | Value |\n|---|---|\n"
        a += f"| Current Actual Delay | {actual_d} days |\n"
        a += f"| **ML Predicted Delay** | **{int(ml_delay)} days** |\n"
        a += f"| ML Predicted Cost Overrun | {ml_cost:+.1f}% |\n"
        a += f"| ML Risk Level | `{ml_risk}` |\n"
        a += f"| Risk Confidence | {ml_conf:.0f}% |\n"
        return a

    def _project_progress_detail(self, r):
        code = r.get('Project Code', 'N/A')
        name = r.get('Project Name', 'Unknown')
        progress = r.get('Physical Progress (%)', 0)
        status = "Completed" if progress >= 100 else ("Ongoing" if progress > 0 else "Not Started")
        a = f"**Progress for P{code}** ({name}):\n\n"
        a += f"- **Physical Progress:** {progress:.1f}%\n"
        a += f"- **Status:** {status}\n"
        return a

    def _project_location_detail(self, r):
        code = r.get('Project Code', 'N/A')
        name = r.get('Project Name', 'Unknown')
        state = r.get('State', 'N/A')
        a = f"**Location of P{code}** ({name}):\n\n- **State/Region:** {state}\n"
        return a

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Project Search
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_project_search(self, msg, df):
        lower = msg.lower()
        total = len(df)

        # Search by state
        state = self._extract_state(msg, df)
        if state:
            matched = df[df['State'].str.lower() == state.lower()]
            if len(matched) == 0:
                return f"No projects found in **{state}**."
            delayed = matched[matched['Actual_Delay_Months'] > 0]
            a = f"There are **{len(matched)}** projects in **{state}** ({len(delayed)} delayed).\n\n"
            a += self._top_n_table(matched.sort_values('Actual_Delay_Months', ascending=False).head(5))
            return a

        # Search by sector
        sector = self._extract_sector(msg, df)
        if sector:
            matched = df[df['Sector'].str.lower() == sector.lower()]
            if len(matched) == 0:
                return f"No projects found in sector **{sector}**."
            a = f"There are **{len(matched)}** projects in the **{sector}** sector.\n\n"
            a += self._top_n_table(matched.sort_values('Actual_Delay_Months', ascending=False).head(5))
            return a

        # Check for status-based searches
        if any(w in lower for w in ["completed", "done", "finished"]):
            done = df[df['Physical Progress (%)'] >= 100]
            a = f"**{len(done)}** projects are completed (100% progress) out of {total}.\n\n"
            a += self._top_n_table(done.head(5))
            return a

        if any(w in lower for w in ["ongoing", "running", "active", "in progress", "under construction"]):
            ongoing = df[(df['Physical Progress (%)'] > 0) & (df['Physical Progress (%)'] < 100)]
            a = f"**{len(ongoing)}** projects are currently ongoing.\n\n"
            a += self._top_n_table(ongoing.sort_values('Physical Progress (%)').head(5))
            return a

        # Generic search: return count
        a = f"The PAIMANA dataset contains **{total:,}** projects across {df['State'].nunique()} states and {df['Sector'].nunique()} sectors.\n"
        a += "You can narrow your search by specifying a state, sector, or status. For example: *'Show projects in Maharashtra'*."
        return a

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Delay Analysis
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_delay(self, msg, df, preds):
        lower = msg.lower()
        total = len(df)
        delayed = df[df['Actual_Delay_Months'] > 0].copy()
        delay_count = len(delayed)

        # "most delayed" / "highest delay"
        if any(w in lower for w in ["most delayed", "highest delay", "maximum delay", "worst delay"]):
            top = delayed.sort_values('Actual_Delay_Months', ascending=False).head(5)
            a = f"**Top 5 Most Delayed Projects:**\n\n"
            for i, (_, r) in enumerate(top.iterrows(), 1):
                d = round(r['Actual_Delay_Months'] * 30)
                a += f"{i}. **P{r['Project Code']}** ({r['Project Name'][:50]}): **{d} days** delay | Progress: {r['Physical Progress (%)']:.1f}%\n"
            return a

        # "average delay"
        if "average" in lower or "mean" in lower:
            avg = delayed['Actual_Delay_Months'].mean() * 30
            median = delayed['Actual_Delay_Months'].median() * 30
            return f"The **average delay** across {delay_count:,} delayed projects is **{avg:.0f} days** (median: {median:.0f} days)."

        # "how many delayed"
        if any(w in lower for w in ["how many", "count", "total"]):
            pct = delay_count / total * 100
            return f"**{delay_count:,}** out of {total:,} projects are delayed ({pct:.1f}%)."

        # "delay greater than X"
        threshold = self._extract_number(msg)
        if threshold:
            filtered = delayed[delayed['Actual_Delay_Months'] * 30 > threshold]
            a = f"**{len(filtered)}** projects have delay greater than {int(threshold)} days.\n\n"
            if len(filtered) > 0:
                a += self._top_n_table(filtered.sort_values('Actual_Delay_Months', ascending=False).head(5))
            return a

        # "which category has the most delays"
        if any(w in lower for w in ["category", "sector"]):
            cat_delay = delayed.groupby('Sector').agg(
                count=('Actual_Delay_Months', 'size'),
                avg_delay=('Actual_Delay_Months', 'mean'),
            ).sort_values('count', ascending=False).head(5)
            a = "**Most Delayed Sectors:**\n\n"
            for sector, row in cat_delay.iterrows():
                a += f"- **{sector}**: {int(row['count'])} delayed projects (avg {row['avg_delay']*30:.0f} days)\n"
            return a

        # Default: delay overview
        pct = delay_count / total * 100
        a = f"**{delay_count:,}** projects are delayed ({pct:.1f}% of {total:,}).\n\n"
        a += "**Top 5 Most Delayed:**\n\n"
        top = delayed.sort_values('Actual_Delay_Months', ascending=False).head(5)
        for i, (_, r) in enumerate(top.iterrows(), 1):
            d = round(r['Actual_Delay_Months'] * 30)
            a += f"{i}. **P{r['Project Code']}** ({r['Project Name'][:50]}): {d} days\n"
        return a

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Cost Analysis
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_cost(self, msg, df):
        lower = msg.lower()
        overrun = df[df['Cost_Overrun_Ratio'] > 0].copy()

        if any(w in lower for w in ["biggest", "highest", "maximum", "worst", "most"]):
            top = overrun.sort_values('Cost_Overrun_Ratio', ascending=False).head(5)
            a = "**Top 5 Cost Overrun Projects:**\n\n"
            for i, (_, r) in enumerate(top.iterrows(), 1):
                pct = r['Cost_Overrun_Ratio'] * 100
                orig = r.get('Original Cost (Rs. Crore)', 0)
                rev = r.get('Revised Cost (Rs. Crore)', 0)
                a += f"{i}. **P{r['Project Code']}** ({r['Project Name'][:50]}): **+{pct:.1f}%** overrun (₹{orig:,.1f} → ₹{rev:,.1f} Cr)\n"
            return a

        if any(w in lower for w in ["how many", "count", "total"]):
            return f"**{len(overrun):,}** projects have cost overruns ({len(overrun)/len(df)*100:.1f}% of total)."

        if any(w in lower for w in ["how much", "spending", "total cost", "total expenditure"]):
            total_sanc = df['Original Cost (Rs. Crore)'].sum()
            total_exp = df['Cumulative Expenditure (Rs. Crore)'].sum()
            return f"**Total sanctioned cost:** ₹{total_sanc:,.2f} Crore\n**Total expenditure:** ₹{total_exp:,.2f} Crore"

        if any(w in lower for w in ["category", "sector"]):
            cat = overrun.groupby('Sector')['Cost_Overrun_Ratio'].agg(['count', 'mean']).sort_values('count', ascending=False).head(5)
            a = "**Cost Overruns by Sector:**\n\n"
            for sector, row in cat.iterrows():
                a += f"- **{sector}**: {int(row['count'])} projects (avg +{row['mean']*100:.1f}% overrun)\n"
            return a

        # Default
        a = f"**{len(overrun):,}** projects have cost overruns.\n\n"
        top = overrun.sort_values('Cost_Overrun_Ratio', ascending=False).head(5)
        a += "**Top 5:**\n"
        for i, (_, r) in enumerate(top.iterrows(), 1):
            pct = r['Cost_Overrun_Ratio'] * 100
            a += f"{i}. **P{r['Project Code']}** ({r['Project Name'][:45]}): +{pct:.1f}%\n"
        return a

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Progress Analysis
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_progress(self, msg, df):
        lower = msg.lower()
        avg = df['Physical Progress (%)'].mean()
        median = df['Physical Progress (%)'].median()

        if any(w in lower for w in ["average", "mean", "overall"]):
            return f"The **average physical progress** across all {len(df):,} projects is **{avg:.1f}%** (median: {median:.1f}%)."

        if any(w in lower for w in ["below 50", "under 50", "less than 50"]):
            low = df[df['Physical Progress (%)'] < 50]
            a = f"**{len(low)}** projects have progress below 50%.\n\n"
            a += self._top_n_table(low.sort_values('Physical Progress (%)').head(5))
            return a

        if any(w in lower for w in ["almost", "nearly", "above 80", "above 90", "almost complete"]):
            near = df[(df['Physical Progress (%)'] >= 80) & (df['Physical Progress (%)'] < 100)]
            a = f"**{len(near)}** projects are near completion (80–99% progress).\n\n"
            a += self._top_n_table(near.sort_values('Physical Progress (%)', ascending=False).head(5))
            return a

        if any(w in lower for w in ["completed", "done", "finished", "100"]):
            done = df[df['Physical Progress (%)'] >= 100]
            return f"**{len(done)}** projects are completed (100% progress)."

        if any(w in lower for w in ["slow", "stalled", "sluggish"]):
            slow = df[(df['Physical Progress (%)'] < 30) & (df['Actual_Delay_Months'] > 12)]
            a = f"**{len(slow)}** projects have slow progress (<30%) with significant delay (>12 months).\n\n"
            if len(slow) > 0:
                a += self._top_n_table(slow.sort_values('Physical Progress (%)').head(5))
            return a

        # Default
        completed = len(df[df['Physical Progress (%)'] >= 100])
        ongoing = len(df[(df['Physical Progress (%)'] > 0) & (df['Physical Progress (%)'] < 100)])
        not_started = len(df[df['Physical Progress (%)'] == 0])
        a = f"**Project Progress Summary:**\n\n"
        a += f"- Average Progress: **{avg:.1f}%**\n"
        a += f"- Completed (100%): **{completed}** projects\n"
        a += f"- Ongoing (1-99%): **{ongoing}** projects\n"
        a += f"- Not Started (0%): **{not_started}** projects\n"
        return a

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Category Analysis
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_category(self, msg, df):
        lower = msg.lower()
        cats = df.groupby('Sector').agg(
            count=('Project Code', 'size'),
            avg_progress=('Physical Progress (%)', 'mean'),
            avg_delay=('Actual_Delay_Months', 'mean'),
            delayed_count=('Actual_Delay_Months', lambda x: (x > 0).sum()),
            avg_overrun=('Cost_Overrun_Ratio', 'mean'),
        ).sort_values('count', ascending=False)

        if any(w in lower for w in ["best", "top performing", "least delay"]):
            best = cats.sort_values('avg_delay').head(5)
            a = "**Best Performing Sectors (by lowest avg delay):**\n\n"
            for sector, row in best.iterrows():
                a += f"- **{sector}**: Avg delay {row['avg_delay']*30:.0f} days | {int(row['count'])} projects | {row['avg_progress']:.1f}% avg progress\n"
            return a

        if any(w in lower for w in ["worst", "most delay", "most problem"]):
            worst = cats.sort_values('avg_delay', ascending=False).head(5)
            a = "**Worst Performing Sectors (by highest avg delay):**\n\n"
            for sector, row in worst.iterrows():
                a += f"- **{sector}**: Avg delay {row['avg_delay']*30:.0f} days | {int(row['delayed_count'])}/{int(row['count'])} delayed\n"
            return a

        if "cost" in lower or "overrun" in lower:
            by_cost = cats.sort_values('avg_overrun', ascending=False).head(5)
            a = "**Sectors by Cost Overrun:**\n\n"
            for sector, row in by_cost.iterrows():
                a += f"- **{sector}**: Avg overrun +{row['avg_overrun']*100:.1f}% | {int(row['count'])} projects\n"
            return a

        # Default: overview
        a = f"**Sector Overview** ({len(cats)} sectors):\n\n"
        for sector, row in cats.head(8).iterrows():
            a += f"- **{sector}**: {int(row['count'])} projects | {row['avg_progress']:.1f}% progress | {int(row['delayed_count'])} delayed\n"
        if len(cats) > 8:
            a += f"\n_...and {len(cats) - 8} more sectors._\n"
        return a

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Location Analysis
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_location(self, msg, df):
        state = self._extract_state(msg, df)
        if state:
            matched = df[df['State'].str.lower() == state.lower()]
            if len(matched) == 0:
                return f"No projects found in **{state}**."
            delayed = matched[matched['Actual_Delay_Months'] > 0]
            avg_p = matched['Physical Progress (%)'].mean()
            a = f"**{state}** has **{len(matched)}** projects:\n\n"
            a += f"- Delayed: {len(delayed)} ({len(delayed)/len(matched)*100:.1f}%)\n"
            a += f"- Avg Progress: {avg_p:.1f}%\n"
            a += f"- Sectors: {', '.join(matched['Sector'].unique()[:5])}\n\n"
            a += self._top_n_table(matched.sort_values('Actual_Delay_Months', ascending=False).head(5))
            return a

        # Geographic distribution overview
        states = df.groupby('State').agg(
            count=('Project Code', 'size'),
            delayed=('Actual_Delay_Months', lambda x: (x > 0).sum()),
        ).sort_values('count', ascending=False)
        a = "**Top States by Project Count:**\n\n"
        for state, row in states.head(10).iterrows():
            a += f"- **{state}**: {int(row['count'])} projects ({int(row['delayed'])} delayed)\n"
        return a

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Risk Analysis
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_risk(self, msg, df, preds):
        lower = msg.lower()

        if preds is not None and 'ML_Risk_Level' in preds.columns:
            risk_df = preds
        else:
            risk_df = df

        if 'ML_Risk_Level' not in risk_df.columns:
            risk_df = df
            if 'Risk_Level' in risk_df.columns:
                risk_df = risk_df.rename(columns={'Risk_Level': 'ML_Risk_Level'})

        high_risk = risk_df[risk_df['ML_Risk_Level'].isin(['HIGH', 'MEDIUM-HIGH'])]

        if any(w in lower for w in ["top", "worst", "highest", "most"]):
            n = 10
        else:
            n = 5

        if any(w in lower for w in ["need attention", "prioritize", "priority", "urgent", "management"]):
            # Management view: high risk + high delay + high cost overrun
            priority = df.copy()
            priority['priority_score'] = (
                priority['Actual_Delay_Months'].clip(0) / 120 * 40 +
                priority['Cost_Overrun_Ratio'].clip(0) * 30 +
                (100 - priority['Physical Progress (%)']) / 100 * 30
            )
            top = priority.sort_values('priority_score', ascending=False).head(n)
            a = f"**Top {n} Projects Requiring Management Attention:**\n\n"
            for i, (_, r) in enumerate(top.iterrows(), 1):
                d = round(max(r['Actual_Delay_Months'], 0) * 30)
                a += f"{i}. **P{r['Project Code']}** ({r['Project Name'][:45]}): Delay {d}d | Progress {r['Physical Progress (%)']:.1f}% | Overrun +{r['Cost_Overrun_Ratio']*100:.1f}%\n"
            return a

        a = f"**{len(high_risk)}** projects are classified as HIGH or MEDIUM-HIGH risk.\n\n"
        top = high_risk.sort_values('Actual_Delay_Months', ascending=False).head(n)
        if len(top) > 0:
            a += f"**Top {min(n, len(top))} High-Risk Projects:**\n\n"
            for i, (_, r) in enumerate(top.iterrows(), 1):
                d = round(max(r.get('Actual_Delay_Months', 0), 0) * 30)
                a += f"{i}. **P{r['Project Code']}** ({r.get('Project Name', 'Unknown')[:45]}): Risk `{r.get('ML_Risk_Level', 'N/A')}` | Delay {d}d\n"
        return a

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: ML Prediction
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_ml(self, msg, df, preds):
        if preds is None or 'ML_Predicted_Delay_Days' not in preds.columns:
            return "ML predictions are not currently available."

        lower = msg.lower()

        # Specific project prediction
        code = self._extract_project_code(msg)
        if code:
            row = self._find_project(code, df)
            if row is not None:
                return self._project_prediction_detail(row, preds)

        # "highest predicted delay"
        if any(w in lower for w in ["highest", "most", "worst", "top"]):
            top = preds.sort_values('ML_Predicted_Delay_Days', ascending=False).head(10)
            a = "**Top 10 Projects by ML Predicted Delay:**\n\n"
            for i, (_, r) in enumerate(top.iterrows(), 1):
                ml_d = int(r['ML_Predicted_Delay_Days'])
                actual_d = round(max(r.get('Actual_Delay_Months', 0), 0) * 30)
                a += f"{i}. **P{r['Project Code']}** ({r['Project Name'][:45]}): **ML: {ml_d}d** | Actual: {actual_d}d | Risk: `{r.get('ML_Risk_Level', 'N/A')}`\n"
            return a

        # "compare predicted vs actual"
        if any(w in lower for w in ["compare", "vs", "versus", "actual"]):
            sample = preds.sample(min(10, len(preds)), random_state=42)
            a = "**Predicted vs Actual Delay (sample):**\n\n| Project | Actual Delay | ML Predicted | Risk |\n|---|---|---|---|\n"
            for _, r in sample.iterrows():
                actual = round(max(r.get('Actual_Delay_Months', 0), 0) * 30)
                pred = int(r['ML_Predicted_Delay_Days'])
                a += f"| P{r['Project Code']} | {actual}d | {pred}d | `{r.get('ML_Risk_Level', 'N/A')}` |\n"
            return a

        # "high risk" projects from ML
        if any(w in lower for w in ["high risk", "risky"]):
            high = preds[preds['ML_Risk_Level'].isin(['HIGH', 'MEDIUM-HIGH'])]
            a = f"**{len(high)}** projects are predicted as HIGH or MEDIUM-HIGH risk by the ML model.\n\n"
            top = high.sort_values('ML_Predicted_Delay_Days', ascending=False).head(5)
            for i, (_, r) in enumerate(top.iterrows(), 1):
                a += f"{i}. **P{r['Project Code']}** ({r['Project Name'][:45]}): ML Delay {int(r['ML_Predicted_Delay_Days'])}d | Risk `{r['ML_Risk_Level']}`\n"
            return a

        # Default: ML summary
        avg = preds['ML_Predicted_Delay_Days'].mean()
        risk_dist = preds['ML_Risk_Level'].value_counts().to_dict()
        a = f"**ML Prediction Summary:**\n\n"
        a += f"- Average predicted delay: **{avg:.0f} days**\n"
        a += f"- Risk distribution: {', '.join(f'{k}: {v}' for k, v in sorted(risk_dist.items()))}\n\n"
        a += "Ask about specific projects or try: *'Show highest predicted delays'* or *'Compare predicted vs actual delay'*."
        return a

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Portfolio Summary
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_summary(self, msg, df, preds):
        lower = msg.lower()
        total = len(df)
        delayed = len(df[df['Actual_Delay_Months'] > 0])
        overruns = len(df[df['Cost_Overrun_Ratio'] > 0])
        avg_progress = df['Physical Progress (%)'].mean()
        completed = len(df[df['Physical Progress (%)'] >= 100])
        total_cost = df['Original Cost (Rs. Crore)'].sum()
        total_exp = df['Cumulative Expenditure (Rs. Crore)'].sum()

        if any(w in lower for w in ["how many", "count", "total projects"]):
            return f"There are **{total:,}** projects in the PAIMANA dataset across {df['State'].nunique()} states and {df['Sector'].nunique()} sectors."

        if any(w in lower for w in ["biggest problem", "major issue", "key concern"]):
            a = "**Top Concerns in PAIMANA Portfolio:**\n\n"
            a += f"1. **{delayed:,}** projects ({delayed/total*100:.1f}%) are delayed\n"
            a += f"2. **{overruns:,}** projects ({overruns/total*100:.1f}%) have cost overruns\n"
            slow = len(df[(df['Physical Progress (%)'] < 30) & (df['Actual_Delay_Months'] > 12)])
            a += f"3. **{slow}** projects have critically slow progress (<30%) with >12-month delay\n"
            if preds is not None and 'ML_Risk_Level' in preds.columns:
                high_risk = len(preds[preds['ML_Risk_Level'].isin(['HIGH', 'MEDIUM-HIGH'])])
                a += f"4. **{high_risk}** projects classified as HIGH/MEDIUM-HIGH risk by ML model\n"
            return a

        if any(w in lower for w in ["trend", "pattern"]):
            by_year = df.groupby('Approval_Year').agg(
                count=('Project Code', 'size'),
                avg_delay=('Actual_Delay_Months', 'mean'),
            ).sort_index()
            a = "**Project Trends by Approval Year:**\n\n"
            for year, row in by_year.tail(8).iterrows():
                a += f"- **{int(year)}**: {int(row['count'])} projects, avg delay {row['avg_delay']*30:.0f} days\n"
            return a

        # Default: portfolio overview
        a = f"**PAIMANA Portfolio Summary:**\n\n"
        a += f"- **Total Projects:** {total:,}\n"
        a += f"- **Delayed:** {delayed:,} ({delayed/total*100:.1f}%)\n"
        a += f"- **On-Time:** {total - delayed:,}\n"
        a += f"- **Completed:** {completed:,}\n"
        a += f"- **Cost Overruns:** {overruns:,} ({overruns/total*100:.1f}%)\n"
        a += f"- **Avg Progress:** {avg_progress:.1f}%\n"
        a += f"- **Total Sanctioned Cost:** ₹{total_cost:,.0f} Crore\n"
        a += f"- **Total Expenditure:** ₹{total_exp:,.0f} Crore\n"

        if preds is not None and 'ML_Risk_Level' in preds.columns:
            risk_dist = preds['ML_Risk_Level'].value_counts().to_dict()
            a += f"\n**ML Risk Distribution:** {', '.join(f'{k}: {v}' for k, v in sorted(risk_dist.items()))}\n"

        return a

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Comparison
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_comparison(self, msg, df, preds):
        # Try to find two project codes
        codes = re.findall(r'P?(\d{5,7})', msg, re.IGNORECASE)
        if len(codes) >= 2:
            rows = [self._find_project(c, df) for c in codes[:2]]
            if all(r is not None for r in rows):
                a = "**Project Comparison:**\n\n"
                a += "| Metric | P{} | P{} |\n|---|---|---|\n".format(codes[0], codes[1])
                for field, label in [('Physical Progress (%)', 'Progress'), ('Actual_Delay_Months', 'Delay (months)'),
                                      ('Cost_Overrun_Ratio', 'Cost Overrun'), ('Risk_Level', 'Risk')]:
                    v1 = rows[0].get(field, 'N/A')
                    v2 = rows[1].get(field, 'N/A')
                    if isinstance(v1, float):
                        v1 = f"{v1:.1f}"
                        v2 = f"{v2:.1f}"
                    a += f"| {label} | {v1} | {v2} |\n"
                return a

        return self._handle_category(msg, df)

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Ranking
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_ranking(self, msg, df, preds):
        lower = msg.lower()
        n = 10 if "10" in msg else 5

        if any(w in lower for w in ["delay", "late", "behind"]):
            return self._handle_delay(msg, df, preds)
        if any(w in lower for w in ["cost", "overrun", "budget", "expensive"]):
            return self._handle_cost(msg, df)
        if any(w in lower for w in ["risk", "risky", "danger"]):
            return self._handle_risk(msg, df, preds)
        if any(w in lower for w in ["progress", "slow", "fast"]):
            return self._handle_progress(msg, df)
        if any(w in lower for w in ["predict", "ml", "forecast"]):
            return self._handle_ml(msg, df, preds)

        # Default: by delay
        return self._handle_delay("top " + str(n) + " most delayed", df, preds)

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Filtering
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_filtering(self, msg, df):
        lower = msg.lower()
        threshold = self._extract_number(msg)
        if threshold is None:
            threshold = 100

        if any(w in lower for w in ["delay", "days"]):
            if any(w in lower for w in ["greater", "more", "above", "over"]):
                filtered = df[df['Actual_Delay_Months'] * 30 > threshold]
            else:
                filtered = df[df['Actual_Delay_Months'] * 30 < threshold]
            a = f"**{len(filtered)}** projects match the delay filter.\n\n"
            a += self._top_n_table(filtered.sort_values('Actual_Delay_Months', ascending=False).head(5))
            return a

        if any(w in lower for w in ["progress", "percent", "%"]):
            if any(w in lower for w in ["below", "less", "under"]):
                filtered = df[df['Physical Progress (%)'] < threshold]
            else:
                filtered = df[df['Physical Progress (%)'] > threshold]
            a = f"**{len(filtered)}** projects match the progress filter.\n\n"
            a += self._top_n_table(filtered.sort_values('Physical Progress (%)').head(5))
            return a

        return f"Please specify what to filter. Example: *'Show projects with delay greater than 200 days'*."

    # ═══════════════════════════════════════════════════════════════════════
    # HANDLER: Help & Platform Info
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_help(self):
        return """I'm the **PAIMANA Intelligence Assistant**. I can answer questions about the project portfolio using real data. Here are some examples:

**📊 Project Info:** *"Tell me about project P108841"* or *"Show projects in Maharashtra"*
**⏱ Delays:** *"Which project has the highest delay?"* or *"How many projects are delayed?"*
**💰 Cost:** *"Show cost overruns"* or *"Which project is most over budget?"*
**📈 Progress:** *"What is the average progress?"* or *"Show projects below 50%"*
**🏷 Categories:** *"Compare sectors"* or *"Which sector has the most delays?"*
**📍 Location:** *"Projects in Gujarat"* or *"Geographic distribution"*
**🤖 ML Predictions:** *"Show predicted delays"* or *"Which projects are high risk?"*
**📋 Summary:** *"Portfolio summary"* or *"What are the biggest problems?"*

You can also ask follow-up questions — I'll remember the context!"""

    def _handle_platform_info(self):
        return ("**PAIMANA** (Project Analytics & Infrastructure Management Network Application) "
                "is an intelligence platform that combines real-time project data with Machine Learning "
                "to monitor infrastructure project delays, cost overruns, and risk levels. "
                "The ML pipeline uses trained Gradient Boosting and Random Forest models on 10 features "
                "(Sector, State, Ministry, Original Cost, Approval Year, Physical Progress, Expenditure Ratio, "
                "Cost Per Progress, Large Project flag, Budget Utilization) to predict delay, cost overrun, and risk tier.")

    # ═══════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════════

    def _reply(self, answer, sources=None):
        return {"answer": answer, "sources": sources or ["PAIMANA Dataset"]}

    def _normalize_history(self, history):
        if not history or not isinstance(history, list):
            return []
        result = []
        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                result.append({"role": role, "content": content})
        return result

    def _resolve_pronouns(self, msg, history):
        """Replace pronouns like 'it', 'its', 'this project' with last project code."""
        lower = msg.lower()
        has_pronoun = any(p in lower for p in ["its ", "it's ", " it ", "this project", "that project", "that one", "the same"])
        if not has_pronoun:
            return msg

        # Try to find project code from history
        last_code = self._last_project_code
        if not last_code:
            for item in reversed(history):
                code_match = re.search(r'P?(\d{5,7})', item.get("content", ""))
                if code_match:
                    last_code = code_match.group(1)
                    break

        if last_code:
            self._last_project_code = last_code
            return f"{msg} [context: project P{last_code}]"
        return msg

    def _extract_project_code(self, msg):
        """Extract a 5-7 digit project code from message."""
        # Check for [context: project P123456]
        ctx = re.search(r'\[context: project P?(\d{5,7})\]', msg)
        if ctx:
            return ctx.group(1)
        m = re.search(r'P?(\d{5,7})', msg)
        return m.group(1) if m else None

    def _find_project(self, code, df):
        """Find project by code (numeric)."""
        code_str = str(code).strip().lstrip('Pp')
        try:
            code_num = int(code_str)
        except ValueError:
            return None
        match = df[df['Project Code'] == code_num]
        if len(match) == 0:
            match = df[df['Project Code'].astype(str) == code_str]
        if len(match) > 0:
            return match.iloc[0]
        return None

    def _find_prediction(self, code, preds):
        """Find ML prediction row for a project."""
        if preds is None:
            return None
        code_str = str(code).strip().lstrip('Pp')
        try:
            code_num = int(code_str)
        except ValueError:
            return None
        match = preds[preds['Project Code'] == code_num]
        if len(match) == 0:
            match = preds[preds['Project Code'].astype(str) == code_str]
        return match.iloc[0] if len(match) > 0 else None

    def _fuzzy_find_project(self, msg, df):
        """Try to match a project name from the message using fuzzy matching."""
        lower = msg.lower()
        # Remove common words
        for stop in ["tell me about", "show me", "details of", "info on", "what is",
                      "what about", "how about", "find", "search", "project"]:
            lower = lower.replace(stop, "")
        lower = lower.strip()
        if len(lower) < 4:
            return None

        best_score = 0
        best_code = None
        for _, row in df.iterrows():
            name = str(row.get('Project Name', '')).lower()
            ratio = SequenceMatcher(None, lower, name).ratio()
            if ratio > best_score and ratio > 0.5:
                best_score = ratio
                best_code = str(row.get('Project Code', ''))
        return best_code

    def _extract_state(self, msg, df):
        """Extract a state name from message."""
        lower = msg.lower()
        # Check aliases
        for alias, full in self.STATE_ALIASES.items():
            if alias in lower.split():
                return full
        # Check actual state names in dataset
        for state in df['State'].unique():
            if state.lower() in lower:
                return state
        return None

    def _extract_sector(self, msg, df):
        """Extract a sector name from message."""
        lower = msg.lower()
        for sector in df['Sector'].unique():
            if sector.lower() in lower:
                return sector
        # Check partial matches
        sector_keywords = {
            "road": "Roads & Highways", "highway": "Roads & Highways",
            "railway": "Railways", "rail": "Railways",
            "power": "Power", "energy": "Power",
            "water": "Water Resources", "irrigation": "Water Resources",
            "telecom": "Telecommunication",
            "petroleum": "Petroleum",
            "coal": "Coal",
        }
        for kw, sector in sector_keywords.items():
            if kw in lower:
                # Verify sector exists in data
                if sector in df['Sector'].values:
                    return sector
                # Try partial match
                for s in df['Sector'].unique():
                    if kw in s.lower():
                        return s
        return None

    def _extract_number(self, msg):
        """Extract a number from message."""
        m = re.search(r'(\d+(?:\.\d+)?)', msg)
        return float(m.group(1)) if m else None

    def _top_n_table(self, subset):
        """Generate a markdown table for a project subset."""
        if len(subset) == 0:
            return "_No projects to display._\n"
        a = "| Project | Name | State | Delay (days) | Progress |\n|---|---|---|---|---|\n"
        for _, r in subset.iterrows():
            d = round(max(r.get('Actual_Delay_Months', 0), 0) * 30)
            p = r.get('Physical Progress (%)', 0)
            name = str(r.get('Project Name', 'Unknown'))[:40]
            a += f"| P{r.get('Project Code', 'N/A')} | {name} | {r.get('State', 'N/A')} | {d} | {p:.1f}% |\n"
        return a

    # ═══════════════════════════════════════════════════════════════════════
    # LLM Integration (optional, if API key configured)
    # ═══════════════════════════════════════════════════════════════════════

    def _generate_llm_response(self, message, history, calc_result):
        """Optional LLM synthesis layer. Only used if an API key is configured."""
        api_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
                   or os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY"))
        if not api_key:
            return None

        prompt = f"""You are the PAIMANA Intelligence AI Assistant.
The user asked: "{message}"

Authoritative facts from the PAIMANA dataset:
{calc_result.get('text_answer', calc_result.get('answer', ''))}

INSTRUCTIONS:
1. Format a clear answer using ONLY the provided data.
2. DO NOT invent numbers, names, or statistics.
3. Use clean Markdown formatting.
"""
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are PAIMANA AI. Use only provided data."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        try:
            req = urllib.request.Request(base_url, data=json.dumps(payload).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return data.get('choices', [{}])[0].get('message', {}).get('content', '').strip() or None
        except Exception:
            return None
