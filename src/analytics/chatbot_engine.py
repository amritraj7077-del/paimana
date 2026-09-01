"""
PAIMANA Dataset-Aware Chatbot Engine
Authoritative Data Retrieval, Statistical Calculations, and LLM Integration
"""

import os
import re
import json
import urllib.request
import pandas as pd
import numpy as np


class PAIMANAChatbotEngine:
    def __init__(self):
        pass

    def process_chat(self, message: str, history: list, df: pd.DataFrame, analytics: dict = None) -> dict:
        """
        Process chat message with conversation history context and authoritative dataset analytics.
        Returns: {"answer": str, "sources": list}
        """
        if not message or not isinstance(message, str):
            return {
                "answer": "I couldn't find that information in the PAIMANA dataset.",
                "sources": []
            }

        message_clean = message.strip()
        history_normalized = self._normalize_history(history)

        # 1. Intent Recognition & Authoritative Data Calculation
        calculation_result = self._calculate_answer(message_clean, history_normalized, df, analytics)

        # 2. Check if LLM API key is present for natural language response generation
        llm_answer = self._generate_llm_response(message_clean, history_normalized, calculation_result)

        if llm_answer:
            return {
                "answer": llm_answer,
                "sources": calculation_result.get("sources", ["PAIMANA Dataset"])
            }

        # 3. Fallback to authoritative template response if no LLM configured
        return {
            "answer": calculation_result.get("text_answer", "I couldn't find that information in the PAIMANA dataset."),
            "sources": calculation_result.get("sources", ["PAIMANA Dataset"])
        }

    def _normalize_history(self, history: list) -> list:
        """
        Normalize various history formats into a standard list of {"role": "user"/"assistant", "content": "..."}
        """
        normalized = []
        if not history or not isinstance(history, list):
            return normalized

        for item in history:
            if isinstance(item, dict):
                role = item.get("role") or item.get("sender") or ("user" if "question" in item else "assistant")
                content = item.get("content") or item.get("text") or item.get("message") or item.get("question") or item.get("answer") or ""
                if role and content:
                    normalized.append({"role": role.lower(), "content": str(content)})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.append({"role": "user", "content": str(item[0])})
                normalized.append({"role": "assistant", "content": str(item[1])})

        return normalized

    def _extract_project_ids(self, text: str) -> list:
        """
        Extract project IDs/codes from text. Handles P108841, 108841, project 108841, etc.
        """
        if not text:
            return []
        
        # Match explicit project patterns
        matches = re.findall(r'\b(?:[Pp]roject\s*(?:[Ii][Dd])?\s*#?)?\s*([Pp]?\d{3,8})\b', text)
        extracted = []
        for m in matches:
            # Clean leading P/p if present or keep raw
            clean_code = re.sub(r'^[Pp]', '', m)
            if clean_code.isdigit() and len(clean_code) >= 3:
                extracted.append(clean_code)
        
        # Deduplicate preserving order
        res = []
        for x in extracted:
            if x not in res:
                res.append(x)
        return res

    def _get_active_project_id(self, message: str, history: list, df: pd.DataFrame) -> str:
        """
        Identify target project ID from current message or backward history context.
        """
        # 1. Direct mention in current message
        current_ids = self._extract_project_ids(message)
        for pid in current_ids:
            if self._project_exists(df, pid):
                return pid

        # 2. Check for pronouns or context references in message
        pronouns = ["it", "its", "this project", "that project", "the project", "this", "that"]
        msg_lower = message.lower()
        has_context_ref = any(re.search(rf'\b{re.escape(p)}\b', msg_lower) for p in pronouns) or "delay" in msg_lower or "cost" in msg_lower or "progress" in msg_lower or "status" in msg_lower or "risk" in msg_lower

        if has_context_ref and history:
            # Search backward in history
            for item in reversed(history):
                content = item.get("content", "")
                h_ids = self._extract_project_ids(content)
                for pid in h_ids:
                    if self._project_exists(df, pid):
                        return pid

        return None

    def _project_exists(self, df: pd.DataFrame, project_id: str) -> bool:
        """Check if project exists in dataset"""
        if not project_id:
            return False
        
        # Check Project Code (numeric or string)
        if 'Project Code' in df.columns:
            if project_id.isdigit():
                if not df[df['Project Code'] == int(project_id)].empty:
                    return True
            if not df[df['Project Code'].astype(str).str.strip().str.upper() == project_id.upper()].empty:
                return True

        if 'project_id' in df.columns:
            if not df[df['project_id'].astype(str).str.strip().str.upper() == project_id.upper()].empty:
                return True

        return False

    def _get_project_row(self, df: pd.DataFrame, project_id: str):
        """Retrieve single project Pandas Series from dataset"""
        if not project_id:
            return None
        
        if 'Project Code' in df.columns and project_id.isdigit():
            match = df[df['Project Code'] == int(project_id)]
            if not match.empty:
                return match.iloc[0]

        if 'Project Code' in df.columns:
            match = df[df['Project Code'].astype(str).str.strip().str.upper() == project_id.upper()]
            if not match.empty:
                return match.iloc[0]

        if 'project_id' in df.columns:
            match = df[df['project_id'].astype(str).str.strip().str.upper() == project_id.upper()]
            if not match.empty:
                return match.iloc[0]

        # Name search fallback
        name_matches = df[df['Project Name'].astype(str).str.lower().str.contains(project_id.lower(), na=False)]
        if not name_matches.empty:
            return name_matches.iloc[0]

        return None

    def _extract_state(self, message: str, df: pd.DataFrame) -> str:
        """Extract state/district mentioned in message"""
        if 'State' not in df.columns:
            return None
        
        unique_states = df['State'].dropna().unique()
        msg_lower = message.lower()

        # Sort by length descending to match longer names first
        for state in sorted(unique_states, key=lambda s: len(str(s)), reverse=True):
            state_clean = str(state).replace('\n', ' ').strip()
            if len(state_clean) > 2 and state_clean.lower() in msg_lower:
                return state

        return None

    def _extract_sector(self, message: str, df: pd.DataFrame) -> str:
        """Extract sector/category mentioned in message"""
        if 'Sector' not in df.columns:
            return None

        unique_sectors = df['Sector'].dropna().unique()
        msg_lower = message.lower()

        for sector in sorted(unique_sectors, key=lambda s: len(str(s)), reverse=True):
            sector_clean = str(sector).strip()
            if len(sector_clean) > 2 and sector_clean.lower() in msg_lower:
                return sector

        return None

    def _format_project_detail(self, row) -> str:
        """Format full details for a project row"""
        p_code = str(row.get('Project Code', row.get('project_id', 'N/A')))
        p_name = str(row.get('Project Name', row.get('project_name', 'Unknown')))
        state = str(row.get('State', row.get('district', 'N/A'))).replace('\n', ' ')
        sector = str(row.get('Sector', row.get('category', 'N/A')))
        agency = str(row.get('Agency', 'N/A'))
        ministry = str(row.get('Ministry', 'N/A'))

        s_cost = row.get('Original Cost (Rs. Crore)', row.get('sanctioned_cost', 0))
        if s_cost > 1000000: # Convert raw rupees if needed
            s_cost_crore = s_cost / 10000000
        else:
            s_cost_crore = s_cost

        c_exp = row.get('Cumulative Expenditure (Rs. Crore)', row.get('expenditure_to_date', 0))
        if c_exp > 1000000:
            c_exp_crore = c_exp / 10000000
        else:
            c_exp_crore = c_exp

        progress = float(row.get('Physical Progress (%)', row.get('physical_progress_percent', 0)))
        delay_m = float(row.get('Actual_Delay_Months', row.get('Delay_Months', 0)))
        delay_days = int(delay_m * 30) if delay_m > 0 else int(row.get('delay_days', 0))

        cost_overrun_pct = float(row.get('Cost_Overrun_Ratio', 0)) * 100
        risk = str(row.get('Risk_Level', row.get('risk_level', 'LOW')))

        target_doc = str(row.get('Original/Target DoC', 'N/A'))
        revised_doc = str(row.get('Revised DoC', 'N/A'))
        actual_doc = str(row.get('Actual Date of Completion', 'N/A'))

        status = "Delayed" if delay_m > 0 else ("Completed" if progress >= 100 else "In Progress")

        ans = f"### 📌 Project Details: Project {p_code}\n\n"
        ans += f"**Project Name:** {p_name}\n"
        ans += f"**State/Location:** {state}\n"
        ans += f"**Sector/Category:** {sector}\n"
        ans += f"**Ministry/Agency:** {ministry} ({agency})\n"
        ans += f"**Status:** {status}\n\n"
        ans += f"#### 📊 Progress & Financials:\n"
        ans += f"- **Physical Progress:** {progress:.1f}%\n"
        ans += f"- **Sanctioned Cost:** ₹{s_cost_crore:,.2f} Crore\n"
        ans += f"- **Cumulative Expenditure:** ₹{c_exp_crore:,.2f} Crore\n"
        ans += f"- **Cost Overrun:** {cost_overrun_pct:.1f}%\n\n"
        ans += f"#### ⏱️ Delay & ML Predictions:\n"
        ans += f"- **Delay:** {delay_days} days ({delay_m:.1f} months)\n"
        ans += f"- **Target Completion:** {target_doc}\n"
        if revised_doc and revised_doc != '-':
            ans += f"- **Revised Completion:** {revised_doc}\n"
        if actual_doc and str(actual_doc) != 'nan':
            ans += f"- **Actual Completion:** {actual_doc}\n"
        ans += f"- **Predicted Risk Level:** `{risk}`\n"

        return ans

    def _calculate_answer(self, message: str, history: list, df: pd.DataFrame, analytics: dict = None) -> dict:
        """
        Core authoritative query handler using Pandas calculations on dataset.
        """
        msg_lower = message.lower()
        sources = ["PAIMANA Dataset (df_reference.csv)"]

        # Helper calculations
        total_projects = len(df)
        delayed_df = df[df['Actual_Delay_Months'] > 0]
        cost_overrun_df = df[df['Cost_Overrun_Ratio'] > 0]
        high_risk_df = df[df['Risk_Level'].isin(['HIGH', 'MEDIUM-HIGH'])]

        # --- A. Check for Comparison Query ---
        comp_ids = self._extract_project_ids(message)
        if ("compare" in msg_lower or "versus" in msg_lower or " vs " in msg_lower) and len(comp_ids) >= 1:
            # If user says "compare it with P120250", get active project for first
            if len(comp_ids) == 1:
                active_id = self._get_active_project_id(message, history, df)
                if active_id and active_id != comp_ids[0]:
                    comp_ids = [active_id, comp_ids[0]]
            
            if len(comp_ids) >= 2:
                p1_row = self._get_project_row(df, comp_ids[0])
                p2_row = self._get_project_row(df, comp_ids[1])
                if p1_row is not None and p2_row is not None:
                    txt = f"### ⚖️ Project Comparison: P{comp_ids[0]} vs P{comp_ids[1]}\n\n"
                    txt += f"| Metric | Project {comp_ids[0]} | Project {comp_ids[1]} |\n"
                    txt += f"|---|---|---|\n"
                    txt += f"| **Project Name** | {p1_row.get('Project Name', 'N/A')[:40]} | {p2_row.get('Project Name', 'N/A')[:40]} |\n"
                    txt += f"| **State/Location** | {str(p1_row.get('State', 'N/A')).replace(chr(10), ' ')} | {str(p2_row.get('State', 'N/A')).replace(chr(10), ' ')} |\n"
                    txt += f"| **Sector** | {p1_row.get('Sector', 'N/A')} | {p2_row.get('Sector', 'N/A')} |\n"
                    txt += f"| **Sanctioned Cost** | ₹{p1_row.get('Original Cost (Rs. Crore)', 0):,.2f} Cr | ₹{p2_row.get('Original Cost (Rs. Crore)', 0):,.2f} Cr |\n"
                    txt += f"| **Expenditure** | ₹{p1_row.get('Cumulative Expenditure (Rs. Crore)', 0):,.2f} Cr | ₹{p2_row.get('Cumulative Expenditure (Rs. Crore)', 0):,.2f} Cr |\n"
                    txt += f"| **Physical Progress** | {p1_row.get('Physical Progress (%)', 0):.1f}% | {p2_row.get('Physical Progress (%)', 0):.1f}% |\n"
                    txt += f"| **Delay (Months)** | {p1_row.get('Actual_Delay_Months', 0):.1f} months | {p2_row.get('Actual_Delay_Months', 0):.1f} months |\n"
                    txt += f"| **Risk Level** | `{p1_row.get('Risk_Level', 'LOW')}` | `{p2_row.get('Risk_Level', 'LOW')}` |\n"
                    return {"text_answer": txt, "sources": sources, "intent": "COMPARISON"}

        # --- B. Check for Specific Project Query ---
        target_project_id = self._get_active_project_id(message, history, df)

        # Check if the user is asking specifically about a project (and not just general aggregate stats)
        is_general_stat_query = any(q in msg_lower for q in [
            "how many projects", "total projects", "which district", "which project has highest",
            "highest delay", "average physical progress", "average progress", "show cost overruns",
            "delayed projects", "which projects have cost overruns", "most delayed", "most projects"
        ])

        # If user explicitly provided an unknown project ID e.g. P999999
        raw_ids_in_msg = self._extract_project_ids(message)
        if raw_ids_in_msg and not target_project_id and not is_general_stat_query:
            return {
                "text_answer": f"I couldn't find project **{raw_ids_in_msg[0]}** in the PAIMANA dataset.",
                "sources": sources,
                "intent": "NOT_FOUND"
            }

        if target_project_id and not is_general_stat_query:
            row = self._get_project_row(df, target_project_id)
            if row is not None:
                p_code = str(row.get('Project Code', target_project_id))
                delay_m = float(row.get('Actual_Delay_Months', 0))
                delay_d = int(delay_m * 30)
                s_cost = row.get('Original Cost (Rs. Crore)', 0)
                c_exp = row.get('Cumulative Expenditure (Rs. Crore)', 0)
                prog = float(row.get('Physical Progress (%)', 0))

                # Check sub-queries for project
                if "delay" in msg_lower or "late" in msg_lower:
                    ans = f"Project **P{p_code}** ({row.get('Project Name')}) has a delay of **{delay_d} days** ({delay_m:.1f} months)."
                    if delay_m > 0:
                        ans += f"\n- Target DoC: {row.get('Original/Target DoC', 'N/A')}\n- ML Predicted Risk: `{row.get('Risk_Level', 'LOW')}`"
                    else:
                        ans += " The project is currently on schedule."
                    return {"text_answer": ans, "sources": sources, "intent": "PROJECT_DELAY"}

                if "cost" in msg_lower or "expenditure" in msg_lower or "spent" in msg_lower or "budget" in msg_lower:
                    overrun = float(row.get('Cost_Overrun_Ratio', 0)) * 100
                    ans = f"Financial details for **Project P{p_code}** ({row.get('Project Name')}):\n"
                    ans += f"- **Sanctioned Cost:** ₹{s_cost:,.2f} Crore\n"
                    ans += f"- **Cumulative Expenditure:** ₹{c_exp:,.2f} Crore\n"
                    ans += f"- **Cost Overrun:** {overrun:.1f}%\n"
                    ans += f"- **Budget Utilization / Exp Ratio:** {float(row.get('Expenditure_Ratio', 0))*100:.1f}%"
                    return {"text_answer": ans, "sources": sources, "intent": "PROJECT_COST"}

                if "progress" in msg_lower:
                    ans = f"Project **P{p_code}** has a physical progress of **{prog:.1f}%**."
                    return {"text_answer": ans, "sources": sources, "intent": "PROJECT_PROGRESS"}

                if "risk" in msg_lower or "prediction" in msg_lower:
                    ans = f"ML Risk Assessment for **Project P{p_code}**:\n"
                    ans += f"- **Risk Level:** `{row.get('Risk_Level', 'LOW')}`\n"
                    ans += f"- **Actual/Predicted Delay:** {delay_m:.1f} months ({delay_d} days)\n"
                    ans += f"- **Cost Overrun Ratio:** {float(row.get('Cost_Overrun_Ratio', 0)):.2f}"
                    return {"text_answer": ans, "sources": sources, "intent": "PROJECT_RISK"}

                # Full details
                return {"text_answer": self._format_project_detail(row), "sources": sources, "intent": "PROJECT_DETAIL"}
            else:
                return {
                    "text_answer": "I couldn't find that information in the PAIMANA dataset.",
                    "sources": sources,
                    "intent": "NOT_FOUND"
                }

        # --- C. Query 1: Total project count ---
        if any(p in msg_lower for p in ["how many projects are there", "total projects", "count of projects", "how many total projects"]):
            total_cost = df['Original Cost (Rs. Crore)'].sum()
            total_exp = df['Cumulative Expenditure (Rs. Crore)'].sum()
            ans = f"There are **{total_projects:,} projects** in the PAIMANA dataset.\n\n"
            ans += f"**Key Breakdown:**\n"
            ans += f"- **Delayed Projects:** {len(delayed_df):,} ({len(delayed_df)/total_projects*100:.1f}%)\n"
            ans += f"- **On-Time / Ahead Projects:** {total_projects - len(delayed_df):,}\n"
            ans += f"- **Projects with Cost Overrun:** {len(cost_overrun_df):,} ({len(cost_overrun_df)/total_projects*100:.1f}%)\n"
            ans += f"- **High/Medium-High Risk Projects:** {len(high_risk_df):,}\n"
            ans += f"- **Total Sanctioned Cost:** ₹{total_cost:,.2f} Crore\n"
            ans += f"- **Total Cumulative Expenditure:** ₹{total_exp:,.2f} Crore"
            return {"text_answer": ans, "sources": sources, "intent": "TOTAL_PROJECTS"}

        # --- D. Query 2: Highest Delay ---
        if "highest delay" in msg_lower or "most delayed project" in msg_lower or "max delay" in msg_lower or "maximum delay" in msg_lower:
            max_idx = df['Actual_Delay_Months'].idxmax()
            top_row = df.loc[max_idx]
            top_code = str(top_row.get('Project Code'))
            top_name = str(top_row.get('Project Name'))
            top_delay_m = float(top_row.get('Actual_Delay_Months'))
            top_delay_d = int(top_delay_m * 30)
            top_state = str(top_row.get('State')).replace('\n', ' ')
            top_cost = top_row.get('Original Cost (Rs. Crore)', 0)

            ans = f"The project with the **highest delay** in the PAIMANA dataset is **Project P{top_code}**: **{top_name}**.\n\n"
            ans += f"- **Delay:** {top_delay_d:,} days ({top_delay_m:.1f} months / {top_delay_m/12:.1f} years)\n"
            ans += f"- **State/Location:** {top_state}\n"
            ans += f"- **Sector:** {top_row.get('Sector')}\n"
            ans += f"- **Physical Progress:** {top_row.get('Physical Progress (%)', 0):.1f}%\n"
            ans += f"- **Sanctioned Cost:** ₹{top_cost:,.2f} Crore\n"
            ans += f"- **ML Risk Level:** `{top_row.get('Risk_Level', 'HIGH')}`"
            return {"text_answer": ans, "sources": sources, "intent": "HIGHEST_DELAY"}

        # --- E. Query 3 & Delayed Projects List ---
        if "how many projects are delayed" in msg_lower or "show me the delayed projects" in msg_lower or "delayed projects" in msg_lower or "list delayed" in msg_lower:
            del_count = len(delayed_df)
            pct = (del_count / total_projects) * 100
            top_5 = delayed_df.sort_values('Actual_Delay_Months', ascending=False).head(5)

            ans = f"There are **{del_count:,} delayed projects** out of {total_projects:,} total projects ({pct:.1f}%).\n\n"
            ans += f"**Top 5 Most Delayed Projects:**\n"
            for i, (_, r) in enumerate(top_5.iterrows(), 1):
                pcode = r.get('Project Code')
                pname = r.get('Project Name')
                d_m = float(r.get('Actual_Delay_Months', 0))
                st = str(r.get('State')).replace('\n', ' ')
                ans += f"{i}. **P{pcode}** ({pname[:45]}...): **{d_m:.1f} months delay** | State: {st} | Progress: {r.get('Physical Progress (%)', 0):.1f}%\n"
            return {"text_answer": ans, "sources": sources, "intent": "DELAYED_PROJECTS"}

        # --- F. Query 4: Average Physical Progress ---
        if "average physical progress" in msg_lower or "average progress" in msg_lower or "mean progress" in msg_lower:
            avg_prog = df['Physical Progress (%)'].mean()
            median_prog = df['Physical Progress (%)'].median()
            min_prog = df['Physical Progress (%)'].min()
            max_prog = df['Physical Progress (%)'].max()

            ans = f"The **average physical progress** across all {total_projects:,} projects is **{avg_prog:.2f}%** (median: {median_prog:.1f}%).\n\n"
            ans += f"- **Minimum Progress:** {min_prog:.1f}%\n"
            ans += f"- **Maximum Progress:** {max_prog:.1f}%\n"
            ans += f"- **Completed Projects (100%):** {len(df[df['Physical Progress (%)'] >= 100]):,} projects"
            return {"text_answer": ans, "sources": sources, "intent": "AVERAGE_PROGRESS"}

        # --- G. Query 5: District / State Statistics ---
        if "which district" in msg_lower or "which state" in msg_lower or "district has the most" in msg_lower or "state has the most" in msg_lower or "district-wise" in msg_lower or "state-wise" in msg_lower:
            if "delay" in msg_lower:
                top_states_del = delayed_df['State'].value_counts()
                top_state = top_states_del.index[0].replace('\n', ' ')
                top_cnt = top_states_del.iloc[0]

                ans = f"The state/district with the most **delayed projects** is **{top_state}** with **{top_cnt} delayed projects**.\n\n"
                ans += f"**Top 5 States by Delayed Projects:**\n"
                for idx, (st, cnt) in enumerate(top_states_del.head(5).items(), 1):
                    st_clean = str(st).replace('\n', ' ')
                    ans += f"{idx}. **{st_clean}**: {cnt} delayed projects\n"
                return {"text_answer": ans, "sources": sources, "intent": "DISTRICT_DELAY_RANKING"}

            else:
                top_states_all = df['State'].value_counts()
                top_state = top_states_all.index[0].replace('\n', ' ')
                top_cnt = top_states_all.iloc[0]

                ans = f"The state/district with the **most projects** is **{top_state}** with **{top_cnt} projects**.\n\n"
                ans += f"**Top 5 States by Project Count:**\n"
                for idx, (st, cnt) in enumerate(top_states_all.head(5).items(), 1):
                    st_clean = str(st).replace('\n', ' ')
                    ans += f"{idx}. **{st_clean}**: {cnt} projects\n"
                return {"text_answer": ans, "sources": sources, "intent": "DISTRICT_COUNT_RANKING"}

        # --- Check for specific state mentioned ---
        state_found = self._extract_state(message, df)
        if state_found:
            st_df = df[df['State'] == state_found]
            st_clean = str(state_found).replace('\n', ' ')
            st_del = st_df[st_df['Actual_Delay_Months'] > 0]
            st_overrun = st_df[st_df['Cost_Overrun_Ratio'] > 0]
            st_cost = st_df['Original Cost (Rs. Crore)'].sum()
            st_exp = st_df['Cumulative Expenditure (Rs. Crore)'].sum()

            ans = f"### 📍 Project Summary for **{st_clean}**\n\n"
            ans += f"- **Total Projects:** {len(st_df):,}\n"
            ans += f"- **Delayed Projects:** {len(st_del):,} ({len(st_del)/len(st_df)*100:.1f}%)\n"
            ans += f"- **Projects with Cost Overrun:** {len(st_overrun):,}\n"
            ans += f"- **Average Physical Progress:** {st_df['Physical Progress (%)'].mean():.1f}%\n"
            ans += f"- **Average Delay:** {st_df['Actual_Delay_Months'].mean():.1f} months\n"
            ans += f"- **Total Sanctioned Budget:** ₹{st_cost:,.2f} Crore\n"
            ans += f"- **Total Cumulative Expenditure:** ₹{st_exp:,.2f} Crore"
            return {"text_answer": ans, "sources": sources, "intent": "STATE_SUMMARY"}

        # --- H. Query 6: Cost Overruns ---
        if "cost overrun" in msg_lower or "cost overruns" in msg_lower or "projects have cost overruns" in msg_lower or "budget overrun" in msg_lower:
            co_cnt = len(cost_overrun_df)
            pct = (co_cnt / total_projects) * 100
            top_co = cost_overrun_df.sort_values('Cost_Overrun_Ratio', ascending=False).head(5)

            ans = f"There are **{co_cnt:,} projects with cost overruns** ({pct:.1f}% of total projects).\n\n"
            ans += f"**Top 5 Cost Overrun Projects:**\n"
            for i, (_, r) in enumerate(top_co.iterrows(), 1):
                pcode = r.get('Project Code')
                pname = r.get('Project Name')
                ratio = float(r.get('Cost_Overrun_Ratio', 0)) * 100
                orig = r.get('Original Cost (Rs. Crore)', 0)
                rev = r.get('Revised Cost (Rs. Crore)', 0)
                ans += f"{i}. **P{pcode}** ({pname[:40]}...): **+{ratio:.1f}% overrun** | Orig: ₹{orig:,.1f} Cr → Rev: ₹{rev:,.1f} Cr\n"
            return {"text_answer": ans, "sources": sources, "intent": "COST_OVERRUNS"}

        # --- Check for Category / Sector mentioned ---
        sector_found = self._extract_sector(message, df)
        if sector_found or "category" in msg_lower or "sector" in msg_lower:
            if sector_found:
                sec_df = df[df['Sector'] == sector_found]
                sec_del = sec_df[sec_df['Actual_Delay_Months'] > 0]
                ans = f"### 🏗️ Category Summary: **{sector_found}**\n\n"
                ans += f"- **Total Projects:** {len(sec_df):,}\n"
                ans += f"- **Delayed Projects:** {len(sec_del):,} ({len(sec_del)/len(sec_df)*100:.1f}%)\n"
                ans += f"- **Average Physical Progress:** {sec_df['Physical Progress (%)'].mean():.1f}%\n"
                ans += f"- **Average Delay:** {sec_df['Actual_Delay_Months'].mean():.1f} months\n"
                ans += f"- **Total Sanctioned Budget:** ₹{sec_df['Original Cost (Rs. Crore)'].sum():,.2f} Crore"
                return {"text_answer": ans, "sources": sources, "intent": "CATEGORY_SUMMARY"}
            else:
                sec_counts = df['Sector'].value_counts()
                ans = f"PAIMANA tracks projects across **{len(sec_counts)} sectors/categories**.\n\n"
                ans += f"**Top Sectors by Project Count:**\n"
                for idx, (sec, cnt) in enumerate(sec_counts.head(5).items(), 1):
                    ans += f"{idx}. **{sec}**: {cnt} projects\n"
                return {"text_answer": ans, "sources": sources, "intent": "CATEGORY_BREAKDOWN"}

        # --- I. Critical / High-Risk Projects ---
        if "high risk" in msg_lower or "critical" in msg_lower or "risk level" in msg_lower:
            hr_cnt = len(high_risk_df)
            top_hr = high_risk_df.sort_values('Actual_Delay_Months', ascending=False).head(5)

            ans = f"PAIMANA's ML risk model has identified **{hr_cnt:,} High / Medium-High Risk projects**.\n\n"
            ans += f"**Top Critical High-Risk Projects:**\n"
            for i, (_, r) in enumerate(top_hr.iterrows(), 1):
                pcode = r.get('Project Code')
                pname = r.get('Project Name')
                rk = r.get('Risk_Level')
                dm = float(r.get('Actual_Delay_Months', 0))
                ans += f"{i}. **P{pcode}** ({pname[:40]}...): Risk `{rk}` | Delay: {dm:.1f} months | Progress: {r.get('Physical Progress (%)', 0):.1f}%\n"
            return {"text_answer": ans, "sources": sources, "intent": "HIGH_RISK"}

        # --- J. General Platform Questions ---
        if "what is paimana" in msg_lower or "about paimana" in msg_lower:
            ans = "PAIMANA is an infrastructure intelligence platform that combines project dataset analytics, GIS mapping, and trained Machine Learning models to predict cost overruns, delays, and risk levels across national infrastructure projects."
            return {"text_answer": ans, "sources": sources, "intent": "PLATFORM_INFO"}

        if "how does risk prediction work" in msg_lower or "machine learning" in msg_lower or "ml model" in msg_lower or "algorithm" in msg_lower:
            ans = "PAIMANA utilizes Machine Learning models trained on historical project indicators (sector, budget, physical progress, expenditure ratio, cost-per-progress) to classify project risk into LOW, MEDIUM, MEDIUM-HIGH, and HIGH tiers, while predicting potential delay months and cost overrun ratios."
            return {"text_answer": ans, "sources": sources, "intent": "PLATFORM_INFO"}

        if "help" in msg_lower or "what can you ask" in msg_lower or "what can you do" in msg_lower:
            ans = "I am the PAIMANA Dataset-Aware AI Assistant. You can ask me dynamic questions such as:\n"
            ans += "- *'How many projects are there?'*\n"
            ans += "- *'Which project has the highest delay?'*\n"
            ans += "- *'Show me the delayed projects.'*\n"
            ans += "- *'What is the average physical progress?'*\n"
            ans += "- *'Which district has the most delayed projects?'*\n"
            ans += "- *'Which projects have cost overruns?'*\n"
            ans += "- *'Tell me about project P108841.'*\n"
            ans += "- Follow-ups: *'What is its delay?'*, *'What about its cost?'*\n"
            ans += "- *'Compare P108841 and P120250.'*"
            return {"text_answer": ans, "sources": sources, "intent": "HELP"}

        # --- K. Default Dataset Overview Fallback ---
        ans = f"The PAIMANA dataset contains **{total_projects:,} active project records** across {df['State'].nunique()} states/districts and {df['Sector'].nunique()} sectors.\n\n"
        ans += f"- **Delayed Projects:** {len(delayed_df):,} ({len(delayed_df)/total_projects*100:.1f}%)\n"
        ans += f"- **Average Progress:** {df['Physical Progress (%)'].mean():.1f}%\n"
        ans += f"- **Cost Overrun Projects:** {len(cost_overrun_df):,}\n\n"
        ans += "You can ask me about specific project IDs (e.g. *'Tell me about project 108841'*), rankings (e.g. *'Which project has highest delay?'*), or statistics."
        return {"text_answer": ans, "sources": sources, "intent": "DATASET_OVERVIEW"}

    def _generate_llm_response(self, message: str, history: list, calc_result: dict) -> str:
        """
        Synthesize natural language response using an external LLM API if key is set in environment.
        Supports OPENAI_API_KEY, GEMINI_API_KEY, or generic LLM_API_KEY.
        """
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None  # No LLM API key configured; fallback to authoritative python answer

        prompt = f"""You are the official PAIMANA Intelligence AI Assistant.
The user asked: "{message}"

Authoritative facts calculated directly from the PAIMANA project dataset:
{calc_result.get('text_answer')}

INSTRUCTIONS:
1. Format a clear, polite, and authoritative answer based ONLY on the provided dataset facts above.
2. DO NOT invent or alter any numerical values, project IDs, delays, costs, or statistics.
3. Keep the answer concise and easy to read using clean Markdown formatting.
"""

        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are PAIMANA AI Assistant. Always rely strictly on provided authoritative dataset values."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        try:
            req = urllib.request.Request(
                base_url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_data = json.loads(resp.read().decode('utf-8'))
                choices = res_data.get('choices', [])
                if choices and len(choices) > 0:
                    return choices[0].get('message', {}).get('content', '').strip()
        except Exception as e:
            print(f"LLM API call warning (falling back to direct calculation): {e}")

        return None
