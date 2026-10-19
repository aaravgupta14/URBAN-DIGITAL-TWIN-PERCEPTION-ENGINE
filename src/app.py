import streamlit as st
import sqlite3
import os

# Configure professional layout framework
st.set_page_config(
    page_title="Urban Traffic Digital Twin Analytics Engine", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def parse_and_query(user_input):
    """
    Core natural language keyword routing engine.
    Queries the SQLite state repository and formats clean tabular layouts.
    """
    db_path = "twin_state.db"
    if not os.path.exists(db_path):
        return "System Status: Data source error. 'twin_state.db' repository connection missing."
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = user_input.lower().strip()
    
    try:
        # 1. CRITICAL SAFETY ALERTS
        if "critical" in query and "alert" in query:
            cursor.execute("SELECT type, severity, tracks, message FROM alerts WHERE severity='critical'")
            rows = cursor.fetchall()
            if not rows:
                return "Analysis Output: Zero critical conflicts logged in the active sequence."
            
            output = "### Active Critical Safety Conflict Logs\n\n"
            output += "| Conflict Type | Target Track Scope | System Telemetry Message |\n"
            output += "| :--- | :--- | :--- |\n"
            for r in rows:
                output += f"| {r[0].upper()} | Tracks: {r[2]} | {r[3]} |\n"
            return output
            
        # 2. ALL SYSTEM ALERTS / WARNING SUMMARY
        elif "alert" in query or "warning" in query:
            cursor.execute("SELECT type, severity, message FROM alerts")
            rows = cursor.fetchall()
            if not rows:
                return "Analysis Output: No alerts or incident parameters present in this run log."
            
            output = f"### System Alert Summary Ledger (Total Incidents: {len(rows)})\n\n"
            output += "| Threat Level | Classification | Incident Details |\n"
            output += "| :--- | :--- | :--- |\n"
            for r in rows:
                output += f"| {r[1].upper()} | {r[0]} | {r[2]} |\n"
            return output

        # 3. VELOCITY & TRAJECTORY STATS (KINEMATICS)
        elif "speed" in query or "overspeed" in query or "velocity" in query or "kinematics" in query:
            cursor.execute("SELECT tracks, speed FROM alerts WHERE type='overspeed'")
            alert_rows = cursor.fetchall()
            cursor.execute("SELECT MAX(speed_kmh) FROM kinematics")
            max_kin = cursor.fetchone()
            
            output = "### Vehicle Telemetry and Velocity Report\n\n"
            output += "| Parameter Description | Metric Evaluated |\n"
            output += "| :--- | :--- |\n"
            if alert_rows:
                for r in alert_rows:
                    output += f"| Active Overspeed Infraction (Track ID {r[0]}) | {r[1]} km/h |\n"
            if max_kin and max_kin[0] is not None:
                output += f"| Peak Recorded Telemetry Track Speed | {max_kin[0]:.2f} km/h |\n"
            else:
                output += "| Peak Recorded Telemetry Track Speed | No valid kinematics tracking rows |\n"
            return output
            
        # 4. DATABASE METADATA ARCHITECTURE & COLUMNS
        elif "table" in query or "schema" in query or "structure" in query:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]
            
            output = "### Database Schema Validation Overview\n"
            output += f"**Active Data Collections:** {', '.join([f'`{t}`' for t in tables])}\n\n"
            
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                cols = [c[1] for c in cursor.fetchall()]
                output += f"*   **Schema [{table.upper()}]:** `{', '.join(cols)}`\n"
            return output
            
        # 5. GENERAL SYSTEM INFORMATION FALLBACK
        else:
            return ("System Insight Engine:\nInput parameters acknowledged. "
                    "Please structure your query to target 'critical alerts', 'velocity telemetry reports', "
                    "or 'database schema structures' to query the digital twin data store.")
                    
    except Exception as e:
        return f"System Runtime Exception: {str(e)}"
    finally:
        conn.close()

# --- WEBAPP PAGE LAYOUT ---
st.title("Urban Traffic Digital Twin Perception Engine")
st.caption("Analytical Query Layer & Vehicle Telemetry Assessment Dashboard")
st.markdown("---")

# Professional, clean sidebar setup
with st.sidebar:
    st.markdown("### Workspace Infrastructure")
    st.markdown("System Status: **Connected to twin_state.db**")
    st.markdown("Architecture Mode: **Local Offline Processing Layer**")
    st.markdown("---")
    if st.button("Reset Session History", use_container_width=True):
        st.session_state.app_chat_history = []
        st.rerun()

# Maintain localized layout memory tracking
if "app_chat_history" not in st.session_state:
    st.session_state.app_chat_history = []

# Output conversation flow cleanly
for entry in st.session_state.app_chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# User Chat Input Prompt Interaction
if input_prompt := st.chat_input("Enter analytical query parameter..."):
    # Render user prompt cleanly
    with st.chat_message("user"):
        st.markdown(input_prompt)
    st.session_state.app_chat_history.append({"role": "user", "content": input_prompt})
    
    # Process queries and output clean layout tables
    with st.chat_message("assistant"):
        with st.spinner("Processing local tracking data matrix..."):
            processed_response = parse_and_query(input_prompt)
            st.markdown(processed_response)
    st.session_state.app_chat_history.append({"role": "assistant", "content": processed_response})
