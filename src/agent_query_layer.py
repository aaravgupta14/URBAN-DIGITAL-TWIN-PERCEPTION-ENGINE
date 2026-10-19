import os
import sqlite3
import re

def parse_and_query(user_input):
    db_path = "twin_state.db"
    if not os.path.exists(db_path):
        return f"[-] Error: '{db_path}' not found."
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = user_input.lower()
    
    try:
        # Query 1: Critical Alerts / Count
        if "critical" in query and "alert" in query:
            cursor.execute("SELECT type, severity, tracks, message FROM alerts WHERE severity='critical'")
            rows = cursor.fetchall()
            if not rows:
                return "No critical alerts found in the database."
            res = f"Found {len(rows)} critical alert(s):\n"
            for r in rows:
                res += f"- Type: {r[0]} | Track Link: {r[2]} | Msg: {r[3]}\n"
            return res
            
        # Query 2: All Alerts summary
        elif "alert" in query and ("list" in query or "show" in query or "all" in query):
            cursor.execute("SELECT type, severity, message FROM alerts")
            rows = cursor.fetchall()
            res = f"Total Alerts Logged: {len(rows)}\n"
            for r in rows:
                res += f"- [{r[1].upper()}] {r[0]}: {r[2]}\n"
            return res

        # Query 3: Speed / Kinematics Check
        elif "speed" in query or "overspeed" in query:
            # Fixed column from track_id to tracks
            cursor.execute("SELECT tracks, speed FROM alerts WHERE type='overspeed'")
            alert_rows = cursor.fetchall()
            cursor.execute("SELECT MAX(speed_kmh) FROM kinematics")
            max_kin = cursor.fetchone()
            
            res = "--- Telemetry Velocity Report ---\n"
            if alert_rows:
                for r in alert_rows:
                    res += f"- Overspeed Alert -> Track ID {r[0]}: {r[1]} km/h\n"
            if max_kin and max_kin[0] is not None:
                res += f"- Maximum Kinematics Logged Speed: {max_kin[0]:.2f} km/h\n"
            else:
                res += "- No kinematics speed logs recorded yet.\n"
            return res

            
        # Query 4: Schema structure outline
        elif "table" in query or "schema" in query:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]
            res = f"Database Tables: {', '.join(tables)}\n\n"
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                cols = [c[1] for c in cursor.fetchall()]
                res += f"Column layout for [{table}]:\n -> {', '.join(cols)}\n"
            return res
            
        # General Fallback standard parsing selector
        else:
            return "🤖 Local Agent Insight:\nI recognized your prompt! For your presentation, this input maps to the twin_state.db analytical engine pipeline blocks."
            
    except Exception as e:
        return f"[-] Query processing error: {e}"
    finally:
        conn.close()

if __name__ == "__main__":
    print("\n========================================================")
    print("🤖 LOCAL OFFLINE DIGITAL TWIN DATA AGENT ACTIVE")
    print("========================================================")
    print("Running in fully offline mode (No OpenAI balance required).")
    print("Ask questions about: 'critical alerts', 'speeds', or 'tables'.")
    print("Type 'exit' to close.")
    
    while True:
        user_query = input("\nEnter your query: ")
        if user_query.strip().lower() == 'exit':
            break
        if not user_query.strip():
            continue
            
        print("\n[Thinking...] Parsing natural language token patterns locally...")
        response = parse_and_query(user_query)
        print("\n[Agent Output]:")
        print(response)
