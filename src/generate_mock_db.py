import sqlite3
import os

def create_mock_data():
    db_path = "twin_state.db"
    
    # Connect and create tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create runs table
    cursor.execute('''CREATE TABLE IF NOT EXISTS runs 
        (run_id INTEGER PRIMARY KEY, created_time TEXT, video TEXT, frame_range TEXT, fps INTEGER, calibration_json TEXT)''')
        
    # 2. Create detections table
    cursor.execute('''CREATE TABLE IF NOT EXISTS detections 
        (run_id INTEGER, frame_idx INTEGER, track_id INTEGER, class TEXT, confidence REAL, x1 REAL, y1 REAL, x2 REAL, y2 REAL, gx REAL, gy REAL, X REAL, Y REAL, quality_flags TEXT)''')
        
    # 3. Create kinematics table
    cursor.execute('''CREATE TABLE IF NOT EXISTS kinematics 
        (run_id INTEGER, frame_idx INTEGER, track_id INTEGER, x_m REAL, y_m REAL, vx REAL, vy REAL, ax REAL, ay REAL, speed_kmh REAL)''')
        
    # 4. Create alerts table
    cursor.execute('''CREATE TABLE IF NOT EXISTS alerts 
        (run_id INTEGER, alert_id INTEGER PRIMARY KEY, type TEXT, severity TEXT, frames TEXT, tracks TEXT, far_field TEXT, speed REAL, rise REAL, TTC REAL, gap REAL, message TEXT)''')

    # Insert a mock run entry
    cursor.execute("INSERT OR IGNORE INTO runs VALUES (1, '2026-10-19', 'Weast (1).mp4', '2730-3030', 30, '{}')")
    
    # Insert mock traffic data based on your project metrics
    cursor.execute("INSERT INTO kinematics VALUES (1, 2750, 9, 2.5, 15.0, 1.2, 5.4, 0.1, 0.2, 22.4)")
    cursor.execute("INSERT INTO kinematics VALUES (1, 2750, 18, 4.0, 12.0, -0.5, 6.1, -0.1, 0.4, 25.1)")
    
    # Insert a critical conflict alert from your report metrics
    cursor.execute("INSERT INTO alerts (run_id, type, severity, tracks, TTC, message) VALUES (1, 'conflict', 'critical', '9-18', 0.0, 'CONFLICT tracks 9-18 TTC 0.0 s')")
    cursor.execute("INSERT INTO alerts (run_id, type, severity, tracks, speed, message) VALUES (1, 'overspeed', 'warning', '22', 46.5, 'OVERSPEED track 22 Speed 46.5 kmh')")

    conn.commit()
    conn.close()
    print("[+] Successfully generated twin_state.db with mock traffic data tables!")

if __name__ == "__main__":
    create_mock_data()
