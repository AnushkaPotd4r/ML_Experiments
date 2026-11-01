"""
SQLite experiment logging database for ML Studio.
Maintains history of model runs, user query inputs, predictions, and timestamps.
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    algorithm_id TEXT NOT NULL,
                    algorithm_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    prediction_result TEXT NOT NULL,
                    metrics_summary TEXT
                )
            """)
    finally:
        conn.close()


def log_experiment(algorithm_id, algorithm_name, category, inputs, prediction, metrics_summary=None):
    """Log an execution of an algorithm to the history."""
    conn = None
    try:
        init_db()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db()
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiment_history (
                    timestamp, algorithm_id, algorithm_name, category, inputs_json, prediction_result, metrics_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                now_str,
                algorithm_id,
                algorithm_name,
                category,
                json.dumps(inputs),
                str(prediction),
                json.dumps(metrics_summary) if metrics_summary else None
            ))
    except Exception as e:
        print(f"Error logging experiment: {e}")
    finally:
        if conn:
            conn.close()


def get_recent_experiments(limit=10):
    """Retrieve recent experiment records."""
    conn = None
    try:
        init_db()
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, algorithm_id, algorithm_name, category, inputs_json, prediction_result, metrics_summary
            FROM experiment_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            results.append({
                'id': r['id'],
                'timestamp': r['timestamp'],
                'algorithm_id': r['algorithm_id'],
                'algorithm_name': r['algorithm_name'],
                'category': r['category'],
                'inputs': json.loads(r['inputs_json']) if r['inputs_json'] else {},
                'prediction_result': r['prediction_result'],
                'metrics_summary': json.loads(r['metrics_summary']) if r['metrics_summary'] else {}
            })
        return results
    except Exception as e:
        print(f"Error reading history: {e}")
        return []
    finally:
        if conn:
            conn.close()


def clear_history():
    """Clear all experiment history."""
    conn = None
    try:
        init_db()
        conn = get_db()
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiment_history")
    except Exception as e:
        print(f"Error clearing history: {e}")
    finally:
        if conn:
            conn.close()
