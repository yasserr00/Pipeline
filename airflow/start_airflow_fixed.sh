#!/bin/bash

# Fixed Airflow Startup Script
# This script properly initializes and starts Airflow

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_HOME="${PROJECT_ROOT}/airflow"

echo "🚀 Starting Airflow..."
echo "   Project Root: ${PROJECT_ROOT}"
echo "   Airflow Home: ${AIRFLOW_HOME}"
echo ""

# Set Airflow home
export AIRFLOW_HOME="${AIRFLOW_HOME}"

# Check if Airflow is installed
if ! command -v airflow &> /dev/null; then
    echo "❌ Airflow is not installed!"
    echo "   Please install it with: pip install apache-airflow==2.8.0"
    exit 1
fi

echo "✅ Airflow is installed"
echo ""

# Check if database exists
if [ ! -f "${AIRFLOW_HOME}/airflow.db" ]; then
    echo "📊 Database not found. Initializing..."
    airflow db init
    echo "✅ Database initialized"
    echo ""
    
    # Create admin user
    echo "👤 Creating admin user..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin 2>/dev/null || echo "⚠️  User might already exist"
    echo ""
fi

# Disable example DAGs if not already done
if grep -q "load_examples = True" "${AIRFLOW_HOME}/airflow.cfg" 2>/dev/null; then
    echo "🔧 Disabling example DAGs..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/load_examples = True/load_examples = False/' "${AIRFLOW_HOME}/airflow.cfg"
    else
        sed -i 's/load_examples = True/load_examples = False/' "${AIRFLOW_HOME}/airflow.cfg"
    fi
    echo "✅ Example DAGs disabled"
    echo ""
fi

echo "=" * 70
echo "Starting Airflow Services..."
echo "=" * 70
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping Airflow services..."
    kill $SCHEDULER_PID $WEBSERVER_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start scheduler in background
echo "📅 Starting scheduler..."
airflow scheduler &
SCHEDULER_PID=$!
echo "   Scheduler PID: $SCHEDULER_PID"
sleep 3

# Start webserver in background
echo "🌐 Starting webserver..."
airflow webserver --port 8080 &
WEBSERVER_PID=$!
echo "   Webserver PID: $WEBSERVER_PID"

echo ""
echo "✅ Airflow services started!"
echo ""
echo "🌐 Open your browser to: http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for processes
wait

