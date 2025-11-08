#!/bin/bash

# Start Airflow Services
# This script starts both the scheduler and webserver

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_HOME="${PROJECT_ROOT}/airflow"

export AIRFLOW_HOME="${AIRFLOW_HOME}"

echo "🚀 Starting Airflow services..."
echo "   Airflow Home: ${AIRFLOW_HOME}"
echo ""

# Check if database exists
if [ ! -f "${AIRFLOW_HOME}/airflow.db" ]; then
    echo "⚠️  Database not found. Running setup..."
    bash "${PROJECT_ROOT}/airflow/setup_airflow.sh"
    echo ""
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping Airflow services..."
    kill $SCHEDULER_PID $WEBSERVER_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start scheduler in background
echo "📅 Starting scheduler..."
airflow scheduler &
SCHEDULER_PID=$!
echo "   Scheduler PID: $SCHEDULER_PID"

# Wait a bit for scheduler to start
sleep 3

# Start webserver in background
echo "🌐 Starting webserver..."
airflow webserver --port 8080 &
WEBSERVER_PID=$!
echo "   Webserver PID: $WEBSERVER_PID"

echo ""
echo "✅ Airflow services started!"
echo "   Scheduler: Running (PID: $SCHEDULER_PID)"
echo "   Webserver: Running (PID: $WEBSERVER_PID)"
echo ""
echo "🌐 Open your browser to: http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for processes
wait

