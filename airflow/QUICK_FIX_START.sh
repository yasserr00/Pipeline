#!/bin/bash

# Quick Fix to Start Airflow
# Run this script to fix common issues and start Airflow

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_HOME="${PROJECT_ROOT}/airflow"

echo "🔧 Quick Fix for Airflow Startup"
echo ""

# Set AIRFLOW_HOME
export AIRFLOW_HOME="${AIRFLOW_HOME}"
echo "✅ AIRFLOW_HOME set to: ${AIRFLOW_HOME}"

# Stop any existing processes
echo ""
echo "🛑 Stopping any existing Airflow processes..."
pkill -f "airflow scheduler" 2>/dev/null || true
pkill -f "airflow webserver" 2>/dev/null || true
sleep 2

# Check if database exists
if [ ! -f "${AIRFLOW_HOME}/airflow.db" ]; then
    echo ""
    echo "📊 Initializing database..."
    airflow db init
    echo "✅ Database initialized"
    
    echo ""
    echo "👤 Creating admin user..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin 2>/dev/null || echo "⚠️  User might already exist"
else
    echo "✅ Database exists"
fi

# Disable examples
if grep -q "load_examples = True" "${AIRFLOW_HOME}/airflow.cfg" 2>/dev/null; then
    echo ""
    echo "🔧 Disabling example DAGs..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/load_examples = True/load_examples = False/' "${AIRFLOW_HOME}/airflow.cfg"
    else
        sed -i 's/load_examples = True/load_examples = False/' "${AIRFLOW_HOME}/airflow.cfg"
    fi
    echo "✅ Example DAGs disabled"
fi

echo ""
echo "=" * 70
echo "✅ Setup complete! Now starting Airflow..."
echo "=" * 70
echo ""
echo "📝 Instructions:"
echo "   1. Open a NEW terminal window"
echo "   2. Run: export AIRFLOW_HOME=${AIRFLOW_HOME}"
echo "   3. Run: airflow scheduler"
echo ""
echo "   4. Open ANOTHER terminal window"
echo "   5. Run: export AIRFLOW_HOME=${AIRFLOW_HOME}"
echo "   6. Run: airflow webserver --port 8080"
echo ""
echo "   7. Open browser: http://localhost:8080"
echo "      Username: admin"
echo "      Password: admin"
echo ""

