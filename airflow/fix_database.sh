#!/bin/bash

# Fix Airflow Database Issues
# This script resets and reinitializes the Airflow database

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_HOME="${PROJECT_ROOT}/airflow"

export AIRFLOW_HOME="${AIRFLOW_HOME}"

echo "🔧 Fixing Airflow Database..."
echo "   Airflow Home: ${AIRFLOW_HOME}"
echo ""

# Stop any running Airflow processes
echo "🛑 Stopping any running Airflow processes..."
pkill -f "airflow scheduler" 2>/dev/null || true
pkill -f "airflow webserver" 2>/dev/null || true
sleep 2

# Remove existing database
if [ -f "${AIRFLOW_HOME}/airflow.db" ]; then
    echo "🗑️  Removing existing database..."
    rm -f "${AIRFLOW_HOME}/airflow.db"
    echo "✅ Database removed"
fi

# Remove any lock files
rm -f "${AIRFLOW_HOME}/*.lock" 2>/dev/null || true

# Initialize database
echo ""
echo "📊 Initializing Airflow database..."
airflow db init

if [ $? -eq 0 ]; then
    echo "✅ Database initialized successfully!"
else
    echo "❌ Database initialization failed!"
    exit 1
fi

# Create admin user
echo ""
echo "👤 Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin 2>/dev/null || echo "⚠️  User might already exist (this is okay)"

echo ""
echo "=" * 70
echo "✅ Database fix complete!"
echo "=" * 70
echo ""
echo "📝 You can now start Airflow:"
echo "   1. airflow scheduler"
echo "   2. airflow webserver --port 8080"
echo ""

