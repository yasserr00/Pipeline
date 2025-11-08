#!/bin/bash

# Airflow Setup Script
# This script initializes Airflow for the ML Pipeline project

set -e  # Exit on error

echo "🚀 Setting up Apache Airflow..."
echo ""

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRFLOW_HOME="${PROJECT_ROOT}/airflow"

echo "📁 Project Root: ${PROJECT_ROOT}"
echo "📁 Airflow Home: ${AIRFLOW_HOME}"
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

# Create necessary directories
echo "📂 Creating directories..."
mkdir -p "${AIRFLOW_HOME}/dags"
mkdir -p "${AIRFLOW_HOME}/logs"
mkdir -p "${AIRFLOW_HOME}/plugins"
mkdir -p "${AIRFLOW_HOME}/config"
echo "✅ Directories created"
echo ""

# Check if database exists
if [ -f "${AIRFLOW_HOME}/airflow.db" ]; then
    echo "⚠️  Database already exists at ${AIRFLOW_HOME}/airflow.db"
    read -p "Do you want to reset it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing database..."
        rm -f "${AIRFLOW_HOME}/airflow.db"
        echo "✅ Database removed"
    else
        echo "📊 Using existing database"
    fi
fi

# Initialize Airflow database
echo ""
echo "📊 Initializing Airflow database..."
airflow db init

# Check if user exists
echo ""
echo "👤 Checking for admin user..."
if airflow users list | grep -q "admin"; then
    echo "✅ Admin user already exists"
else
    echo "👤 Creating admin user..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin
    echo "✅ Admin user created"
    echo "   Username: admin"
    echo "   Password: admin"
fi

echo ""
echo "=" * 70
echo "✅ Airflow setup complete!"
echo "=" * 70
echo ""
echo "📝 Next steps:"
echo "   1. Start scheduler: airflow scheduler"
echo "   2. Start webserver: airflow webserver --port 8080"
echo "   3. Open browser: http://localhost:8080"
echo ""
echo "💡 Tip: Add this to your ~/.zshrc or ~/.bashrc:"
echo "   export AIRFLOW_HOME=${AIRFLOW_HOME}"
echo ""

