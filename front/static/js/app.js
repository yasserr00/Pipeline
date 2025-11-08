/**
 * ML Model Predictor - Main JavaScript
 * Handles API communication, form submission, and UI updates
 */

const API_URL = 'http://localhost:5050';

// DOM Elements
const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('resultSection');
const errorMessage = document.getElementById('errorMessage');
const statusDiv = document.getElementById('status');

/**
 * Check API connection status
 */
async function checkConnection() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        if (data.status === 'healthy') {
            statusDiv.className = 'status connected';
            statusDiv.innerHTML = `✅ Connected to ${data.model} model (${data.problem_type})`;
        } else {
            throw new Error('API not healthy');
        }
    } catch (error) {
        statusDiv.className = 'status disconnected';
        statusDiv.innerHTML = `❌ Cannot connect to API. Make sure the server is running: <code>python serve_model.py</code>`;
    }
}

/**
 * Handle form submission and prediction
 */
async function handlePrediction(e) {
    e.preventDefault();
    
    // Hide previous results and errors
    resultSection.classList.remove('show');
    errorMessage.classList.remove('show');
    
    // Show loading
    loading.classList.add('show');
    predictBtn.disabled = true;

    try {
        // Collect form data
        const formData = new FormData(form);
        const features = {};
        
        // Convert form data to features object
        for (const [key, value] of formData.entries()) {
            const numValue = parseFloat(value);
            features[key] = isNaN(numValue) ? value : numValue;
        }

        // Add interaction feature if needed
        const squareFeet = parseFloat(formData.get('square_feet'));
        const bedrooms = parseFloat(formData.get('bedrooms'));
        if (squareFeet && bedrooms) {
            features['square_feet_x_bedrooms'] = squareFeet * bedrooms;
        }

        console.log('Sending prediction request:', { features });
        console.log('API URL:', `${API_URL}/predict`);

        // Make API call
        let response;
        try {
            response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ features })
            });
            console.log('Response status:', response.status, response.statusText);
        } catch (fetchError) {
            // Network error (server not running, CORS, etc.)
            console.error('Fetch error:', fetchError);
            throw new Error(`Cannot connect to server. Make sure the server is running at ${API_URL}. Error: ${fetchError.message}`);
        }

        // Check if response is ok
        if (!response.ok) {
            let errorMsg = `Server error (${response.status})`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.error || errorMsg;
            } catch (e) {
                errorMsg = await response.text() || errorMsg;
            }
            throw new Error(errorMsg);
        }

        const data = await response.json();

        // Display results
        if (data.prediction !== undefined) {
            const prediction = parseFloat(data.prediction);
            document.getElementById('predictionValue').textContent = 
                `$${prediction.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            
            // Show probabilities if available (classification)
            if (data.probabilities) {
                const probSection = document.getElementById('probabilitiesSection');
                const probList = document.getElementById('probabilitiesList');
                probSection.style.display = 'block';
                probList.innerHTML = '';
                
                for (const [label, prob] of Object.entries(data.probabilities)) {
                    const probBar = document.createElement('div');
                    probBar.className = 'prob-bar';
                    probBar.innerHTML = `
                        <div class="prob-label">${label}</div>
                        <div class="prob-bar-container">
                            <div class="prob-bar-fill" style="width: ${prob * 100}%">${(prob * 100).toFixed(1)}%</div>
                        </div>
                        <div class="prob-value">${(prob * 100).toFixed(2)}%</div>
                    `;
                    probList.appendChild(probBar);
                }
            } else {
                document.getElementById('probabilitiesSection').style.display = 'none';
            }
            
            resultSection.classList.add('show');
        }

    } catch (error) {
        errorMessage.textContent = `❌ Error: ${error.message}`;
        errorMessage.classList.add('show');
    } finally {
        loading.classList.remove('show');
        predictBtn.disabled = false;
    }
}

/**
 * Reset form to default values
 */
function resetForm() {
    form.reset();
    resultSection.classList.remove('show');
    errorMessage.classList.remove('show');
    // Reset to default values
    document.getElementById('square_feet').value = 2000;
    document.getElementById('bedrooms').value = 3;
    document.getElementById('bathrooms').value = 2;
    document.getElementById('age').value = 10;
    document.getElementById('location_score').value = 7.5;
    document.getElementById('garage').value = 1;
    document.getElementById('pool').value = 0;
    document.getElementById('school_rating').value = 4;
    document.getElementById('crime_rate').value = 30;
    document.getElementById('distance_to_city').value = 15;
    document.getElementById('lot_size').value = 8000;
    document.getElementById('year_built').value = 2010;
    document.getElementById('energy_efficiency').value = 1;
    document.getElementById('neighborhood').value = 2;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Check connection on page load
    checkConnection();
    setInterval(checkConnection, 30000); // Check every 30 seconds

    // Event listeners
    form.addEventListener('submit', handlePrediction);
    resetBtn.addEventListener('click', resetForm);
});

