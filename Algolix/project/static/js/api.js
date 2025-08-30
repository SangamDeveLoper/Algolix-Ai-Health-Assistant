document.addEventListener('DOMContentLoaded', () => {
    // ... (rest of the script) ...

    // Handle the main symptom form submission
    if (symptomForm) {
        symptomForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // ... (loading state, payload creation) ...

            try {
                // MAKE THE FETCH CALL TO YOUR WEB APP'S PROXY ENDPOINT
                // This calls YOUR webapp/app.py, which is running on port 5001
                const response = await fetch('/api/predict', { 
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                displayPredictionResult(result);
                // ... (display messages) ...

            } catch (error) {
                console.error('Error getting prediction from proxy:', error);
                // IMPORTANT: Error message updated to guide the user if the AI API is down
                showDashboardMessage(`Error: ${error.message}. Please ensure you are logged in and both Flask apps are running.`, 'error');
            } finally {
                // ... (hide loading, re-enable button) ...
            }
        });
    }

    // ... (rest of the script) ...
});
