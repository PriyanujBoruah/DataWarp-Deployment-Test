/**
 * scaling.js
 * 
 * This module handles the UI and logic for the "Feature Scaling"
 * panel in the Data Cleaner view. It provides several common scaling
 * methods to transform numeric columns.
 */

/**
 * Handles the asynchronous API call to the backend to apply a scaling method.
 * @param {HTMLButtonElement} button - The button that was clicked.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {string} targetColumn - The specific column to be scaled.
 * @param {Function} refreshCallback - The function to refresh the dashboard on success.
 */
async function handleScaling(button, statusP, targetColumn, refreshCallback) {
    const method = document.getElementById('scaling-method').value;

    button.disabled = true;
    button.textContent = 'Scaling...';
    statusP.textContent = 'Applying scaler...';

    try {
        const response = await fetch('/feature_scaling', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Send the targetColumn to the backend.
            // The backend is designed to scale all numeric cols if this is null,
            // but our current UI always provides it.
            body: JSON.stringify({ method: method, column_name: targetColumn })
        });
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        statusP.textContent = result.message;
        if (window.updateUndoRedoStatus) window.updateUndoRedoStatus();
        // Refresh the main dashboard to show the scaled data
        if (refreshCallback) {
            refreshCallback();
        }

    } catch(error) {
        statusP.textContent = `Error: ${error.message}`;
    } finally {
        button.disabled = false;
        button.textContent = `Apply Scaling to ${targetColumn}`;
    }
}

/**
 * The main export function for this module.
 * It builds the HTML for the sidebar panel and attaches event listeners.
 * @param {HTMLElement} panel - The sidebar panel element to render content into.
 * @param {Array} columnsWithTypes - The list of all columns in the dataset.
 * @param {string} targetColumn - The specific column selected by the user from a card.
 * @param {Function} refreshCallback - The function to call to refresh the main dashboard.
 */
export function render(panel, columnsWithTypes, targetColumn, refreshCallback) {
    // This panel is context-aware. It receives the targetColumn directly
    // from the card click, so no further column selection is needed.
    
    panel.innerHTML = `
        <h3>Feature Scaling</h3>
        <div class="options-group">
            <p style="color: #a095c0; font-size: 14px;">
                You are applying a scaling method to the column:
                <strong style="color: white;">${targetColumn}</strong>
            </p>
            <p style="color: #a095c0; font-size: 12px; margin-top: 15px;">
                This transforms the data to a common scale, which can improve the performance of many machine learning algorithms.
            </p>
        </div>
        <div class="options-group">
            <label for="scaling-method">Select a Scaling Method</label>
            <select id="scaling-method">
                <option value="standard">StandardScaler</option>
                <option value="minmax">MinMaxScaler (scales to [0, 1])</option>
                <option value="robust">RobustScaler (handles outliers)</option>
                <option value="absmax">Absolute Maximum Scaling (scales to [-1, 1])</option>
                <option value="mean_norm">Mean Normalization</option>
                <option value="vector_norm">Vector Normalization (L2)</option>
            </select>
        </div>
        <button id="scaling-btn" class="action-btn">Apply Scaling to ${targetColumn}</button>
        <p id="scaling-status" style="color:#a095c0; margin-top:15px; text-align:center;"></p>
    `;

    // --- Attach Event Listeners ---
    const scalingBtn = document.getElementById('scaling-btn');
    const statusP = document.getElementById('scaling-status');

    if (scalingBtn) {
        scalingBtn.addEventListener('click', () => handleScaling(scalingBtn, statusP, targetColumn, refreshCallback));
    }
}