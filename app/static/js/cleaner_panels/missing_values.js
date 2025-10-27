/**
 * missing_values.js
 * 
 * This module handles the UI and logic for the "Missing Value Imputation"
 * panel in the Data Cleaner view. It provides standard methods for numeric/categorical
 * columns and a global, exportable "Smart-Impute" feature.
 */

/**
 * Polls for job progress and updates the UI accordingly.
 * @param {string} jobId - The ID of the job to poll.
 * @param {HTMLButtonElement} button - The button that triggered the action.
 * @param {HTMLElement} progressBar - The progress bar element.
 * @param {HTMLElement} progressText - The text element inside the progress bar.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {Function} onComplete - Callback function on job success.
 */

/**
 * Handles the asynchronous API call for standard imputation methods on a single column.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {HTMLButtonElement} button - The button that was clicked.
 * @param {string} targetColumn - The name of the column to process.
 * @param {Function} refreshCallback - Function to refresh the dashboard on success.
 */
async function handleStandardImpute(statusP, button, targetColumn, refreshCallback) {
    const method = document.getElementById('impute-method-select').value;
    const valueInput = document.getElementById('specific-value-input');
    const specificValue = valueInput ? valueInput.value : null;

    button.disabled = true;
    button.textContent = 'Applying...';
    statusP.textContent = 'Processing missing values...';
    statusP.style.color = '#a095c0';

    try {
        const response = await fetch('/handle_missing_values', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                column_name: targetColumn,
                method: method,
                value: specificValue
            }),
            // --- ADDED for robustness ---
            credentials: 'same-origin'
        });
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error);
        }
        
        statusP.textContent = result.message;
        statusP.style.color = '#28a745';
        if (window.updateUndoRedoStatus) window.updateUndoRedoStatus();
        if (refreshCallback) {
            refreshCallback();
        }

    } catch (error) {
        statusP.textContent = `Error: ${error.message}`;
        statusP.style.color = '#dc3545';
    } finally {
        button.disabled = false;
        button.textContent = 'Apply Method';
    }
}


/**
 * Handles the asynchronous API call to the backend for smart imputation.
 * This function is EXPORTED so other modules can use it.
 * @param {HTMLElement} statusP - The paragraph element to display status messages.
 * @param {HTMLButtonElement} button - The button that triggered the action.
 * @param {Function} refreshCallback - The function to call to refresh the main dashboard on success.
 */
// --- 2. REPLACE your `handleSmartImpute` function with this one ---
export async function handleSmartImpute(statusP, button, refreshCallback) {
    button.disabled = true;
    button.textContent = 'Submitting...';
    statusP.textContent = 'Sending job to the background worker...';
    statusP.style.color = '#a095c0';

    try {
        const response = await fetch('/smart_impute', { 
            method: 'POST',
            credentials: 'same-origin' // Also good practice to add here
        });
        if (response.status !== 202) {
            const errorResult = await response.json();
            throw new Error(errorResult.error || 'Failed to start the job.');
        }
        
        const result = await response.json();
        
        const jobStartedEvent = new CustomEvent('job-started', {
            detail: {
                jobId: result.job_id,
                jobName: 'Smart-Impute'
            }
        });
        document.dispatchEvent(jobStartedEvent);

        button.textContent = 'Job is running in the background...';
        statusP.textContent = 'You can now navigate to other pages.';

    } catch(error) {
        statusP.textContent = `Error: ${error.message}`;
        statusP.style.color = '#dc3545';
        button.disabled = false;
        button.textContent = 'Smart-Impute All Columns';
    }
}

/**
 * The main export function for this module.
 * It builds the HTML for the sidebar panel and attaches the necessary event listeners.
 * @param {HTMLElement} panel - The sidebar panel element to render content into.
 * @param {Array} columnsWithTypes - The list of all columns in the dataset.
 * @param {string} targetColumn - The specific column selected by the user (if any).
 * @param {Function} refreshCallback - The function to call to refresh the main dashboard.
 */
export function render(panel, columnsWithTypes, targetColumn, refreshCallback) {
    // Find the full column object to determine its type
    const column = columnsWithTypes.find(c => c.name === targetColumn);
    const isNumeric = column && column.type === 'numeric';

    // Define the different sets of options for the dropdown menu
    const numericMethods = `
        <option value="median">Median</option>
        <option value="mean">Mean</option>
        <option value="mode">Mode</option>
        <option value="specific_value">Specific Value</option>
        <option value="ffill">Forward Fill (Next)</option>
        <option value="bfill">Backward Fill (Previous)</option>
        <option value="drop_rows">Drop Rows with Missing</option>
    `;
    const categoricalMethods = `
        <option value="mode">Mode</option>
        <option value="ffill">Forward Fill (Next)</option>
        <option value="bfill">Backward Fill (Previous)</option>
        <option value="specific_value">Specific Value</option>
        <option value="drop_rows">Drop Rows with Missing</option>
    `;

    // --- Main HTML structure for the panel ---
    panel.innerHTML = `
        <h3>Missing Value Imputation</h3>
        
        <!-- Section 1: Standard, single-column methods -->
        <div class="options-group">
            <label>Standard Methods for <strong style="color:white;">${targetColumn}</strong></label>
            <p class="panel-description">
                Choose a simple strategy to fill missing values only in this column.
            </p>
            <select id="impute-method-select">
                ${isNumeric ? numericMethods : categoricalMethods}
            </select>
            <input type="text" id="specific-value-input" placeholder="Enter value..." style="display:none; margin-top:10px;">
            <button id="standard-impute-btn" class="action-btn">Apply Method</button>
            <p id="standard-impute-status" class="status-message"></p>
        </div>

        <hr class="panel-divider">

        <!-- Section 2: AI-powered, multi-column method with progress bar -->
        <div class="options-group">
            <label>AI-Powered Smart Imputation</label>
            <p class="panel-description">
                Automatically find and fill all missing values in the entire dataset using a K-Nearest Neighbors approach. This is a background task.
            </p>
            <button id="smart-impute-btn" class="action-btn">Smart-Impute All Columns</button>
            
            <p id="impute-status" class="status-message"></p>
        </div>
    `;

    // --- Get references to all the new elements in the DOM ---
    const smartImputeBtn = document.getElementById('smart-impute-btn');
    const smartImputeStatusP = document.getElementById('impute-status');
    
    const standardImputeBtn = document.getElementById('standard-impute-btn');
    const standardImputeStatusP = document.getElementById('standard-impute-status');
    const methodSelect = document.getElementById('impute-method-select');
    const valueInput = document.getElementById('specific-value-input');

    // --- Attach event listeners to the interactive elements ---

    // Listener for the Smart Impute button
    if (smartImputeBtn) {
        smartImputeBtn.addEventListener('click', () => {
            // Call the handler, passing the elements it needs to control
            handleSmartImpute(smartImputeStatusP, smartImputeBtn, refreshCallback);
        });
    }
    
    // Listener for the Standard Impute button
    if (standardImputeBtn) {
        standardImputeBtn.addEventListener('click', () => {
            // Call the handler, passing its own UI elements and the target column
            handleStandardImpute(standardImputeStatusP, standardImputeBtn, targetColumn, refreshCallback);
        });
    }

    // Listener for the dropdown menu to show/hide the specific value input
    if (methodSelect && valueInput) {
        methodSelect.addEventListener('change', () => {
            valueInput.style.display = (methodSelect.value === 'specific_value') ? 'block' : 'none';
        });
    }
}