/**
 * datetime_impute.js
 *
 * This module handles the UI for imputing missing values in DATETIME columns.
 * It provides datetime-specific methods and re-uses the global "Smart-Impute" feature.
 */

// --- Import the shared function from the other module ---
import { handleSmartImpute } from "./missing_values.js";

/**
 * Handles the API call for imputing missing datetime values using specific methods.
 * @param {HTMLButtonElement} button - The button that was clicked.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {string} targetColumn - The name of the column to impute.
 * @param {Function} refreshCallback - Function to refresh the dashboard on success.
 */
async function handleImpute(button, statusP, targetColumn, refreshCallback) {
    const method = document.getElementById('dt-impute-method').value;
    const valueInput = document.getElementById('dt-specific-value');
    const specificValue = valueInput ? valueInput.value : null;

    button.disabled = true;
    button.textContent = 'Imputing...';
    statusP.textContent = 'Applying imputation...';
    statusP.style.color = '#a095c0';

    try {
        const response = await fetch('/handle_datetime_impute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ column_name: targetColumn, method: method, value: specificValue })
        });
        const result = await response.json();
        if (result.error) throw new Error(result.error);

        statusP.textContent = result.message;
        statusP.style.color = '#28a745';
        if (window.updateUndoRedoStatus) window.updateUndoRedoStatus();
        if (refreshCallback) refreshCallback();
    } catch (error) {
        statusP.textContent = `Error: ${error.message}`;
        statusP.style.color = '#dc3545';
    } finally {
        button.disabled = false;
        button.textContent = 'Apply Imputation';
    }
}

/**
 * The main export function for this module.
 * It builds the HTML for the sidebar panel and attaches event listeners.
 * @param {HTMLElement} panel - The sidebar panel element to render content into.
 * @param {Array} columnsWithTypes - The list of all columns in the dataset.
 * @param {string} targetColumn - The specific column selected by the user.
 * @param {Function} refreshCallback - The function to call to refresh the main dashboard.
 */
export function render(panel, columnsWithTypes, targetColumn, refreshCallback) {
    panel.innerHTML = `
        <h3>Impute Datetime: <strong style="color:white;">${targetColumn}</strong></h3>
        
        <div class="options-group">
            <label>Standard Methods for this Datetime Column</label>
            <p class="panel-description">
                Choose a simple strategy to fill missing values in this column.
            </p>
            <select id="dt-impute-method">
                <option value="median">Median</option>
                <option value="mean">Mean</option>
                <option value="mode">Mode (Most Frequent)</option>
                <option value="interpolation">Interpolate (Time-based)</option>
                <option value="specific_value">Specific Datetime</option>
                <option value="ffill">Forward Fill</option>
                <option value="bfill">Backward Fill</option>
                <option value="drop_rows">Drop Rows</option>
            </select>
            <input type="text" id="dt-specific-value" placeholder="e.g., 2023-01-01 14:30:00" style="display:none; margin-top:10px;">
            <button id="impute-btn" class="action-btn">Apply Imputation</button>
            <p id="impute-status" class="status-message"></p>
        </div>

        <hr class="panel-divider">

        <div class="options-group">
            <label>AI-Powered Smart Imputation</label>
            <p class="panel-description">Fills all missing values in the dataset using a K-Nearest Neighbors approach.</p>
            <button id="smart-impute-btn" class="action-btn">Smart-Impute All Columns</button>
            
            <!-- This HTML block is the only new addition -->
            <div id="smart-impute-progress-container" style="display: none; width: 100%; background-color: #1f1f1f; border-radius: 5px; margin-top: 15px; border: 1px solid #2e2e2e;">
                <div id="smart-impute-progress-bar" style="width: 0%; height: 24px; background-color: #4a3f6e; border-radius: 5px; transition: width 0.4s ease;">
                    <span id="smart-impute-progress-text" style="display: flex; align-items: center; justify-content: center; height: 100%; color: white; font-size: 12px; font-weight: 500;">0%</span>
                </div>
            </div>
            
            <p id="smart-impute-status" class="status-message"></p>
        </div>
    `;

    const imputeBtn = document.getElementById('impute-btn');
    const imputeStatus = document.getElementById('impute-status');
    const dtImputeMethod = document.getElementById('dt-impute-method');
    const dtSpecificValue = document.getElementById('dt-specific-value');

    dtImputeMethod.addEventListener('change', () => {
        dtSpecificValue.style.display = (dtImputeMethod.value === 'specific_value') ? 'block' : 'none';
    });
    
    imputeBtn.addEventListener('click', () => handleImpute(imputeBtn, imputeStatus, targetColumn, refreshCallback));
    
    // Attach the imported function to the Smart Impute button
    const smartImputeBtn = document.getElementById('smart-impute-btn');
    const smartImputeStatus = document.getElementById('smart-impute-status');
    if (smartImputeBtn) {
        smartImputeBtn.addEventListener('click', () => handleSmartImpute(smartImputeStatus, smartImputeBtn, refreshCallback));
    }
}