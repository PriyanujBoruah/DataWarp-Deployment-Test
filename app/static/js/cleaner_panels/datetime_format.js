/**
 * datetime_format.js
 *
 * This module handles the UI and logic for the "Format Datetime"
 * panel in the Data Cleaner view. It provides tools for standardizing
 * inconsistent date/time strings and for extracting date or time components.
 */

/**
 * Handles the API call for formatting a datetime column.
 * It reads the state of the 'dayfirst' checkbox to handle ambiguous formats.
 * @param {HTMLButtonElement} button - The button that was clicked.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {string} targetColumn - The name of the column to format.
 * @param {Function} refreshCallback - Function to refresh the dashboard on success.
 */
async function handleFormat(button, statusP, targetColumn, refreshCallback) {
    // Get the state of the new checkbox to handle DD/MM/YYYY formats
    const dayfirst = document.getElementById('dayfirst-checkbox').checked;

    button.disabled = true;
    button.textContent = 'Formatting...';
    statusP.textContent = 'Attempting to standardize formats...';
    statusP.style.color = '#a095c0';

    try {
        const response = await fetch('/format_datetime_column', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Add the dayfirst flag to the payload sent to the backend
            body: JSON.stringify({
                column_name: targetColumn,
                dayfirst: dayfirst
            })
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
        button.disabled = false;
        button.textContent = 'Attempt to Fix All Formats';
    }
}

/**
 * Handles the API call for extracting the date or time part from a datetime column.
 * @param {HTMLButtonElement} button - The button that was clicked.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {string} targetColumn - The name of the column to extract from.
 * @param {Function} refreshCallback - Function to refresh the dashboard on success.
 */
async function handleExtract(button, statusP, targetColumn, refreshCallback) {
    const part_to_extract = document.getElementById('dt-extract-part').value;

    button.disabled = true;
    button.textContent = 'Extracting...';
    statusP.textContent = 'Applying extraction...';
    statusP.style.color = '#a095c0';

    try {
        const response = await fetch('/extract_datetime_part', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                column_name: targetColumn,
                part_to_extract: part_to_extract
            })
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
        button.textContent = 'Apply Extraction';
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
        <h3>Format Datetime: <strong style="color:white;">${targetColumn}</strong></h3>

        <!-- 1. Standardize Format Section -->
        <div class="options-group">
            <label>Standardize Format</label>
            <p class="panel-description">Attempt to parse all values into a consistent datetime format (YYYY-MM-DD HH:MM:SS).</p>

            <div class="inline-checkbox" style="margin-bottom: 15px; padding: 10px; border-radius: 5px;">
                <input type="checkbox" id="dayfirst-checkbox">
                <label for="dayfirst-checkbox">Assume Day-First Format (e.g., DD/MM/YYYY)</label>
            </div>

            <button id="format-btn" class="action-btn">Attempt to Fix All Formats</button>
            <p id="format-status" class="status-message"></p>
        </div>
        <hr class="panel-divider">

        <!-- 2. Extract Component Section -->
        <div class="options-group">
            <label>Extract Component</label>
            <p class="panel-description">Replace the column with just its date or time component. This will change the column's data type.</p>
            <select id="dt-extract-part">
                <option value="date">Date Only (e.g., 2025-10-10)</option>
                <option value="time">Time Only (e.g., 18:00:00)</option>
            </select>
            <button id="extract-btn" class="action-btn">Apply Extraction</button>
            <p id="extract-status" class="status-message"></p>
        </div>
    `;

    // --- Attach Event Listeners ---
    const formatBtn = document.getElementById('format-btn');
    const formatStatus = document.getElementById('format-status');
    formatBtn.addEventListener('click', () => handleFormat(formatBtn, formatStatus, targetColumn, refreshCallback));

    const extractBtn = document.getElementById('extract-btn');
    const extractStatus = document.getElementById('extract-status');
    extractBtn.addEventListener('click', () => handleExtract(extractBtn, extractStatus, targetColumn, refreshCallback));
}