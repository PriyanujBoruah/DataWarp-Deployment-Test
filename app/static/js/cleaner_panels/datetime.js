/**
 * datetime.js
 *
 * This module handles the UI and logic for the "Datetime Cleaning"
 * panel in the Data Cleaner view. It provides tools for formatting,
 * imputing missing values, shifting time, and extracting components
 * for datetime columns.
 */

/**
 * Handles the API call for formatting a datetime column.
 * @param {HTMLButtonElement} button - The button that was clicked.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {string} targetColumn - The name of the column to format.
 * @param {Function} refreshCallback - Function to refresh the dashboard on success.
 */
async function handleFormat(button, statusP, targetColumn, refreshCallback) {
    button.disabled = true;
    button.textContent = 'Formatting...';
    statusP.textContent = 'Attempting to standardize formats...';
    statusP.style.color = '#a095c0';

    try {
        const response = await fetch('/format_datetime_column', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ column_name: targetColumn })
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
 * Handles the API call for imputing missing datetime values.
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
        button.disabled = false;
        button.textContent = 'Apply Imputation';
    }
}

/**
 * Handles the API call for shifting datetime values forward or backward.
 * @param {HTMLButtonElement} button - The button that was clicked.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {string} targetColumn - The name of the column to shift.
 * @param {Function} refreshCallback - Function to refresh the dashboard on success.
 */
async function handleShift(button, statusP, targetColumn, refreshCallback) {
    const direction = document.getElementById('dt-shift-direction').value;
    const hours = document.getElementById('dt-shift-hours').value || 0;
    const minutes = document.getElementById('dt-shift-minutes').value || 0;

    button.disabled = true;
    button.textContent = 'Applying Shift...';
    statusP.textContent = 'Shifting datetime values...';
    statusP.style.color = '#a095c0';

    try {
        const response = await fetch('/shift_datetime_column', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                column_name: targetColumn,
                direction: direction,
                hours: hours,
                minutes: minutes
            })
        });
        const result = await response.json();
        if (result.error) throw new Error(result.error);

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
        button.textContent = 'Apply Shift';
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
        if (refreshCallback) {
            refreshCallback();
        }

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
 * @param {string|null} focusSection - The ID of the section to scroll into view (e.g., 'impute', 'format', 'shift').
 */
export function render(panel, columnsWithTypes, targetColumn, refreshCallback, focusSection = null) {
    // Note: The 'panel-divider' and 'panel-description' classes should be defined in cleaner.html's CSS for consistent styling.
    panel.innerHTML = `
        <h3>Datetime Cleaning: <strong style="color:white;">${targetColumn}</strong></h3>

        <!-- 1. FORMATTING SECTION -->
        <div class="options-group" id="dt-section-format">
            <label>1. Standardize / Extract</label>
            <p class="panel-description">Attempt to parse all values into a consistent datetime format (YYYY-MM-DD HH:MM:SS).</p>
            <button id="format-btn" class="action-btn">Attempt to Fix All Formats</button>
            <p id="format-status" class="status-message"></p>
        
            <p class="panel-description" style="margin-top: 20px;">Or, replace the column with just its date or time component.</p>
            <select id="dt-extract-part" style="margin-top: 5px;">
                <option value="date">Date Only (e.g., 2025-10-10)</option>
                <option value="time">Time Only (e.g., 18:00:00)</option>
            </select>
            <button id="extract-btn" class="action-btn">Apply Extraction</button>
            <p id="extract-status" class="status-message"></p>
        </div>
        <hr class="panel-divider">

        <!-- 2. IMPUTATION SECTION -->
        <div class="options-group" id="dt-section-impute">
            <label>2. Impute Missing Values</label>
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

        <!-- 3. SHIFT DATETIME SECTION -->
        <div class="options-group" id="dt-section-shift">
            <label>3. Shift Datetime</label>
            <p class="panel-description">Move all datetime values forward or backward by a specified duration to correct offsets.</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; align-items: center;">
                <select id="dt-shift-direction">
                    <option value="forward">Forward by</option>
                    <option value="backward">Backward by</option>
                </select>
                <input type="number" id="dt-shift-hours" placeholder="Hours" min="0" value="0">
                <input type="number" id="dt-shift-minutes" placeholder="Minutes" min="0" value="0">
            </div>
            <button id="shift-btn" class="action-btn">Apply Shift</button>
            <p id="shift-status" class="status-message"></p>
        </div>
    `;

    // --- Scroll-to-focus logic ---
    if (focusSection) {
        // Use a slight delay to ensure the element is painted before scrolling
        setTimeout(() => {
            // Note: "Extract" is part of the "format" section visually.
            const sectionToFocus = focusSection === 'extract' ? 'format' : focusSection;
            const targetId = `dt-section-${sectionToFocus}`;
            const elementToFocus = document.getElementById(targetId);

            if (elementToFocus) {
                // Scroll the element into view
                elementToFocus.scrollIntoView({ behavior: 'smooth', block: 'start' });
                // Add a temporary highlight for better user feedback
                elementToFocus.style.transition = 'background-color 0.5s ease-out';
                elementToFocus.style.backgroundColor = '#4a3f6e'; // A subtle highlight color
                setTimeout(() => {
                    elementToFocus.style.backgroundColor = ''; // Reset after animation
                }, 1500);
            }
        }, 100);
    }

    // --- Attach Event Listeners ---
    const formatBtn = document.getElementById('format-btn');
    const imputeBtn = document.getElementById('impute-btn');
    const shiftBtn = document.getElementById('shift-btn');
    const extractBtn = document.getElementById('extract-btn');

    const formatStatus = document.getElementById('format-status');
    const imputeStatus = document.getElementById('impute-status');
    const shiftStatus = document.getElementById('shift-status');
    const extractStatus = document.getElementById('extract-status');

    const dtImputeMethod = document.getElementById('dt-impute-method');
    const dtSpecificValue = document.getElementById('dt-specific-value');

    // Show/hide the 'specific value' input based on the dropdown selection
    dtImputeMethod.addEventListener('change', () => {
        dtSpecificValue.style.display = (dtImputeMethod.value === 'specific_value') ? 'block' : 'none';
    });

    // Wire up all the buttons to their respective handler functions
    formatBtn.addEventListener('click', () => handleFormat(formatBtn, formatStatus, targetColumn, refreshCallback));
    imputeBtn.addEventListener('click', () => handleImpute(imputeBtn, imputeStatus, targetColumn, refreshCallback));
    shiftBtn.addEventListener('click', () => handleShift(shiftBtn, shiftStatus, targetColumn, refreshCallback));
    extractBtn.addEventListener('click', () => handleExtract(extractBtn, extractStatus, targetColumn, refreshCallback));
}