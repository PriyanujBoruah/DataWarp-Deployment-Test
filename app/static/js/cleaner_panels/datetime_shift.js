/**
 * datetime_shift.js
 *
 * This module handles the UI and logic for the "Shift Datetime"
 * panel in the Data Cleaner view. It provides tools for shifting
 * datetime values by both time (hours/minutes) and date (days/months/years).
 */

/**
 * Handles the API call for shifting datetime values by hours and minutes.
 * @param {HTMLButtonElement} button - The button that was clicked.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {string} targetColumn - The name of the column to shift.
 * @param {Function} refreshCallback - Function to refresh the dashboard on success.
 */
async function handleTimeShift(button, statusP, targetColumn, refreshCallback) {
    const direction = document.getElementById('dt-time-shift-direction').value;
    const hours = document.getElementById('dt-shift-hours').value || 0;
    const minutes = document.getElementById('dt-shift-minutes').value || 0;

    button.disabled = true;
    button.textContent = 'Applying Shift...';
    statusP.textContent = 'Shifting time values...';
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
        if (refreshCallback) refreshCallback();
    } catch (error) {
        statusP.textContent = `Error: ${error.message}`;
        statusP.style.color = '#dc3545';
    } finally {
        button.disabled = false;
        button.textContent = 'Apply Time Shift';
    }
}

/**
 * Handles the API call for shifting datetime values by days, months, or years.
 * @param {HTMLButtonElement} button - The button that was clicked.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {string} targetColumn - The name of the column to shift.
 * @param {Function} refreshCallback - Function to refresh the dashboard on success.
 */
async function handleDateShift(button, statusP, targetColumn, refreshCallback) {
    const direction = document.getElementById('dt-date-shift-direction').value;
    const amount = document.getElementById('dt-shift-amount').value || 0;
    const unit = document.getElementById('dt-shift-unit').value;

    button.disabled = true;
    button.textContent = 'Applying Shift...';
    statusP.textContent = 'Shifting date values...';
    statusP.style.color = '#a095c0';

    try {
        const response = await fetch('/shift_date_column', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                column_name: targetColumn,
                direction: direction,
                amount: amount,
                unit: unit
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
        button.textContent = 'Apply Date Shift';
    }
}

/**
 * The main export function for this module.
 * It builds the HTML for the sidebar panel for both time and date shifting.
 * @param {HTMLElement} panel - The sidebar panel element to render content into.
 * @param {Array} columnsWithTypes - The list of all columns in the dataset.
 * @param {string} targetColumn - The specific column selected by the user.
 * @param {Function} refreshCallback - The function to call to refresh the main dashboard.
 */
export function render(panel, columnsWithTypes, targetColumn, refreshCallback) {
    panel.innerHTML = `
        <h3>Shift Datetime: <strong style="color:white;">${targetColumn}</strong></h3>

        <!-- 1. SHIFT TIME SECTION -->
        <div class="options-group">
            <label>Shift Time</label>
            <p class="panel-description">Move time forward or backward by hours and minutes.</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; align-items: center;">
                <select id="dt-time-shift-direction">
                    <option value="forward">Forward by</option>
                    <option value="backward">Backward by</option>
                </select>
                <input type="number" id="dt-shift-hours" placeholder="Hours" min="0" value="0">
                <input type="number" id="dt-shift-minutes" placeholder="Minutes" min="0" value="0">
            </div>
            <button id="time-shift-btn" class="action-btn">Apply Time Shift</button>
            <p id="time-shift-status" class="status-message"></p>
        </div>
        <hr class="panel-divider">

        <!-- 2. SHIFT DATE SECTION -->
        <div class="options-group">
            <label>Shift Date</label>
            <p class="panel-description">Move date forward or backward by days, months, or years.</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; align-items: center;">
                <select id="dt-date-shift-direction">
                    <option value="forward">Forward by</option>
                    <option value="backward">Backward by</option>
                </select>
                <input type="number" id="dt-shift-amount" placeholder="Amount" min="0" value="0">
                <select id="dt-shift-unit">
                    <option value="days">Days</option>
                    <option value="months">Months</option>
                    <option value="years">Years</option>
                </select>
            </div>
            <button id="date-shift-btn" class="action-btn">Apply Date Shift</button>
            <p id="date-shift-status" class="status-message"></p>
        </div>
    `;

    // --- Attach Event Listeners ---
    const timeShiftBtn = document.getElementById('time-shift-btn');
    const timeShiftStatus = document.getElementById('time-shift-status');
    timeShiftBtn.addEventListener('click', () => handleTimeShift(timeShiftBtn, timeShiftStatus, targetColumn, refreshCallback));

    const dateShiftBtn = document.getElementById('date-shift-btn');
    const dateShiftStatus = document.getElementById('date-shift-status');
    dateShiftBtn.addEventListener('click', () => handleDateShift(dateShiftBtn, dateShiftStatus, targetColumn, refreshCallback));
}