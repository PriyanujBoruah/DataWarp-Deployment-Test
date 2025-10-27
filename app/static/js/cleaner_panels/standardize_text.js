/**
 * standardize_text.js
 *
 * This module handles the UI and logic for the "Standardize Text"
 * panel in the Data Cleaner view. It provides one-click actions for
 * common text cleaning operations.
 */

/**
 * A generic handler for all standardization operations.
 * @param {string} operation - The specific operation to perform (e.g., 'trim_whitespace').
 * @param {HTMLButtonElement} button - The button that was clicked.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {string} targetColumn - The name of the column to process.
 * @param {Function} refreshCallback - Function to refresh the dashboard on success.
 * @param {HTMLElement} panel - The parent panel element, used to re-enable buttons.
 */
async function handleStandardize(operation, button, statusP, targetColumn, refreshCallback, panel) {
    // Disable just the clicked button to give immediate feedback
    button.disabled = true;
    button.textContent = 'Applying...';
    statusP.textContent = 'Processing request...';
    statusP.style.color = '#a095c0';

    try {
        const response = await fetch('/standardize_text_column', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ column_name: targetColumn, operation: operation })
        });
        const result = await response.json();
        if (result.error) throw new Error(result.error);

        statusP.textContent = result.message;
        statusP.style.color = '#28a745'; // Green for success
        if (window.updateUndoRedoStatus) window.updateUndoRedoStatus();
        if (refreshCallback) {
            refreshCallback();
        }

    } catch (error) {
        statusP.textContent = `Error: ${error.message}`;
        statusP.style.color = '#dc3545'; // Red for error
    } finally {
        // --- THIS IS THE FIX ---
        // This 'finally' block will run regardless of success or failure.
        // It finds ALL buttons within the panel and resets their state.
        if (panel) {
            panel.querySelectorAll('.action-btn').forEach(btn => {
                btn.disabled = false;
                const op = btn.dataset.op;
                if (op === 'trim_whitespace') btn.textContent = 'Trim Whitespace';
                else if (op === 'remove_punctuation') btn.textContent = 'Remove Punctuation';
                else if (op === 'remove_numbers') btn.textContent = 'Remove Numbers';
                else if (op === 'remove_special_chars') btn.textContent = 'Remove Special Characters';
            });
        }
        // --- END OF FIX ---
    }
}

/**
 * The main export function for this module.
 * @param {HTMLElement} panel - The sidebar panel element to render content into.
 * @param {Array} columnsWithTypes - The list of all columns in the dataset.
 * @param {string} targetColumn - The specific column selected by the user.
 * @param {Function} refreshCallback - The function to call to refresh the main dashboard.
 */
export function render(panel, columnsWithTypes, targetColumn, refreshCallback) {
    panel.innerHTML = `
        <h3>Standardize Text: <strong style="color:white;">${targetColumn}</strong></h3>
        
        <div class="options-group">
            <p class="panel-description">Apply a single cleaning function to all values in this column.</p>
            
            <button data-op="trim_whitespace" class="action-btn" style="margin-top: 5px;">Trim Whitespace</button>
            <p class="panel-description" style="margin-top: 5px; font-size: 11px;">Removes spaces from the beginning and end.</p>
        </div>

        <div class="options-group">
            <button data-op="remove_punctuation" class="action-btn">Remove Punctuation</button>
            <p class="panel-description" style="margin-top: 5px; font-size: 11px;">Removes characters like !, ?, ., etc.</p>
        </div>

        <div class="options-group">
            <button data-op="remove_numbers" class="action-btn">Remove Numbers</button>
            <p class="panel-description" style="margin-top: 5px; font-size: 11px;">Removes all digits 0-9.</p>
        </div>
        
        <div class="options-group">
            <button data-op="remove_special_chars" class="action-btn">Remove Special Characters</button>
            <p class="panel-description" style="margin-top: 5px; font-size: 11px;">Removes any non-alphanumeric characters.</p>
        </div>

        <p id="standardize-status" class="status-message"></p>
    `;

    // --- Attach Event Listeners ---
    const statusP = document.getElementById('standardize-status');
    
    panel.querySelectorAll('.action-btn').forEach(button => {
        button.addEventListener('click', () => {
            const operation = button.dataset.op;
            // Pass the `panel` element itself to the handler function so it knows
            // which buttons to find and re-enable.
            handleStandardize(operation, button, statusP, targetColumn, refreshCallback, panel);
        });
    });
}