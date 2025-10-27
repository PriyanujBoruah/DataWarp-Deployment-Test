/**
 * encoding.js
 * 
 * This module handles the UI and logic for the "Categorical Encoding"
 * panel in the Data Cleaner view. It provides options for both One-Hot
 * and Hashing encoding methods for a single target column.
 */

/**
 * A generic handler for both encoding methods. It gathers the required info
 * and calls the backend API to perform the transformation.
 * @param {string} method - The encoding method ('onehot' or 'hashing').
 * @param {HTMLButtonElement} button - The button that was clicked.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {string} targetColumn - The name of the column to encode.
 * @param {Function} refreshCallback - The function to refresh the dashboard on success.
 */
async function handleEncoding(method, button, statusP, targetColumn, refreshCallback) {
    // 1. Validate that a target column is actually provided.
    if (!targetColumn) {
        statusP.textContent = 'Error: No target column specified.';
        return;
    }

    // 2. Update the UI to a loading state.
    button.disabled = true;
    button.textContent = 'Encoding...';
    statusP.textContent = 'Applying encoding and rebuilding dataset...';

    try {
        // 3. Make the API call to the backend.
        const response = await fetch('/categorical_encoding', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ method, column_name: targetColumn })
        });
        const result = await response.json();
        
        // 4. Handle any errors returned from the server.
        if (result.error) {
            throw new Error(result.error);
        }
        
        // 5. Display the success message.
        statusP.textContent = result.message;
        if (window.updateUndoRedoStatus) window.updateUndoRedoStatus();
        // 6. If the operation was successful, call the refresh function
        //    to rebuild the main dashboard with the new data structure.
        if (refreshCallback) {
            refreshCallback();
        }

    } catch (error) {
        // 7. Display any caught errors.
        statusP.textContent = `Error: ${error.message}`;
    } finally {
        // 8. No need to re-enable the button, as the panel will be rebuilt
        //    on a successful refresh. If it fails, the user can try again.
        if (button) { // Check if button still exists
            button.disabled = false;
            button.textContent = `Apply ${method === 'onehot' ? 'One-Hot' : 'Hashing'} Encoding`;
        }
    }
}

/**
 * The main export function for this module.
 * It builds the HTML for the sidebar panel and attaches the necessary event listeners.
 * @param {HTMLElement} panel - The sidebar panel element to render content into.
 * @param {Array} columnsWithTypes - The list of all columns in the dataset.
 * @param {string} targetColumn - The specific column selected by the user.
 * @param {Function} refreshCallback - The function to call to refresh the main dashboard.
 */
export function render(panel, columnsWithTypes, targetColumn, refreshCallback) {
    // Define the HTML structure for this panel, confirming the target column.
    panel.innerHTML = `
        <h3>Categorical Encoding</h3>
        <div class="options-group">
            <p style="color: #a095c0; font-size: 14px;">
                You are encoding the column: 
                <strong style="color: white;">${targetColumn}</strong>
            </p>
        </div>
        
        <div class="options-group">
            <label>One-Hot Encoding (for low-cardinality)</label>
            <p style="font-size: 12px; color: #a095c0; margin-top: -5px; margin-bottom: 10px;">
                Creates a new column for each unique category. Best for columns with few distinct values.
            </p>
            <button id="onehot-btn" class="action-btn" style="margin-top:10px;">Apply One-Hot Encoding</button>
        </div>

        <hr style="border-color: #2C2541; margin: 30px 0;">

        <div class="options-group">
            <label>Hashing Encoding (for high-cardinality)</label>
            <p style="font-size: 12px; color: #a095c0; margin-top: -5px; margin-bottom: 10px;">
                Converts categories into a fixed number of numerical features. Good for columns with many unique values.
            </p>
            <button id="hashing-btn" class="action-btn" style="margin-top:10px;">Apply Hashing Encoding</button>
        </div>

        <p id="encoding-status" style="color:#a095c0; margin-top:15px; text-align:center;"></p>
    `;

    // --- Attach Event Listeners ---
    // Get references to the newly created elements within this panel.
    const oneHotBtn = document.getElementById('onehot-btn');
    const hashingBtn = document.getElementById('hashing-btn');
    const statusP = document.getElementById('encoding-status');

    // These listeners are now correctly inside the render function's scope,
    // so they have access to `targetColumn` and `refreshCallback`.
    if (oneHotBtn) {
        oneHotBtn.addEventListener('click', () => handleEncoding('onehot', oneHotBtn, statusP, targetColumn, refreshCallback));
    }
    if (hashingBtn) {
        hashingBtn.addEventListener('click', () => handleEncoding('hashing', hashingBtn, statusP, targetColumn, refreshCallback));
    }
}