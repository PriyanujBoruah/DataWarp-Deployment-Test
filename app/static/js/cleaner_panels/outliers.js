/**
 * outliers.js
 * 
 * This module handles the UI and logic for the "Outlier Management"
 * panel in the Data Cleaner view. It provides a diagnostic dashboard with
 * sensitivity analysis charts and controls for clipping or trimming outliers.
 */

let previewCharts = {}; // Store chart instances to prevent memory leaks

/**
 * Renders a sensitivity analysis line chart in a given canvas.
 * @param {string} id - The base ID for the chart canvas (e.g., 'iqr').
 * @param {string} xAxisLabel - The label for the X-axis (e.g., 'IQR Multiplier').
 * @param {Array} data - The array of {threshold, count} objects from the API.
 * @returns {Chart|null} The new Chart.js instance.
 */
function renderSensitivityChart(id, xAxisLabel, data) {
    const container = document.getElementById(`container-${id}`);
    if (container) {
        container.innerHTML = `<canvas id="preview-chart-${id}"></canvas>`;
    }
    const ctx = document.getElementById(`preview-chart-${id}`)?.getContext('2d');
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.threshold),
            datasets: [{
                label: 'Outlier Count',
                data: data.map(d => d.count),
                borderColor: '#9D4EDD',
                backgroundColor: 'rgba(157, 78, 221, 0.2)',
                fill: true,
                tension: 0.1,
                pointBackgroundColor: '#FFFFFF',
                pointRadius: 4
            }]
        },
        options: {
            plugins: {
                legend: { display: false },
                title: { display: true, text: `Sensitivity to ${xAxisLabel}`, color: '#E0D9F0', font: { size: 14 } }
            },
            scales: {
                x: { 
                    title: { display: true, text: xAxisLabel, color: '#a095c0' },
                    ticks: { color: '#a095c0' } 
                },
                y: { 
                    beginAtZero: true, 
                    title: { display: true, text: 'No. of Outliers Detected', color: '#a095c0' },
                    ticks: { color: '#a095c0' } 
                }
            }
        }
    });
}

/**
 * Handles the asynchronous API call to the backend to apply outlier changes.
 * @param {HTMLElement} statusP - The paragraph element for status messages.
 * @param {HTMLButtonElement} button - The button that triggered the action.
 * @param {string} columnName - The name of the column to process.
 * @param {Function} refreshCallback - The function to refresh the main dashboard.
 */
async function handleOutlierApply(statusP, button, columnName, refreshCallback) {
    const definition = document.getElementById('outlier-definition').value;
    const threshold = parseFloat(document.getElementById('outlier-threshold').value);
    const action = document.querySelector('.toggle-switch-label.active').dataset.action;

    if (!columnName) {
        statusP.textContent = 'Error: No target column specified.';
        return;
    }

    button.disabled = true;
    button.textContent = 'Processing...';
    statusP.textContent = 'Applying changes...';

    try {
        const response = await fetch('/handle_outliers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ column_name: columnName, definition, threshold, action })
        });
        const result = await response.json();
        if (result.error) throw new Error(result.error);
        
        statusP.textContent = result.message;
        if (window.updateUndoRedoStatus) window.updateUndoRedoStatus();
        if (refreshCallback) refreshCallback();

    } catch(error) {
        statusP.textContent = `Error: ${error.message}`;
    } finally {
        button.disabled = false;
        button.textContent = `Handle Outliers for ${columnName}`;
    }
}

/**
 * The main export function for this module.
 * It builds the HTML for the sidebar panel and attaches all event listeners.
 * @param {HTMLElement} panel - The sidebar panel element to render content into.
 * @param {Array} columnsWithTypes - The list of all columns in the dataset.
 * @param {string} targetColumn - The specific column selected by the user from the dashboard card.
 * @param {Function} refreshCallback - The function to call to refresh the main dashboard.
 */
export function render(panel, columnsWithTypes, targetColumn, refreshCallback) {
    panel.innerHTML = `
        <h3>Outlier Management</h3>
        <div class="options-group">
            <p style="color: #a095c0; font-size: 14px;">
                You are managing outliers for the column: 
                <strong style="color: white;">${targetColumn}</strong>
            </p>
        </div>
        
        <div id="outlier-controls">
            <hr style="border-color: #2C2541; margin: 20px 0;">
            <div class="options-group">
                <label for="outlier-definition">1. Choose Method & Threshold</label>
                <select id="outlier-definition">
                    <option value="iqr">IQR (Interquartile Range)</option>
                    <option value="zscore">Z-score</option>
                    <option value="percentile">Percentile</option>
                </select>
            </div>
            <div class="options-group">
                <label id="threshold-label" for="outlier-threshold">Multiplier</label>
                <input type="range" id="outlier-threshold">
                <div style="text-align:center; font-weight:bold; color:white; margin-top:5px;" id="threshold-value"></div>
            </div>
            <div class="options-group">
                <label>2. Choose Action</label>
                <div class="toggle-switch" id="outlier-action-toggle">
                    <div class="toggle-switch-button" id="toggle-button"></div>
                    <div class="toggle-switch-label active" data-action="clip">Clip</div>
                    <div class="toggle-switch-label" data-action="trim">Trim</div>
                </div>
            </div>
            <button id="outlier-btn" class="action-btn">Handle Outliers for ${targetColumn}</button>
            <p id="outlier-status" style="color:#a095c0; margin-top:15px; text-align:center;"></p>
        </div>
    `;

    // --- Attach Event Listeners ---
    const outlierPreviewContainer = document.getElementById('outlier-preview-container');

    const loadPreviews = async (selectedColumn) => {
        if (!selectedColumn) {
            outlierPreviewContainer.style.display = 'none';
            Object.values(previewCharts).forEach(chart => chart.destroy());
            return;
        }
        
        outlierPreviewContainer.style.display = 'flex';
        document.querySelectorAll('.preview-chart-container').forEach(c => c.innerHTML = '<p style="text-align:center;color:#a095c0;">Running analysis...</p>');
        
        try {
            const response = await fetch(`/outlier_previews/${encodeURIComponent(selectedColumn)}`);
            const result = await response.json();
            if (result.error) throw new Error(result.error);
            Object.values(previewCharts).forEach(chart => chart.destroy());

            if (result.previews.iqr) previewCharts['iqr'] = renderSensitivityChart('iqr', 'IQR Multiplier', result.previews.iqr);
            if (result.previews.zscore) previewCharts['zscore'] = renderSensitivityChart('zscore', 'Z-score Threshold', result.previews.zscore);
            if (result.previews.percentile) previewCharts['percentile'] = renderSensitivityChart('percentile', 'Percentile Threshold', result.previews.percentile);

        } catch (error) {
             outlierPreviewContainer.innerHTML = `<p style="color:#ffc107; padding: 20px;">Error: ${error.message}</p>`;
        }
    };

    // --- Dynamic Slider and Toggle Logic ---
    const outlierDefinitionSelect = document.getElementById('outlier-definition');
    const thresholdLabel = document.getElementById('threshold-label');
    const thresholdSlider = document.getElementById('outlier-threshold');
    const thresholdValue = document.getElementById('threshold-value');
    const outlierActionToggle = document.getElementById('outlier-action-toggle');
    const outlierBtn = document.getElementById('outlier-btn');
    const statusP = document.getElementById('outlier-status');

    const updateSlider = () => {
        const method = outlierDefinitionSelect.value;
        if (method === 'iqr') {
            thresholdLabel.textContent = 'Multiplier (1.5 is standard)';
            thresholdSlider.min = 1; thresholdSlider.max = 3; thresholdSlider.step = 0.1; thresholdSlider.value = 1.5;
        } else if (method === 'zscore') {
            thresholdLabel.textContent = 'Z-score Threshold (3.0 is standard)';
            thresholdSlider.min = 2; thresholdSlider.max = 4; thresholdSlider.step = 0.1; thresholdSlider.value = 3.0;
        } else if (method === 'percentile') {
            thresholdLabel.textContent = 'Threshold (e.g., 0.01 for 1%/99%)';
            thresholdSlider.min = 0.001; thresholdSlider.max = 0.1; thresholdSlider.step = 0.001; thresholdSlider.value = 0.01;
        }
        thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(3);
    };
    
    outlierDefinitionSelect.addEventListener('change', updateSlider);
    thresholdSlider.addEventListener('input', () => { thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(3); });
    
    outlierActionToggle.addEventListener('click', (event) => {
        if (event.target.classList.contains('toggle-switch-label')) {
            const action = event.target.dataset.action;
            document.querySelectorAll('.toggle-switch-label').forEach(lbl => lbl.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('toggle-button').style.transform = (action === 'trim') ? 'translateX(100%)' : 'translateX(0%)';
        }
    });

    outlierBtn.addEventListener('click', () => handleOutlierApply(statusP, outlierBtn, targetColumn, refreshCallback));

    // --- Initial State ---
    loadPreviews(targetColumn);
    updateSlider();
}