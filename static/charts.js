/**
 * EnerSense Chart.js Utility
 * Reusable chart rendering function for line, bar, and mixed charts.
 * 
 * Author: EnerSense Dev
 */

function renderChart(canvasId, chartType, labels, datasets, options = {}) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    return new Chart(ctx, {
        type: chartType,
        data: {
            labels: labels,
            datasets: datasets
        },
        options: Object.assign({
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                tooltip: { mode: 'index', intersect: false }
            },
            interaction: { mode: 'nearest', axis: 'x', intersect: false },
            scales: {
                x: { title: { display: true, text: 'Timestamp' } },
                y: { title: { display: true, text: 'kWh' }, beginAtZero: true }
            }
        }, options)
    });
}

/**
 * Helper to create styled datasets
 */
function createDataset(label, data, color, fill = false) {
    return {
        label: label,
        data: data,
        borderColor: color,
        backgroundColor: fill ? color.replace('1)', '0.2)') : color,
        borderWidth: 2,
        fill: fill,
        tension: 0.3
    };
}
