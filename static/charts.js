function renderChart(canvasId, labels, data, label, color) {
    new Chart(document.getElementById(canvasId), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: color,
                borderColor: color,
                fill: false,
                tension: 0.1
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

function renderBarChart(canvasId, labels, data, label, color) {
    new Chart(document.getElementById(canvasId), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: color,
                borderColor: color,
                borderWidth: 1
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}
