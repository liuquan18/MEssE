// ICON ML Training Monitor - Dashboard JavaScript

// Configuration
const CONFIG = {
    updateInterval: 5000, // 5 seconds
    historyLimit: 100,
    apiBaseUrl: window.location.origin
};

// Chart instances
let charts = {
    history: null,
    distribution: null,
    batch: null,
    timestepStats: null
};

// Global state
let state = {
    isConnected: false,
    lastUpdate: null,
    currentData: null
};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    startDataUpdates();
    setupEventListeners();
    logMessage('Dashboard initialized', 'success');
});

// Setup event listeners
function setupEventListeners() {
    const historyLimit = document.getElementById('history-limit');
    if (historyLimit) {
        historyLimit.addEventListener('change', (e) => {
            CONFIG.historyLimit = parseInt(e.target.value);
            updateDashboard();
            logMessage(`History limit changed to ${CONFIG.historyLimit}`, 'success');
        });
    }
}

// Initialize all charts
function initializeCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                labels: {
                    color: '#94a3b8'
                }
            }
        },
        scales: {
            x: {
                ticks: { color: '#94a3b8' },
                grid: { color: '#475569' }
            },
            y: {
                ticks: { color: '#94a3b8' },
                grid: { color: '#475569' }
            }
        }
    };

    // Loss History Chart
    const historyCtx = document.getElementById('lossHistoryChart').getContext('2d');
    charts.history = new Chart(historyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Mean Loss',
                    data: [],
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Min Loss',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Max Loss',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: chartOptions
    });

    // Loss Distribution Chart
    const distCtx = document.getElementById('lossDistributionChart').getContext('2d');
    charts.distribution = new Chart(distCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Frequency',
                data: [],
                backgroundColor: 'rgba(124, 58, 237, 0.6)',
                borderColor: '#7c3aed',
                borderWidth: 1
            }]
        },
        options: {
            ...chartOptions,
            scales: {
                ...chartOptions.scales,
                y: {
                    ...chartOptions.scales.y,
                    beginAtZero: true
                }
            }
        }
    });

    // Batch Loss Chart
    const batchCtx = document.getElementById('batchLossChart').getContext('2d');
    charts.batch = new Chart(batchCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Batch Loss',
                data: [],
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.2)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                pointRadius: 3
            }]
        },
        options: chartOptions
    });

    // Timestep Statistics Chart
    const statsCtx = document.getElementById('timestepStatsChart').getContext('2d');
    charts.timestepStats = new Chart(statsCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Mean',
                    data: [],
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.3)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Mean: ${context.parsed.y.toFixed(6)}`;
                        }
                    }
                }
            }
        }
    });
}

// Start periodic data updates
function startDataUpdates() {
    updateDashboard();
    setInterval(updateDashboard, CONFIG.updateInterval);
}

// Update entire dashboard
async function updateDashboard() {
    try {
        // Fetch all data in parallel
        const [status, current, history, statistics] = await Promise.all([
            fetchAPI('/api/status'),
            fetchAPI('/api/loss/current'),
            fetchAPI('/api/loss/history', { limit: CONFIG.historyLimit }),
            fetchAPI('/api/loss/statistics')
        ]);

        // Update state
        state.isConnected = true;
        state.lastUpdate = new Date();
        state.currentData = { status, current, history, statistics };

        // Update UI components
        updateStatusBar(status);
        updateStatCards(current, statistics);
        updateCharts(history, current, statistics);
        updateStatisticsTable(statistics);

        // Update connection status
        document.getElementById('connection-status').textContent = 'Connected';
        document.getElementById('connection-status').style.color = '#10b981';

    } catch (error) {
        console.error('Error updating dashboard:', error);
        state.isConnected = false;
        document.getElementById('connection-status').textContent = 'Disconnected';
        document.getElementById('connection-status').style.color = '#ef4444';
        logMessage(`Error: ${error.message}`, 'error');
    }
}

// Fetch data from API
async function fetchAPI(endpoint, params = {}) {
    const url = new URL(`${CONFIG.apiBaseUrl}${endpoint}`);
    Object.keys(params).forEach(key => url.searchParams.append(key, params[key]));
    
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
}

// Update status bar
function updateStatusBar(status) {
    const lastUpdate = document.getElementById('last-update');
    if (lastUpdate) {
        lastUpdate.textContent = `Last update: ${new Date().toLocaleTimeString()}`;
    }
}

// Update statistic cards
function updateStatCards(current, statistics) {
    // Current Loss
    if (current && !current.error) {
        document.getElementById('current-loss').textContent = 
            current.latest_loss !== null ? current.latest_loss.toFixed(6) : '--';
        
        if (statistics && statistics.first_loss && current.latest_loss) {
            const change = ((current.latest_loss - statistics.first_loss) / statistics.first_loss * 100);
            const changeElem = document.getElementById('loss-change');
            changeElem.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}% from start`;
            changeElem.className = `stat-change ${change < 0 ? 'positive' : 'negative'}`;
        }
        
        document.getElementById('mean-loss').textContent = current.mean_loss.toFixed(6);
    }

    // Total Timesteps and Batches
    if (statistics && !statistics.error) {
        document.getElementById('total-timesteps').textContent = statistics.total_timesteps || '--';
        document.getElementById('total-batches').textContent = 
            `${statistics.total_batches || '--'} batches`;
        
        // Trend
        const trendElem = document.getElementById('trend');
        if (statistics.trend) {
            trendElem.textContent = statistics.trend === 'decreasing' ? '↓ Decreasing' : '↑ Increasing';
            trendElem.style.color = statistics.trend === 'decreasing' ? '#10b981' : '#ef4444';
        }
        
        const improvementElem = document.getElementById('improvement');
        if (statistics.improvement !== null) {
            improvementElem.textContent = `Δ ${statistics.improvement.toFixed(6)}`;
        }
    }
}

// Update all charts
function updateCharts(history, current, statistics) {
    if (!history || history.error) return;

    // Update History Chart
    if (history.timestamps && history.mean_losses) {
        // Format timestamps for display
        const labels = history.timestamps.map(ts => {
            const date = new Date(ts);
            return date.toLocaleTimeString();
        });

        charts.history.data.labels = labels;
        charts.history.data.datasets[0].data = history.mean_losses;
        charts.history.data.datasets[1].data = history.min_losses;
        charts.history.data.datasets[2].data = history.max_losses;
        charts.history.update('none');
    }

    // Update Distribution Chart (histogram of all losses)
    if (history.all_losses) {
        const bins = createHistogram(history.all_losses, 20);
        charts.distribution.data.labels = bins.labels;
        charts.distribution.data.datasets[0].data = bins.counts;
        charts.distribution.update('none');
        
        document.getElementById('dist-info').textContent = 
            `${history.all_losses.length} data points`;
    }

    // Update Batch Loss Chart (current timestep)
    if (current && current.num_batches) {
        // Fetch current timestep batch data
        fetchAPI(`/api/loss/by_timestep/${encodeURIComponent(current.timestamp)}`)
            .then(data => {
                if (data && data.batch_indices) {
                    charts.batch.data.labels = data.batch_indices;
                    charts.batch.data.datasets[0].data = data.losses;
                    charts.batch.update('none');
                    
                    document.getElementById('batch-info').textContent = 
                        `${data.num_batches} batches at ${current.timestamp}`;
                }
            });
    }

    // Update Timestep Statistics Chart
    if (history.timestamps && history.mean_losses) {
        const labels = history.timestamps.map(ts => {
            const date = new Date(ts);
            return date.toLocaleTimeString();
        });

        charts.timestepStats.data.labels = labels;
        charts.timestepStats.data.datasets[0].data = history.mean_losses;
        charts.timestepStats.update('none');
    }
}

// Create histogram bins
function createHistogram(data, numBins) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / numBins;
    
    const bins = Array(numBins).fill(0);
    const labels = [];
    
    // Create bin labels
    for (let i = 0; i < numBins; i++) {
        const binStart = min + i * binWidth;
        labels.push(binStart.toFixed(4));
    }
    
    // Count data points in each bin
    data.forEach(value => {
        const binIndex = Math.min(Math.floor((value - min) / binWidth), numBins - 1);
        bins[binIndex]++;
    });
    
    return { labels, counts: bins };
}

// Update statistics table
function updateStatisticsTable(statistics) {
    const tbody = document.getElementById('stats-tbody');
    if (!statistics || statistics.error) {
        tbody.innerHTML = '<tr><td colspan="2">No data available</td></tr>';
        return;
    }

    const rows = [
        { label: 'Total Batches', value: statistics.total_batches },
        { label: 'Total Timesteps', value: statistics.total_timesteps },
        { label: 'Overall Mean', value: statistics.overall_mean?.toFixed(6) },
        { label: 'Overall Std Dev', value: statistics.overall_std?.toFixed(6) },
        { label: 'Overall Min', value: statistics.overall_min?.toFixed(6) },
        { label: 'Overall Max', value: statistics.overall_max?.toFixed(6) },
        { label: '25th Percentile', value: statistics.percentile_25?.toFixed(6) },
        { label: '50th Percentile (Median)', value: statistics.percentile_50?.toFixed(6) },
        { label: '75th Percentile', value: statistics.percentile_75?.toFixed(6) },
        { label: '90th Percentile', value: statistics.percentile_90?.toFixed(6) },
        { label: '95th Percentile', value: statistics.percentile_95?.toFixed(6) },
        { label: '99th Percentile', value: statistics.percentile_99?.toFixed(6) },
        { label: 'First Loss', value: statistics.first_loss?.toFixed(6) },
        { label: 'Latest Loss', value: statistics.latest_loss?.toFixed(6) },
        { label: 'Improvement', value: statistics.improvement?.toFixed(6) },
        { label: 'Trend', value: statistics.trend },
        { label: 'Trend Slope', value: statistics.trend_slope?.toExponential(4) }
    ];

    tbody.innerHTML = rows.map(row => 
        `<tr><td>${row.label}</td><td>${row.value ?? '--'}</td></tr>`
    ).join('');
}

// Log message to activity log
function logMessage(message, type = 'info') {
    const logContainer = document.getElementById('log-container');
    const timestamp = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${timestamp}] ${message}`;
    
    logContainer.insertBefore(entry, logContainer.firstChild);
    
    // Keep only last 50 entries
    while (logContainer.children.length > 50) {
        logContainer.removeChild(logContainer.lastChild);
    }
}

// Export for debugging
window.dashboardDebug = {
    state,
    charts,
    config: CONFIG,
    updateDashboard,
    logMessage
};
