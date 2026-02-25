/**
 * Philosophical Text Analysis — SPA Router & View Manager
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const AppState = {
    texts: {},
    results: null,
    vizData: null,
    currentView: null,
    networkSimulation: null,    // D3 simulation reference
    selectedNode: null,         // currently selected network node
};

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

const Router = {
    routes: {
        '/upload':    renderUploadView,
        '/dashboard': renderDashboardView,
        '/temporal':  renderTemporalView,
        '/network':   renderNetworkView,
        '/compare':   renderCompareView,
    },

    init() {
        window.addEventListener('hashchange', () => this.navigate());
        if (!location.hash || location.hash === '#/') {
            location.hash = '#/upload';
        }
        this.navigate();
    },

    navigate() {
        const path = location.hash.slice(1) || '/upload';
        const renderFn = this.routes[path];
        if (!renderFn) { location.hash = '#/upload'; return; }

        // Stop any running D3 simulation
        if (AppState.networkSimulation) {
            AppState.networkSimulation.stop();
            AppState.networkSimulation = null;
        }

        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.view === path.slice(1));
        });

        document.getElementById('sidebar').classList.remove('open');
        document.getElementById('sidebar-overlay').classList.remove('active');

        AppState.currentView = path.slice(1);
        const main = document.getElementById('main-content');
        main.innerHTML = '';
        const view = document.createElement('div');
        view.className = 'view';
        main.appendChild(view);
        renderFn(view);

        requestAnimationFrame(() => {
            requestAnimationFrame(() => view.classList.add('active'));
        });
    },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const icon = type === 'success' ? '\u2713' : type === 'error' ? '\u2717' : '\u2139';
    toast.innerHTML = `<span class="toast-icon">${icon}</span> ${escHtml(message)}`;
    toast.className = `toast ${type} visible`;
    const delay = type === 'error' ? 5000 : 3000;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => toast.classList.remove('visible'), delay);
}

function showOverlay(show, status = '') {
    const overlay = document.getElementById('analysis-overlay');
    overlay.classList.toggle('active', show);
    if (status) document.getElementById('analysis-status').textContent = status;
}

function setProgress(pct) {
    const bar = document.getElementById('analysis-progress');
    bar.style.width = `${pct}%`;
    const progressBar = bar.closest('.progress-bar');
    if (progressBar) progressBar.setAttribute('aria-valuenow', Math.round(pct));
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    const hamburger = document.querySelector('.hamburger');
    const isOpen = sidebar.classList.toggle('open');
    overlay.classList.toggle('active', isOpen);
    if (hamburger) hamburger.setAttribute('aria-expanded', String(isOpen));
}

function fmt(n, d = 3) {
    if (n === null || n === undefined || isNaN(n)) return '—';
    return Number(n).toFixed(d);
}

// Alias for templates
const formatNumber = fmt;

function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

/** Normalize a value to 0-1 given a min and max */
function normalize(val, min, max) {
    return max === min ? 0.5 : (val - min) / (max - min);
}

// ---------------------------------------------------------------------------
// Chart palette & theme
// ---------------------------------------------------------------------------

const CHART_COLORS = [
    '#C9A96E', '#E5E5E7', '#8E8E93', '#D4BA85',
    '#64D2FF', '#BF5AF2', '#FF9F0A', '#30D158', '#FF375F',
];

function colorFor(index) {
    return CHART_COLORS[index % CHART_COLORS.length];
}

function plotlyDefaults(overrides = {}) {
    return Object.assign({
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            family: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
            color: '#E5E5E7',
            size: 13,
        },
        margin: { t: 40, r: 30, b: 50, l: 60 },
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            zerolinecolor: 'rgba(255,255,255,0.08)',
            tickfont: { color: '#A1A1A6', size: 11 },
        },
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            zerolinecolor: 'rgba(255,255,255,0.08)',
            tickfont: { color: '#A1A1A6', size: 11 },
        },
        legend: {
            font: { color: '#E5E5E7', size: 12 },
            bgcolor: 'rgba(0,0,0,0)',
            borderwidth: 0,
        },
        colorway: CHART_COLORS,
    }, overrides);
}

const PL_CFG = { displayModeBar: false, responsive: true };

// Shared metric definitions
const METRICS = [
    { key: 'first_order_coherence',  label: 'Coherence' },
    { key: 'second_order_coherence', label: '2nd order' },
    { key: 'syntactic_complexity',   label: 'Complexity' },
    { key: 'determiner_frequency',   label: 'Determiners' },
    { key: 'avg_sentence_length',    label: 'Sentence len' },
];

const METRIC_KEYS   = METRICS.map(m => m.key);
const METRIC_LABELS = METRICS.map(m => m.label);

// =====================================================================
// VIEW: Upload & Analyze
// =====================================================================

async function renderUploadView(container) {
    container.innerHTML = `
        <div class="page-header">
            <h1>Upload & analyze</h1>
            <p class="page-description">
                Upload philosophical texts or use sample data to discover patterns
                in different thinking styles using Latent Semantic Analysis.
            </p>
        </div>
        <div class="page-body">
            <div class="pill-group" style="margin-bottom: var(--space-6);">
                <button class="pill active" data-tab="upload" onclick="switchUploadTab('upload')">Upload files</button>
                <button class="pill" data-tab="paste" onclick="switchUploadTab('paste')">Paste text</button>
                <button class="pill" data-tab="samples" onclick="switchUploadTab('samples')">Sample texts</button>
            </div>

            <!-- Upload tab -->
            <div id="tab-upload" class="section">
                <div class="upload-zone glass-card no-hover" id="drop-zone"
                     ondragover="event.preventDefault(); this.classList.add('dragover')"
                     ondragleave="this.classList.remove('dragover')"
                     ondrop="handleDrop(event)"
                     onclick="document.getElementById('file-input').click()">
                    <div class="upload-zone-icon">
                        <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                            <path d="M24 6v28M14 16l10-10 10 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity="0.4"/>
                            <path d="M8 34v4a4 4 0 004 4h24a4 4 0 004-4v-4" stroke="currentColor" stroke-width="2" stroke-linecap="round" opacity="0.4"/>
                        </svg>
                    </div>
                    <div class="upload-zone-title">Drop text files here or click to browse</div>
                    <div class="upload-zone-subtitle">.txt files supported</div>
                </div>
                <input type="file" id="file-input" multiple accept=".txt" style="display:none" onchange="handleFiles(this.files)">
            </div>

            <!-- Paste tab -->
            <div id="tab-paste" class="section" style="display:none;">
                <div class="glass-card no-hover">
                    <div class="grid grid-2 gap-4" style="margin-bottom: var(--space-4);">
                        <div class="input-group">
                            <label class="input-label">Text name</label>
                            <input class="input" id="paste-name" type="text" placeholder="e.g. kant_critique">
                        </div>
                        <div class="input-group">
                            <label class="input-label">Author (optional)</label>
                            <input class="input" id="paste-author" type="text" placeholder="e.g. Immanuel Kant">
                        </div>
                    </div>
                    <div class="input-group">
                        <label class="input-label">Philosophical text</label>
                        <textarea class="input" id="paste-content" placeholder="Paste the philosophical text here..." style="min-height: 200px;"></textarea>
                    </div>
                    <button class="btn btn-secondary" onclick="submitPastedText()">Add text</button>
                </div>
            </div>

            <!-- Samples tab -->
            <div id="tab-samples" class="section" style="display:none;">
                <div class="glass-card no-hover" style="text-align:center; padding:var(--space-8);">
                    <h3 style="color:var(--color-text-primary); margin-bottom:var(--space-3);">Sample philosophical texts</h3>
                    <p class="text-secondary" style="max-width:500px; margin:0 auto var(--space-6);">
                        Load pre-selected excerpts from Nietzsche, Kant, and Hume to explore
                        the analysis capabilities.
                    </p>
                    <button class="btn btn-primary btn-lg" onclick="loadSamples()">Load samples</button>
                </div>
            </div>

            <!-- Loaded texts -->
            <div id="loaded-texts-section" class="section" style="display:none;">
                <h2 style="margin-bottom:var(--space-4);">Loaded texts</h2>
                <div class="glass-card no-hover">
                    <table class="data-table" id="texts-table">
                        <thead><tr><th>Name</th><th>Length</th><th></th></tr></thead>
                        <tbody id="texts-table-body"></tbody>
                    </table>
                </div>
            </div>

            <!-- Analysis config -->
            <div id="analysis-config" class="section" style="display:none;">
                <h2 style="margin-bottom:var(--space-4);">Analysis parameters</h2>
                <div class="glass-card no-hover">
                    <div class="grid grid-2 gap-5">
                        <div class="range-group">
                            <div class="range-header">
                                <label class="input-label">LSA components</label>
                                <span class="range-value" id="lsa-value">10</span>
                            </div>
                            <input type="range" min="5" max="100" value="10"
                                   oninput="document.getElementById('lsa-value').textContent=this.value">
                        </div>
                        <div class="range-group">
                            <div class="range-header">
                                <label class="input-label">Coherence window</label>
                                <span class="range-value" id="window-value">5</span>
                            </div>
                            <input type="range" min="2" max="10" value="5"
                                   oninput="document.getElementById('window-value').textContent=this.value">
                        </div>
                    </div>
                    <hr class="separator">
                    <button class="btn btn-primary btn-lg" style="width:100%;" onclick="runAnalysis()">Run analysis</button>
                </div>
            </div>
        </div>
    `;

    // Fetch current texts from server on page load
    try {
        const data = await API.getTexts();
        AppState.texts = data.texts || {};
    } catch { /* first visit, no texts yet */ }
    refreshTextsTable();
}

function switchUploadTab(tab) {
    ['upload', 'paste', 'samples'].forEach(t => {
        const el = document.getElementById(`tab-${t}`);
        if (el) el.style.display = t === tab ? '' : 'none';
    });
    document.querySelectorAll('.pill[data-tab]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
}

async function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    if (e.dataTransfer.files.length) await handleFiles(e.dataTransfer.files);
}

async function handleFiles(files) {
    try {
        const result = await API.uploadFiles(files);
        AppState.texts = result.texts || {};
        showToast(`Uploaded ${Object.keys(result.texts).length} file(s)`, 'success');
        refreshTextsTable();
    } catch (err) { showToast(err.message, 'error'); }
}

async function submitPastedText() {
    const name = document.getElementById('paste-name').value.trim();
    const content = document.getElementById('paste-content').value.trim();
    if (!name || !content) { showToast('Provide a name and text', 'error'); return; }
    try {
        const result = await API.submitText(name, content);
        AppState.texts = result.texts || {};
        document.getElementById('paste-name').value = '';
        document.getElementById('paste-content').value = '';
        showToast(`Added "${name}"`, 'success');
        refreshTextsTable();
    } catch (err) { showToast(err.message, 'error'); }
}

async function loadSamples() {
    try {
        const result = await API.loadSamples();
        AppState.texts = result.texts || {};
        showToast('Sample texts loaded', 'success');
        refreshTextsTable();
    } catch (err) { showToast(err.message, 'error'); }
}

async function removeText(name) {
    try {
        const result = await API.removeText(name);
        AppState.texts = result.texts || {};
        refreshTextsTable();
    } catch (err) { showToast(err.message, 'error'); }
}

function refreshTextsTable() {
    const keys = Object.keys(AppState.texts);
    const section = document.getElementById('loaded-texts-section');
    const config = document.getElementById('analysis-config');
    if (!section || !config) return;

    if (keys.length === 0) { section.style.display = 'none'; config.style.display = 'none'; return; }
    section.style.display = '';
    config.style.display = '';

    document.getElementById('texts-table-body').innerHTML = keys.map(name => `
        <tr>
            <td>${escHtml(name)}</td>
            <td class="text-secondary">${(AppState.texts[name] || '').length.toLocaleString()} chars</td>
            <td style="text-align:right;"><button class="btn btn-ghost btn-sm" onclick="removeText('${escHtml(name)}')">Remove</button></td>
        </tr>`).join('');
}

async function runAnalysis() {
    const lsa = parseInt(document.getElementById('lsa-value').textContent);
    const win = parseInt(document.getElementById('window-value').textContent);
    showOverlay(true, 'Preparing analysis pipeline...'); setProgress(10);

    try {
        setProgress(30); showOverlay(true, 'Running Latent Semantic Analysis...');
        const result = await API.runAnalysis({ lsa_components: lsa, coherence_window: win });
        setProgress(80); showOverlay(true, 'Generating visualizations...');
        AppState.results = result.results;
        AppState.vizData  = result.viz_data;
        setProgress(100); showOverlay(false);
        showToast('Analysis complete', 'success');
        location.hash = '#/dashboard';
    } catch (err) {
        showOverlay(false);
        showToast(`Analysis failed: ${err.message}`, 'error');
    }
}

// =====================================================================
// VIEW: Dashboard
// =====================================================================

async function renderDashboardView(container) {
    if (!AppState.vizData) {
        container.innerHTML = `<div class="page-header"><h1>Dashboard</h1></div><div class="page-body">${chartSkeleton(400)}</div>`;
        try { AppState.vizData = await API.getAllVizData(); } catch {
            container.innerHTML = emptyState('Dashboard', 'No analysis results yet', 'Upload texts and run an analysis to see your dashboard.', 'upload');
            return;
        }
    }

    const dashboard = AppState.vizData.dashboard || {};
    const philosophers = dashboard.philosophers || {};
    const stats = dashboard.stats || {};
    const names = Object.keys(philosophers);
    if (names.length === 0) {
        container.innerHTML = emptyState('Dashboard', 'No analysis results yet', 'Upload texts and run an analysis to see your dashboard.');
        return;
    }

    // Compute avg coherence if not provided
    const avgCoherence = stats.avg_coherence ||
        (names.reduce((s, n) => s + (philosophers[n].first_order_coherence || 0), 0) / names.length);

    container.innerHTML = `
        <div class="page-header">
            <h1>Dashboard</h1>
            <p class="page-description">Comparative overview of philosophical text analysis results.</p>
        </div>
        <div class="page-body">
            <div class="grid grid-4 stagger section" id="dash-metrics">
                <div class="glass-card metric-card">
                    <div class="metric-value" data-target="${names.length}" data-decimals="0">${names.length}</div>
                    <div class="metric-label">Philosophers</div>
                </div>
                <div class="glass-card metric-card">
                    <div class="metric-value" data-target="${stats.highest_coherence_value || 0}" data-decimals="3">${fmt(stats.highest_coherence_value)}</div>
                    <div class="metric-label">Highest coherence</div>
                    <div class="text-tertiary" style="margin-top:4px;font-size:var(--text-xs)">${stats.highest_coherence || ''}</div>
                </div>
                <div class="glass-card metric-card">
                    <div class="metric-value" data-target="${stats.most_complex_value || 0}" data-decimals="3">${fmt(stats.most_complex_value)}</div>
                    <div class="metric-label">Max complexity</div>
                    <div class="text-tertiary" style="margin-top:4px;font-size:var(--text-xs)">${stats.most_complex_syntax || ''}</div>
                </div>
                <div class="glass-card metric-card">
                    <div class="metric-value" data-target="${avgCoherence || 0}" data-decimals="3">${fmt(avgCoherence)}</div>
                    <div class="metric-label">Avg coherence</div>
                </div>
            </div>

            <div class="philosopher-legend section">
                ${names.map((n, i) => `<div class="legend-item"><span class="legend-dot" style="background:${colorFor(i)}"></span>${n}</div>`).join('')}
            </div>

            <div class="chart-controls" id="dash-controls">
                <div class="pill-group">
                    <button class="pill active" onclick="switchDashChart('radar')">Radar</button>
                    <button class="pill" onclick="switchDashChart('bar')">Bars</button>
                    <button class="pill" onclick="switchDashChart('scatter')">Scatter</button>
                    <button class="pill" onclick="switchDashChart('heatmap')">Heatmap</button>
                </div>
            </div>

            <div class="glass-card no-hover section">
                <div id="dashboard-chart" class="chart-container chart-container-lg"></div>
            </div>

            <div class="section">
                <h2 style="margin-bottom:var(--space-4);">Detailed metrics</h2>
                <div class="glass-card no-hover">
                    <table class="data-table"><thead><tr>
                        <th>Philosopher</th>${METRICS.map(m => `<th>${m.label}</th>`).join('')}
                    </tr></thead><tbody>
                        ${names.map(n => { const p = philosophers[n]; return `<tr><td>${n}</td>${METRIC_KEYS.map(k => `<td class="mono">${fmt(p[k], k === 'avg_sentence_length' ? 1 : 3)}</td>`).join('')}</tr>`; }).join('')}
                    </tbody></table>
                </div>
            </div>
        </div>`;

    // Animate metric counters on entry
    animateMetricValues(document.getElementById('dash-metrics'));

    renderDashboardChart('radar', philosophers);
}

function switchDashChart(mode) {
    document.querySelectorAll('#dash-controls .pill').forEach(p =>
        p.classList.toggle('active', p.textContent.toLowerCase() === mode));
    renderDashboardChart(mode, (AppState.vizData?.dashboard?.philosophers) || {});
}

function renderDashboardChart(mode, philosophers) {
    const el = document.getElementById('dashboard-chart');
    if (!el) return;
    const names = Object.keys(philosophers);
    if (names.length === 0) return;

    // Pre-compute per-metric min/max for normalization
    const ranges = {};
    METRIC_KEYS.forEach(k => {
        const vals = names.map(n => philosophers[n][k] || 0);
        ranges[k] = { min: Math.min(...vals), max: Math.max(...vals) };
    });

    if (mode === 'radar') {
        const traces = names.map((name, i) => {
            const p = philosophers[name];
            const norm = METRIC_KEYS.map(k => {
                const r = ranges[k];
                return r.max === r.min ? 0.5 : (p[k] || 0 - r.min) / (r.max - r.min || 1);
            });
            return {
                type: 'scatterpolar', r: [...norm, norm[0]], theta: [...METRIC_LABELS, METRIC_LABELS[0]],
                fill: 'toself', fillcolor: colorFor(i) + '15',
                line: { color: colorFor(i), width: 2.5, shape: 'spline' },
                marker: { size: 5, color: colorFor(i) },
                name,
            };
        });
        Plotly.newPlot(el, traces, plotlyDefaults({
            polar: {
                bgcolor: 'rgba(0,0,0,0)',
                radialaxis: { visible: true, gridcolor: 'rgba(255,255,255,0.06)', tickfont: { color: '#A1A1A6', size: 10 }, range: [0, 1.05] },
                angularaxis: { gridcolor: 'rgba(255,255,255,0.08)', tickfont: { color: '#E5E5E7', size: 11 } },
            },
            showlegend: true,
        }), PL_CFG);

    } else if (mode === 'bar') {
        const traces = names.map((name, i) => ({
            type: 'bar',
            x: METRIC_LABELS,
            y: METRIC_KEYS.map(k => philosophers[name][k] || 0),
            name,
            marker: { color: colorFor(i), opacity: 0.85, line: { color: colorFor(i), width: 1 } },
            texttemplate: '%{y:.3f}', textposition: 'auto',
            textfont: { color: '#E5E5E7', size: 10 },
        }));
        Plotly.newPlot(el, traces, plotlyDefaults({ barmode: 'group', showlegend: true }), PL_CFG);

    } else if (mode === 'scatter') {
        // Bubble scatter: x=coherence, y=complexity, size=sentence_length
        const sentLens = names.map(n => philosophers[n].avg_sentence_length || 15);
        const maxSent  = Math.max(...sentLens);
        const traces = names.map((name, i) => {
            const p = philosophers[name];
            const sl = p.avg_sentence_length || 15;
            return {
                type: 'scatter', mode: 'markers+text',
                x: [p.first_order_coherence || 0],
                y: [p.syntactic_complexity || 0],
                text: [name], textposition: 'top center', textfont: { color: '#E5E5E7', size: 11 },
                marker: {
                    color: colorFor(i), size: 12 + (sl / maxSent) * 30,
                    opacity: 0.85,
                    line: { color: '#fff', width: 1.5 },
                },
                name,
                hovertemplate: `<b>${name}</b><br>Coherence: %{x:.3f}<br>Complexity: %{y:.3f}<br>Avg sentence: ${fmt(sl, 1)}<extra></extra>`,
            };
        });
        Plotly.newPlot(el, traces, plotlyDefaults({
            xaxis: { title: { text: 'First-order coherence', font: { color: '#E5E5E7', size: 13 } } },
            yaxis: { title: { text: 'Syntactic complexity', font: { color: '#E5E5E7', size: 13 } } },
            showlegend: false,
        }), PL_CFG);

    } else if (mode === 'heatmap') {
        // Normalize each metric column independently for the heatmap
        const z = names.map(name =>
            METRIC_KEYS.map(k => {
                const r = ranges[k];
                return r.max === r.min ? 0.5 : ((philosophers[name][k] || 0) - r.min) / (r.max - r.min);
            })
        );
        // Text overlay with raw values
        const text = names.map(name => METRIC_KEYS.map(k => fmt(philosophers[name][k])));

        Plotly.newPlot(el, [{
            type: 'heatmap', z, x: METRIC_LABELS, y: names,
            text, texttemplate: '%{text}', textfont: { size: 11 },
            colorscale: [[0, '#0a0a0a'], [0.35, '#4a3a20'], [0.7, '#A08850'], [1, '#C9A96E']],
            showscale: true,
            colorbar: { tickfont: { color: '#A1A1A6' }, title: { text: 'Normalized', font: { color: '#A1A1A6', size: 11 } } },
        }], plotlyDefaults({ showlegend: false, margin: { t: 40, r: 80, b: 60, l: 100 } }), PL_CFG);
    }
}

// =====================================================================
// VIEW: Temporal Analysis
// =====================================================================

// Track current temporal display mode
let _temporalMode = 'temporal';

async function renderTemporalView(container) {
    if (!AppState.vizData) {
        container.innerHTML = `<div class="page-header"><h1>Temporal analysis</h1></div><div class="page-body">${chartSkeleton(400)}</div>`;
        try { AppState.vizData = await API.getAllVizData(); } catch {
            container.innerHTML = emptyState('Temporal analysis', 'No analysis results yet', 'Upload texts and run an analysis to explore coherence over time.', 'upload');
            return;
        }
    }
    const temporal = AppState.vizData?.temporal || {};
    const names = Object.keys(temporal);

    if (names.length === 0) {
        container.innerHTML = emptyState('Temporal analysis', 'No temporal data', 'Run an analysis to see coherence patterns over time.');
        return;
    }

    _temporalMode = 'temporal';

    container.innerHTML = `
        <div class="page-header">
            <h1>Temporal analysis</h1>
            <p class="page-description">How semantic coherence evolves across segments of each text.</p>
        </div>
        <div class="page-body">
            <div class="chart-controls">
                <div class="pill-group" id="temporal-pills">
                    <button class="pill active" onclick="selectTemporal('__ALL__')">All</button>
                    ${names.map(n => `<button class="pill" onclick="selectTemporal('${escHtml(n)}')">${n}</button>`).join('')}
                </div>
                <div class="pill-group" id="temporal-mode">
                    <button class="pill active" onclick="setTemporalMode('temporal')">Timeline</button>
                    <button class="pill" onclick="setTemporalMode('segments')">Segments</button>
                </div>
            </div>

            <div class="glass-card no-hover section">
                <div id="temporal-chart" class="chart-container chart-container-lg"></div>
            </div>

            <div class="stats-row stagger section" id="temporal-stats"></div>

            <div id="temporal-details" class="section"></div>
        </div>`;

    selectTemporal('__ALL__');
}

function setTemporalMode(mode) {
    _temporalMode = mode;
    document.querySelectorAll('#temporal-mode .pill').forEach(p =>
        p.classList.toggle('active', p.textContent.trim().toLowerCase() === (mode === 'temporal' ? 'timeline' : 'segments')));
    // Re-render with current selection
    const activePill = document.querySelector('#temporal-pills .pill.active');
    const name = activePill ? activePill.textContent.trim() : '__ALL__';
    selectTemporal(name === 'All' ? '__ALL__' : name);
}

function selectTemporal(name) {
    document.querySelectorAll('#temporal-pills .pill').forEach(p => {
        const label = p.textContent.trim();
        p.classList.toggle('active', label === name || (name === '__ALL__' && label === 'All'));
    });

    const temporal = AppState.vizData?.temporal || {};
    const el = document.getElementById('temporal-chart');
    const statsEl = document.getElementById('temporal-stats');
    const detailsEl = document.getElementById('temporal-details');
    if (!el) return;

    if (name === '__ALL__') {
        renderTemporalAll(temporal, el, statsEl, detailsEl);
    } else {
        renderTemporalSingle(name, temporal[name], el, statsEl, detailsEl);
    }
}

function renderTemporalAll(temporal, chartEl, statsEl, detailsEl) {
    const names = Object.keys(temporal);

    if (_temporalMode === 'segments') {
        // Heatmap: each row = philosopher, columns = segments
        const maxLen = Math.max(...names.map(n => (temporal[n].coherence_timeline || []).length));
        const z = names.map(n => {
            const tl = temporal[n].coherence_timeline || [];
            return tl.concat(new Array(Math.max(0, maxLen - tl.length)).fill(null));
        });
        Plotly.newPlot(chartEl, [{
            type: 'heatmap', z, y: names,
            x: Array.from({ length: maxLen }, (_, i) => i + 1),
            colorscale: [[0, '#0a0a0a'], [0.5, '#A08850'], [1, '#C9A96E']],
            colorbar: { title: { text: 'Coherence', font: { color: '#A1A1A6', size: 11 } }, tickfont: { color: '#A1A1A6' } },
        }], plotlyDefaults({
            xaxis: { title: 'Segment' },
            margin: { t: 20, r: 80, b: 50, l: 100 },
        }), PL_CFG);
    } else {
        // Overlay line chart with spline and markers
        const traces = names.map((n, i) => {
            const tl = temporal[n].coherence_timeline || [];
            return {
                type: 'scatter', mode: 'lines+markers',
                x: tl.map((_, j) => j + 1), y: tl,
                name: n,
                line: { color: colorFor(i), width: 2, shape: 'spline' },
                marker: { color: colorFor(i), size: 3 },
            };
        });
        Plotly.newPlot(chartEl, traces, plotlyDefaults({
            xaxis: { title: 'Segment' },
            yaxis: { title: 'Coherence', range: [0, 1] },
            showlegend: true,
        }), PL_CFG);
    }

    statsEl.innerHTML = names.map((n, i) => `
        <div class="glass-card metric-card">
            <div class="metric-value" style="color:${colorFor(i)}">${fmt(temporal[n].avg_coherence)}</div>
            <div class="metric-label">${n}</div>
            <div class="text-tertiary" style="margin-top:4px;font-size:var(--text-xs)">vol. ${fmt(temporal[n].volatility)}</div>
        </div>`).join('');

    detailsEl.innerHTML = '';
}

function renderTemporalSingle(name, data, chartEl, statsEl, detailsEl) {
    if (!data) return;
    const tl = data.coherence_timeline || [];
    if (tl.length === 0) return;
    const avg = data.avg_coherence || 0;

    if (_temporalMode === 'segments') {
        // Grid heatmap for single philosopher
        const gridSize = Math.ceil(Math.sqrt(tl.length));
        const z = [];
        for (let r = 0; r < gridSize; r++) {
            const row = [];
            for (let c = 0; c < gridSize; c++) {
                const idx = r * gridSize + c;
                row.push(idx < tl.length ? tl[idx] : null);
            }
            z.push(row);
        }
        Plotly.newPlot(chartEl, [{
            type: 'heatmap', z: z.reverse(),
            colorscale: [[0, '#0a0a0a'], [0.5, '#A08850'], [1, '#C9A96E']],
            colorbar: { title: { text: 'Coherence', font: { color: '#A1A1A6', size: 11 } }, tickfont: { color: '#A1A1A6' } },
        }], plotlyDefaults({
            xaxis: { title: 'Column', showticklabels: false },
            yaxis: { title: 'Row', showticklabels: false },
            title: { text: `${name} — Segment grid`, font: { color: '#C9A96E', size: 16, weight: 300 } },
        }), PL_CFG);
    } else {
        // Line chart with area fill + mean reference line
        const mainTrace = {
            type: 'scatter', mode: 'lines+markers',
            x: tl.map((_, j) => j + 1), y: tl,
            line: { color: '#C9A96E', width: 2, shape: 'spline' },
            marker: { color: '#C9A96E', size: 4 },
            fill: 'tozeroy', fillcolor: 'rgba(201,169,110,0.06)',
            name: 'Coherence',
        };
        const meanTrace = {
            type: 'scatter', mode: 'lines',
            x: [1, tl.length], y: [avg, avg],
            line: { color: '#E5E5E7', width: 1, dash: 'dash' },
            name: `Mean (${fmt(avg)})`,
            hoverinfo: 'skip',
        };
        Plotly.newPlot(chartEl, [mainTrace, meanTrace], plotlyDefaults({
            xaxis: { title: 'Segment' },
            yaxis: { title: 'Coherence', range: [0, 1] },
            showlegend: true,
            title: { text: name, font: { color: '#C9A96E', size: 16, weight: 300 } },
        }), PL_CFG);
    }

    statsEl.innerHTML = `
        <div class="glass-card metric-card"><div class="metric-value">${fmt(avg)}</div><div class="metric-label">Avg coherence</div></div>
        <div class="glass-card metric-card"><div class="metric-value">${fmt(data.volatility)}</div><div class="metric-label">Volatility</div></div>
        <div class="glass-card metric-card"><div class="metric-value">${fmt(data.peak_coherence)}</div><div class="metric-label">Peak</div></div>
        <div class="glass-card metric-card"><div class="metric-value">${data.coherent_segments || 0}</div><div class="metric-label">Coherent segments</div></div>
    `;

    // High-coherence segments list
    const highSegs = tl.map((v, i) => ({ seg: i + 1, val: v })).filter(s => s.val > 0.6).sort((a, b) => b.val - a.val).slice(0, 8);
    if (highSegs.length > 0) {
        detailsEl.innerHTML = `
            <h3 style="margin-bottom:var(--space-3);">High coherence segments</h3>
            <div class="glass-card no-hover">
                <ul class="segment-list">
                    ${highSegs.map(s => `<li class="segment-item"><span class="segment-value">${fmt(s.val)}</span><span class="segment-label">Segment ${s.seg}</span></li>`).join('')}
                </ul>
            </div>`;
    } else {
        detailsEl.innerHTML = '';
    }
}

// =====================================================================
// VIEW: Semantic Network
// =====================================================================

async function renderNetworkView(container) {
    if (!AppState.vizData) {
        container.innerHTML = `<div class="page-header"><h1>Semantic network</h1></div><div class="page-body">${chartSkeleton(500)}</div>`;
        try { AppState.vizData = await API.getAllVizData(); } catch {
            container.innerHTML = emptyState('Semantic network', 'No network data', 'Upload texts and run an analysis to generate the semantic network.', 'upload');
            return;
        }
    }
    const network = AppState.vizData?.network;
    if (!network || !network.nodes || network.nodes.length === 0) {
        container.innerHTML = emptyState('Semantic network', 'No network data', 'Run an analysis with texts to generate the semantic network.', 'upload');
        return;
    }

    const meta = network.metadata || {};
    // Gather unique philosophers and categories
    const allPhilosophers = [...new Set(network.nodes.map(n => n.philosopher).filter(Boolean))];
    const allCategories   = [...new Set(network.nodes.map(n => n.category).filter(Boolean))];

    container.innerHTML = `
        <div class="page-header">
            <h1>Semantic network</h1>
            <p class="page-description">Concept relationships discovered across philosophical texts.</p>
        </div>
        <div class="page-body">
            <div class="grid grid-3 stagger section">
                <div class="glass-card metric-card">
                    <div class="metric-value">${meta.total_concepts || 0}</div><div class="metric-label">Concepts</div>
                </div>
                <div class="glass-card metric-card">
                    <div class="metric-value">${meta.total_relationships || 0}</div><div class="metric-label">Relationships</div>
                </div>
                <div class="glass-card metric-card">
                    <div class="metric-value">${fmt(meta.density)}</div><div class="metric-label">Density</div>
                </div>
            </div>

            <div class="network-layout section">
                <!-- Left: Filters -->
                <div class="network-sidebar">
                    <div class="glass-card no-hover" style="padding:var(--space-4);">
                        <div class="filter-title">Philosophers</div>
                        <div class="checkbox-group" id="net-phil-filters">
                            ${allPhilosophers.map(p => `<label class="checkbox-item"><input type="checkbox" checked data-philosopher="${escHtml(p)}" onchange="applyNetworkFilters()"> ${escHtml(p)}</label>`).join('')}
                        </div>
                    </div>
                    <div class="glass-card no-hover" style="padding:var(--space-4);">
                        <div class="filter-title">Categories</div>
                        <div class="checkbox-group" id="net-cat-filters">
                            ${allCategories.map(c => `<label class="checkbox-item"><input type="checkbox" checked data-category="${escHtml(c)}" onchange="applyNetworkFilters()"> ${escHtml(c)}</label>`).join('')}
                        </div>
                    </div>
                    <div class="glass-card no-hover" style="padding:var(--space-4);">
                        <div class="filter-title">Min strength</div>
                        <div class="range-group" style="margin-bottom:0;">
                            <div class="range-header">
                                <span class="range-value" id="net-strength-val">0.3</span>
                            </div>
                            <input type="range" min="0" max="100" value="30" id="net-strength-slider"
                                   oninput="document.getElementById('net-strength-val').textContent=(this.value/100).toFixed(2); applyNetworkFilters()">
                        </div>
                    </div>
                    <button class="btn btn-secondary btn-sm" style="width:100%;" onclick="resetNetworkFilters()">Reset filters</button>
                </div>

                <!-- Center: Graph -->
                <div class="glass-card no-hover network-main" style="padding:var(--space-2); overflow:hidden;">
                    <div id="network-graph" class="network-container"></div>
                </div>

                <!-- Right: Node info -->
                <div class="network-info" id="network-info">
                    <div class="glass-card no-hover" style="padding:var(--space-4);">
                        <div class="filter-title">Selected node</div>
                        <div id="node-detail" class="text-secondary" style="font-size:var(--text-sm);">Click a node to see details</div>
                    </div>
                    <div class="glass-card no-hover" style="padding:var(--space-4);">
                        <div class="filter-title">Connections</div>
                        <div id="node-connections" class="text-secondary" style="font-size:var(--text-sm);">—</div>
                    </div>
                </div>
            </div>
        </div>`;

    initD3Network(network);
}

/** Full D3 network with interactivity */
function initD3Network(rawData) {
    const container = document.getElementById('network-graph');
    if (!container) return;

    const width  = container.clientWidth || 700;
    const height = 600;

    // Store raw data for filtering
    AppState._networkRaw = rawData;

    const svg = d3.select(container).html('').append('svg')
        .attr('width', width).attr('height', height)
        .attr('viewBox', [0, 0, width, height]);

    // Groups for links and nodes (links behind nodes)
    svg.append('g').attr('class', 'links-group');
    svg.append('g').attr('class', 'nodes-group');

    renderNetworkGraph(rawData.nodes, rawData.links, svg, width, height);

    // Resize handler
    const onResize = () => {
        const w = container.clientWidth;
        svg.attr('width', w).attr('viewBox', [0, 0, w, height]);
        if (AppState.networkSimulation) {
            AppState.networkSimulation.force('center', d3.forceCenter(w / 2, height / 2));
            AppState.networkSimulation.alpha(0.3).restart();
        }
    };
    window.addEventListener('resize', onResize);
}

function renderNetworkGraph(nodes, links, svg, width, height) {
    // Stop previous
    if (AppState.networkSimulation) AppState.networkSimulation.stop();

    const nodesData = nodes.map(d => ({ ...d }));
    const linksData = links.map(d => ({ ...d }));

    // Category color map
    const catColors = { metaphysics: '#C9A96E', ethics: '#64D2FF', epistemology: '#E5E5E7', logic: '#BF5AF2' };
    const nodeColor = d => catColors[d.category] || '#8E8E93';

    const simulation = d3.forceSimulation(nodesData)
        .force('link', d3.forceLink(linksData).id(d => d.id).distance(80))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => 8 + (d.importance || 0.5) * 15));

    AppState.networkSimulation = simulation;

    // Links
    const linkSel = svg.select('.links-group').selectAll('line').data(linksData, d => `${d.source.id || d.source}-${d.target.id || d.target}`);
    linkSel.exit().remove();
    const linkEnter = linkSel.enter().append('line').attr('class', 'network-link');
    const link = linkEnter.merge(linkSel)
        .attr('stroke-width', d => Math.max(1, Math.sqrt((d.strength || 0.5) * 5)))
        .attr('stroke-opacity', d => 0.2 + (d.strength || 0.5) * 0.4);

    // Nodes
    const nodeSel = svg.select('.nodes-group').selectAll('g.network-node').data(nodesData, d => d.id);
    nodeSel.exit().remove();
    const nodeEnter = nodeSel.enter().append('g').attr('class', 'network-node');

    nodeEnter.append('circle')
        .attr('r', d => 5 + (d.importance || 0.5) * 15)
        .attr('fill', d => nodeColor(d));

    nodeEnter.append('text')
        .attr('dy', d => -(8 + (d.importance || 0.5) * 15))
        .attr('text-anchor', 'middle')
        .style('font-size', d => Math.max(9, (d.importance || 0.5) * 13) + 'px')
        .text(d => d.label || d.id);

    const node = nodeEnter.merge(nodeSel);

    // Drag
    node.call(d3.drag()
        .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end',   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );

    // Click to select
    node.on('click', (e, d) => {
        e.stopPropagation();
        AppState.selectedNode = d;

        // Highlight connected links
        link.classed('highlighted', l => l.source.id === d.id || l.target.id === d.id);

        // Dim unconnected nodes
        const connectedIds = new Set();
        connectedIds.add(d.id);
        linksData.forEach(l => {
            if ((l.source.id || l.source) === d.id) connectedIds.add(l.target.id || l.target);
            if ((l.target.id || l.target) === d.id) connectedIds.add(l.source.id || l.source);
        });
        node.style('opacity', n => connectedIds.has(n.id) ? 1 : 0.25);

        // Update info panels
        const detail = document.getElementById('node-detail');
        if (detail) {
            detail.innerHTML = `
                <div style="font-size:var(--text-lg); color:var(--color-gold); margin-bottom:var(--space-2);">${escHtml(d.label || d.id)}</div>
                <div>Category: <span class="badge">${escHtml(d.category || '—')}</span></div>
                <div style="margin-top:var(--space-2);">Philosopher: ${escHtml(d.philosopher || '—')}</div>
                <div>Importance: <span class="text-gold">${fmt(d.importance, 2)}</span></div>`;
        }

        const conns = document.getElementById('node-connections');
        if (conns) {
            const connected = linksData
                .filter(l => (l.source.id || l.source) === d.id || (l.target.id || l.target) === d.id)
                .map(l => {
                    const other = (l.source.id || l.source) === d.id ? (l.target.label || l.target.id || l.target) : (l.source.label || l.source.id || l.source);
                    return `<li class="segment-item"><span class="segment-value">${fmt(l.strength, 2)}</span><span class="segment-label">${escHtml(String(other))}</span></li>`;
                });
            conns.innerHTML = connected.length ? `<ul class="segment-list">${connected.join('')}</ul>` : 'No connections';
        }
    });

    // Click background to deselect
    svg.on('click', () => {
        AppState.selectedNode = null;
        link.classed('highlighted', false);
        node.style('opacity', 1);
        const detail = document.getElementById('node-detail');
        if (detail) detail.innerHTML = 'Click a node to see details';
        const conns = document.getElementById('node-connections');
        if (conns) conns.innerHTML = '—';
    });

    // Tick
    simulation.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}

function applyNetworkFilters() {
    const raw = AppState._networkRaw;
    if (!raw) return;

    const checkedPhils = [...document.querySelectorAll('#net-phil-filters input:checked')].map(i => i.dataset.philosopher);
    const checkedCats  = [...document.querySelectorAll('#net-cat-filters input:checked')].map(i => i.dataset.category);
    const minStrength  = parseInt(document.getElementById('net-strength-slider').value) / 100;

    const filteredNodes = raw.nodes.filter(n =>
        (checkedPhils.length === 0 || checkedPhils.includes(n.philosopher)) &&
        (checkedCats.length === 0  || checkedCats.includes(n.category))
    );
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredLinks = raw.links.filter(l =>
        nodeIds.has(l.source.id || l.source) && nodeIds.has(l.target.id || l.target) && (l.strength || 0) >= minStrength
    );

    const svg = d3.select('#network-graph svg');
    const w = parseInt(svg.attr('width'));
    renderNetworkGraph(filteredNodes, filteredLinks, svg, w, 600);
}

function resetNetworkFilters() {
    document.querySelectorAll('#net-phil-filters input, #net-cat-filters input').forEach(i => i.checked = true);
    document.getElementById('net-strength-slider').value = 30;
    document.getElementById('net-strength-val').textContent = '0.30';
    applyNetworkFilters();
}

// =====================================================================
// VIEW: Compare
// =====================================================================

async function renderCompareView(container) {
    if (!AppState.vizData) {
        container.innerHTML = `<div class="page-header"><h1>Compare</h1></div><div class="page-body">${chartSkeleton(400)}</div>`;
        try { AppState.vizData = await API.getAllVizData(); } catch {
            container.innerHTML = emptyState('Compare', 'No data to compare', 'Upload texts and run an analysis to compare philosophers.', 'upload');
            return;
        }
    }
    const philosophers = AppState.vizData?.dashboard?.philosophers || {};
    const names = Object.keys(philosophers);
    if (names.length === 0) {
        container.innerHTML = emptyState('Compare', 'No data to compare', 'Run an analysis first to compare philosophers.', 'upload');
        return;
    }

    container.innerHTML = `
        <div class="page-header">
            <h1>Compare</h1>
            <p class="page-description">Multi-dimensional comparison across philosophers.</p>
        </div>
        <div class="page-body">
            <div class="chart-controls" id="compare-controls">
                <div class="pill-group">
                    <button class="pill active" data-mode="parallel" onclick="switchCompare('parallel')">Parallel coordinates</button>
                    <button class="pill" data-mode="smallmultiples" onclick="switchCompare('smallmultiples')">Small multiples</button>
                    <button class="pill" data-mode="ranked" onclick="switchCompare('ranked')">Ranked dot plot</button>
                    <button class="pill" data-mode="ridgeline" onclick="switchCompare('ridgeline')">Ridgeline</button>
                </div>
            </div>
            <div class="glass-card no-hover section">
                <div id="compare-chart" class="chart-container chart-container-xl"></div>
            </div>
        </div>`;

    switchCompare('parallel');
}

function switchCompare(mode) {
    document.querySelectorAll('#compare-controls .pill').forEach(p =>
        p.classList.toggle('active', p.dataset.mode === mode));

    const philosophers = AppState.vizData?.dashboard?.philosophers || {};
    const temporal = AppState.vizData?.temporal || {};
    const el = document.getElementById('compare-chart');
    if (!el) return;

    const names = Object.keys(philosophers);

    if (mode === 'parallel') {
        const dimensions = METRICS.map(m => {
            const vals = names.map(n => philosophers[n][m.key] || 0);
            return { label: m.label, values: vals, range: [Math.min(...vals) * 0.9, Math.max(...vals) * 1.1] };
        });
        Plotly.newPlot(el, [{
            type: 'parcoords',
            line: {
                color: names.map((_, i) => i),
                colorscale: names.map((_, i) => [i / Math.max(1, names.length - 1), colorFor(i)]),
            },
            dimensions,
            labelfont: { color: '#E5E5E7', size: 12 },
            tickfont:  { color: '#A1A1A6', size: 10 },
            rangefont: { color: '#6E6E73', size: 9 },
        }], plotlyDefaults({ margin: { t: 40, r: 60, b: 30, l: 60 } }), PL_CFG);

    } else if (mode === 'smallmultiples') {
        const cols = Math.min(names.length, 3);
        const rows = Math.ceil(names.length / cols);
        const traces = [];
        const layout = plotlyDefaults({ showlegend: false, height: rows * 280 });

        // Compute global normalization
        const maxVals = METRIC_KEYS.map(k => Math.max(...names.map(n => philosophers[n][k] || 0)));

        names.forEach((name, i) => {
            const p = philosophers[name];
            const norm = METRIC_KEYS.map((k, j) => maxVals[j] ? (p[k] || 0) / maxVals[j] : 0);
            const row = Math.floor(i / cols);
            const col = i % cols;
            const pk = i === 0 ? 'polar' : `polar${i + 1}`;

            traces.push({
                type: 'scatterpolar',
                r: [...norm, norm[0]],
                theta: [...METRIC_LABELS, METRIC_LABELS[0]],
                fill: 'toself', fillcolor: colorFor(i) + '20',
                line: { color: colorFor(i), width: 2 },
                name, subplot: pk,
            });

            layout[pk] = {
                bgcolor: 'rgba(0,0,0,0)',
                radialaxis: { visible: true, gridcolor: 'rgba(255,255,255,0.05)', showticklabels: false, range: [0, 1.1] },
                angularaxis: { gridcolor: 'rgba(255,255,255,0.08)', tickfont: { color: '#A1A1A6', size: 9 } },
                domain: {
                    x: [col / cols + 0.02, (col + 1) / cols - 0.02],
                    y: [1 - (row + 1) / rows + 0.06, 1 - row / rows - 0.06],
                },
            };
            layout.annotations = layout.annotations || [];
            layout.annotations.push({
                text: name, x: (col + 0.5) / cols, y: 1 - row / rows - 0.02,
                xref: 'paper', yref: 'paper', showarrow: false,
                font: { color: colorFor(i), size: 12, weight: 300 },
            });
        });

        Plotly.newPlot(el, traces, layout, PL_CFG);

    } else if (mode === 'ranked') {
        // Dot plot with connecting lines per philosopher
        const traces = [];

        // First, draw connecting lines for each philosopher
        names.forEach((name, ci) => {
            const vals = METRIC_KEYS.map(k => philosophers[name][k] || 0);
            traces.push({
                type: 'scatter', mode: 'lines',
                x: vals, y: METRIC_LABELS,
                line: { color: colorFor(ci), width: 1, dash: 'dot' },
                showlegend: false, hoverinfo: 'skip',
            });
        });

        // Then dots on top
        names.forEach((name, ci) => {
            traces.push({
                type: 'scatter', mode: 'markers',
                x: METRIC_KEYS.map(k => philosophers[name][k] || 0),
                y: METRIC_LABELS,
                marker: { color: colorFor(ci), size: 10, line: { color: '#fff', width: 1 } },
                name,
                hovertemplate: `<b>${name}</b><br>%{y}: %{x:.3f}<extra></extra>`,
            });
        });

        Plotly.newPlot(el, traces, plotlyDefaults({
            yaxis: { type: 'category' },
            xaxis: { title: 'Value' },
            showlegend: true,
            margin: { l: 120 },
        }), PL_CFG);

    } else if (mode === 'ridgeline') {
        // Ridgeline plot using temporal coherence distributions
        const tNames = Object.keys(temporal);
        if (tNames.length === 0) {
            el.innerHTML = '<div class="empty-state" style="padding:var(--space-8);"><div class="empty-state-title">No temporal data for ridgeline</div></div>';
            return;
        }

        const traces = [];
        tNames.forEach((name, i) => {
            const tl = temporal[name].coherence_timeline || [];
            if (tl.length < 2) return;
            traces.push({
                type: 'violin', y0: name,
                x: tl,
                orientation: 'h',
                side: 'positive',
                line: { color: colorFor(i), width: 2 },
                fillcolor: colorFor(i) + '30',
                meanline: { visible: true, color: '#E5E5E7', width: 1 },
                name,
                spanmode: 'hard',
                bandwidth: 0.05,
            });
        });

        Plotly.newPlot(el, traces, plotlyDefaults({
            xaxis: { title: 'Coherence', range: [0, 1] },
            yaxis: { title: '' },
            showlegend: false,
            violinmode: 'overlay',
            title: { text: 'Coherence distribution by philosopher', font: { color: '#C9A96E', size: 16, weight: 300 } },
            margin: { l: 120, t: 60 },
        }), PL_CFG);
    }
}

// =====================================================================
// Shared: Empty state template
// =====================================================================

function emptyState(title, heading, description, targetView = 'upload') {
    const labels = { upload: 'Go to upload', dashboard: 'View dashboard', temporal: 'View temporal', network: 'View network', compare: 'View compare' };
    const btnLabel = labels[targetView] || 'Go to upload';
    return `
        <div class="page-header"><h1>${title}</h1></div>
        <div class="page-body">
            <div class="empty-state">
                <div class="empty-state-icon" style="font-size:var(--text-5xl); opacity:0.2;">&#9673;</div>
                <div class="empty-state-title">${heading}</div>
                <div class="empty-state-description">${description}</div>
                <a href="#/${targetView}" class="btn btn-primary" style="margin-top:var(--space-5);">${btnLabel}</a>
            </div>
        </div>`;
}

// =====================================================================
// Init
// =====================================================================

// ---------------------------------------------------------------------------
// Keyboard navigation
// ---------------------------------------------------------------------------

function initKeyboardNav() {
    document.addEventListener('keydown', (e) => {
        // Escape closes mobile sidebar
        if (e.key === 'Escape') {
            const sidebar = document.getElementById('sidebar');
            if (sidebar.classList.contains('open')) {
                toggleSidebar();
                document.querySelector('.hamburger')?.focus();
            }
        }

        // Arrow keys navigate sidebar items when focused within sidebar
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            const active = document.activeElement;
            if (active && active.closest('.sidebar-nav')) {
                e.preventDefault();
                const items = Array.from(document.querySelectorAll('.sidebar-nav .nav-item'));
                const idx = items.indexOf(active);
                if (idx === -1) return;
                const next = e.key === 'ArrowDown'
                    ? items[(idx + 1) % items.length]
                    : items[(idx - 1 + items.length) % items.length];
                next.focus();
            }
        }

        // Left/right arrow keys navigate within pill groups
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            const active = document.activeElement;
            if (active && active.classList.contains('pill')) {
                e.preventDefault();
                const group = active.closest('.pill-group');
                if (!group) return;
                const pills = Array.from(group.querySelectorAll('.pill'));
                const idx = pills.indexOf(active);
                if (idx === -1) return;
                const next = e.key === 'ArrowRight'
                    ? pills[(idx + 1) % pills.length]
                    : pills[(idx - 1 + pills.length) % pills.length];
                next.focus();
                next.click();
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Metric counter animation
// ---------------------------------------------------------------------------

function animateMetricValues(container) {
    const metrics = container.querySelectorAll('.metric-value[data-target]');
    metrics.forEach(el => {
        const target = parseFloat(el.dataset.target);
        if (isNaN(target)) return;

        const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 3;
        const duration = 600;
        const start = performance.now();

        function step(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = eased * target;
            el.textContent = current.toFixed(decimals);
            if (progress < 1) requestAnimationFrame(step);
        }

        el.textContent = '0.' + '0'.repeat(decimals);
        requestAnimationFrame(step);
    });
}

// ---------------------------------------------------------------------------
// Chart skeleton helper
// ---------------------------------------------------------------------------

function chartSkeleton(height = 400) {
    return `<div class="chart-skeleton" style="min-height:${height}px">
        <div class="chart-skeleton-bars">
            <div class="chart-skeleton-bar" style="height:60%"></div>
            <div class="chart-skeleton-bar" style="height:85%"></div>
            <div class="chart-skeleton-bar" style="height:45%"></div>
            <div class="chart-skeleton-bar" style="height:70%"></div>
            <div class="chart-skeleton-bar" style="height:55%"></div>
        </div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    Router.init();
    initKeyboardNav();
});
