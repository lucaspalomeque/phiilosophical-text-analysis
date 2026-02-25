/**
 * API Client — Communicates with the FastAPI backend
 */

const API = {
    baseUrl: '',

    /**
     * Generic fetch wrapper with error handling
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaults = {
            headers: { 'Content-Type': 'application/json' },
        };

        const config = { ...defaults, ...options };

        // Don't set Content-Type for FormData (browser sets multipart boundary)
        if (config.body instanceof FormData) {
            delete config.headers['Content-Type'];
        }

        try {
            const response = await fetch(url, config);
            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(error.detail || `Request failed: ${response.status}`);
            }
            return await response.json();
        } catch (err) {
            console.error(`API Error [${endpoint}]:`, err);
            throw err;
        }
    },

    /**
     * Upload text files for analysis
     */
    async uploadFiles(files) {
        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }
        return this.request('/api/upload', {
            method: 'POST',
            body: formData,
        });
    },

    /**
     * Submit pasted text
     */
    async submitText(name, content) {
        return this.request('/api/texts', {
            method: 'POST',
            body: JSON.stringify({ name, content }),
        });
    },

    /**
     * Load sample texts
     */
    async loadSamples() {
        return this.request('/api/samples', { method: 'POST' });
    },

    /**
     * Get currently loaded texts
     */
    async getTexts() {
        return this.request('/api/texts');
    },

    /**
     * Remove a loaded text
     */
    async removeText(name) {
        return this.request(`/api/texts/${encodeURIComponent(name)}`, {
            method: 'DELETE',
        });
    },

    /**
     * Run analysis on loaded texts
     */
    async runAnalysis(params = {}) {
        return this.request('/api/analyze', {
            method: 'POST',
            body: JSON.stringify(params),
        });
    },

    /**
     * Get analysis results
     */
    async getResults() {
        return this.request('/api/results');
    },

    /**
     * Get dashboard visualization data
     */
    async getDashboardData() {
        return this.request('/api/viz/dashboard');
    },

    /**
     * Get temporal coherence data
     */
    async getTemporalData() {
        return this.request('/api/viz/temporal');
    },

    /**
     * Get semantic network data
     */
    async getNetworkData() {
        return this.request('/api/viz/network');
    },

    /**
     * Get all visualization data at once
     */
    async getAllVizData() {
        return this.request('/api/viz/all');
    },

    /**
     * Health check
     */
    async health() {
        return this.request('/api/health');
    },
};
