/**
 * Number Generator Web App - Main JavaScript
 */

class NumberGenApp {
    constructor() {
        this.apiBase = '/api/v1';
        this.currentData = [];
        this.history = [];
    }

    async fetch(endpoint, options = {}) {
        const url = this.apiBase + endpoint;
        const resp = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...options
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ message: 'Unknown error' }));
            throw new Error(err.message || `HTTP ${resp.status}`);
        }
        return resp.json();
    }

    async generateRandom(algorithm = 'mersenne', count = 10, min = 0, max = 1000, seed = null) {
        let url = `/generate/random?algorithm=${algorithm}&count=${count}&min=${min}&max=${max}`;
        if (seed !== null) url += `&seed=${seed}`;
        const data = await this.fetch(url);
        this.currentData = data.data.numbers;
        this.addToHistory('random', algorithm, this.currentData);
        return this.currentData;
    }

    async generatePrime(count = 20, start = 2) {
        const data = await this.fetch(`/generate/prime?count=${count}&start=${start}`);
        this.currentData = data.data.primes;
        this.addToHistory('prime', 'sieve', this.currentData);
        return this.currentData;
    }

    async generateFibonacci(count = 20, type = 'fibonacci', variant = 'iterative') {
        const data = await this.fetch(`/generate/fibonacci?count=${count}&type=${type}&variant=${variant}`);
        this.currentData = data.data.sequence;
        this.addToHistory('fibonacci', type, this.currentData);
        return this.currentData;
    }

    async generateSequence(type, count = 15, params = {}) {
        let url = `/generate/sequence?type=${type}&count=${count}`;
        for (const [k, v] of Object.entries(params)) {
            url += `&${k}=${v}`;
        }
        const data = await this.fetch(url);
        this.currentData = data.data.sequence;
        this.addToHistory('sequence', type, this.currentData);
        return this.currentData;
    }

    async analyze(data, tests = 'statistical') {
        return this.fetch('/analyze', {
            method: 'POST',
            body: JSON.stringify({ data, tests })
        });
    }

    async transform(data, operation, params = {}) {
        return this.fetch('/transform', {
            method: 'POST',
            body: JSON.stringify({ data, operation, params })
        });
    }

    addToHistory(type, subtype, data) {
        this.history.unshift({
            type, subtype,
            count: data.length,
            preview: data.slice(0, 5),
            timestamp: new Date().toISOString()
        });
        if (this.history.length > 50) this.history.pop();
    }

    computeStats(data) {
        if (!data.length) return null;
        const n = data.length;
        const sum = data.reduce((a, b) => a + b, 0);
        const mean = sum / n;
        const variance = data.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
        const sorted = [...data].sort((a, b) => a - b);
        return {
            count: n,
            sum: Math.round(sum * 1000) / 1000,
            mean: Math.round(mean * 10000) / 10000,
            stdDev: Math.round(Math.sqrt(variance) * 10000) / 10000,
            min: sorted[0],
            max: sorted[n - 1],
            median: sorted[Math.floor(n / 2)],
            range: sorted[n - 1] - sorted[0]
        };
    }

    formatNumber(n, decimals = 4) {
        if (Number.isInteger(n)) return n.toString();
        return n.toFixed(decimals);
    }
}

// Global app instance
const app = new NumberGenApp();

// Utility functions
function debounce(fn, delay = 300) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    };
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!');
    }).catch(() => {
        // Fallback
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showToast('Copied!');
    });
}

function showToast(message, type = 'success', duration = 3000) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.style.cssText = `
        position: fixed; bottom: 2rem; right: 2rem; z-index: 1000;
        background: ${type === 'success' ? '#064e3b' : '#7f1d1d'};
        color: ${type === 'success' ? '#6ee7b7' : '#f87171'};
        padding: 0.75rem 1.5rem; border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        font-size: 0.9rem; transition: opacity 0.3s;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function renderNumberGrid(container, numbers, maxDisplay = 200) {
    const display = numbers.slice(0, maxDisplay);
    container.innerHTML = display.map(n =>
        `<div class="number-cell">${app.formatNumber(n)}</div>`
    ).join('');
    if (numbers.length > maxDisplay) {
        const more = document.createElement('div');
        more.style.cssText = 'text-align:center; color:#555; padding:0.5rem; font-size:0.8rem;';
        more.textContent = `... and ${numbers.length - maxDisplay} more`;
        container.appendChild(more);
    }
}

function renderMiniHistogram(container, data, bins = 10) {
    if (!data.length) return;
    const min = Math.min(...data);
    const max = Math.max(...data);
    if (min === max) return;
    const bw = (max - min) / bins;
    const counts = new Array(bins).fill(0);
    data.forEach(v => {
        const idx = Math.min(Math.floor((v - min) / bw), bins - 1);
        counts[idx]++;
    });
    const maxC = Math.max(...counts);
    const barHeight = 60;
    const barWidth = Math.floor(container.clientWidth / bins) || 20;

    container.innerHTML = `
        <svg width="${bins * barWidth}" height="${barHeight + 20}" style="overflow:visible">
            ${counts.map((c, i) => {
                const h = Math.round(c / maxC * barHeight);
                return `<rect x="${i * barWidth + 1}" y="${barHeight - h}" width="${barWidth - 2}" height="${h}"
                              fill="#7c3aed" rx="2" opacity="0.8"/>`;
            }).join('')}
        </svg>
    `;
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            const btn = document.querySelector('.btn[onclick*="Generate"]') ||
                        document.querySelector('.btn[onclick*="generate"]');
            if (btn) btn.click();
        }
    });
});
