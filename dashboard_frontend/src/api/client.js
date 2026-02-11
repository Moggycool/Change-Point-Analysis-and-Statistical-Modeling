const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";

export async function fetchPrices(params) {
    const url = new URL(`${API_BASE}/api/prices`);
    Object.entries(params || {}).forEach(([k, v]) => v && url.searchParams.set(k, v));
    return fetch(url).then(r => r.json());
}

export async function fetchChangePoints() {
    return fetch(`${API_BASE}/api/changepoints`).then(r => r.json());
}

export async function fetchEvents() {
    return fetch(`${API_BASE}/api/events`).then(r => r.json());
}

export async function fetchEventCorrelation(params) {
    const url = new URL(`${API_BASE}/api/event-correlation`);
    Object.entries(params || {}).forEach(([k, v]) => v && url.searchParams.set(k, v));
    return fetch(url).then(r => r.json());
}
