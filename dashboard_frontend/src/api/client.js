const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";

function withParams(url, params) {
    const u = new URL(url);
    Object.entries(params || {}).forEach(([k, v]) => {
        if (v !== undefined && v !== null && v !== "") u.searchParams.set(k, v);
    });
    return u.toString();
}

export async function fetchPrices(params) {
    const url = withParams(`${API_BASE}/api/prices`, params);
    const res = await fetch(url);
    return res.json();
}

export async function fetchChangePoints() {
    const res = await fetch(`${API_BASE}/api/changepoints`);
    return res.json();
}

export async function fetchEvents(params) {
    const url = withParams(`${API_BASE}/api/events`, params);
    const res = await fetch(url);
    return res.json();
}

export async function fetchEventCorrelation(params) {
    const url = withParams(`${API_BASE}/api/event-correlation`, params);
    const res = await fetch(url);
    return res.json();
}
