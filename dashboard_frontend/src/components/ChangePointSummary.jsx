import React from "react";

function KV({ k, v }) {
    return (
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, padding: "4px 0" }}>
            <div style={{ opacity: 0.75 }}>{k}</div>
            <div style={{ fontFamily: "monospace" }}>{String(v ?? "")}</div>
        </div>
    );
}

export default function ChangePointSummary({ title, payload }) {
    const tau = payload?.tau_summary || {};
    const impact = payload?.impact_summary || {};

    return (
        <div style={card}>
            <h3 style={{ marginTop: 0 }}>{title}</h3>
            <div style={{ display: "grid", gap: 12 }}>
                <div>
                    <strong>τ (date summary)</strong>
                    <KV k="mode" v={tau.tau_mode_date} />
                    <KV k="94% HDI low" v={tau.tau_hdi_low_date} />
                    <KV k="94% HDI high" v={tau.tau_hdi_high_date} />
                </div>
                <div>
                    <strong>Impact (summary)</strong>
                    {Object.keys(impact).slice(0, 8).map((k) => (
                        <KV key={k} k={k} v={impact[k]} />
                    ))}
                </div>
            </div>
        </div>
    );
}

const card = {
    border: "1px solid #eee",
    borderRadius: 10,
    padding: 14,
    background: "#fff",
};
