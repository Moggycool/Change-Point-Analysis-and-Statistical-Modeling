import React from "react";

export default function DateRangePicker({ start, end, onChange }) {
    return (
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
            <label>
                Start:&nbsp;
                <input
                    type="date"
                    value={start || ""}
                    onChange={(e) => onChange({ start: e.target.value, end })}
                />
            </label>
            <label>
                End:&nbsp;
                <input
                    type="date"
                    value={end || ""}
                    onChange={(e) => onChange({ start, end: e.target.value })}
                />
            </label>
        </div>
    );
}
