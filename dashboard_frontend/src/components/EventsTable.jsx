import React from "react";

export default function EventsTable({ events, selectedEventId, onSelect }) {
    return (
        <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                    <tr>
                        <th style={th}>Date</th>
                        <th style={th}>Event</th>
                    </tr>
                </thead>
                <tbody>
                    {events.map((e) => {
                        const active = String(e.event_id) === String(selectedEventId);
                        return (
                            <tr
                                key={e.event_id}
                                onClick={() => onSelect(e)}
                                style={{
                                    cursor: "pointer",
                                    background: active ? "rgba(22,119,255,0.10)" : "transparent",
                                }}
                            >
                                <td style={td}>{e.event_date}</td>
                                <td style={td}>{e.event_title || e.title || e.event || "Event"}</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

const th = { textAlign: "left", borderBottom: "1px solid #ddd", padding: "8px" };
const td = { borderBottom: "1px solid #eee", padding: "8px", verticalAlign: "top" };
