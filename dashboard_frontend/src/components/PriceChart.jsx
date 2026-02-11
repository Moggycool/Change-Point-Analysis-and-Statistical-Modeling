import React from "react";
import {
    ResponsiveContainer,
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ReferenceLine, ReferenceArea, Legend,
} from "recharts";

export default function PriceChart({
    data,
    cpDate,
    hdiLow,
    hdiHigh,
    selectedEvent,
    highlightWindowDays = 7,
}) {
    const eventDate = selectedEvent?.event_date;

    // compute event window bounds (string ISO dates). We'll just display the line; shading uses strings too.
    function shiftISO(iso, days) {
        if (!iso) return null;
        const d = new Date(iso);
        d.setDate(d.getDate() + days);
        return d.toISOString().slice(0, 10);
    }

    const evLow = eventDate ? shiftISO(eventDate, -highlightWindowDays) : null;
    const evHigh = eventDate ? shiftISO(eventDate, highlightWindowDays) : null;

    return (
        <div style={{ width: "100%", height: 380 }}>
            <ResponsiveContainer>
                <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" minTickGap={30} />
                    <YAxis domain={["auto", "auto"]} />
                    <Tooltip />
                    <Legend />

                    <Line type="monotone" dataKey="price" strokeWidth={2} dot={false} name="Brent price" />

                    {/* Change point + HDI band */}
                    {hdiLow && hdiHigh && (
                        <ReferenceArea x1={hdiLow} x2={hdiHigh} fillOpacity={0.12} label="τ 94% HDI" />
                    )}
                    {cpDate && (
                        <ReferenceLine x={cpDate} strokeWidth={2} stroke="#ff4d4f" label="Change point (mode)" />
                    )}

                    {/* Event highlight */}
                    {eventDate && (
                        <>
                            {evLow && evHigh && (
                                <ReferenceArea x1={evLow} x2={evHigh} fill="#1677ff" fillOpacity={0.08} />
                            )}
                            <ReferenceLine x={eventDate} stroke="#1677ff" strokeWidth={2} label="Selected event" />
                        </>
                    )}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
