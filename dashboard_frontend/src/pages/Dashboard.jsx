import React, { useEffect, useMemo, useState } from "react";
import DateRangePicker from "../components/DateRangePicker.jsx";
import PriceChart from "../components/PriceChart.jsx";
import EventsTable from "../components/EventsTable.jsx";
import ChangePointSummary from "../components/ChangePointSummary.jsx";
import { fetchPrices, fetchChangePoints, fetchEvents, fetchEventCorrelation } from "../api/client.js";

export default function Dashboard() {
    const [start, setStart] = useState("2014-01-01");
    const [end, setEnd] = useState("");
    const [prices, setPrices] = useState([]);
    const [cp, setCp] = useState(null);
    const [events, setEvents] = useState([]);
    const [selectedEvent, setSelectedEvent] = useState(null);

    const [windowDays, setWindowDays] = useState(7);
    const [eventStats, setEventStats] = useState([]);

    useEffect(() => {
        async function load() {
            const [p, c, e] = await Promise.all([
                fetchPrices({ start, end, include: "volatility" }),
                fetchChangePoints(),
                fetchEvents({ start, end }),
            ]);
            setPrices(p);
            setCp(c);
            setEvents(e);
        }
        load().catch(console.error);
    }, [start, end]);

    useEffect(() => {
        async function loadStats() {
            const s = await fetchEventCorrelation({ start, end, window: windowDays, metric: "mean_abs_return" });
            setEventStats(s);
        }
        loadStats().catch(console.error);
    }, [start, end, windowDays]);

    const m1 = cp?.mean_switch;
    const tau = m1?.tau_summary || {};
    const cpDate = tau?.tau_mode_date;
    const hdiLow = tau?.tau_hdi_low_date;
    const hdiHigh = tau?.tau_hdi_high_date;

    const selectedEventId = selectedEvent?.event_id;

    const selectedEventStat = useMemo(() => {
        if (!selectedEventId) return null;
        return eventStats.find((x) => String(x.event_id) === String(selectedEventId)) || null;
    }, [selectedEventId, eventStats]);

    return (
        <div style={page}>
            <header style={header}>
                <h2 style={{ margin: 0 }}>Brent Oil — Change Points & Event Impacts</h2>
                <div style={{ opacity: 0.7 }}>Task 3 Dashboard (Flask + React)</div>
            </header>

            <section style={toolbar}>
                <DateRangePicker
                    start={start}
                    end={end}
                    onChange={({ start: s, end: e }) => {
                        setStart(s);
                        setEnd(e);
                    }}
                />
                <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                    <label>
                        Event window (±days):&nbsp;
                        <input
                            type="number"
                            value={windowDays}
                            min={1}
                            max={60}
                            onChange={(e) => setWindowDays(Number(e.target.value))}
                            style={{ width: 80 }}
                        />
                    </label>
                    <button onClick={() => setSelectedEvent(null)}>Clear event highlight</button>
                </div>
            </section>

            <section style={grid}>
                <div style={panel}>
                    <h3 style={h3}>Historical price + change point + event highlight</h3>
                    <PriceChart
                        data={prices}
                        cpDate={cpDate}
                        hdiLow={hdiLow}
                        hdiHigh={hdiHigh}
                        selectedEvent={selectedEvent}
                        highlightWindowDays={windowDays}
                    />
                    {selectedEvent && (
                        <div style={{ marginTop: 10, padding: 12, border: "1px solid #eee", borderRadius: 10 }}>
                            <div><strong>Selected event:</strong> {selectedEvent.event_date} — {selectedEvent.event_title}</div>
                            {selectedEventStat && (
                                <div style={{ marginTop: 6, fontFamily: "monospace" }}>
                                    mean_abs_return={selectedEventStat.mean_abs_return.toFixed(6)} | max_abs_return={selectedEventStat.max_abs_return.toFixed(6)} | n={selectedEventStat.n_obs}
                                </div>
                            )}
                        </div>
                    )}
                </div>

                <div style={side}>
                    <ChangePointSummary title="Model 1 — Mean switch" payload={cp?.mean_switch} />
                    <ChangePointSummary title="Model 2 — Sigma switch" payload={cp?.sigma_switch} />

                    <div style={panel}>
                        <h3 style={h3}>Events (click to highlight)</h3>
                        <EventsTable
                            events={events}
                            selectedEventId={selectedEventId}
                            onSelect={(e) => setSelectedEvent(e)}
                        />
                    </div>

                    <div style={panel}>
                        <h3 style={h3}>Top event windows by mean abs return</h3>
                        <div style={{ maxHeight: 240, overflow: "auto" }}>
                            <ol style={{ margin: 0, paddingLeft: 18 }}>
                                {eventStats.slice(0, 10).map((x) => (
                                    <li key={x.event_id} style={{ marginBottom: 6 }}>
                                        <span style={{ fontFamily: "monospace" }}>{x.event_date}</span>{" "}
                                        — {x.event_title}{" "}
                                        <span style={{ opacity: 0.75, fontFamily: "monospace" }}>
                                            ({x.mean_abs_return.toFixed(6)})
                                        </span>
                                    </li>
                                ))}
                            </ol>
                        </div>
                    </div>
                </div>
            </section>

            <footer style={{ opacity: 0.6, marginTop: 20 }}>
                Tip: Use the date filter + click events to see spikes/drops around real-world events.
            </footer>
        </div>
    );
}

const page = { padding: 16, background: "#fafafa", minHeight: "100vh" };
const header = { marginBottom: 12 };
const toolbar = {
    display: "flex",
    justifyContent: "space-between",
    gap: 14,
    flexWrap: "wrap",
    alignItems: "center",
    padding: 12,
    border: "1px solid #eee",
    borderRadius: 10,
    background: "#fff",
    marginBottom: 12,
};
const grid = {
    display: "grid",
    gridTemplateColumns: "1.6fr 1fr",
    gap: 12,
};
const panel = { border: "1px solid #eee", borderRadius: 10, padding: 14, background: "#fff" };
const side = { display: "grid", gap: 12 };
const h3 = { marginTop: 0, marginBottom: 10 };

