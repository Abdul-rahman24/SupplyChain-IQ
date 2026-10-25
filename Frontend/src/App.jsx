
import { useState } from "react";
import ForecastPanel from "./ForecastPanel";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api/predict/";

const initialForm = {
  Type: "DEBIT",
  "Days for shipment (scheduled)": 4,
  "Category Name": "Sporting Goods",
  "Customer Segment": "Consumer",
  "Department Name": "Fitness",
  Market: "Pacific Asia",
  "Order Country": "India",
  "Order Region": "South Asia",
  "Shipping Mode": "Standard Class",
  "Order Item Quantity": 1,
  "Order Item Product Price": 327.75,
  "Order Item Discount Rate": 0.05,
  order_year: 2018,
  order_month: 1,
  order_day_of_week: 5,
};

const fields = [
  {
    title: "Order details",
    items: [
      { key: "Type", label: "Payment type", options: ["DEBIT", "TRANSFER", "CASH", "PAYMENT"] },
      { key: "Category Name", label: "Product category", options: ["Sporting Goods", "Cleats", "Women's Apparel", "Electronics", "Fishing", "Camping & Hiking"] },
      { key: "Customer Segment", label: "Customer segment", options: ["Consumer", "Corporate", "Home Office"] },
      { key: "Department Name", label: "Department", options: ["Fitness", "Footwear", "Apparel", "Golf", "Outdoors", "Technology"] },
      { key: "Order Item Quantity", label: "Quantity", type: "number", min: 1, step: 1 },
      { key: "Order Item Product Price", label: "Product price", type: "number", min: 0, step: "0.01" },
      { key: "Order Item Discount Rate", label: "Discount rate (0–1)", type: "number", min: 0, max: 1, step: "0.01" },
    ],
  },
  {
    title: "Shipping and destination",
    items: [
      { key: "Shipping Mode", label: "Shipping mode", options: ["Standard Class", "Second Class", "First Class", "Same Day"] },
      { key: "Days for shipment (scheduled)", label: "Scheduled shipping days", type: "number", min: 0, step: 1 },
      { key: "Market", label: "Market", options: ["Pacific Asia", "Europe", "LATAM", "USCA", "Africa", "EMEA", "Asia Pacific"] },
      { key: "Order Country", label: "Destination country", options: ["India", "Indonesia", "Australia", "China", "Japan", "United States", "Germany", "France"] },
      { key: "Order Region", label: "Destination region", options: ["South Asia", "Southeast Asia", "Oceania", "Eastern Asia", "Western Europe", "Central America", "West of USA"] },
    ],
  },
  {
    title: "Order timing",
    items: [
      { key: "order_year", label: "Order year", type: "number", min: 2015, max: 2018, step: 1 },
      { key: "order_month", label: "Order month", type: "number", min: 1, max: 12, step: 1 },
      { key: "order_day_of_week", label: "Day of week (0–6)", type: "number", min: 0, max: 6, step: 1 },
    ],
  },
];

const categoricalKeys = [
  "Type",
  "Category Name",
  "Customer Segment",
  "Department Name",
  "Market",
  "Order Country",
  "Order Region",
  "Shipping Mode",
];

function FormField({ field, value, onChange }) {
  const inputClass =
    "mt-2 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-800 outline-none focus:border-teal-600 focus:ring-4 focus:ring-teal-600/10";

  return (
    <label className="block text-sm font-medium text-slate-600">
      {field.label}
      {field.options ? (
        <select
          className={inputClass}
          value={value}
          onChange={(event) => onChange(field.key, event.target.value)}
        >
          {field.options.map((option) => (
            <option key={option} value={option}>{option}</option>
          ))}
        </select>
      ) : (
        <input
          className={inputClass}
          type="number"
          min={field.min}
          max={field.max}
          step={field.step}
          required
          value={value}
          onChange={(event) => onChange(field.key, event.target.value)}
        />
      )}
    </label>
  );
}

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("delivery");

  function updateField(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const payload = Object.fromEntries(
      Object.entries(form).map(([key, value]) => [
        key,
        categoricalKeys.includes(key) ? value : Number(value),
      ])
    );

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        const details = data.missing_features
          ? `Missing fields: ${data.missing_features.join(", ")}`
          : data.errors
            ? data.errors.map((item) => `${item.field}: ${item.error}`).join("; ")
            : data.error || "The prediction request failed.";

        throw new Error(details);
      }

      setResult(data);
    } catch (err) {
      if (err instanceof TypeError) {
        setError(
          "Cannot connect to Django. Check that the backend is running and CORS is configured."
        );
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

  const probability = result
    ? result.late_delivery_risk_probability * 100
    : null;

  const highRisk = result?.predicted_late_delivery_risk === 1;

  return (
    <div className="min-h-screen bg-[#f4f7fa] text-slate-800">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 md:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#103b3b] text-sm font-bold text-white">
              SQ
            </div>
            <div>
              <h1 className="font-bold tracking-tight text-slate-900">
                SupplyChain-IQ
              </h1>
              <p className="text-xs text-slate-500">
                Predictive operations
              </p>
            </div>
          </div>
          <span className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-800">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            Development
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-7 px-5 py-8 md:px-8 md:py-10">
        <nav aria-label="Prediction tools" className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => setActiveTab("delivery")}
            aria-current={activeTab === "delivery" ? "page" : undefined}
            className={`rounded-lg px-5 py-3 text-sm font-semibold transition ${
              activeTab === "delivery"
                ? "bg-[#12665b] text-white"
                : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
            }`}
          >
            Late Delivery Risk
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("forecast")}
            aria-current={activeTab === "forecast" ? "page" : undefined}
            className={`rounded-lg px-5 py-3 text-sm font-semibold transition ${
              activeTab === "forecast"
                ? "bg-[#12665b] text-white"
                : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
            }`}
          >
            Demand Forecasting
          </button>
        </nav>

        {activeTab === "delivery" ? (
          <>
        <section className="rounded-2xl bg-[#103b3b] p-6 text-white md:p-9">
          <p className="text-xs font-semibold uppercase tracking-[0.17em] text-teal-200">
            Prediction workspace
          </p>
          <h2 className="mt-3 max-w-2xl text-3xl font-semibold tracking-tight md:text-4xl">
            Late-delivery risk analysis
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
            Submit order, shipping, destination, and timing information to
            estimate late-delivery risk using the trained DataCo model.
          </p>
        </section>

        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1.5fr)_minmax(300px,0.8fr)]">
          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm md:p-7">
            <h3 className="text-lg font-semibold text-slate-900">
              Order information
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              Complete all 15 model inputs.
            </p>

            <form onSubmit={handleSubmit} className="mt-7 space-y-7">
              {fields.map((group, index) => (
                <fieldset key={group.title}>
                  <legend className="mb-4 flex items-center gap-3 text-sm font-semibold text-slate-800">
                    <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#e8f4f1] text-xs text-[#12665b]">
                      0{index + 1}
                    </span>
                    {group.title}
                  </legend>
                  <div className="grid gap-4 sm:grid-cols-2">
                    {group.items.map((field) => (
                      <FormField
                        key={field.key}
                        field={field}
                        value={form[field.key]}
                        onChange={updateField}
                      />
                    ))}
                  </div>
                </fieldset>
              ))}

              {error && (
                <div
                  role="alert"
                  className="break-words rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm leading-6 text-rose-800"
                >
                  <p className="font-semibold">Prediction failed</p>
                  <p className="mt-1">{error}</p>
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-lg bg-[#12665b] px-5 py-3.5 text-sm font-semibold text-white transition hover:bg-[#0f554c] disabled:cursor-wait disabled:opacity-60"
              >
                {loading ? "Analyzing order..." : "Predict delivery risk"}
              </button>
            </form>
          </section>

          <aside
            aria-live="polite"
            className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm xl:sticky xl:top-5"
          >
            <div className="border-b border-slate-100 px-5 py-5">
              <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-400">
                Django API
              </p>
              <h3 className="mt-1.5 text-lg font-semibold text-slate-900">
                Prediction result
              </h3>
            </div>

            {!result ? (
              <div className="px-6 py-12 text-center">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-xl bg-slate-100 text-2xl text-slate-500">
                  ◉
                </div>
                <h4 className="mt-4 font-semibold text-slate-800">
                  Awaiting prediction
                </h4>
                <p className="mt-2 text-sm leading-6 text-slate-500">
                  Submit the form to display the response returned by your
                  model.
                </p>
              </div>
            ) : (
              <div className="p-5">
                <p className="text-sm text-slate-500">
                  Estimated risk probability
                </p>
                <p className="mt-3 text-5xl font-semibold tracking-tight text-slate-900">
                  {probability.toFixed(2)}%
                </p>

                <div className="mt-5 h-2 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full ${
                      highRisk ? "bg-amber-500" : "bg-emerald-500"
                    }`}
                    style={{
                      width: `${Math.min(100, Math.max(0, probability))}%`,
                    }}
                  />
                </div>

                <div
                  className={`mt-6 rounded-lg border p-4 ${
                    highRisk
                      ? "border-amber-200 bg-amber-50"
                      : "border-emerald-200 bg-emerald-50"
                  }`}
                >
                  <p className="text-xs uppercase tracking-wide text-slate-500">
                    Model classification
                  </p>
                  <p
                    className={`mt-2 text-sm font-semibold ${
                      highRisk ? "text-amber-900" : "text-emerald-900"
                    }`}
                  >
                    {result.interpretation}
                  </p>
                </div>

                <dl className="mt-5 divide-y divide-slate-100 text-sm">
                  <div className="flex justify-between gap-3 py-3">
                    <dt className="text-slate-500">Threshold</dt>
                    <dd className="font-medium">
                      {(result.threshold * 100).toFixed(0)}%
                    </dd>
                  </div>
                  <div className="flex justify-between gap-3 py-3">
                    <dt className="text-slate-500">Predicted label</dt>
                    <dd className="font-medium">
                      {result.predicted_late_delivery_risk}
                    </dd>
                  </div>
                  <div className="py-3">
                    <dt className="text-slate-500">Model</dt>
                    <dd className="mt-1 break-words font-medium">
                      {result.model}
                    </dd>
                  </div>
                </dl>

                <p className="mt-4 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-500">
                  This is an estimate of the model's target label, not a
                  guarantee that a shipment will be late.
                </p>
                <button
                  type="button"
                  onClick={() => setResult(null)}
                  className="mt-4 w-full rounded-lg border border-slate-200 px-4 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  Clear result
                </button>
              </div>
            )}
          </aside>
        </div>
          </>
        ) : (
          <ForecastPanel />
        )}

        <footer className="border-t border-slate-200 py-5 text-xs text-slate-400">
          SupplyChain-IQ · DataCo late-delivery prediction
        </footer>
      </main>
    </div>
  );
}
