
import { useEffect, useState } from "react";

const API_BASE = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api/predict/"
).replace(/\/predict\/?$/, "");

const inputClass =
  "mt-2 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-800 outline-none focus:border-teal-600 focus:ring-4 focus:ring-teal-600/10";

function ForecastChart({ points }) {
  const width = 700;
  const height = 280;
  const padding = { top: 24, right: 24, bottom: 48, left: 48 };

  const values = points.map((point) => point.predicted_order_frequency);
  const maxValue = Math.max(1, ...values) * 1.15;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const coordinates = points.map((point, index) => ({
    x:
      padding.left +
      (points.length === 1 ? plotWidth / 2 : (index / (points.length - 1)) * plotWidth),
    y:
      padding.top +
      plotHeight -
      (point.predicted_order_frequency / maxValue) * plotHeight,
    ...point,
  }));

  const line = coordinates
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");

  return (
    <div className="w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Monthly predicted order frequency line chart"
        className="w-full min-w-[480px]"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const y = padding.top + plotHeight * fraction;
          const label = (maxValue * (1 - fraction)).toFixed(1);

          return (
            <g key={fraction}>
              <line
                x1={padding.left}
                x2={width - padding.right}
                y1={y}
                y2={y}
                stroke="#e2e8f0"
                strokeDasharray="4 4"
              />
              <text
                x={padding.left - 8}
                y={y + 4}
                fontSize="11"
                fill="#64748b"
                textAnchor="end"
              >
                {label}
              </text>
            </g>
          );
        })}

        <path d={line} fill="none" stroke="#0f766e" strokeWidth="3" />

        {coordinates.map((point) => (
          <circle
            key={point.month}
            cx={point.x}
            cy={point.y}
            r="4"
            fill="#0f766e"
            stroke="white"
            strokeWidth="2"
          >
            <title>
              {point.month}: {point.predicted_order_frequency}
            </title>
          </circle>
        ))}

        {coordinates.map((point, index) => {
          const showLabel =
            points.length <= 12 ||
            index === 0 ||
            index === points.length - 1 ||
            index % 3 === 0;

          return showLabel ? (
            <text
              key={point.month}
              x={point.x}
              y={height - 17}
              fontSize="10"
              fill="#64748b"
              textAnchor="middle"
            >
              {point.month}
            </text>
          ) : null;
        })}
      </svg>
    </div>
  );
}

export default function ForecastPanel() {
  const [productModels, setProductModels] = useState([]);
  const [productId, setProductId] = useState("");
  const [horizon, setHorizon] = useState("12");
  const [result, setResult] = useState(null);
  const [loadingModels, setLoadingModels] = useState(true);
  const [loadingForecast, setLoadingForecast] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadProductModels() {
      setLoadingModels(true);
      setError("");

      try {
        const response = await fetch(`${API_BASE}/product-models/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Unable to load product models.");
        }

        if (!cancelled) {
          setProductModels(data.product_models || []);
          setProductId((data.product_models || [])[0] || "");
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof TypeError
              ? "Cannot connect to Django. Check that the backend is running."
              : err.message
          );
        }
      } finally {
        if (!cancelled) setLoadingModels(false);
      }
    }

    loadProductModels();

    return () => {
      cancelled = true;
    };
  }, []);

  async function handleForecast(event) {
    event.preventDefault();
    setLoadingForecast(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/forecast/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          product_model_id: productId,
          horizon: Number(horizon),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Forecast request failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Cannot connect to Django. Check that the backend is running and CORS is configured."
          : err.message
      );
    } finally {
      setLoadingForecast(false);
    }
  }

  const points = result?.forecast || [];
  const totalForecast = points.reduce(
    (sum, point) => sum + point.predicted_order_frequency,
    0
  );
  const averageForecast = points.length ? totalForecast / points.length : 0;

  return (
    <section className="space-y-6">
      <div className="rounded-2xl bg-[#103b3b] p-6 text-white md:p-9">
        <p className="text-xs font-semibold uppercase tracking-[0.17em] text-teal-200">
          Predictive planning
        </p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight md:text-4xl">
          Demand forecasting
        </h2>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
          Forecast monthly order frequency for a product model using ARIMA.
          Explore the expected trend to support planning decisions.
        </p>
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(280px,0.75fr)_minmax(0,1.5fr)]">
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm md:p-7">
          <h3 className="text-lg font-semibold text-slate-900">
            Forecast settings
          </h3>
          <p className="mt-1 text-sm text-slate-500">
            Choose a product model and forecast period.
          </p>

          <form onSubmit={handleForecast} className="mt-6 space-y-5">
            <label className="block text-sm font-medium text-slate-600">
              Product model ID
              <select
                className={inputClass}
                value={productId}
                onChange={(event) => {
                  setProductId(event.target.value);
                  setResult(null);
                }}
                disabled={loadingModels || productModels.length === 0}
                required
              >
                {loadingModels && <option value="">Loading products...</option>}
                {productModels.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </label>

            <label className="block text-sm font-medium text-slate-600">
              Forecast horizon
              <select
                className={inputClass}
                value={horizon}
                onChange={(event) => setHorizon(event.target.value)}
              >
                {[1, 3, 6, 9, 12, 18, 24].map((months) => (
                  <option key={months} value={months}>
                    {months} {months === 1 ? "month" : "months"}
                  </option>
                ))}
              </select>
            </label>

            <button
              type="submit"
              disabled={
                loadingModels ||
                loadingForecast ||
                !productId ||
                productModels.length === 0
              }
              className="w-full rounded-lg bg-[#12665b] px-5 py-3.5 text-sm font-semibold text-white transition hover:bg-[#0f554c] disabled:cursor-wait disabled:opacity-60"
            >
              {loadingForecast ? "Generating forecast..." : "Generate forecast"}
            </button>
          </form>

          <p className="mt-5 text-xs leading-5 text-slate-500">
            Model: ARIMA(1, 1, 1). Forecasts are estimates based on historical
            data; they are not guaranteed outcomes.
          </p>
        </section>

        <section className="min-w-0 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm md:p-7">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-400">
              Forecast results
            </p>
            <h3 className="mt-1 text-lg font-semibold text-slate-900">
              Monthly order frequency
            </h3>
          </div>

          {error && (
            <div
              role="alert"
              className="mt-5 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm leading-6 text-rose-800"
            >
              <p className="font-semibold">Forecast request failed</p>
              <p>{error}</p>
            </div>
          )}

          {!result ? (
            <div className="py-16 text-center">
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-xl bg-slate-100 text-2xl text-slate-500">
                ↗
              </div>
              <h4 className="mt-4 font-semibold text-slate-800">
                {loadingModels ? "Loading product models" : "Ready to forecast"}
              </h4>
              <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-slate-500">
                Choose your settings and generate a forecast to see the chart
                and monthly values here.
              </p>
            </div>
          ) : (
            <div className="mt-6 space-y-6">
              <div className="grid gap-3 sm:grid-cols-3">
                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-xs text-slate-500">Product model</p>
                  <p className="mt-2 break-all text-sm font-semibold text-slate-900">
                    {result.product_model_id}
                  </p>
                </div>
                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-xs text-slate-500">Average per month</p>
                  <p className="mt-2 text-2xl font-semibold text-slate-900">
                    {averageForecast.toFixed(2)}
                  </p>
                </div>
                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-xs text-slate-500">Forecast period</p>
                  <p className="mt-2 text-sm font-semibold text-slate-900">
                    {result.forecast_horizon_months} months
                  </p>
                </div>
              </div>

              <div>
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <h4 className="font-semibold text-slate-800">
                    Predicted monthly trend
                  </h4>
                  <span className="text-xs text-slate-500">
                    History: {result.history_start} – {result.history_end}
                  </span>
                </div>
                <ForecastChart points={points} />
              </div>

              <div className="overflow-x-auto rounded-xl border border-slate-200">
                <table className="w-full text-left text-sm">
                  <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                    <tr>
                      <th className="px-4 py-3 font-semibold">Month</th>
                      <th className="px-4 py-3 text-right font-semibold">
                        Predicted order frequency
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {points.map((point) => (
                      <tr key={point.month}>
                        <td className="px-4 py-3 text-slate-700">
                          {point.month}
                        </td>
                        <td className="px-4 py-3 text-right font-medium text-slate-900">
                          {point.predicted_order_frequency.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="rounded-lg bg-amber-50 p-4 text-xs leading-5 text-amber-900">
                These predictions represent monthly order frequency, not
                physical units required for inventory replenishment. Use them
                as estimates alongside other planning information.
              </p>
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
