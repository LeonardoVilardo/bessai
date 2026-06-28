"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { FormEvent, ReactNode, useMemo, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

const COLOR = {
  solar: "#d97706",
  battery: "#16a34a",
  grid: "#dc2626",
  load: "#2563eb",
  axis: "#94a3b8",
  gridline: "#e2e8f0",
  tickLabel: "#64748b",
};

type ScenarioInput = {
  location_name: string;
  latitude: number;
  longitude: number;
  timezone: string;
  pv_size_kwp: number;
  average_daily_consumption_kwh: number;
  critical_load_kw: number;
  load_profile_type: string;
  custom_load_shape: number[] | null;
  battery_cost_per_kwh: number;
  grid_tariff_per_kwh: number;
  peak_tariff_per_kwh: number;
  export_credit_per_kwh: number;
  peak_start_hour: number;
  peak_end_hour: number;
  outage_duration_hours_per_month: number;
  currency: string;
};

type Recommendation = {
  battery_size_kwh: number;
  grid_import_kwh: number;
  no_battery_exported_pv_kwh: number;
  exported_pv_kwh: number;
  daily_cost_without_battery: number;
  daily_cost_with_battery: number;
  representative_24h_savings: number;
  annualised_savings_estimate: number;
  average_annual_savings: number | null;
  p50_annual_savings: number | null;
  p90_annual_savings: number | null;
  annual_cost_without_battery: number | null;
  annual_cost_with_battery: number | null;
  battery_cost: number;
  payback_years: number | null;
  outage_backup_hours: number;
  score: number;
  passes_payback_threshold: boolean;
};

type DispatchPoint = {
  time: string;
  pv_generation_kwh: number;
  load_kwh: number;
  battery_soc_kwh: number;
  grid_import_kwh: number;
  no_battery_grid_import_kwh: number;
  battery_charge_kwh: number;
  battery_discharge_kwh: number;
  pv_exported_kwh: number;
  no_battery_pv_exported_kwh: number;
};

type ForecastUncertaintyPoint = {
  time: string;
  raw_shortwave_radiation_w_m2: number;
  p90_uncertainty_w_m2: number | null;
  mae_uncertainty_w_m2: number | null;
};

type SeasonalForecastProfilePoint = {
  week_of_year: number;
  is_current_week: boolean;
  mean_daylight_irradiance_w_m2: number | null;
  p90_uncertainty_w_m2: number | null;
  mean_uncertainty_w_m2: number | null;
  uncertainty_rows: number;
};

type AnnualSimulationDiagnostics = {
  source: string;
  annual_pv_generation_kwh: number;
  annual_load_kwh: number;
  annual_grid_import_without_battery_kwh: number;
  annual_grid_import_with_battery_kwh: number;
  annual_exported_without_battery_kwh: number;
  annual_exported_with_battery_kwh: number;
  self_consumed_pv_kwh: number;
  self_consumption_ratio: number;
  average_annual_savings: number;
  p50_annual_savings: number;
  p90_annual_savings: number;
};

type MlDiagnostics = {
  model_source: string;
  training_rows: number;
  training_period: string;
  mae_before_correction_w_m2: number | null;
  mae_after_correction_w_m2: number | null;
  mean_forecast_error_w_m2: number | null;
  mean_forecast_uncertainty_w_m2: number | null;
  max_forecast_uncertainty_w_m2: number | null;
  evaluation_basis: string;
  uncertainty_basis: string;
  correction_applied: boolean;
  correction_reason: string;
};

type Assumptions = {
  annual_economics_year: number | null;
  annual_economics_years: number[];
  forecast_source_for_dispatch: string;
  ml_correction_applied: boolean;
  ml_correction_reason: string;
  battery_round_trip_efficiency: number;
  reserve_soc_fraction: number;
  load_profile_type: string;
  custom_load_shape_note: string;
  export_credit_modeled: boolean;
  export_credit_per_kwh: number;
  export_credit_note: string;
};

type OptimizeResponse = {
  forecast_source: string;
  ml_correction_applied: boolean;
  ml_correction_reason: string;
  ml_training_source: string;
  ml_training_rows: number;
  ml_training_period: string;
  ml_model_note: string;
  economics_source: string;
  economics_note: string;
  model_diagnostics: {
    annual_simulation: AnnualSimulationDiagnostics;
    ml: MlDiagnostics;
  };
  assumptions: Assumptions;
  scenario: ScenarioInput;
  recommendation: Recommendation;
  comparison: Recommendation[];
  dispatch: DispatchPoint[];
  forecast_uncertainty: ForecastUncertaintyPoint[];
  seasonal_forecast_profile: SeasonalForecastProfilePoint[];
};

const defaultScenario: ScenarioInput = {
  location_name: "Brasilia, Brazil",
  latitude: -15.826016,
  longitude: -47.812539,
  timezone: "America/Sao_Paulo",
  pv_size_kwp: 4,
  average_daily_consumption_kwh: 18,
  critical_load_kw: 1.5,
  load_profile_type: "residential_evening",
  custom_load_shape: null,
  battery_cost_per_kwh: 2500,
  grid_tariff_per_kwh: 0.95,
  peak_tariff_per_kwh: 1.5,
  export_credit_per_kwh: 0,
  peak_start_hour: 18,
  peak_end_hour: 21,
  outage_duration_hours_per_month: 8,
  currency: "BRL",
};

const loadProfileOptions = [
  { value: "residential_evening", label: "Residential evening" },
  { value: "residential_daytime", label: "Residential daytime" },
  { value: "small_business", label: "Small business" },
  { value: "flat", label: "Flat" },
  { value: "custom", label: "Custom" },
];

const locationPresets = [
  {
    id: "brasilia",
    label: "Brasilia",
    location_name: "Brasilia, Brazil",
    latitude: -15.826016,
    longitude: -47.812539,
    timezone: "America/Sao_Paulo",
  },
  {
    id: "west_bahia",
    label: "West of Bahia",
    location_name: "West of Bahia, Brazil",
    latitude: -13.792761,
    longitude: -46.104032,
    timezone: "America/Sao_Paulo",
  },
];

const MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function monthFromIsoWeek(week: number) {
  const weekMidpoint = new Date(Date.UTC(2024, 0, 4 + (week - 1) * 7));
  return weekMidpoint.getUTCMonth();
}

function averageOrNull(values: number[]) {
  if (!values.length) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function flatCustomLoadShape(dailyKwh: number) {
  return Array.from({ length: 24 }, () => dailyKwh / 24);
}

function formatLoadProfile(value: string) {
  return loadProfileOptions.find((option) => option.value === value)?.label ?? value;
}

function formatCurrency(value: number, currency: string) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatTariff(value: number, currency: string) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  }).format(value);
}

function formatNumber(value: number, digits = 1) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

function formatPercent(value: number, digits = 0) {
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

function formatOptionalIrradiance(value: number | null, digits = 1) {
  return value === null ? "Unavailable" : `${formatNumber(value, digits)} W/m²`;
}

function localHour(timestamp: string) {
  return timestamp.slice(11, 16);
}

function setCustomLoadWeight(
  shape: number[] | null,
  hour: number,
  value: number,
  fallbackDailyKwh: number,
) {
  const next = shape && shape.length === 24 ? [...shape] : flatCustomLoadShape(fallbackDailyKwh);
  next[hour] = Number.isFinite(value) ? Math.max(0, value) : 0;
  return next;
}

function scaleCustomLoadShape(
  shape: number[] | null,
  nextDailyKwh: number,
) {
  const safeDailyKwh = Math.max(0, nextDailyKwh);
  const current = shape && shape.length === 24 ? [...shape] : flatCustomLoadShape(safeDailyKwh);
  const currentTotal = current.reduce((total, value) => total + Math.max(0, value), 0);
  if (currentTotal <= 0) {
    return flatCustomLoadShape(safeDailyKwh);
  }
  return current.map((value) => (Math.max(0, value) / currentTotal) * safeDailyKwh);
}

function scenarioFingerprint(scenario: ScenarioInput) {
  return JSON.stringify(scenario);
}

function HelpTooltip({ text }: { text: string }) {
  return (
    <span
      className="group/help relative inline-flex shrink-0 items-center"
      tabIndex={0}
      aria-label={text}
    >
      <span className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 bg-white text-[10px] font-semibold leading-none text-slate-500">
        ?
      </span>
      <span className="pointer-events-none absolute left-1/2 top-full z-50 mt-1 hidden w-56 -translate-x-1/2 rounded-md border border-slate-200 bg-white px-2.5 py-2 text-left text-xs font-normal normal-case leading-relaxed tracking-normal text-slate-600 shadow-lg group-focus/help:inline-block group-hover/help:inline-block">
        {text}
      </span>
    </span>
  );
}

function LabelWithHelp({
  children,
  help,
  className,
}: {
  children: ReactNode;
  help?: string;
  className?: string;
}) {
  return (
    <span className={`inline-flex items-center gap-1.5 ${className ?? ""}`}>
      <span>{children}</span>
      {help ? <HelpTooltip text={help} /> : null}
    </span>
  );
}

function MetricCard({
  label,
  value,
  unit,
  hint,
  help,
  children,
}: {
  label: string;
  value: string;
  unit?: string;
  hint?: string;
  help?: string;
  children?: ReactNode;
}) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-4">
      <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-500">
        <LabelWithHelp help={help}>{label}</LabelWithHelp>
      </p>
      <p className="mt-2 flex items-baseline gap-1.5">
        <span className="text-3xl font-semibold tabular-nums tracking-tight text-slate-900">
          {value}
        </span>
        {unit ? (
          <span className="text-sm font-medium text-slate-500">{unit}</span>
        ) : null}
      </p>
      <p className="mt-1 min-h-[1rem] text-xs text-slate-500">{hint ?? ""}</p>
      {children}
    </div>
  );
}

function AnnualSavingsBand({
  recommendation,
  currency,
}: {
  recommendation: Recommendation | undefined;
  currency: string;
}) {
  if (
    !recommendation ||
    recommendation.average_annual_savings === null ||
    recommendation.p90_annual_savings === null
  ) {
    return null;
  }

  const p50 = recommendation.annualised_savings_estimate;
  const average = recommendation.average_annual_savings;
  const p90 = recommendation.p90_annual_savings;
  const min = Math.min(p90, p50, average);
  const max = Math.max(p90, p50, average);
  const spread = max - min;
  const position = (value: number) => (spread === 0 ? 50 : ((value - min) / spread) * 100);

  return (
    <div className="mt-3 border-t border-slate-100 pt-3">
      <div className="relative h-2 rounded-full bg-slate-100">
        <div
          className="absolute top-0 h-2 rounded-full bg-green-200"
          style={{
            left: `${Math.min(position(p90), position(p50))}%`,
            width: `${Math.abs(position(p50) - position(p90))}%`,
          }}
        />
        <span
          aria-hidden
          className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 rounded-full bg-slate-500"
          style={{ left: `${position(p90)}%` }}
        />
        <span
          aria-hidden
          className="absolute top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-green-700"
          style={{ left: `${position(p50)}%` }}
        />
        <span
          aria-hidden
          className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 rounded-full bg-blue-500"
          style={{ left: `${position(average)}%` }}
        />
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2 text-[10px] leading-tight text-slate-500">
        <span>P90 {formatCurrency(p90, currency)}</span>
        <span className="text-center">P50 {formatCurrency(p50, currency)}</span>
        <span className="text-right">Avg {formatCurrency(average, currency)}</span>
      </div>
    </div>
  );
}

function NumberField({
  label,
  value,
  step,
  suffix,
  help,
  onChange,
}: {
  label: string;
  value: number;
  step?: number;
  suffix?: string;
  help?: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="grid gap-1">
      <LabelWithHelp
        help={help}
        className="text-xs font-medium text-slate-600"
      >
        {label}
      </LabelWithHelp>
      <div className="relative">
        <input
          type="number"
          step={step ?? 0.1}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          className={`h-9 w-full rounded-md border border-slate-300 bg-white px-2.5 ${suffix ? "pr-12" : "pr-2.5"} text-sm tabular-nums text-slate-900 outline-none transition focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10`}
        />
        {suffix ? (
          <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[11px] font-medium text-slate-500">
            {suffix}
          </span>
        ) : null}
      </div>
    </label>
  );
}

function FormSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className="grid gap-2.5 border-t border-slate-200 pt-3 first:border-0 first:pt-0">
      <legend className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
        {title}
      </legend>
      {children}
    </fieldset>
  );
}

function DisclosureSection({
  title,
  summary,
  children,
  defaultOpen = false,
}: {
  title: string;
  summary: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details
      className="group/disclosure rounded-md border border-slate-200 bg-slate-50/70"
      open={defaultOpen}
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-2.5">
        <span>
          <span className="block text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-600">
            {title}
          </span>
          <span className="mt-0.5 block text-[11px] leading-relaxed text-slate-500">
            {summary}
          </span>
        </span>
        <span className="text-lg leading-none text-slate-400 transition group-open/disclosure:rotate-45">
          +
        </span>
      </summary>
      <div className="grid gap-2.5 border-t border-slate-200 bg-white p-3">
        {children}
      </div>
    </details>
  );
}

function LegendChip({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-slate-600">
      <span
        aria-hidden
        className="inline-block h-2 w-2 rounded-full"
        style={{ background: color }}
      />
      {label}
    </span>
  );
}

function DiagnosticItem({
  label,
  value,
  hint,
  help,
}: {
  label: string;
  value: string;
  hint?: string;
  help?: string;
}) {
  return (
    <div className="grid gap-1">
      <dt className="text-[11px] font-medium uppercase tracking-[0.1em] text-slate-500">
        <LabelWithHelp help={help}>{label}</LabelWithHelp>
      </dt>
      <dd className="text-sm font-semibold tabular-nums text-slate-900">
        {value}
      </dd>
      {hint ? <dd className="text-xs text-slate-500">{hint}</dd> : null}
    </div>
  );
}

function AssumptionItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[140px_1fr] gap-3 border-t border-slate-100 py-2 first:border-0">
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="text-xs font-medium leading-relaxed text-slate-800">
        {value}
      </dd>
    </div>
  );
}

function buildRationale(
  rec: Recommendation,
  currency: string,
  candidates: Recommendation[],
  economicsSource: string,
  criticalLoadKw: number,
) {
  const lines: string[] = [];
  const size = rec.battery_size_kwh;

  if (size === 0) {
    lines.push(
      "No battery size beat the no-battery baseline for this scenario.",
    );
    lines.push(
      "Either the tariff is too low, the daily PV surplus is small, or the battery cost per kWh is too high to justify storage.",
    );
    return lines;
  }

  const sizes = candidates.map((c) => c.battery_size_kwh);
  const sizeRange =
    sizes.length > 0
      ? `${Math.min(...sizes)} – ${Math.max(...sizes)} kWh`
      : "the candidate set";
  lines.push(
    `Selected ${size} kWh after sweeping ${candidates.length} candidate sizes across ${sizeRange}.`,
  );

  if (rec.passes_payback_threshold && rec.payback_years) {
    lines.push(
      `Payback of ${formatNumber(rec.payback_years, 1)} years sits inside the 10-year threshold, with annual savings of ${formatCurrency(rec.annualised_savings_estimate, currency)} based on ${economicsSource}.`,
    );
  } else if (rec.payback_years) {
    lines.push(
      `Payback is ${formatNumber(rec.payback_years, 1)} years — outside the 10-year threshold, so the financial case is weak even if the resilience is useful.`,
    );
  } else {
    lines.push(
      "Payback could not be computed because annualised savings are zero or negative — this size was picked on resilience and score only.",
    );
  }

  lines.push(
    `Provides about ${formatNumber(rec.outage_backup_hours, 1)} hours of backup at the configured ${formatNumber(criticalLoadKw, 1)} kW critical-load draw. Resilience is capped at 8 h when scoring so the optimiser does not over-size for outages alone.`,
  );

  const next = candidates
    .filter(
      (row) => row.battery_size_kwh !== size && row.battery_size_kwh !== 0,
    )
    .sort((a, b) => b.score - a.score)[0];
  if (next) {
    const margin = rec.score - next.score;
    lines.push(
      `Closest alternative is ${next.battery_size_kwh} kWh (score ${formatNumber(next.score, 2)} vs ${formatNumber(rec.score, 2)}), so the choice is ${margin > 0.1 ? "clear" : "narrow"}.`,
    );
  }

  return lines;
}

export default function Home() {
  const [scenario, setScenario] = useState(defaultScenario);
  const [data, setData] = useState<OptimizeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dispatchMode, setDispatchMode] = useState<"battery" | "no_battery">("battery");

  const chartData = useMemo(
    () =>
      data?.dispatch.map((point) => ({
        hour: localHour(point.time),
        pv: point.pv_generation_kwh,
        load: point.load_kwh,
        soc: dispatchMode === "battery" ? point.battery_soc_kwh : null,
        grid: dispatchMode === "battery" ? point.grid_import_kwh : point.no_battery_grid_import_kwh,
      })) ?? [],
    [data, dispatchMode],
  );

  const seasonalProfileData = useMemo(
    () => {
      const monthlyBuckets = MONTH_LABELS.map((label, index) => ({
        month: index + 1,
        label,
        irradiance: [] as number[],
        p90: [] as number[],
        mean: [] as number[],
        rows: 0,
      }));

      data?.seasonal_forecast_profile.forEach((point) => {
        const bucket = monthlyBuckets[monthFromIsoWeek(point.week_of_year)];
        if (point.mean_daylight_irradiance_w_m2 !== null) {
          bucket.irradiance.push(point.mean_daylight_irradiance_w_m2);
        }
        if (point.p90_uncertainty_w_m2 !== null) {
          bucket.p90.push(point.p90_uncertainty_w_m2);
        }
        if (point.mean_uncertainty_w_m2 !== null) {
          bucket.mean.push(point.mean_uncertainty_w_m2);
        }
        bucket.rows += point.uncertainty_rows;
      });

      return monthlyBuckets.map((bucket) => ({
        month: bucket.month,
        label: bucket.label,
        irradiance: averageOrNull(bucket.irradiance),
        p90: averageOrNull(bucket.p90),
        mean: averageOrNull(bucket.mean),
        rows: bucket.rows,
      })).map((point) => {
        const band =
          point.irradiance === null || point.p90 === null
            ? null
            : [Math.max(0, point.irradiance - point.p90), point.irradiance + point.p90];
        return { ...point, band };
      });
    },
    [data],
  );

  const currentSeasonalPoint = data?.seasonal_forecast_profile.find((point) => point.is_current_week);
  const currentSeasonalMonth = currentSeasonalPoint ? monthFromIsoWeek(currentSeasonalPoint.week_of_year) + 1 : null;

  const currentLocationPreset =
    locationPresets.find(
      (preset) =>
        preset.location_name === scenario.location_name &&
        preset.latitude === scenario.latitude &&
        preset.longitude === scenario.longitude,
    ) ?? locationPresets[0];

  const customLoadPreview = useMemo(() => {
    const shape = scenario.custom_load_shape ?? flatCustomLoadShape(scenario.average_daily_consumption_kwh);
    const hourlyLoads = shape.map((value) => Math.max(0, value));
    const totalKwh = hourlyLoads.reduce((total, value) => total + value, 0);
    if (scenario.load_profile_type !== "custom" || totalKwh <= 0) {
      return null;
    }

    const peakLoad = Math.max(...hourlyLoads);
    const peakHour = hourlyLoads.findIndex((value) => value === peakLoad);

    return {
      totalKwh,
      peakLoad,
      peakHour,
    };
  }, [
    scenario.average_daily_consumption_kwh,
    scenario.custom_load_shape,
    scenario.load_profile_type,
  ]);

  const inputsChangedSinceRun =
    data !== null && scenarioFingerprint(data.scenario) !== scenarioFingerprint(scenario);

  async function runOptimization(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario }),
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(payload?.detail ?? `API returned ${response.status}`);
      }

      setData((await response.json()) as OptimizeResponse);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Request failed";
      setError(`${message}. Check that the FastAPI server is running on ${API_URL}.`);
    } finally {
      setIsLoading(false);
    }
  }

  const recommendation = data?.recommendation;
  const rationale = data
    ? buildRationale(
        data.recommendation,
        scenario.currency,
        data.comparison,
        data.economics_source,
        data.scenario.critical_load_kw,
      )
    : null;

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-3 border-b border-slate-200 pb-4 md:flex-row md:items-end md:justify-between">
          <div className="grid gap-1">
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-slate-500">
              BESSAi · battery decision-support
            </p>
            <h1 className="text-2xl font-semibold tracking-tight text-slate-900 sm:text-[26px]">
              Sizing &amp; dispatch dashboard
            </h1>
          </div>
          <dl className="grid gap-1 text-xs text-slate-600 md:text-right">
            <div className="flex items-center gap-2 md:justify-end">
              <span
                aria-hidden
                className={`inline-block h-1.5 w-1.5 rounded-full ${data ? "bg-green-600" : "bg-slate-300"}`}
              />
              <dt className="sr-only">Forecast source</dt>
              <dd className="font-mono text-slate-700">
                {data?.forecast_source ?? "Awaiting run"}
              </dd>
            </div>
            <div>
              <dt className="sr-only">Economics source</dt>
              <dd className="font-mono text-slate-700">
                {data?.economics_source ?? "Annual economics pending"}
              </dd>
            </div>
            <div>
              <dt className="sr-only">Site</dt>
              <dd className="tabular-nums">
                {scenario.location_name} ·{" "}
                {formatNumber(scenario.latitude, 3)},{" "}
                {formatNumber(scenario.longitude, 3)}
              </dd>
            </div>
          </dl>
        </header>

        <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
          <aside>
            <form
              onSubmit={runOptimization}
              className="grid gap-4 rounded-md border border-slate-200 bg-white p-4 lg:sticky lg:top-4"
            >
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-900">
                  Scenario inputs
                </h2>
                <button
                  type="button"
                  onClick={() => setScenario(defaultScenario)}
                  className="text-[11px] font-medium text-slate-500 underline-offset-2 hover:text-slate-900 hover:underline"
                >
                  Reset
                </button>
              </div>

              <FormSection title="Site">
                <label className="grid gap-1">
                  <span className="text-xs font-medium text-slate-600">
                    Location
                  </span>
                  <select
                    value={currentLocationPreset.id}
                    onChange={(event) => {
                      const selected =
                        locationPresets.find((preset) => preset.id === event.target.value) ??
                        locationPresets[0];
                      setScenario((current) => ({
                        ...current,
                        location_name: selected.location_name,
                        latitude: selected.latitude,
                        longitude: selected.longitude,
                        timezone: selected.timezone,
                      }));
                    }}
                    className="h-9 rounded-md border border-slate-300 bg-white px-2.5 text-sm text-slate-900 outline-none transition focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10"
                  >
                    {locationPresets.map((preset) => (
                      <option key={preset.id} value={preset.id}>
                        {preset.label}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <div className="grid gap-1">
                    <span className="text-xs font-medium text-slate-600">
                      Latitude
                    </span>
                    <div className="flex h-9 items-center rounded-md border border-slate-200 bg-slate-50 px-2.5 text-sm tabular-nums text-slate-700">
                      {formatNumber(scenario.latitude, 6)}
                    </div>
                  </div>
                  <div className="grid gap-1">
                    <span className="text-xs font-medium text-slate-600">
                      Longitude
                    </span>
                    <div className="flex h-9 items-center rounded-md border border-slate-200 bg-slate-50 px-2.5 text-sm tabular-nums text-slate-700">
                      {formatNumber(scenario.longitude, 6)}
                    </div>
                  </div>
                </div>
                <p className="text-[11px] leading-relaxed text-slate-500">
                  Annual economics load from local cache or NASA POWER when optimisation runs.
                </p>
              </FormSection>

              <FormSection title="System">
                <NumberField
                  label="PV size"
                  value={scenario.pv_size_kwp}
                  suffix="kWp"
                  help="Installed solar PV capacity in kilowatt-peak. This scales the PV generation estimate from irradiance."
                  onChange={(value) =>
                    setScenario((current) => ({ ...current, pv_size_kwp: value }))
                  }
                />
                <NumberField
                  label="Daily consumption"
                  value={scenario.average_daily_consumption_kwh}
                  suffix="kWh"
                  help="Total electricity used in a typical day. In custom mode, this is calculated from the 24 hourly kWh values."
                  onChange={(value) =>
                    setScenario((current) => ({
                      ...current,
                      average_daily_consumption_kwh: value,
                      custom_load_shape:
                        current.load_profile_type === "custom"
                          ? scaleCustomLoadShape(current.custom_load_shape, value)
                          : current.custom_load_shape,
                    }))
                  }
                />
                <label className="grid gap-1">
                  <LabelWithHelp
                    help="Presets use the daily consumption total. Custom mode uses exact hourly kWh values."
                    className="text-xs font-medium text-slate-600"
                  >
                    Load profile
                  </LabelWithHelp>
                  <select
                    value={scenario.load_profile_type}
                    onChange={(event) =>
                      setScenario((current) => ({
                        ...current,
                        load_profile_type: event.target.value,
                        custom_load_shape:
                          event.target.value === "custom"
                            ? current.custom_load_shape ?? flatCustomLoadShape(current.average_daily_consumption_kwh)
                            : current.custom_load_shape,
                      }))
                    }
                    className="h-9 rounded-md border border-slate-300 bg-white px-2.5 text-sm text-slate-900 outline-none transition focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10"
                  >
                    {loadProfileOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  {scenario.load_profile_type === "custom" ? (
                    <span className="text-[11px] text-slate-500">
                      Enter exact hourly kWh. Daily consumption is their sum.
                    </span>
                  ) : null}
                </label>
                {scenario.load_profile_type === "custom" ? (
                  <div className="grid w-full min-w-0 gap-2 overflow-hidden rounded-md border border-slate-200 bg-slate-50 p-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-medium text-slate-600">
                        Hourly kWh
                      </span>
                      <button
                        type="button"
                        onClick={() =>
                          setScenario((current) => ({
                            ...current,
                            custom_load_shape: flatCustomLoadShape(current.average_daily_consumption_kwh),
                          }))
                        }
                        className="text-[11px] font-medium text-slate-500 underline-offset-2 hover:text-slate-900 hover:underline"
                      >
                        Flat
                      </button>
                    </div>
                    <div className="grid min-w-0 grid-cols-6 gap-1">
                      {(scenario.custom_load_shape ?? flatCustomLoadShape(scenario.average_daily_consumption_kwh)).map(
                        (weight, hour) => (
                          <label key={hour} className="grid min-w-0 gap-0.5">
                            <span className="text-[10px] font-medium tabular-nums text-slate-500">
                              {hour.toString().padStart(2, "0")}
                            </span>
                            <input
                              type="number"
                              min={0}
                              step={0.01}
                              value={weight}
                              onChange={(event) =>
                                setScenario((current) => {
                                  const custom_load_shape = setCustomLoadWeight(
                                    current.custom_load_shape,
                                    hour,
                                    Number(event.target.value),
                                    current.average_daily_consumption_kwh,
                                  );
                                  return {
                                    ...current,
                                    custom_load_shape,
                                    average_daily_consumption_kwh:
                                      custom_load_shape.reduce((total, item) => total + item, 0),
                                  };
                                })
                              }
                              className="h-7 w-full min-w-0 rounded border border-slate-300 bg-white px-1 text-xs tabular-nums text-slate-900 outline-none transition focus:border-slate-900 focus:ring-2 focus:ring-slate-900/10"
                            />
                          </label>
                        ),
                      )}
                    </div>
                    <p className="text-[11px] leading-relaxed text-slate-500">
                      {customLoadPreview
                        ? `Daily total: ${formatNumber(customLoadPreview.totalKwh, 1)} kWh/day. Peak hour: ${customLoadPreview.peakHour.toString().padStart(2, "0")}:00 at ${formatNumber(customLoadPreview.peakLoad, 2)} kWh.`
                        : "All hourly values are zero; the backend will use a flat 1 kWh/hour fallback."}
                    </p>
                  </div>
                ) : null}
                <NumberField
                  label="Critical load"
                  value={scenario.critical_load_kw}
                  suffix="kW"
                  help="Essential load the battery should support during an outage, in kW. Higher critical load reduces backup hours."
                  onChange={(value) =>
                    setScenario((current) => ({ ...current, critical_load_kw: value }))
                  }
                />
              </FormSection>

              <DisclosureSection
                title="Economics &amp; tariffs"
                summary="Use defaults for a quick run, then tune costs when comparing cases."
              >
                <NumberField
                  label="Battery cost"
                  value={scenario.battery_cost_per_kwh}
                  step={50}
                  suffix={`${scenario.currency}/kWh`}
                  help="Installed battery cost per usable kWh. Include installation and inverter costs here if you want a conservative payback."
                  onChange={(value) =>
                    setScenario((current) => ({
                      ...current,
                      battery_cost_per_kwh: value,
                    }))
                  }
                />
                <div className="grid grid-cols-2 gap-2">
                  <NumberField
                    label="Grid tariff"
                    value={scenario.grid_tariff_per_kwh}
                    step={0.01}
                    suffix="/kWh"
                    help="Standard import price paid for grid electricity outside the peak window."
                    onChange={(value) =>
                      setScenario((current) => ({
                        ...current,
                        grid_tariff_per_kwh: value,
                      }))
                    }
                  />
                  <NumberField
                    label="Peak tariff"
                    value={scenario.peak_tariff_per_kwh}
                    step={0.01}
                    suffix="/kWh"
                    help="Higher import price during the configured peak window, currently 18:00-21:00."
                    onChange={(value) =>
                      setScenario((current) => ({
                        ...current,
                        peak_tariff_per_kwh: value,
                      }))
                    }
                  />
                </div>
                <NumberField
                  label="Export credit"
                  value={scenario.export_credit_per_kwh}
                  step={0.01}
                  suffix="/kWh"
                  help="Value received for each kWh of surplus PV exported to the grid. This is a simplified input, not country-specific tariff policy."
                  onChange={(value) =>
                    setScenario((current) => ({
                      ...current,
                      export_credit_per_kwh: value,
                    }))
                  }
                />
              </DisclosureSection>

              <button
                type="submit"
                disabled={isLoading}
                className="mt-1 inline-flex h-10 items-center justify-center rounded-md bg-slate-900 px-4 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
              >
                {isLoading
                  ? "Running optimisation…"
                  : inputsChangedSinceRun
                    ? "Re-run optimisation"
                    : "Run optimisation"}
              </button>

              {inputsChangedSinceRun ? (
                <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-800">
                  Inputs changed since the last run. Re-run optimisation to update
                  the charts, recommendation, and diagnostics.
                </p>
              ) : null}

              {error ? (
                <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                  {error}
                </p>
              ) : null}
            </form>
          </aside>

          <section className="grid min-w-0 gap-6">
            <section className="rounded-md border border-slate-200 bg-white">
              <header className="border-b border-slate-200 px-5 py-3">
                <h2 className="text-sm font-semibold text-slate-900">
                  Quick recommendation
                </h2>
                <p className="text-xs text-slate-500">
                  The simple answer first; detailed economics and diagnostics stay below.
                </p>
              </header>
              <div className="grid gap-4 p-5 xl:grid-cols-[minmax(0,1fr)_320px]">
                <div className="grid gap-3 sm:grid-cols-2">
                  <MetricCard
                    label="Recommended size"
                    value={recommendation ? `${recommendation.battery_size_kwh}` : "—"}
                    unit={recommendation ? "kWh" : undefined}
                    help="Battery size selected from the discrete candidate set using savings, payback, and backup resilience."
                    hint={
                      recommendation
                        ? recommendation.passes_payback_threshold
                          ? "Inside 10-year payback"
                          : "Outside 10-year payback"
                        : "Run optimisation to populate"
                    }
                  />
                  <MetricCard
                    label="P50 annual savings"
                    help="Median annual savings across historical weather years when multi-year NASA data is available."
                    value={
                      recommendation
                        ? formatCurrency(
                            recommendation.annualised_savings_estimate,
                            scenario.currency,
                          )
                        : "—"
                    }
                    hint={
                      recommendation?.p90_annual_savings !== null &&
                      recommendation?.p90_annual_savings !== undefined
                        ? `P90: ${formatCurrency(recommendation.p90_annual_savings, scenario.currency)}`
                        : recommendation
                          ? "Historical-year model when available"
                          : ""
                    }
                  >
                    <AnnualSavingsBand
                      recommendation={recommendation}
                      currency={scenario.currency}
                    />
                  </MetricCard>
                  <MetricCard
                    label="Payback"
                    help="Battery cost divided by estimated annual savings. Lower is better."
                    value={
                      recommendation?.payback_years
                        ? formatNumber(recommendation.payback_years, 1)
                        : "—"
                    }
                    unit={recommendation?.payback_years ? "years" : undefined}
                    hint="Threshold: 10 years"
                  />
                  <MetricCard
                    label="Critical-load backup"
                    help="Estimated hours the final battery state can serve the configured critical load."
                    value={
                      recommendation
                        ? formatNumber(recommendation.outage_backup_hours, 1)
                        : "—"
                    }
                    unit={recommendation ? "hours" : undefined}
                    hint={`At ${scenario.critical_load_kw} kW critical load`}
                  />
                </div>

                <section className="overflow-hidden rounded-md border border-slate-200 bg-slate-50">
                  <header className="border-b border-slate-200 bg-white px-4 py-3">
                    <h3 className="text-sm font-semibold text-slate-900">
                      Before vs after
                    </h3>
                    <p className="text-xs text-slate-500">
                      Current 24 h forecast window
                    </p>
                  </header>
                  <dl className="grid divide-y divide-slate-200">
                    <div className="grid grid-cols-[1fr_auto] items-baseline gap-4 px-4 py-3">
                      <dt className="text-[11px] uppercase tracking-[0.1em] text-slate-500">
                        Without battery
                      </dt>
                      <dd className="text-lg font-semibold tabular-nums text-slate-900">
                        {recommendation
                          ? formatCurrency(
                              recommendation.daily_cost_without_battery,
                              scenario.currency,
                            )
                          : "—"}
                      </dd>
                    </div>
                    <div className="grid grid-cols-[1fr_auto] items-baseline gap-4 px-4 py-3">
                      <dt className="text-[11px] uppercase tracking-[0.1em] text-slate-500">
                        With battery
                      </dt>
                      <dd className="text-lg font-semibold tabular-nums text-slate-900">
                        {recommendation
                          ? formatCurrency(
                              recommendation.daily_cost_with_battery,
                              scenario.currency,
                            )
                          : "—"}
                      </dd>
                    </div>
                    <div className="grid grid-cols-[1fr_auto] items-baseline gap-4 bg-white px-4 py-3">
                      <dt className="text-[11px] font-medium uppercase tracking-[0.1em] text-slate-700">
                        24 h saving
                      </dt>
                      <dd className="text-lg font-semibold tabular-nums text-slate-900">
                        {recommendation
                          ? formatCurrency(
                              recommendation.representative_24h_savings,
                              scenario.currency,
                            )
                          : "—"}
                      </dd>
                    </div>
                  </dl>
                </section>
              </div>
            </section>

            <section className="min-w-0 overflow-hidden rounded-md border border-slate-200 bg-white">
              <header className="flex flex-col gap-3 border-b border-slate-200 px-5 py-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className="text-sm font-semibold text-slate-900">
                    Tomorrow&apos;s dispatch
                  </h2>
                  <p className="text-xs text-slate-500">
                    Hourly profile · {dispatchMode === "battery" ? "recommended battery" : "no-battery baseline"}
                  </p>
                </div>
                <div className="grid gap-2 sm:justify-items-end">
                  <div className="inline-flex rounded-md border border-slate-200 bg-slate-50 p-0.5 text-xs">
                    <button
                      type="button"
                      onClick={() => setDispatchMode("battery")}
                      className={`rounded px-2.5 py-1 font-medium transition ${
                        dispatchMode === "battery"
                          ? "bg-white text-slate-900 shadow-sm"
                          : "text-slate-500 hover:text-slate-900"
                      }`}
                    >
                      With battery
                    </button>
                    <button
                      type="button"
                      onClick={() => setDispatchMode("no_battery")}
                      className={`rounded px-2.5 py-1 font-medium transition ${
                        dispatchMode === "no_battery"
                          ? "bg-white text-slate-900 shadow-sm"
                          : "text-slate-500 hover:text-slate-900"
                      }`}
                    >
                      No battery
                    </button>
                  </div>
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
                    <LegendChip color={COLOR.solar} label="PV" />
                    <LegendChip color={COLOR.load} label="Load" />
                    <LegendChip color={COLOR.grid} label="Grid import" />
                    {dispatchMode === "battery" ? (
                      <LegendChip color={COLOR.battery} label="Battery SOC" />
                    ) : null}
                  </div>
                </div>
              </header>
              <div className="h-[340px] min-w-0 px-2 pb-3 pt-4 sm:h-[460px] sm:px-4">
                {chartData.length ? (
                  <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
                    <ComposedChart
                      data={chartData}
                      margin={{ left: -8, right: 8, top: 4, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient id="socFill" x1="0" y1="0" x2="0" y2="1">
                          <stop
                            offset="0%"
                            stopColor={COLOR.battery}
                            stopOpacity={0.18}
                          />
                          <stop
                            offset="100%"
                            stopColor={COLOR.battery}
                            stopOpacity={0.02}
                          />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke={COLOR.gridline} vertical={false} />
                      <XAxis
                        dataKey="hour"
                        stroke={COLOR.axis}
                        tick={{ fontSize: 11, fill: COLOR.tickLabel }}
                        tickLine={false}
                        axisLine={{ stroke: COLOR.gridline }}
                        interval="preserveStartEnd"
                        minTickGap={28}
                      />
                      <YAxis
                        yAxisId="energy"
                        stroke={COLOR.axis}
                        tick={{ fontSize: 11, fill: COLOR.tickLabel }}
                        tickLine={false}
                        axisLine={false}
                        width={44}
                      />
                      {dispatchMode === "battery" ? (
                        <YAxis
                          yAxisId="soc"
                          orientation="right"
                          stroke={COLOR.axis}
                          tick={{ fontSize: 11, fill: COLOR.tickLabel }}
                          tickLine={false}
                          axisLine={false}
                          width={36}
                        />
                      ) : null}
                      <Tooltip
                        cursor={{ stroke: "#cbd5e1", strokeDasharray: "3 3" }}
                        contentStyle={{
                          borderRadius: 6,
                          border: "1px solid #e2e8f0",
                          background: "#fff",
                          fontSize: 12,
                          boxShadow: "0 1px 2px rgb(15 23 42 / 0.06)",
                          padding: "8px 10px",
                        }}
                        labelStyle={{
                          color: "#475569",
                          fontSize: 11,
                          marginBottom: 4,
                        }}
                      />
                      {dispatchMode === "battery" ? (
                        <Area
                          yAxisId="soc"
                          type="monotone"
                          dataKey="soc"
                          name="Battery SOC"
                          stroke={COLOR.battery}
                          strokeWidth={1.5}
                          fill="url(#socFill)"
                        />
                      ) : null}
                      <Line
                        yAxisId="energy"
                        type="monotone"
                        dataKey="pv"
                        name="PV"
                        stroke={COLOR.solar}
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        yAxisId="energy"
                        type="monotone"
                        dataKey="load"
                        name="Load"
                        stroke={COLOR.load}
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        yAxisId="energy"
                        type="monotone"
                        dataKey="grid"
                        name="Grid import"
                        stroke={COLOR.grid}
                        strokeWidth={2}
                        dot={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center rounded-md border border-dashed border-slate-200 bg-slate-50/60 text-sm text-slate-500">
                    Run optimisation to populate the dispatch profile.
                  </div>
                )}
              </div>
            </section>

            <div className="grid min-w-0 gap-6">
              <section className="min-w-0 overflow-hidden rounded-md border border-slate-200 bg-white">
                <header className="flex items-center justify-between border-b border-slate-200 px-5 py-3">
                  <h2 className="text-sm font-semibold text-slate-900">
                    Economics sweep
                  </h2>
                  <span className="text-[11px] uppercase tracking-wide text-slate-500">
                    Sweep results
                  </span>
                </header>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[640px] border-collapse text-left text-sm tabular-nums">
                    <thead className="text-[11px] uppercase tracking-[0.08em] text-slate-500">
                      <tr>
                        <th className="px-5 py-2.5 font-medium">Size</th>
                        <th className="px-3 py-2.5 font-medium">Annual grid</th>
                        <th className="px-3 py-2.5 font-medium">Annual cost</th>
                        <th className="px-3 py-2.5 font-medium">24 h saving</th>
                        <th className="px-3 py-2.5 font-medium">Annualised</th>
                        <th className="px-3 py-2.5 font-medium">Payback</th>
                        <th className="px-3 py-2.5 font-medium">Backup</th>
                        <th className="px-5 py-2.5 font-medium">Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(data?.comparison ?? []).map((row) => {
                        const isPick =
                          row.battery_size_kwh ===
                          recommendation?.battery_size_kwh;
                        return (
                          <tr
                            key={row.battery_size_kwh}
                            className={`border-t border-slate-100 ${isPick ? "bg-slate-50" : ""}`}
                          >
                            <td
                              className={`px-5 py-2 font-medium ${isPick ? "text-slate-900" : "text-slate-700"}`}
                            >
                              <span className="inline-flex items-center gap-2">
                                {row.battery_size_kwh} kWh
                                {isPick ? (
                                  <span className="rounded-full border border-slate-300 bg-white px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-slate-700">
                                    pick
                                  </span>
                                ) : null}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-slate-700">
                              {formatNumber(row.grid_import_kwh, 0)} kWh
                            </td>
                            <td className="px-3 py-2 text-slate-700">
                              {row.annual_cost_with_battery === null
                                ? "n/a"
                                : formatCurrency(
                                    row.annual_cost_with_battery,
                                    scenario.currency,
                                  )}
                            </td>
                            <td className="px-3 py-2 text-slate-700">
                              {formatCurrency(
                                row.representative_24h_savings,
                                scenario.currency,
                              )}
                            </td>
                            <td className="px-3 py-2 text-slate-700">
                              {formatCurrency(
                                row.annualised_savings_estimate,
                                scenario.currency,
                              )}
                            </td>
                            <td className="px-3 py-2 text-slate-700">
                              {row.payback_years
                                ? `${formatNumber(row.payback_years, 1)} yr`
                                : "n/a"}
                            </td>
                            <td className="px-3 py-2 text-slate-700">
                              {formatNumber(row.outage_backup_hours, 1)} h
                            </td>
                            <td className="px-5 py-2 text-slate-700">
                              {formatNumber(row.score, 2)}
                            </td>
                          </tr>
                        );
                      })}
                      {!data ? (
                        <tr>
                          <td
                            colSpan={8}
                            className="px-5 py-6 text-center text-xs text-slate-500"
                          >
                            Run optimisation to populate this comparison.
                          </td>
                        </tr>
                      ) : null}
                    </tbody>
                  </table>
                </div>
                {data?.economics_note ? (
                  <p className="border-t border-slate-200 px-5 py-3 text-xs leading-relaxed text-slate-500">
                    {data.economics_note}
                  </p>
                ) : null}
              </section>
            </div>

            <section className="rounded-md border border-slate-200 bg-white">
              <header className="border-b border-slate-200 px-5 py-3">
                <h2 className="text-sm font-semibold text-slate-900">
                  Why this battery?
                </h2>
                <p className="text-xs text-slate-500">
                  Plain-English rationale derived from the deterministic optimisation.
                </p>
              </header>
              <div className="px-5 py-4">
                {rationale ? (
                  <ul className="grid gap-2.5 text-sm leading-relaxed text-slate-700">
                    {rationale.map((line, idx) => (
                      <li key={idx} className="grid grid-cols-[10px_1fr] gap-2">
                        <span
                          aria-hidden
                          className="mt-2 inline-block h-1 w-1 rounded-full bg-slate-400"
                        />
                        <span>{line}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-slate-500">
                    Run an optimisation to see why a particular battery size is recommended.
                  </p>
                )}
                <div className="mt-4 grid gap-1.5 border-t border-slate-100 pt-3 text-xs text-slate-500 sm:grid-cols-2">
                  <p>Score = 0.5 · savings + 0.3 · resilience + 0.2 · payback.</p>
                  <p>Backup hours are capped at 8 h when scoring.</p>
                  <p>ML estimates forecast uncertainty; dispatch uses the raw forecast.</p>
                  <p>
                    Annual savings use real NASA POWER historical weather years.
                  </p>
                </div>
              </div>
            </section>

            <details className="group/disclosure rounded-md border border-slate-200 bg-white">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-4 border-b border-slate-200 px-5 py-3">
                <span>
                  <span className="block text-sm font-semibold text-slate-900">
                    Advanced diagnostics
                  </span>
                  <span className="mt-0.5 block text-xs text-slate-500">
                    Annual simulation checks, forecast uncertainty status, and model notes.
                  </span>
                </span>
                <span className="text-xl leading-none text-slate-400 transition group-open/disclosure:rotate-45">
                  +
                </span>
              </summary>
              {data ? (
                <div className="grid gap-6 px-5 py-4">
                  <div className="grid gap-3">
                    <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
                      Annual simulation
                    </h3>
                    <dl className="grid gap-4 sm:grid-cols-2">
                      <DiagnosticItem
                        label="PV generation"
                        help="Annual PV energy generated by the configured array under the historical irradiance simulation."
                        value={`${formatNumber(data.model_diagnostics.annual_simulation.annual_pv_generation_kwh, 0)} kWh`}
                      />
                      <DiagnosticItem
                        label="Load"
                        value={`${formatNumber(data.model_diagnostics.annual_simulation.annual_load_kwh, 0)} kWh`}
                      />
                      <DiagnosticItem
                        label="Grid import · no battery"
                        value={`${formatNumber(data.model_diagnostics.annual_simulation.annual_grid_import_without_battery_kwh, 0)} kWh`}
                      />
                      <DiagnosticItem
                        label="Grid import · with battery"
                        value={`${formatNumber(data.model_diagnostics.annual_simulation.annual_grid_import_with_battery_kwh, 0)} kWh`}
                      />
                      <DiagnosticItem
                        label="Exported PV · no battery"
                        help="Surplus PV exported to the grid in the no-battery baseline."
                        value={`${formatNumber(data.model_diagnostics.annual_simulation.annual_exported_without_battery_kwh, 0)} kWh`}
                      />
                      <DiagnosticItem
                        label="Exported PV · with battery"
                        help="Surplus PV exported after the recommended battery has charged from available solar."
                        value={`${formatNumber(data.model_diagnostics.annual_simulation.annual_exported_with_battery_kwh, 0)} kWh`}
                      />
                      <DiagnosticItem
                        label="Self-consumed PV"
                        help="PV used onsite directly or after battery storage, divided by total PV generation."
                        value={`${formatNumber(data.model_diagnostics.annual_simulation.self_consumed_pv_kwh, 0)} kWh`}
                        hint={formatPercent(
                          data.model_diagnostics.annual_simulation.self_consumption_ratio,
                        )}
                      />
                      <DiagnosticItem
                        label="Avg annual savings"
                        help="Mean annual savings across the simulated historical weather years."
                        value={formatCurrency(
                          data.model_diagnostics.annual_simulation.average_annual_savings,
                          scenario.currency,
                        )}
                      />
                      <DiagnosticItem
                        label="P50 annual savings"
                        help="Median annual savings across the simulated historical weather years; this is used for recommendation when multi-year economics are available."
                        value={formatCurrency(
                          data.model_diagnostics.annual_simulation.p50_annual_savings,
                          scenario.currency,
                        )}
                      />
                      <DiagnosticItem
                        label="P90 annual savings"
                        help="10th-percentile annual savings: a conservative weather-year estimate where 90% of simulated years are better."
                        value={formatCurrency(
                          data.model_diagnostics.annual_simulation.p90_annual_savings,
                          scenario.currency,
                        )}
                      />
                    </dl>
                    <p className="text-xs leading-relaxed text-slate-500">
                      {data.model_diagnostics.annual_simulation.source}
                    </p>
                  </div>

                  <div className="grid gap-3">
                    <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
                      Forecast uncertainty
                    </h3>
                    {seasonalProfileData.length ? (
                      <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                          <p className="text-xs font-medium text-slate-700">
                            Seasonal uncertainty profile
                          </p>
                          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                            <LegendChip color={COLOR.solar} label="Historical irradiance" />
                            <LegendChip color="#8b5cf6" label="P90 forecast-error band" />
                          </div>
                        </div>
                        <div className="h-64 min-w-0">
                          <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
                            <ComposedChart
                              data={seasonalProfileData}
                              margin={{ left: -8, right: 8, top: 4, bottom: 0 }}
                            >
                              <CartesianGrid stroke={COLOR.gridline} vertical={false} />
                              <XAxis
                                dataKey="month"
                                stroke={COLOR.axis}
                                tick={{ fontSize: 11, fill: COLOR.tickLabel }}
                                tickLine={false}
                                axisLine={{ stroke: COLOR.gridline }}
                                tickFormatter={(value) => MONTH_LABELS[Number(value) - 1] ?? String(value)}
                                minTickGap={8}
                              />
                              <YAxis
                                yAxisId="irradiance"
                                stroke={COLOR.axis}
                                tick={{ fontSize: 11, fill: COLOR.tickLabel }}
                                tickLine={false}
                                axisLine={false}
                                width={44}
                              />
                              <Tooltip
                                cursor={{ stroke: "#cbd5e1", strokeDasharray: "3 3" }}
                                contentStyle={{
                                  borderRadius: 6,
                                  border: "1px solid #e2e8f0",
                                  background: "#fff",
                                  fontSize: 12,
                                  boxShadow: "0 1px 2px rgb(15 23 42 / 0.06)",
                                  padding: "8px 10px",
                                }}
                                labelFormatter={(value) => MONTH_LABELS[Number(value) - 1] ?? `Month ${value}`}
                                formatter={(value, name) => {
                                  if (Array.isArray(value)) {
                                    return [
                                      `${formatNumber(Number(value[0]), 0)}-${formatNumber(Number(value[1]), 0)} W/m²`,
                                      name,
                                    ];
                                  }
                                  if (typeof value === "number") {
                                    return [`${formatNumber(value, 0)} W/m²`, name];
                                  }
                                  return [value, name];
                                }}
                              />
                              {currentSeasonalMonth ? (
                                <ReferenceLine
                                  yAxisId="irradiance"
                                  x={currentSeasonalMonth}
                                  stroke="#0f172a"
                                  strokeDasharray="4 4"
                                  label={{
                                    value: "current",
                                    position: "insideTopRight",
                                    fill: "#334155",
                                    fontSize: 10,
                                  }}
                                />
                              ) : null}
                              <Area
                                yAxisId="irradiance"
                                type="monotone"
                                dataKey="band"
                                name="P90 forecast-error band"
                                stroke="none"
                                fill="#8b5cf6"
                                fillOpacity={0.18}
                                activeDot={false}
                              />
                              <Line
                                yAxisId="irradiance"
                                type="monotone"
                                dataKey="irradiance"
                                name="Historical daylight irradiance"
                                stroke={COLOR.solar}
                                strokeWidth={2}
                                dot={false}
                              />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                        <p className="mt-2 text-xs leading-relaxed text-slate-500">
                          This annual view compares average historical daylight irradiance with archived forecast-error uncertainty.
                          The shaded band is the seasonal P90 forecast-error range around the historical irradiance line:
                          90% of comparable historical daylight forecast misses were smaller than that band width.
                          Higher P90 means more surprise cloud/rain risk relative to forecasts. Dispatch still uses the raw Open-Meteo forecast.
                        </p>
                      </div>
                    ) : null}
                    <dl className="grid gap-4 sm:grid-cols-2">
                      <DiagnosticItem
                        label="Model source"
                        value={data.model_diagnostics.ml.model_source}
                      />
                      <DiagnosticItem
                        label="Training rows"
                        value={
                          data.model_diagnostics.ml.training_rows > 0
                            ? formatNumber(data.model_diagnostics.ml.training_rows, 0)
                            : "Unavailable"
                        }
                        hint={data.model_diagnostics.ml.training_period}
                      />
                      <DiagnosticItem
                        label="Raw forecast MAE"
                        help="Mean absolute irradiance forecast error on the validation holdout before any model estimate."
                        value={formatOptionalIrradiance(data.model_diagnostics.ml.mae_before_correction_w_m2, 1)}
                      />
                      <DiagnosticItem
                        label="Model residual MAE"
                        help="Validation error after estimating forecast error. This is reported for trust, not used to change dispatch."
                        value={formatOptionalIrradiance(data.model_diagnostics.ml.mae_after_correction_w_m2, 1)}
                        hint={data.model_diagnostics.ml.evaluation_basis}
                      />
                      <DiagnosticItem
                        label="Mean forecast error"
                        value={formatOptionalIrradiance(data.model_diagnostics.ml.mean_forecast_error_w_m2, 1)}
                      />
                      <DiagnosticItem
                        label="Mean uncertainty"
                        help="Average P90 forecast uncertainty for the current seasonal window during daylight hours."
                        value={formatOptionalIrradiance(data.model_diagnostics.ml.mean_forecast_uncertainty_w_m2, 1)}
                        hint={
                          data.model_diagnostics.ml.max_forecast_uncertainty_w_m2 === null
                            ? "Max unavailable"
                            : `Max ${formatNumber(data.model_diagnostics.ml.max_forecast_uncertainty_w_m2, 1)} W/m²`
                        }
                      />
                    </dl>
                    <p className="text-xs leading-relaxed text-slate-500">
                      {data.ml_model_note}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="px-5 py-6 text-sm text-slate-500">
                  Run optimisation to populate diagnostics.
                </div>
              )}
            </details>

            <section className="rounded-md border border-slate-200 bg-white">
              <header className="border-b border-slate-200 px-5 py-3">
                <h2 className="text-sm font-semibold text-slate-900">
                  Assumptions
                </h2>
                <p className="text-xs text-slate-500">
                  Current modelling choices used for this run.
                </p>
              </header>
              {data ? (
                <dl className="px-5 py-3">
                  <AssumptionItem
                    label="Annual economics"
                    value={
                      data.assumptions.annual_economics_years.length > 1
                        ? `NASA POWER ${data.assumptions.annual_economics_years[0]}-${data.assumptions.annual_economics_years[data.assumptions.annual_economics_years.length - 1]} with P50/P90 savings`
                        : data.assumptions.annual_economics_year
                        ? `NASA POWER ${data.assumptions.annual_economics_year}`
                        : "NASA POWER historical economics unavailable"
                    }
                  />
                  <AssumptionItem
                    label="Dispatch forecast"
                    value={data.assumptions.forecast_source_for_dispatch}
                  />
                  <AssumptionItem
                    label="ML uncertainty"
                    value={data.assumptions.ml_correction_reason}
                  />
                  <AssumptionItem
                    label="Battery efficiency"
                    value={`${formatPercent(data.assumptions.battery_round_trip_efficiency)} round-trip`}
                  />
                  <AssumptionItem
                    label="Reserve SOC"
                    value={formatPercent(data.assumptions.reserve_soc_fraction)}
                  />
                  <AssumptionItem
                    label="Load profile"
                    value={`${formatLoadProfile(data.assumptions.load_profile_type)}. ${data.assumptions.custom_load_shape_note}`}
                  />
                  <AssumptionItem
                    label="Export credit"
                    value={`${formatTariff(data.assumptions.export_credit_per_kwh, scenario.currency)}/kWh. ${data.assumptions.export_credit_note}`}
                  />
                </dl>
              ) : (
                <div className="px-5 py-6 text-sm text-slate-500">
                  Run optimisation to see the assumptions used.
                </div>
              )}
            </section>
          </section>
        </div>

        <footer className="border-t border-slate-200 pt-4 text-xs text-slate-500">
          BESSAi MVP · deterministic dispatch with historical annual economics · prices in {scenario.currency}.
        </footer>
      </div>
    </main>
  );
}
