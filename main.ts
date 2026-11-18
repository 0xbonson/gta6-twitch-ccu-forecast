/**
 * GTA VI Twitch CCU Launch-Day Hourly Forecast
 *
 * Competition: make.fun "gta6-twitch-ccu-forecast-01"
 *
 * Forecast Methodology:
 * Rather than assuming a random walk around the initial print (the competition baseline),
 * this model derives the empirical trajectory of release-day viewership from historical
 * comparable launches.
 *
 * Key components of the model:
 * 1. Historical Reference Pool:
 *    Extracts release-day trajectories (hours 1 to 24) from past major launches,
 *    excluding pre-release monitoring.
 *
 * 2. Multi-Metric Similarity Weighting:
 *    Weights each historical launch based on similarity to the target game's hour-1
 *    viewer count and active broadcast channel count:
 *      w = exp(-w_c * |ln((c + 1)/(c_ref + 1))|) * exp(-w_v * |ln((v + 1)/(v_ref + 1))|)
 *    with channel weight w_c = 0.25 and viewer weight w_v = 1.0.
 *
 * 3. Relevance Weight Thresholding:
 *    Excludes comparables whose similarity falls below a minimum relevance threshold
 *    (cut-off at 0.05). This filters out dissimilar trajectories that add noise.
 *
 * 4. Temporal Window Smoothing:
 *    Aggregates log-ratios ln(viewers_t / viewers_1) across a temporal window
 *    of +/- 3 hours around each target hour, capturing diurnal phases and time-zone unlocks.
 *
 * 5. Robust Tail Trimming:
 *    Trims the extreme 10% tails from both ends of the empirical ratio distribution
 *    to eliminate idiosyncratic outliers before computing weighted quantiles.
 *
 * 6. Quantile Estimation & Safety Contract:
 *    Computes quantiles at contest levels [0.05, 0.25, 0.50, 0.75, 0.95], rescales to
 *    the case's hour-1 print, enforces monotonicity and boundary constraints, and includes
 *    a robust fallback to the competition baseline if inputs are malformed.
 */

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** Target quantile levels evaluated by the scoring metric */
export const TARGET_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95] as const;

/** Standard normal quantiles corresponding to TARGET_LEVELS */
export const BASELINE_Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645] as const;

/** Scaling factor for channel count log-difference */
export const CHANNEL_SIMILARITY_WEIGHT = 0.25;

/** Scaling factor for viewer count log-difference */
export const VIEWER_SIMILARITY_WEIGHT = 1.0;

/** Neighborhood smoothing radius in hours */
export const TEMPORAL_WINDOW_HOURS = 3;

/** Minimum similarity weight to retain a comparable */
export const MIN_WEIGHT_CUTOFF = 0.05;

/** Fraction of extreme ratio distribution tails to trim */
export const TAIL_TRIM_FRACTION = 0.10;

/** Maximum allowable viewer count according to competition constraints */
export const MAX_VIEWERS_LIMIT = 1e9;

/** Minimum allowable viewer count */
export const MIN_VIEWERS_LIMIT = 0.01;

export interface HourlyObservation {
  hour: number;
  viewers: number;
  channels: number;
}

export interface LaunchTrajectory {
  name: string;
  hourlyPoints: Map<number, HourlyObservation>;
}

export interface RawComparable {
  name: string;
  day_start: string;
  hourly: Array<{
    from: string;
    viewers: number;
    channels?: number | null;
  }>;
}

export interface CaseInput {
  id: string;
  known: Array<{
    hour: number;
    viewers: number;
    channels?: number | null;
  }>;
  forecast: Array<{
    hour: number;
  }>;
  comparables?: RawComparable[];
}

export interface HourlyForecastRecord {
  hour: number;
  q: number[];
}

export interface ForecastResult {
  hours: HourlyForecastRecord[];
}

/** Parses an ISO 8601 timestamp string into epoch milliseconds */
export function parseIsoTimestamp(isoString: string): number {
  return Date.parse(isoString);
}

/** Computes the time elapsed between two ISO timestamps in hours */
export function elapsedHours(startIso: string, pointIso: string): number {
  return (parseIsoTimestamp(pointIso) - parseIsoTimestamp(startIso)) / 3_600_000;
}

/**
 * Maps hourly observations from a comparable's record to launch-day hours 1 through 24.
 */
export function extractLaunchDayHours(comparable: RawComparable): Map<number, HourlyObservation> {
  const launchHours = new Map<number, HourlyObservation>();
  if (!comparable.hourly || !Array.isArray(comparable.hourly) || !comparable.day_start) {
    return launchHours;
  }

  for (const record of comparable.hourly) {
    if (!record || typeof record.from !== 'string' || typeof record.viewers !== 'number') {
      continue;
    }
    const offset = elapsedHours(comparable.day_start, record.from);
    if (offset < 0 || offset >= 24) {
      continue;
    }
    const hour = Math.floor(offset) + 1;
    if (!launchHours.has(hour)) {
      launchHours.set(hour, {
        hour,
        viewers: record.viewers,
        channels: typeof record.channels === 'number' ? record.channels : 0,
      });
    }
  }

  return launchHours;
}

/**
 * Assembles the reference pool of past launches with complete first-hour release data.
 */
export function assembleTrainingPool(caseData: CaseInput): LaunchTrajectory[] {
  const pool: LaunchTrajectory[] = [];
  if (!caseData.comparables || !Array.isArray(caseData.comparables)) {
    return pool;
  }

  for (const comp of caseData.comparables) {
    // Exclude pre-launch observation periods
    if (/before release/i.test(comp.name)) {
      continue;
    }

    const hours = extractLaunchDayHours(comp);
    const hour1 = hours.get(1);
    if (!hour1 || hour1.viewers <= 0) {
      continue;
    }

    pool.push({
      name: comp.name,
      hourlyPoints: hours,
    });
  }

  return pool;
}

/**
 * Computes pairwise similarity weight between a target launch and a comparable launch
 * based on their respective first-hour viewer and channel counts.
 */
export function calculateLaunchSimilarity(
  targetChannels: number,
  targetViewers: number,
  refChannels: number,
  refViewers: number,
  channelWeight = CHANNEL_SIMILARITY_WEIGHT,
  viewerWeight = VIEWER_SIMILARITY_WEIGHT,
): number {
  const channelRatio = (targetChannels > 0 && refChannels > 0)
    ? Math.exp(-channelWeight * Math.abs(Math.log((targetChannels + 1) / (refChannels + 1))))
    : 1.0;

  const viewerRatio = Math.exp(
    -viewerWeight * Math.abs(Math.log((targetViewers + 1) / (refViewers + 1))),
  );

  return channelRatio * viewerRatio;
}

/**
 * Calculates a weighted quantile from a sorted array of [value, weight] pairs.
 */
export function calculateWeightedQuantile(
  sortedPairs: Array<[number, number]>,
  quantileLevel: number,
): number {
  if (sortedPairs.length === 0) {
    return 0;
  }

  let totalWeight = 0;
  for (let i = 0; i < sortedPairs.length; i++) {
    totalWeight += sortedPairs[i][1];
  }

  const targetCumulative = quantileLevel * totalWeight;
  let cumulative = 0;

  for (let i = 0; i < sortedPairs.length; i++) {
    cumulative += sortedPairs[i][1];
    if (cumulative >= targetCumulative) {
      return sortedPairs[i][0];
    }
  }

  return sortedPairs[sortedPairs.length - 1][0];
}

/**
 * Rounds a numeric value to two decimal places half-up.
 */
export function roundToTwoDecimals(val: number): number {
  return Math.round(val * 100) / 100;
}

/**
 * Generates the house baseline forecast for hours 2 to 24:
 *   q_tau = p * exp(z_tau * 0.15 * sqrt(h - 1))
 */
export function generateHouseBaseline(initialViewers: number): HourlyForecastRecord[] {
  const p = Math.max(1, initialViewers);
  const rows: HourlyForecastRecord[] = [];

  for (let h = 2; h <= 24; h++) {
    const spread = 0.15 * Math.sqrt(h - 1);
    const qValues = BASELINE_Z_SCORES.map((z) => {
      const projected = roundToTwoDecimals(p * Math.exp(z * spread));
      return Math.min(MAX_VIEWERS_LIMIT, Math.max(MIN_VIEWERS_LIMIT, projected));
    });

    for (let i = 1; i < qValues.length; i++) {
      if (qValues[i] < qValues[i - 1]) {
        qValues[i] = qValues[i - 1];
      }
    }

    rows.push({ hour: h, q: qValues });
  }

  return rows;
}

export interface ForecastOptions {
  channelWeight?: number;
  viewerWeight?: number;
  windowRadius?: number;
  weightCutoff?: number;
  tailTrim?: number;
  unweighted?: boolean;
}

/**
 * Predicts the five quantiles for a single target hour given the reference pool.
 */
export function predictHourQuantiles(
  pool: LaunchTrajectory[],
  targetChannels: number,
  targetViewers: number,
  hour: number,
  options: ForecastOptions = {},
): number[] {
  const cw = options.channelWeight ?? CHANNEL_SIMILARITY_WEIGHT;
  const vw = options.viewerWeight ?? VIEWER_SIMILARITY_WEIGHT;
  const windowRadius = options.windowRadius ?? TEMPORAL_WINDOW_HOURS;
  const weightCutoff = options.unweighted ? 0 : (options.weightCutoff ?? MIN_WEIGHT_CUTOFF);
  const tailTrim = options.tailTrim ?? TAIL_TRIM_FRACTION;
  const isUnweighted = options.unweighted ?? false;

  let ratioEntries: Array<[number, number]> = [];

  for (const trajectory of pool) {
    const hour1 = trajectory.hourlyPoints.get(1);
    if (!hour1 || hour1.viewers <= 0) continue;

    const weight = isUnweighted
      ? 1.0
      : calculateLaunchSimilarity(targetChannels, targetViewers, hour1.channels, hour1.viewers, cw, vw);

    if (!isUnweighted && weight < weightCutoff) {
      continue;
    }

    const windowStart = Math.max(1, hour - windowRadius);
    const windowEnd = Math.min(24, hour + windowRadius);

    for (let h = windowStart; h <= windowEnd; h++) {
      const pt = trajectory.hourlyPoints.get(h);
      if (!pt || pt.viewers <= 0) continue;
      const logRatio = Math.log(pt.viewers / hour1.viewers);
      ratioEntries.push([logRatio, weight]);
    }
  }

  // Fallback if weight cutoff removed all comparables
  if (ratioEntries.length === 0 && pool.length > 0) {
    for (const trajectory of pool) {
      const hour1 = trajectory.hourlyPoints.get(1);
      if (!hour1 || hour1.viewers <= 0) continue;
      const windowStart = Math.max(1, hour - windowRadius);
      const windowEnd = Math.min(24, hour + windowRadius);
      for (let h = windowStart; h <= windowEnd; h++) {
        const pt = trajectory.hourlyPoints.get(h);
        if (!pt || pt.viewers <= 0) continue;
        ratioEntries.push([Math.log(pt.viewers / hour1.viewers), 1.0]);
      }
    }
  }

  // If pool was empty, return baseline spread around target viewers
  if (ratioEntries.length === 0) {
    const spread = 0.15 * Math.sqrt(Math.max(1, hour - 1));
    return BASELINE_Z_SCORES.map((z) =>
      roundToTwoDecimals(Math.min(MAX_VIEWERS_LIMIT, Math.max(MIN_VIEWERS_LIMIT, targetViewers * Math.exp(z * spread)))),
    );
  }

  // Sort ratios in ascending order
  ratioEntries.sort((a, b) => a[0] - b[0]);

  // Trim distribution tails if specified
  if (tailTrim > 0) {
    const cutCount = Math.floor(ratioEntries.length * tailTrim);
    if (cutCount > 0 && 2 * cutCount < ratioEntries.length) {
      ratioEntries = ratioEntries.slice(cutCount, ratioEntries.length - cutCount);
    }
  }

  // Compute the five quantile estimates
  const quantiles = TARGET_LEVELS.map((level) => {
    const estimatedRatio = calculateWeightedQuantile(ratioEntries, level);
    const projectedViewers = targetViewers * Math.exp(estimatedRatio);
    return roundToTwoDecimals(Math.min(MAX_VIEWERS_LIMIT, Math.max(MIN_VIEWERS_LIMIT, projectedViewers)));
  });

  // Guarantee non-decreasing quantiles as required by competition schema
  for (let i = 1; i < quantiles.length; i++) {
    if (quantiles[i] < quantiles[i - 1]) {
      quantiles[i] = quantiles[i - 1];
    }
  }

  return quantiles;
}

/**
 * Produces the full 23-hour forecast for a given case.
 * Guaranteed never to throw; falls back safely to house baseline on unexpected data.
 */
export function forecastCase(caseData: CaseInput, options: ForecastOptions = {}): ForecastResult {
  try {
    const hour1Data = Array.isArray(caseData?.known)
      ? caseData.known.find((r) => r && r.hour === 1)
      : undefined;

    const initialViewers = (typeof hour1Data?.viewers === 'number' && Number.isFinite(hour1Data.viewers) && hour1Data.viewers >= 1)
      ? hour1Data.viewers
      : 1;

    const initialChannels = (typeof hour1Data?.channels === 'number' && Number.isFinite(hour1Data.channels) && hour1Data.channels > 0)
      ? hour1Data.channels
      : 0;

    const referencePool = assembleTrainingPool(caseData);

    // The output contract requires exactly 23 rows for hours 2 to 24
    const targetHours = Array.from({ length: 23 }, (_, idx) => idx + 2);

    const rows: HourlyForecastRecord[] = targetHours.map((targetHour) => ({
      hour: targetHour,
      q: predictHourQuantiles(referencePool, initialChannels, initialViewers, targetHour, options),
    }));

    return { hours: rows };
  } catch {
    return { hours: generateHouseBaseline(1) };
  }
}

/**
 * Main execution function reading cases from input directory and outputting forecasts.
 */
export function run(inputDir?: string, outputDir?: string): void {
  const resolvedInputDir = inputDir ?? process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const resolvedOutputDir = outputDir ?? process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

  mkdirSync(resolvedOutputDir, { recursive: true });

  let caseFilenames: string[] = [];
  try {
    caseFilenames = readdirSync(resolvedInputDir)
      .filter((filename) => filename.endsWith('.in'))
      .sort();
  } catch (err) {
    // If input directory cannot be read, exit cleanly
    return;
  }

  for (const filename of caseFilenames) {
    const caseId = filename.slice(0, -'.in'.length);
    const inputPath = join(resolvedInputDir, filename);
    const outputPath = join(resolvedOutputDir, `${caseId}.out`);

    let outputResult: ForecastResult;
    try {
      const fileContent = readFileSync(inputPath, 'utf8');
      const caseData = JSON.parse(fileContent) as CaseInput;
      outputResult = forecastCase(caseData);
    } catch {
      outputResult = { hours: generateHouseBaseline(1) };
    }

    writeFileSync(outputPath, JSON.stringify(outputResult));
  }
}

// Execute CLI when directly invoked
const directScriptPath = process.argv[1] ? resolve(process.argv[1]) : '';
const currentFilePath = fileURLToPath(import.meta.url);
if (directScriptPath && directScriptPath === currentFilePath) {
  run();
}
