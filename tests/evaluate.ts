/**
 * Offline evaluation and verification suite for GTA VI launch-day CCU forecast.
 *
 * Verifies:
 * 1. CLI execution contract on practice cases (input parsing, schema compliance, output format, speed)
 * 2. Leave-One-Out (LOO) cross-validation pinball loss across the 37 historical launch comparables
 *    comparing the Shipped Model against the Unweighted Pooled Model and House Baseline.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  BASELINE_Z_SCORES,
  CHANNEL_SIMILARITY_WEIGHT,
  extractLaunchDayHours,
  MAX_VIEWERS_LIMIT,
  MIN_VIEWERS_LIMIT,
  MIN_WEIGHT_CUTOFF,
  predictHourQuantiles,
  roundToTwoDecimals,
  TAIL_TRIM_FRACTION,
  TARGET_LEVELS,
  TEMPORAL_WINDOW_HOURS,
  VIEWER_SIMILARITY_WEIGHT,
} from '../main.ts';
import type { CaseInput, LaunchTrajectory } from '../main.ts';

/**
 * Calculates pinball loss for a single quantile prediction:
 *   loss = 2 * max(tau * d, (tau - 1) * d) where d = ln(y) - ln(q)
 */
export function computePinballLoss(tau: number, actualViewers: number, quantilePrediction: number): number {
  const y = Math.max(1, actualViewers);
  const q = Math.max(MIN_VIEWERS_LIMIT, Math.min(MAX_VIEWERS_LIMIT, quantilePrediction));
  const d = Math.log(y) - Math.log(q);
  return 2.0 * Math.max(tau * d, (tau - 1.0) * d);
}

/**
 * Discovers the practice input directory.
 */
export function findPracticeInputDir(): string {
  const candidates = [
    process.env.PRACTICE_INPUT_DIR,
    process.env.MAKEFUN_INPUT_DIR,
    resolve(join(fileURLToPath(import.meta.url), '../../../../kits/gta6-twitch-ccu-forecast-01/gta6-twitch-ccu-forecast-01-kit/input')),
    resolve('../../kits/gta6-twitch-ccu-forecast-01/gta6-twitch-ccu-forecast-01-kit/input'),
    resolve('/Users/bonson/.deepai-code/scratch/2026-11-17-tes-4dccd960/kits/gta6-twitch-ccu-forecast-01/gta6-twitch-ccu-forecast-01-kit/input'),
  ];

  for (const candidate of candidates) {
    if (candidate && existsSync(candidate)) {
      const files = readdirSync(candidate).filter((f) => f.endsWith('.in'));
      if (files.length > 0) {
        return candidate;
      }
    }
  }

  throw new Error('Unable to locate practice cases directory (p00.in, p01.in, etc.)');
}

/**
 * Tests CLI execution of main.ts on all practice cases in a clean temporary directory.
 */
export function verifyCliExecution(inputDir: string): { totalDurationMs: number; perCaseMs: number; caseCount: number } {
  const tempOutputDir = mkdtempSync(join(tmpdir(), 'gta6-test-'));
  const mainScriptPath = resolve(join(fileURLToPath(import.meta.url), '../../main.ts'));

  const inputFiles = readdirSync(inputDir).filter((f) => f.endsWith('.in')).sort();
  if (inputFiles.length === 0) {
    throw new Error(`No .in files found in ${inputDir}`);
  }

  const startTime = performance.now();
  execFileSync(process.execPath, [mainScriptPath], {
    env: {
      ...process.env,
      MAKEFUN_INPUT_DIR: inputDir,
      MAKEFUN_OUTPUT_DIR: tempOutputDir,
    },
    stdio: 'pipe',
  });
  const totalDurationMs = performance.now() - startTime;
  const perCaseMs = totalDurationMs / inputFiles.length;

  for (const inputFile of inputFiles) {
    const caseId = inputFile.slice(0, -'.in'.length);
    const outputPath = join(tempOutputDir, `${caseId}.out`);

    if (!existsSync(outputPath)) {
      throw new Error(`Expected output file not generated: ${outputPath}`);
    }

    const parsed = JSON.parse(readFileSync(outputPath, 'utf8'));
    const keys = Object.keys(parsed);
    if (keys.length !== 1 || keys[0] !== 'hours') {
      throw new Error(`Invalid output root keys in ${caseId}.out: ${JSON.stringify(keys)}`);
    }

    if (!Array.isArray(parsed.hours) || parsed.hours.length !== 23) {
      throw new Error(`Expected 23 forecast rows, found: ${parsed.hours?.length}`);
    }

    for (let idx = 0; idx < 23; idx++) {
      const row = parsed.hours[idx];
      const expectedHour = idx + 2;
      if (row.hour !== expectedHour) {
        throw new Error(`Row ${idx} has hour ${row.hour}, expected ${expectedHour}`);
      }

      if (!Array.isArray(row.q) || row.q.length !== 5) {
        throw new Error(`Row ${idx} does not contain 5 quantiles`);
      }

      for (let qIdx = 0; qIdx < 5; qIdx++) {
        const val = row.q[qIdx];
        if (typeof val !== 'number' || !Number.isFinite(val) || val <= 0 || val > MAX_VIEWERS_LIMIT) {
          throw new Error(`Invalid quantile value at hour ${expectedHour}, index ${qIdx}: ${val}`);
        }
        if (qIdx > 0 && row.q[qIdx] < row.q[qIdx - 1]) {
          throw new Error(`Monotonicity violation at hour ${expectedHour}: ${row.q.join(', ')}`);
        }
      }
    }
  }

  rmSync(tempOutputDir, { recursive: true, force: true });
  return { totalDurationMs, perCaseMs, caseCount: inputFiles.length };
}

/**
 * Runs leave-one-out validation on the 37 historical comparable launches.
 */
export function runLeaveOneOutEvaluation(inputDir: string): {
  improvedLoss: number;
  pooledLoss: number;
  baselineLoss: number;
  poolSize: number;
} {
  const p00Path = join(inputDir, 'p00.in');
  const casePayload = JSON.parse(readFileSync(p00Path, 'utf8')) as CaseInput;

  // Extract past launches (excluding pre-release monitoring)
  const fullPool: LaunchTrajectory[] = [];
  for (const comp of casePayload.comparables ?? []) {
    if (/before release/i.test(comp.name)) continue;
    const hours = extractLaunchDayHours(comp);
    const h1 = hours.get(1);
    if (!h1 || h1.viewers <= 0) continue;
    fullPool.push({ name: comp.name, hourlyPoints: hours });
  }

  const poolSize = fullPool.length;
  const improvedLosses: number[] = [];
  const pooledLosses: number[] = [];
  const baselineLosses: number[] = [];

  for (let testIdx = 0; testIdx < poolSize; testIdx++) {
    const testLaunch = fullPool[testIdx];
    const trainPool = fullPool.filter((_, idx) => idx !== testIdx);

    const testH1 = testLaunch.hourlyPoints.get(1)!;
    const targetViewers = testH1.viewers;
    const targetChannels = testH1.channels;

    for (let targetHour = 2; targetHour <= 24; targetHour++) {
      const actualPt = testLaunch.hourlyPoints.get(targetHour);
      if (!actualPt || actualPt.viewers <= 0) continue;
      const actualViewers = actualPt.viewers;

      // 1. Shipped Improved Model
      const improvedQuantiles = predictHourQuantiles(
        trainPool,
        targetChannels,
        targetViewers,
        targetHour,
        {
          channelWeight: CHANNEL_SIMILARITY_WEIGHT,
          viewerWeight: VIEWER_SIMILARITY_WEIGHT,
          windowRadius: TEMPORAL_WINDOW_HOURS,
          weightCutoff: MIN_WEIGHT_CUTOFF,
          tailTrim: TAIL_TRIM_FRACTION,
        },
      );

      // 2. Unweighted Pooled Model (weight = 1.0, no cutoff, no trim)
      const pooledQuantiles = predictHourQuantiles(
        trainPool,
        targetChannels,
        targetViewers,
        targetHour,
        {
          unweighted: true,
          windowRadius: TEMPORAL_WINDOW_HOURS,
          weightCutoff: 0,
          tailTrim: 0,
        },
      );

      // 3. House Baseline
      const spread = 0.15 * Math.sqrt(targetHour - 1);
      const baselineQuantiles = BASELINE_Z_SCORES.map((z) =>
        roundToTwoDecimals(Math.max(1, targetViewers) * Math.exp(z * spread)),
      );

      for (let qIdx = 0; qIdx < TARGET_LEVELS.length; qIdx++) {
        const tau = TARGET_LEVELS[qIdx];
        improvedLosses.push(computePinballLoss(tau, actualViewers, improvedQuantiles[qIdx]));
        pooledLosses.push(computePinballLoss(tau, actualViewers, pooledQuantiles[qIdx]));
        baselineLosses.push(computePinballLoss(tau, actualViewers, baselineQuantiles[qIdx]));
      }
    }
  }

  const improvedLoss = improvedLosses.reduce((sum, v) => sum + v, 0) / improvedLosses.length;
  const pooledLoss = pooledLosses.reduce((sum, v) => sum + v, 0) / pooledLosses.length;
  const baselineLoss = baselineLosses.reduce((sum, v) => sum + v, 0) / baselineLosses.length;

  return { improvedLoss, pooledLoss, baselineLoss, poolSize };
}

export function main(): void {
  console.log('=== GTA VI Twitch CCU Forecast Evaluation ===\n');

  const inputDir = findPracticeInputDir();
  console.log(`Using practice cases from: ${inputDir}`);

  // Test 1: CLI Execution and Schema Verification
  console.log('\n--- 1. CLI Execution & Schema Contract Verification ---');
  const cliResult = verifyCliExecution(inputDir);
  console.log(`Successfully verified ${cliResult.caseCount} practice cases.`);
  console.log(`Total execution time: ${cliResult.totalDurationMs.toFixed(2)} ms`);
  console.log(`Average time per case: ${cliResult.perCaseMs.toFixed(2)} ms (Limit: 8000 ms)`);
  console.log('Output format, row counts, monotonicity, and bounds all verified.');

  // Test 2: Leave-One-Out Evaluation
  console.log('\n--- 2. Leave-One-Out Cross-Validation on Comparable Pool ---');
  const evalResult = runLeaveOneOutEvaluation(inputDir);
  console.log(`Number of historical launches in pool: ${evalResult.poolSize}`);
  console.log(`Total scored items: ${evalResult.poolSize * 23 * 5}`);
  console.log(`\nResults (Competition Pinball Loss, lower is better):`);
  console.log(`  - House Baseline:        ${evalResult.baselineLoss.toFixed(6)}`);
  console.log(`  - Unweighted Pool:       ${evalResult.pooledLoss.toFixed(6)}`);
  console.log(`  - Shipped Improved Model: ${evalResult.improvedLoss.toFixed(6)}`);

  const pooledImprovement = ((evalResult.pooledLoss - evalResult.improvedLoss) / evalResult.pooledLoss) * 100;
  const baselineImprovement = ((evalResult.baselineLoss - evalResult.improvedLoss) / evalResult.baselineLoss) * 100;
  console.log(`\nImprovement over Unweighted Pool: +${pooledImprovement.toFixed(2)}%`);
  console.log(`Improvement over House Baseline:  +${baselineImprovement.toFixed(2)}%`);
  console.log('\n=== Evaluation Completed Successfully ===');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
