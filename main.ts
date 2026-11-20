// gta6-trophy-rarity-forecast-01 — earn-rate forecaster (Node 22, standard library only).
//
// For every trophy of GTA VI on PS5, forecast five quantiles of its Exophase earn
// rate at the case's target, from the case's history of reads of the game's trophy
// page. Score is mean pinball loss on the logit scale, so the quantiles must be
// the forecaster's honest beliefs, not point guesses with a cosmetic band.
//
// Model, per trophy, in logit space (logit(p) = ln(p / (100 - p)), p in percent):
//
//   1. Trend. logit(rate) is fit by weighted least squares against
//      x = ln(1 + hours since the game's release), with weights halving every
//      24h so recent reads count most. The slope is shrunk toward its tier's
//      mean slope (platinum/gold/silver/bronze move together) and capped, so a
//      noisy trophy cannot extrapolate wildly over a 4.5-day horizon. This is
//      the drift the house curve baseline fits, extended to the full history
//      with recency weights and tier shrinkage.
//   2. Refresh risk. Exophase refreshes a game's counts on its own schedule:
//      between refreshes the page is frozen, and at a refresh the tracked
//      player count jumps and almost every rate falls, because the new players
//      tracked have played less than the early adopters. When a case's history
//      shows no refresh at all (the practice cases: every read identical), the
//      median is shifted down by p * delta, where p is the chance a refresh
//      lands before the target (refresh rate from history, with a one-per-6h
//      prior) and delta grows with how rare the trophy is (common trophies
//      barely move; the 100% trophy does not move at all). When the history
//      does show refreshes, the trend already carries their drift, so no extra
//      shift is applied.
//   3. Spread. The predictive spread on the logit scale combines the
//      regression's prediction error (pooled residual SD, grown by leverage
//      as the target moves past the history), a random-walk term that widens
//      with the square root of the horizon, and a refresh-mixture term (the
//      variance of the refresh jump when no refresh has been seen yet, one
//      last refresh's worth of jump noise when refreshes have been seen).
//      The spread is floored so a flat history never claims certainty.
//
//   q_tau = 100 / (1 + exp(-(median_logit + z_tau * spread))), rounded to
//   4 decimals, clamped to [0.01, 100], non-decreasing.
//
// The program reads every <id>.in in the input directory (MAKEFUN_INPUT_DIR
// or /input) and writes <id>.out to the output directory (MAKEFUN_OUTPUT_DIR
// or /output). It is deterministic: no randomness, no clock reads, no network.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface Trophy {
  id: number;
  name: string;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date?: string };
  trophies: Trophy[];
  history: Read[];
}

// ---- small numeric helpers -------------------------------------------------

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (y: number): number => 100 / (1 + Math.exp(-y));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Inverse standard normal CDF (Acklam's rational approximation, |err| < 1.2e-9). */
function normInv(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239e0];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0, -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// ---- case parsing ----------------------------------------------------------

function parseTime(s: string): number {
  const t = Date.parse(s);
  if (Number.isNaN(t)) throw new Error(`unparseable time ${s}`);
  return t;
}

/** Hours from the game's release (midnight UTC of release_date) to a timestamp. */
function hoursSinceRelease(c: Case, ms: number): number {
  let release = Date.parse('2026-11-19T00:00:00Z');
  if (c.game?.release_date) {
    const d = Date.parse(c.game.release_date);
    if (!Number.isNaN(d)) release = d;
  }
  return Math.max(0, (ms - release) / 3_600_000);
}

// ---- per-trophy forecast ---------------------------------------------------

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const z = c.levels.map((t) => normInv(t));
  const horizonDays = c.horizon_minutes / 1440;
  const targetMs = parseTime(c.target);
  const xTarget = Math.log(1 + hoursSinceRelease(c, targetMs));

  const reads = c.history
    .map((h) => ({ ms: parseTime(h.at), players: h.players_tracked, rates: h.rates }))
    .sort((a, b) => a.ms - b.ms);
  const lastMs = reads.length > 0 ? reads[reads.length - 1].ms : parseTime(c.at);
  const spanHours = reads.length > 1 ? (lastMs - reads[0].ms) / 3_600_000 : 0;

  // Refreshes: reads where the tracked count jumps or any rate moves beyond
  // rounding (0.01pp). Count them and measure their logit jumps.
  let nRefresh = 0;
  const jumpShifts: number[] = [];
  for (let j = 1; j < reads.length; j++) {
    const prev = reads[j - 1];
    const cur = reads[j];
    const playersJump = Math.abs(cur.players - prev.players) / Math.max(1, prev.players) > 0.01;
    let rateMoved = false;
    for (let i = 0; i < c.trophies.length; i++) {
      const a = prev.rates[i];
      const b = cur.rates[i];
      if (a !== null && b !== null && Math.abs(b - a) > 0.011) rateMoved = true;
    }
    if (playersJump || rateMoved) {
      nRefresh++;
      for (let i = 0; i < c.trophies.length; i++) {
        const a = prev.rates[i];
        const b = cur.rates[i];
        if (a !== null && b !== null && a > 0 && b > 0 && a < 100 && b < 100) {
          jumpShifts.push(logit(b) - logit(a));
        }
      }
    }
  }
  // Refresh rate and the chance one lands before the target. Refreshes are
  // not memoryless: Exophase's page sat frozen for hours on launch night
  // (unchanged from 23:48 through the 04:00 read, refreshed by 04:30), so a
  // long freeze means one is due. Estimate the refresh interval I from the
  // gaps between refreshes seen in the history (a 5h prior when none has been
  // seen yet — the launch-night freeze lasted about that long), and take
  // p = (time frozen + horizon) / I, capped at 0.95: a page frozen most of an
  // interval is almost certain to refresh before a target a good way into
  // the next one.
  const refreshGaps: number[] = [];
  let lastRefreshMs = reads.length > 0 ? reads[0].ms : parseTime(c.at);
  for (let j = 1; j < reads.length; j++) {
    const prev = reads[j - 1];
    const cur = reads[j];
    const playersJump = Math.abs(cur.players - prev.players) / Math.max(1, prev.players) > 0.01;
    let rateMoved = false;
    for (let i = 0; i < c.trophies.length; i++) {
      const a = prev.rates[i];
      const b = cur.rates[i];
      if (a !== null && b !== null && Math.abs(b - a) > 0.011) rateMoved = true;
    }
    if (playersJump || rateMoved) {
      refreshGaps.push((cur.ms - lastRefreshMs) / 3_600_000);
      lastRefreshMs = cur.ms;
    }
  }
  const intervalHours = refreshGaps.length > 0
    ? refreshGaps.reduce((s, v) => s + v, 0) / refreshGaps.length
    : 5;
  const frozenHours = (lastMs - lastRefreshMs) / 3_600_000;
  const pRefresh = nRefresh === 0
    ? Math.min(0.95, Math.max(0, (frozenHours + c.horizon_minutes / 60) / intervalHours))
    : 1 - Math.exp(-(c.horizon_minutes / 60) / intervalHours);
  // Typical refresh jump size (SD of the logit shifts at refreshes).
  const jumpSd = (() => {
    if (jumpShifts.length < 5) return 0.2;
    const m = jumpShifts.reduce((s, v) => s + v, 0) / jumpShifts.length;
    const v = jumpShifts.reduce((s, x) => s + (x - m) * (x - m), 0) / jumpShifts.length;
    return Math.min(0.5, Math.max(0.1, Math.sqrt(v)));
  })();

  // First pass: per-trophy weighted least squares of logit(rate) on x.
  interface Fit {
    a: number;
    b: number;
    sxx: number;
    xBar: number;
    n: number;
    residSq: number;
    lastRate: number;
    lastLogit: number;
  }
  const fits: Fit[] = c.trophies.map((t, i) => {
    const pts: Array<{ x: number; y: number; w: number }> = [];
    for (const r of reads) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined) continue;
      const h = hoursSinceRelease(c, r.ms);
      const w = Math.pow(0.5, (lastMs - r.ms) / (24 * 3_600_000));
      pts.push({ x: Math.log(1 + h), y: logit(rate), w });
    }
    const lastRate = (() => {
      for (let j = reads.length - 1; j >= 0; j--) {
        const v = reads[j].rates[i];
        if (v !== null && v !== undefined) return v;
      }
      return 0;
    })();
    if (pts.length < 3) {
      return { a: logit(lastRate), b: 0, sxx: 0, xBar: 0, n: pts.length, residSq: 0, lastRate, lastLogit: logit(lastRate) };
    }
    const wSum = pts.reduce((s, p) => s + p.w, 0);
    const xBar = pts.reduce((s, p) => s + p.w * p.x, 0) / wSum;
    const yBar = pts.reduce((s, p) => s + p.w * p.y, 0) / wSum;
    let sxx = 0;
    let sxy = 0;
    for (const p of pts) {
      sxx += p.w * (p.x - xBar) * (p.x - xBar);
      sxy += p.w * (p.x - xBar) * (p.y - yBar);
    }
    const b = sxx > 1e-9 ? sxy / sxx : 0;
    const a = yBar - b * xBar;
    let residSq = 0;
    for (const p of pts) residSq += p.w * (p.y - (a + b * p.x)) ** 2;
    return { a, b, sxx, xBar, n: pts.length, residSq, lastRate, lastLogit: logit(lastRate) };
  });

  // Pooled residual SD across trophies (floored: a flat history is not proof
  // the future is flat), and tier mean slopes for shrinkage. Residual noise
  // differs by tier, so each tier gets its own pooled SD when it has enough
  // residual degrees of freedom, else it falls back to the global one.
  const residNum = fits.reduce((s, f) => s + f.residSq, 0);
  const residDen = fits.reduce((s, f) => s + Math.max(0, f.n - 2), 0);
  const sPoolGlobal = Math.min(1.0, Math.max(0.1, residDen > 0 ? Math.sqrt(residNum / residDen) : 0.1));
  const sPoolByTier = new Map<string, number>();
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const idx = c.trophies.map((t, i) => (t.tier === tier ? i : -1)).filter((i) => i >= 0);
    const num = idx.reduce((s, i) => s + fits[i].residSq, 0);
    const den = idx.reduce((s, i) => s + Math.max(0, fits[i].n - 2), 0);
    sPoolByTier.set(tier, den >= 10 ? Math.min(1.0, Math.max(0.1, Math.sqrt(num / den))) : sPoolGlobal);
  }
  const sPool = sPoolGlobal;
  const tierSlope = new Map<string, number>();
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const bs = fits
      .filter((f, i) => c.trophies[i].tier === tier && f.n >= 3 && f.sxx > 1e-9)
      .map((f) => f.b);
    tierSlope.set(tier, bs.length > 0 ? bs.reduce((s, v) => s + v, 0) / bs.length : 0);
  }

  // Second pass: shrunk slope, median, spread, quantiles.
  return c.trophies.map((t, i) => {
    const f = fits[i];
    // Shrink the slope toward the tier mean: the trophy's own fit counts in
    // proportion to its information (sxx / sPool^2), the tier prior counts
    // as 5 effective points.
    const info = f.sxx > 1e-9 ? f.sxx / (sPool * sPool) : 0;
    const prior = 5;
    const tierMean = tierSlope.get(t.tier) ?? 0;
    let b = (f.b * info + tierMean * prior) / (info + prior);
    if (f.n < 3) b = 0;
    b = Math.min(2.5, Math.max(-0.5, b));
    let median = f.a + b * xTarget;
    if (f.n < 3) median = f.lastLogit;

    // Refresh risk when no refresh has been seen yet: the median drops by
    // p * delta (delta larger for rarer trophies; the 100% trophy does not
    // move), and the spread gains the mixture's variance.
    let shift = 0;
    let sMix = 0;
    const rLast = clampPct(f.lastRate);
    if (nRefresh === 0) {
      const delta = rLast >= 99.99 ? 0 : Math.min(0.6, Math.max(0, 0.5 * (1 - rLast / 100)));
      shift = -pRefresh * delta;
      sMix = Math.sqrt(pRefresh * (1 - pRefresh) * delta * delta + pRefresh * 0.04);
    } else {
      sMix = jumpSd * 0.5;
    }
    median += shift;

    // Prediction spread: regression error grown by leverage, plus a
    // random-walk term in sqrt(days), plus refresh noise. The regression
    // error uses the trophy's tier's pooled residual SD.
    const leverage = f.sxx > 1e-9 ? 1 / Math.max(1, f.n) + ((xTarget - f.xBar) ** 2) / f.sxx : 1;
    const sPred = (sPoolByTier.get(t.tier) ?? sPool) * Math.sqrt(1 + leverage);
    const sRw = 0.25 * Math.sqrt(horizonDays);
    let s = Math.sqrt(sPred * sPred + sRw * sRw + sMix * sMix);
    s = Math.min(2.5, Math.max(0.2, s));
    if (f.n < 3) s = Math.max(s, 1.0);

    median = Math.min(logit(99.99), Math.max(logit(0.01), median));
    const qs = z.map((zv) => invLogit(median + zv * s));
    // Round to 4 decimals, clamp to [0.01, 100], keep non-decreasing.
    let prev = 0.01;
    const q = qs.map((v) => {
      const r = Math.min(100, Math.max(0.01, round4(v)));
      const out = Math.max(prev, r);
      prev = out;
      return out;
    });
    return { id: t.id, q };
  });
}

// ---- main ------------------------------------------------------------------

function main(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
  for (const file of inputs) {
    const id = file.slice(0, -'.in'.length);
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
    for (const t of c.levels) {
      if (!(t > 0 && t < 1)) throw new Error(`${file} asks for level ${t}, not in (0,1)`);
    }
    let rows: Array<{ id: number; q: number[] }>;
    if (c.history.length === 0 || c.trophies.length === 0) {
      // No history to learn from: emit a valid, very wide forecast rather
      // than crash (a crash scores the case as last-rate persistence).
      const z = c.levels.map((t) => normInv(t));
      rows = c.trophies.map((t) => ({
        id: t.id,
        q: z.map((zv) => round4(Math.min(100, Math.max(0.01, invLogit(logit(1) + zv * 1.5))))),
      }));
    } else {
      rows = forecastCase(c);
    }
    const out = JSON.stringify({ trophies: rows });
    if (out.length > 65_536) throw new Error(`${file} output ${out.length} bytes exceeds 65536`);
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${out}\n`);
  }
}

main();
