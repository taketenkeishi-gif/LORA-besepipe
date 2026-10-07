import type { DragEvent } from "react";

export function parseIntOr(value: string, fallback: number, min = 0): number {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, n);
}

export function parseFloatOr(
  value: string,
  fallback: number,
  min = 0,
): number {
  const n = Number.parseFloat(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, n);
}

export function etaText(sec?: number | null): string {
  if (sec == null) return "計算中...";
  if (sec < 60) return `${sec}s`;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

// 学習時間予測など、秒〜時間レンジを h m s で整形する（etaText より高精度）。
export function durationText(sec?: number | null): string {
  if (sec == null || !Number.isFinite(sec)) return "—";
  const s = Math.max(0, Math.round(sec));
  if (s < 60) return `${s}秒`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rs = s % 60;
  if (h > 0) return `${h}時間${m}分`;
  if (m >= 10) return `${m}分`;
  return `${m}分${rs}秒`;
}

export function extractDroppedUrl(ev: DragEvent<HTMLDivElement>): string {
  const uri = ev.dataTransfer.getData("text/uri-list");
  if (uri?.trim()) return uri.trim().split("\n")[0].trim();
  const plain = ev.dataTransfer.getData("text/plain");
  if (plain?.trim().startsWith("http")) return plain.trim();
  const html = ev.dataTransfer.getData("text/html");
  const m = html?.match(/https?:\/\/[^"' >]+/i);
  return m?.[0] ?? "";
}

export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(" ");
}
