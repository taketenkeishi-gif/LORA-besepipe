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
