import type { TabId } from "../types";

export function projectIdFromPath(pathname: string): number | null {
  const match = pathname.match(/^\/project\/(\d+)/);
  return match ? Number(match[1]) : null;
}

export function tabFromPath(pathname: string): TabId {
  if (pathname === "/settings") return "integrations";
  if (pathname === "/guide") return "guide";
  // Legacy deep links are kept reachable, but resolve into the four-stage
  // production workspace instead of rendering the retired mock pages.
  if (pathname.includes("/character-factory")) return "dataset";
  if (pathname.includes("/style-curator")) return "dataset";
  if (pathname.includes("/operations")) return "training";
  if (pathname.includes("/compare")) return "runs";
  if (pathname.includes("/runs")) return "runs";
  if (pathname.includes("/dataset")) return "dataset";
  if (pathname.includes("/train")) return "training";
  if (pathname.includes("/library")) return "library";
  if (pathname.includes("/overview")) return "dataset";
  return "home";
}

export function pathForTab(tab: TabId, projectId: number | null): string {
  if (tab === "integrations") return "/settings";
  if (tab === "guide") return "/guide";
  if (projectId == null) return "/";
  const suffix: Record<string, string> = {
    overview: "overview", dataset: "dataset/images", training: "train",
    runs: "runs", compare: "compare", library: "library",
    characterFactory: "character-factory", styleCurator: "style-curator", operations: "operations",
  };
  return `/project/${projectId}/${suffix[tab] ?? "overview"}`;
}
