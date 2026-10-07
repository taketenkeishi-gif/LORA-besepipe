import { useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  CircleAlert,
  Image as ImageIcon,
  LockKeyhole,
  Play,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Table2,
} from "lucide-react";
import { apiDelete, apiGet, apiPost } from "../../lib/api";
import type { Project, RunDetail, TabId, TrainingStatus } from "../../types";

type Props = { project: Project; onNavigate: (tab: TabId) => void };
type SystemStatus = {
  gpu?: {
    target_physical_index?: number;
    target_device?: string;
    target_state?: string;
  };
};
type Workspace = {
  snapshots?: Array<{ id: number; name: string; item_count?: number }>;
  assets?: Array<{ id: number; file_path?: string; review_status?: string; metadata?: { identity_lock?: Record<string, boolean>; role?: string } }>;
};
type RunsResult = {
  runs?: Array<{
    id: number;
    status: string;
    current_stage?: string;
    current_epoch?: number;
    total_epochs?: number;
    dataset_snapshot_id?: number | null;
  }>;
};
type LibraryResult = Array<{
  id: number;
  training_run_id?: number | null;
  dataset_snapshot_id?: number | null;
}>;
type CharacterGenerationRun = {
  id: number;
  status: string;
  current_stage?: string;
  output_manifest?: Record<string, unknown>;
  stage_history?: Array<Record<string, unknown>>;
  requested?: Record<string, unknown>;
  resolved?: Record<string, unknown>;
  observed?: Record<string, unknown>;
  error_detail?: string;
};

const steps = [
  "参照画像登録",
  "バリエーション生成",
  "候補レビュー",
  "データセット確定",
  "学習実行",
  "プレビュー比較",
];

function Card({
  title,
  eyebrow,
  children,
  className = "",
}: {
  title: string;
  eyebrow?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-[#263348] bg-[#101824] overflow-hidden ${className}`}
    >
      <div className="flex items-center justify-between border-b border-[#263348] px-4 py-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.16em] text-[#7f8da3]">
            {eyebrow}
          </div>
          <h2 className="text-sm font-semibold text-[#f4f7fb]">{title}</h2>
        </div>
        <SlidersHorizontal size={15} className="text-[#7f8da3]" />
      </div>
      <div className="p-4">{children}</div>
    </section>
  );
}

function EmptyMedia({
  label,
  accent = false,
}: {
  label: string;
  accent?: boolean;
}) {
  return (
    <div
      className={`flex min-h-28 items-center justify-center rounded-lg border border-dashed ${accent ? "border-[#5d63f2]/70 bg-[#5d63f2]/10" : "border-[#34445b] bg-[#0c1320]"}`}
    >
      <div className="text-center">
        <ImageIcon size={20} className="mx-auto mb-2 text-[#71819a]" />
        <div className="text-xs text-[#a7b2c4]">{label}</div>
      </div>
    </div>
  );
}

function ShellHeader({
  number,
  title,
  subtitle,
}: {
  number: string;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="mb-5 flex items-end justify-between">
      <div>
        <div className="mb-1 text-[11px] font-mono tracking-[0.18em] text-[#737ff4]">
          {number}
        </div>
        <h1 className="text-2xl font-bold text-[#f4f7fb]">{title}</h1>
        <p className="mt-1 text-xs text-[#a7b2c4]">{subtitle}</p>
      </div>
      <div className="rounded-full border border-[#263348] bg-[#141e2c] px-3 py-1.5 text-[11px] text-[#a7b2c4]">
        実データ表示 / 操作時に保存
      </div>
    </div>
  );
}

export function CharacterFactoryPage({ project, onNavigate }: Props) {
  const identityKeys = ["face", "hair", "eyes", "outfit", "body", "color"] as const;
  const identityLabels: Record<(typeof identityKeys)[number], string> = { face: "顔の特徴", hair: "髪型", eyes: "目", outfit: "衣装", body: "体型", color: "色設計" };
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [loadError, setLoadError] = useState("");
  const [selectedAssetIds, setSelectedAssetIds] = useState<number[]>([]);
  const [identityLock, setIdentityLock] = useState<Record<string, boolean>>({ face: true, hair: true, eyes: true, outfit: true, body: true, color: true });
  const [referenceRole, setReferenceRole] = useState("front");
  const [generationSeed, setGenerationSeed] = useState(12345);
  const [preflight, setPreflight] = useState<{
    run_id: number;
    status: string;
    checks: Array<{ label: string; ok: boolean; detail: string }>;
    message: string;
  } | null>(null);
  const [preflightLoading, setPreflightLoading] = useState(false);
  const [prepareMessage, setPrepareMessage] = useState("");
  const [preparedRun, setPreparedRun] = useState<{
    run_id: number;
    status: string;
    requested: Record<string, unknown>;
    resolved: Record<string, unknown>;
    observed: Record<string, unknown>;
  } | null>(null);
  const [latestGenerationRun, setLatestGenerationRun] =
    useState<CharacterGenerationRun | null>(null);
  useEffect(() => {
    void Promise.all([
      apiGet<Workspace>(`/basepipe/projects/${project.id}/workspace`),
      apiGet<{ runs?: CharacterGenerationRun[] }>(
        `/basepipe/projects/${project.id}/character-generation/runs`,
      ),
    ])
      .then(([nextWorkspace, generationRuns]) => {
        setWorkspace(nextWorkspace);
        const latest = generationRuns.runs?.[0] ?? null;
        setLatestGenerationRun(latest);
        if (latest?.requested && latest.resolved && latest.observed)
          setPreparedRun({
            run_id: latest.id,
            status: latest.status,
            requested: latest.requested,
            resolved: latest.resolved,
            observed: latest.observed,
          });
      })
      .catch((e) =>
        setLoadError(e instanceof Error ? e.message : "Workspace取得失敗"),
      );
  }, [project.id]);
  const snapshotCount = workspace?.snapshots?.length ?? 0;
  const assetCount = workspace?.assets?.length ?? 0;
  const toggleAsset = (id: number) =>
    setSelectedAssetIds((current) => {
      if (current.includes(id)) return current.filter((value) => value !== id);
      const asset = workspace?.assets?.find((item) => item.id === id);
      if (asset?.metadata?.identity_lock) setIdentityLock((lock) => ({ ...lock, ...asset.metadata?.identity_lock }));
      if (asset?.metadata?.role) setReferenceRole(asset.metadata.role);
      return [...current, id].slice(-4);
    });
  async function saveIdentityLock() {
    if (!selectedAssetIds.length) return;
    try {
      await Promise.all(selectedAssetIds.map((id) => {
        const asset = workspace?.assets?.find((item) => item.id === id);
        return apiPost(`/basepipe/assets/${id}/review`, {
          review_status: asset?.review_status ?? "pending",
          metadata: { ...(asset?.metadata ?? {}), identity_lock: identityLock, role: id === selectedAssetIds[0] ? referenceRole : (asset?.metadata?.role ?? "detail") },
        }, "PATCH");
      }));
      setWorkspace(await apiGet<Workspace>(`/basepipe/projects/${project.id}/workspace`));
      setPrepareMessage("Identity Lock / Reference roleを保存しました");
    } catch (e) { setLoadError(e instanceof Error ? e.message : "Identity Lock保存失敗"); }
  }
  async function runPreflight() {
    setPreflightLoading(true);
    setPrepareMessage("");
    try {
      setPreflight(
        await apiPost(
          `/basepipe/projects/${project.id}/character-generation/preflight`,
          { asset_ids: selectedAssetIds, mode: "variation", gpu_device_id: 1 },
        ),
      );
    } catch (e) {
      setLoadError(
        e instanceof Error ? e.message : "Character Generation Preflight失敗",
      );
    } finally {
      setPreflightLoading(false);
    }
  }
  async function prepareGeneration() {
    setPrepareMessage("");
    try {
      const result = await apiPost<{
        run_id: number;
        status: string;
        message: string;
        requested: Record<string, unknown>;
        resolved: Record<string, unknown>;
        observed: Record<string, unknown>;
      }>(`/basepipe/projects/${project.id}/character-generation/start`, {
        asset_ids: selectedAssetIds,
        mode: "variation",
        prompt: "",
        seed: generationSeed,
        width: 768,
        height: 512,
        length: 56,
        ref_image_size: "match",
        gpu_device_id: 1,
        execute: false,
        identity_lock: identityLock,
        reference_roles: Object.fromEntries(selectedAssetIds.map((id, index) => [String(id), index === 0 ? referenceRole : (workspace?.assets?.find((asset) => asset.id === id)?.metadata?.role ?? "detail")])),
      });
      setPreparedRun(result);
      setLatestGenerationRun({
        id: result.run_id,
        status: result.status,
        requested: result.requested,
        resolved: result.resolved,
        observed: result.observed,
        current_stage: "planning",
      });
      setPrepareMessage(
        `Run #${result.run_id} を準備しました（GPUジョブ未開始）`,
      );
    } catch (e) {
      setLoadError(
        e instanceof Error ? e.message : "Character Generation準備失敗",
      );
    }
  }
  const generationStatus = latestGenerationRun?.status ?? "not_started";
  const generationStage = latestGenerationRun?.current_stage ?? "—";
  const generationManifestCount = Object.keys(
    latestGenerationRun?.output_manifest ?? {},
  ).length;
  return (
    <div className="mx-auto max-w-[1180px] pb-10">
      <ShellHeader
        number="01"
        title="Character LoRA Factory"
        subtitle={`${project.name} / 高速モード — 参照から候補生成、レビュー、Dataset確定まで`}
      />
      <div className="mb-4 grid grid-cols-6 gap-2">
        {steps.map((step, i) => (
          <div
            key={step}
            className={`rounded-lg border px-3 py-2 text-center text-[11px] ${i === 0 ? "border-[#5d63f2] bg-[#5d63f2]/15 text-white" : "border-[#263348] bg-[#101824] text-[#7f8da3]"}`}
          >
            <span className="mr-1 font-mono">0{i + 1}</span>
            {step}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-12 gap-4">
        <Card
          eyebrow="REFERENCE CARD"
          title="参照画像とIdentity Lock"
          className="col-span-7"
        >
          <div className="mb-3 flex items-center justify-between text-[11px] text-[#7f8da3]">
            <span>Workspace assets: {assetCount}</span>
            <span>Dataset snapshots: {snapshotCount}</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <EmptyMedia
              label={
                assetCount
                  ? `${assetCount} assets / 実データ接続済み`
                  : "Front reference / 未登録"
              }
              accent
            />
            <EmptyMedia
              label={
                snapshotCount
                  ? `${snapshotCount} snapshots / 実データ接続済み`
                  : "Back reference / 未登録"
              }
            />
          </div>
          {assetCount > 0 && (
            <div className="mt-3 grid max-h-28 grid-cols-2 gap-1 overflow-auto">
              {(workspace?.assets ?? []).slice(0, 12).map((asset) => (
                <label
                  key={asset.id}
                  className="flex items-center gap-2 rounded border border-[#263348] px-2 py-1 text-[10px] text-[#a7b2c4]"
                >
                  <input
                    type="checkbox"
                    checked={selectedAssetIds.includes(asset.id)}
                    onChange={() => toggleAsset(asset.id)}
                    className="accent-[#5d63f2]"
                  />
                  Asset #{asset.id}
                </label>
              ))}
            </div>
          )}
          {loadError && (
            <div className="mt-3 text-xs text-[#ef5b68]">{loadError}</div>
          )}
          <div className="mt-4 rounded-lg border border-[#263348] bg-[#0c1320] p-3">
            <div className="mb-3 flex items-center gap-2 text-xs font-semibold text-[#f4f7fb]">
              <LockKeyhole size={14} className="text-[#737ff4]" /> Identity Lock
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs text-[#a7b2c4]">
              {identityKeys.map((key) => (
                <label key={key} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    aria-label={`Identity Lock ${identityLabels[key]}`}
                    checked={identityLock[key]}
                    onChange={(event) => setIdentityLock((current) => ({ ...current, [key]: event.target.checked }))}
                    className="accent-[#5d63f2]"
                  />
                  {identityLabels[key]}
                </label>
              ))}
            </div>
            <div className="mt-3 flex items-end gap-2">
              <label className="flex-1 text-[10px] text-[#7f8da3]">
                Reference role
                <select
                  aria-label="Reference Role"
                  value={referenceRole}
                  onChange={(event) => setReferenceRole(event.target.value)}
                  className="mt-1 w-full rounded border border-[#263348] bg-[#101a2a] px-2 py-1.5 text-xs text-[#f4f7fb]"
                >
                  <option value="front">Front</option>
                  <option value="back">Back</option>
                  <option value="detail">Detail</option>
                </select>
              </label>
              <button
                type="button"
                disabled={!selectedAssetIds.length}
                onClick={() => void saveIdentityLock()}
                className="rounded border border-[#3b4aa0] px-2 py-1.5 text-xs text-[#cdd3ff] disabled:opacity-40"
              >
                保存
              </button>
            </div>
          </div>
        </Card>
        <Card
          eyebrow="VARIATION SETTINGS"
          title="バリエーション設定"
          className="col-span-5"
        >
          <div className="grid grid-cols-2 gap-2">
            {[
              "Camera",
              "Framing",
              "Pose",
              "Expression",
              "Background",
              "Seed",
            ].map((x, i) => (
              <div
                key={x}
                className="rounded-lg border border-[#263348] bg-[#0c1320] p-3"
              >
                <div className="mb-1 text-[10px] text-[#7f8da3]">{x}</div>
                {i === 5 ? (
                  <input
                    aria-label="Generation Seed"
                    type="number"
                    min="0"
                    value={generationSeed}
                    onChange={(event) => setGenerationSeed(Number(event.target.value) || 0)}
                    className="w-full rounded border border-[#263348] bg-[#101a2a] px-2 py-1 text-xs text-[#f4f7fb]"
                  />
                ) : (
                  <div className="text-xs text-[#f4f7fb]">Auto</div>
                )}
              </div>
            ))}
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <button
              onClick={() => void runPreflight()}
              disabled={preflightLoading || selectedAssetIds.length === 0}
              className="flex items-center justify-center gap-2 rounded-lg border border-[#34445b] py-2.5 text-xs font-semibold text-[#dbe3ef] disabled:opacity-40"
            >
              <RefreshCw size={14} />
              {preflightLoading ? "確認中…" : "Preflight"}
            </button>
            <button
              onClick={() => void prepareGeneration()}
              disabled={selectedAssetIds.length === 0}
              className="flex items-center justify-center gap-2 rounded-lg bg-[#5d63f2] py-2.5 text-xs font-semibold text-white disabled:opacity-40"
            >
              <Sparkles size={14} />
              H3 Runを準備（GPU未起動）
            </button>
          </div>
          {prepareMessage && (
            <div className="mt-2 text-xs text-[#2fd38a]">{prepareMessage}</div>
          )}
        </Card>
        {preflight && (
          <Card
            eyebrow="PREFLIGHT EVIDENCE"
            title={`H3 Generation / ${preflight.status}`}
            className="col-span-12"
          >
            <div className="grid gap-2 md:grid-cols-3">
              {preflight.checks.map((check) => (
                <div
                  key={check.label}
                  className="rounded border border-[#263348] bg-[#0c1320] p-3 text-xs"
                >
                  <div className="flex items-center gap-2 text-[#f4f7fb]">
                    {check.ok ? (
                      <CheckCircle2 size={14} className="text-[#2fd38a]" />
                    ) : (
                      <AlertTriangle size={14} className="text-[#ef5b68]" />
                    )}
                    {check.label}
                  </div>
                  <div className="mt-1 text-[#a7b2c4]">{check.detail}</div>
                </div>
              ))}
            </div>
            <div className="mt-3 text-xs text-[#a7b2c4]">
              {preflight.message} / Preflight Run #{preflight.run_id}
            </div>
          </Card>
        )}
        {preparedRun && (
          <Card
            eyebrow="RUN EVIDENCE"
            title={`H3 Run #${preparedRun.run_id} / ${preparedRun.status} / ${latestGenerationRun?.current_stage ?? "planning"}`}
            className="col-span-12"
          >
            <div className="grid gap-2 md:grid-cols-3 text-xs">
              <div className="rounded border border-[#263348] bg-[#0c1320] p-3">
                <div className="text-[#7f8da3]">Requested</div>
                <div className="mt-1 text-[#a7b2c4]">
                  GPU{String(preparedRun.requested.gpu_device_id ?? "—")} / execute=
                  {String(preparedRun.requested.execute ?? "—")}
                </div>
              </div>
              <div className="rounded border border-[#263348] bg-[#0c1320] p-3">
                <div className="text-[#7f8da3]">Resolved</div>
                <div className="mt-1 text-[#a7b2c4]">
                  {String(preparedRun.resolved.engine ?? "—")}
                </div>
              </div>
              <div className="rounded border border-[#263348] bg-[#0c1320] p-3">
                <div className="text-[#7f8da3]">Observed</div>
                <div className="mt-1 text-[#a7b2c4]">
                  queued={String(preparedRun.observed.queued ?? "—")}
                </div>
              </div>
            </div>
          </Card>
        )}
        <Card
          eyebrow="GENERATION PROGRESS"
          title={`生成ステージ / ${generationStatus}`}
          className="col-span-8"
        >
          <div className="mb-4 flex items-center justify-between text-xs">
            <span className="text-[#f4f7fb]">
              {latestGenerationRun
                ? `Run #${latestGenerationRun.id} / ${generationStage}`
                : "Run未作成 — 実行待ち"}
            </span>
            <span className="text-[#7f8da3]">
              manifest {generationManifestCount} keys
            </span>
          </div>
          <div className="h-2 rounded-full bg-[#263348]">
            <div
              className={`h-2 rounded-full ${latestGenerationRun?.status === "completed" ? "w-full" : latestGenerationRun ? "w-1/3" : "w-0"} bg-[#5d63f2]`}
            />
          </div>
          <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
            <div className="rounded border border-[#263348] bg-[#0c1320] p-3">
              <div className="text-[#7f8da3]">Requested</div>
              <div className="mt-1 text-[#a7b2c4]">
                GPU{String(latestGenerationRun?.requested?.gpu_device_id ?? "—")} / execute={String(latestGenerationRun?.requested?.execute ?? "—")}
              </div>
            </div>
            <div className="rounded border border-[#263348] bg-[#0c1320] p-3">
              <div className="text-[#7f8da3]">Observed</div>
              <div className="mt-1 text-[#a7b2c4]">
                queued={String(latestGenerationRun?.observed?.queued ?? "—")}
              </div>
            </div>
            <div className="rounded border border-[#263348] bg-[#0c1320] p-3">
              <div className="text-[#7f8da3]">Output</div>
              <div className="mt-1 text-[#a7b2c4]">
                {generationManifestCount ? "manifestあり" : "未生成"}
              </div>
            </div>
          </div>
          {latestGenerationRun?.error_detail && (
            <div className="mt-3 rounded border border-[#7d2e3a] bg-[#35151c] p-2 text-xs text-[#ff9aa7]">
              FailureInfo: {latestGenerationRun.error_detail}
            </div>
          )}
        </Card>
        <Card
          eyebrow="QUALITY"
          title="Identity / Quality"
          className="col-span-4"
        >
          <Metric label="Identity Score" value="—" />
          <Metric label="Quality Score" value="—" />
          <div className="mt-4 rounded-lg border border-[#34445b] bg-[#0c1320] p-3 text-xs text-[#a7b2c4]">
            生成ログはRunのEvidenceに記録されます。
          </div>
        </Card>
        <Card
          eyebrow="PIPELINE FLOW"
          title="処理パイプライン"
          className="col-span-12"
        >
          <div className="flex items-center gap-2 overflow-x-auto pb-1">
            {[
              "Master Reference",
              "H3 Generation",
              "Frame Filter",
              "Qwen Edit",
              "Identity Gate",
              "Caption",
              "Dataset Snapshot",
            ].map((x, i) => (
              <div key={x} className="flex shrink-0 items-center gap-2">
                {i > 0 && <ArrowRight size={14} className="text-[#5d63f2]" />}
                <div className="rounded-lg border border-[#263348] bg-[#141e2c] px-3 py-2 text-[11px] text-[#dbe3ef]">
                  {x}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <div className="mt-4 flex justify-end">
        <button
          onClick={() => onNavigate("styleCurator")}
          className="flex items-center gap-2 text-xs text-[#9da5ff] hover:text-white"
        >
          Style Curatorへ <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="mb-3 flex items-center justify-between border-b border-[#263348] pb-3 text-xs">
      <span className="text-[#a7b2c4]">{label}</span>
      <span className="font-mono text-[#f4f7fb]">{value}</span>
    </div>
  );
}

export function StyleCuratorPage({ project, onNavigate }: Props) {
  type Asset = {
    id: number;
    review_status?: string;
    caption_original?: string;
    caption_edited?: string;
    caption_processed?: string;
    training_input?: string;
    user_rating?: number | null;
    training_enabled?: boolean | number;
    training_weight?: number;
    group_ids?: number[];
    groups?: Array<{ id: number; name: string }>;
  };
  type Group = { id: number; name: string; asset_count?: number };
  const [assets, setAssets] = useState<Asset[]>([]);
  const [groups, setGroups] = useState<Group[]>([]);
  const [activeGroup, setActiveGroup] = useState<number | "uncategorized" | null>(null);
  const [groupName, setGroupName] = useState("");
  const [selected, setSelected] = useState<Asset | null>(null);
  const [caption, setCaption] = useState("");
  const [loadError, setLoadError] = useState("");
  const [notice, setNotice] = useState("");
  const [saving, setSaving] = useState(false);
  const load = () =>
    void Promise.all([
      apiGet<{ assets?: Asset[] }>(`/basepipe/projects/${project.id}/assets`),
      apiGet<{ groups?: Group[] }>(`/basepipe/projects/${project.id}/groups`),
    ])
      .then(([assetResult, groupResult]) => {
        const next = assetResult.assets ?? [];
        setAssets(next);
        setGroups(groupResult.groups ?? []);
        if (selected)
          setSelected(next.find((asset) => asset.id === selected.id) ?? null);
      })
      .catch((e) =>
        setLoadError(e instanceof Error ? e.message : "Asset取得失敗"),
      );
  useEffect(load, [project.id]);
  function selectAsset(asset: Asset) {
    setSelected(asset);
    setCaption(
      asset.caption_edited ??
        asset.training_input ??
        asset.caption_original ??
        "",
    );
    setNotice("");
  }
  async function saveAsset() {
    if (!selected) return;
    setSaving(true);
    setNotice("");
    try {
      const updated = await apiPost<Asset>(
        `/basepipe/assets/${selected.id}/review`,
        {
          review_status: selected.review_status ?? "pending",
          caption,
          user_rating: selected.user_rating ?? null,
          training_enabled: selected.training_enabled ?? true,
          training_weight: selected.training_weight ?? 1,
        },
        "PATCH",
      );
      setSelected(updated);
      setAssets((current) =>
        current.map((asset) => (asset.id === updated.id ? updated : asset)),
      );
      setCaption(updated.caption_edited ?? updated.training_input ?? "");
      setNotice(`Asset #${updated.id} を保存しました`);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Asset保存失敗");
    } finally {
      setSaving(false);
    }
  }
  async function createGroup() {
    const name = groupName.trim();
    if (!name) return;
    try {
      await apiPost(`/basepipe/projects/${project.id}/groups`, { name });
      setGroupName("");
      await load();
      setNotice(`Group「${name}」を保存しました`);
    } catch (e) { setLoadError(e instanceof Error ? e.message : "Group作成失敗"); }
  }
  async function renameGroup(group: Group) {
    const name = groupName.trim();
    if (!name || name === group.name) return;
    try {
      await apiPost(`/basepipe/projects/${project.id}/groups/${group.id}`, { name }, "PATCH");
      setGroupName("");
      await load();
      setNotice(`Group「${group.name}」をRenameしました`);
    } catch (e) { setLoadError(e instanceof Error ? e.message : "Group Rename失敗"); }
  }
  async function deleteGroup(group: Group) {
    try {
      await apiDelete(`/basepipe/projects/${project.id}/groups/${group.id}`);
      if (activeGroup === group.id) setActiveGroup(null);
      await load();
      setNotice(`Group「${group.name}」を削除しました（Asset本体は保持）`);
    } catch (e) { setLoadError(e instanceof Error ? e.message : "Group削除失敗"); }
  }
  async function assignSelected(group: Group) {
    if (!selected) return;
    try {
      await apiPost(`/basepipe/projects/${project.id}/groups/${group.id}/assets`, { asset_ids: [selected.id] }, "PUT");
      await load();
      setNotice(`Asset #${selected.id} をGroup「${group.name}」へ割り当てました`);
    } catch (e) { setLoadError(e instanceof Error ? e.message : "Group割当失敗"); }
  }
  async function unassignSelected(group: Group) {
    if (!selected) return;
    try {
      await apiDelete(`/basepipe/projects/${project.id}/groups/${group.id}/assets/${selected.id}`);
      await load();
      setNotice(`Asset #${selected.id} のGroup割当を解除しました`);
    } catch (e) { setLoadError(e instanceof Error ? e.message : "Group解除失敗"); }
  }
  const visibleAssets = assets.filter((asset) =>
    activeGroup === null ? true : activeGroup === "uncategorized" ? !(asset.group_ids ?? []).length : (asset.group_ids ?? []).includes(activeGroup),
  );
  return (
    <div className="mx-auto max-w-[1180px] pb-10">
      <ShellHeader
        number="02"
        title="Style Curator"
        subtitle={`${project.name} / Contact Sheet・Inspector・汚染除去`}
      />
      <div className="grid grid-cols-12 gap-4">
        <Card eyebrow="GROUPS" title="Style Groups" className="col-span-2">
          <button
            onClick={() => setActiveGroup("uncategorized")}
            className="mb-2 w-full rounded-lg border border-[#5d63f2] bg-[#5d63f2]/15 px-3 py-2 text-left text-xs text-white"
          >
            未分類 <span className="float-right">{assets.filter((asset) => !(asset.group_ids ?? []).length).length}</span>
          </button>
          <button onClick={() => setActiveGroup(null)} className="mb-2 w-full border-b border-[#263348] px-2 py-2 text-left text-[11px] text-[#7f8da3]">全Asset <span className="float-right">{assets.length}</span></button>
          {groups.map((group) => (
            <div key={group.id} className={`mb-1 rounded border px-2 py-2 text-xs ${activeGroup === group.id ? "border-[#5d63f2] bg-[#5d63f2]/15 text-white" : "border-[#263348] text-[#a7b2c4]"}`}>
              <button onClick={() => { setActiveGroup(group.id); setGroupName(group.name); }} className="w-full text-left">{group.name}<span className="float-right">{group.asset_count ?? 0}</span></button>
              {activeGroup === group.id && <div className="mt-2 flex gap-1"><input aria-label={`Rename Group ${group.id}`} value={groupName} onChange={(e) => setGroupName(e.target.value)} className="min-w-0 flex-1 rounded border border-[#34445b] bg-[#0c1320] px-1 py-1 text-[10px] text-white"/><button onClick={() => void renameGroup(group)} className="rounded border border-[#34445b] px-1 text-[10px]">保存</button><button onClick={() => void deleteGroup(group)} className="rounded border border-[#7d2e3a] px-1 text-[10px] text-[#ff9aa7]">削除</button></div>}
            </div>
          ))}
          <div className="mt-2 flex gap-1"><input aria-label="New Style Group" value={groupName} onChange={(e) => setGroupName(e.target.value)} placeholder="New Group" className="min-w-0 flex-1 rounded border border-[#34445b] bg-[#0c1320] px-1 py-1 text-[10px] text-white"/><button onClick={() => void createGroup()} className="rounded border border-[#5d63f2] px-1 text-[10px] text-[#dbe3ef]">追加</button></div>
        </Card>
        <Card eyebrow="CONTACT SHEET" title="候補一覧" className="col-span-7">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-xs text-[#a7b2c4]">
              {visibleAssets.length} / {assets.length} assets / 実データ接続済み
            </div>
            <button
              onClick={load}
              className="flex items-center gap-1 rounded border border-[#34445b] px-2 py-1 text-[11px] text-[#dbe3ef]"
            >
              <RefreshCw size={12} />
              更新
            </button>
          </div>
          {loadError && (
            <div className="mb-3 text-xs text-[#ef5b68]">{loadError}</div>
          )}
          {notice && (
            <div className="mb-3 text-xs text-[#2fd38a]">{notice}</div>
          )}
          <div className="grid grid-cols-4 gap-3">
            {Array.from({ length: Math.max(8, visibleAssets.length) }, (_, i) =>
              visibleAssets[i] ? (
                <button
                  key={visibleAssets[i].id}
                  onClick={() => selectAsset(visibleAssets[i])}
                  className={`text-left ${selected?.id === visibleAssets[i].id ? "rounded-lg ring-2 ring-[#5d63f2]" : ""}`}
                >
                  <EmptyMedia label={`Asset #${visibleAssets[i].id}`} />
                </button>
              ) : (
                <EmptyMedia
                  key={i}
                  label={`Asset ${String(i + 1).padStart(2, "0")}`}
                />
              ),
            )}
          </div>
        </Card>
        <Card eyebrow="INSPECTOR" title="選択アイテム" className="col-span-3">
          <EmptyMedia
            label={
              selected ? `Asset #${selected.id}` : "Assetを選択してください"
            }
          />
          <div className="mt-4 space-y-3 text-xs">
            <div>
              <div className="mb-1 text-[#7f8da3]">Caption Lineage</div>
              <div className="rounded border border-[#263348] bg-[#0c1320] p-2 text-[#a7b2c4]">
                Original → Edited → Training Input
              </div>
            </div>
            {selected && (
              <>
                <label className="block">
                  <span className="mb-1 block text-[#7f8da3]">
                    Edited caption
                  </span>
                  <textarea
                    value={caption}
                    onChange={(e) => setCaption(e.target.value)}
                    className="min-h-24 w-full rounded border border-[#34445b] bg-[#0c1320] p-2 text-xs text-[#f4f7fb]"
                  />
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <label className="block"><span className="mb-1 block text-[#7f8da3]">Review Status</span><select aria-label="Review Status" value={selected.review_status ?? "pending"} onChange={(e) => setSelected({ ...selected, review_status: e.target.value })} className="w-full rounded border border-[#34445b] bg-[#0c1320] p-2 text-xs text-[#f4f7fb]"><option value="pending">pending</option><option value="needs_review">needs_review</option><option value="approved">approved</option><option value="rejected">rejected</option></select></label>
                  <label className="block"><span className="mb-1 block text-[#7f8da3]">Rating</span><select aria-label="User Rating" value={selected.user_rating ?? ""} onChange={(e) => setSelected({ ...selected, user_rating: e.target.value ? Number(e.target.value) : null })} className="w-full rounded border border-[#34445b] bg-[#0c1320] p-2 text-xs text-[#f4f7fb]"><option value="">—</option>{[1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>{value} / 5</option>)}</select></label>
                </div>
                <div className="flex items-center gap-3 text-[11px] text-[#a7b2c4]"><label className="flex items-center gap-1"><input aria-label="Training Enabled" type="checkbox" checked={selected.training_enabled === true || selected.training_enabled === 1} onChange={(e) => setSelected({ ...selected, training_enabled: e.target.checked })} /> Training Enabled</label><label className="flex items-center gap-1">Weight <input aria-label="Training Weight" type="number" min="0.1" max="10" step="0.05" value={selected.training_weight ?? 1} onChange={(e) => setSelected({ ...selected, training_weight: Number(e.target.value) })} className="w-16 rounded border border-[#34445b] bg-[#0c1320] px-1 py-1 text-xs text-white" /></label></div>
                <button
                  onClick={() => void saveAsset()}
                  disabled={saving}
                  className="w-full rounded-lg bg-[#5d63f2] px-3 py-2 text-xs font-semibold text-white disabled:opacity-50"
                >
                  {saving ? "保存中…" : "Caption / Reviewを保存"}
                </button>
                <div className="rounded border border-[#263348] bg-[#0c1320] p-2 text-[10px] text-[#a7b2c4]">
                  <div className="mb-1 text-[#7f8da3]">Group Assignment</div>
                  <div className="flex flex-wrap gap-1">{(selected.groups ?? []).map((group) => <button key={group.id} onClick={() => void unassignSelected(group)} className="rounded bg-[#263348] px-1.5 py-1 text-[#dbe3ef]">{group.name} ×</button>)}</div>
                  <div className="mt-1 flex flex-wrap gap-1">{groups.filter((group) => !(selected.group_ids ?? []).includes(group.id)).map((group) => <button key={group.id} onClick={() => void assignSelected(group)} className="rounded border border-[#34445b] px-1.5 py-1 text-[#a7b2c4]">+ {group.name}</button>)}</div>
                </div>
              </>
            )}
            <div>
              <div className="mb-1 text-[#7f8da3]">Acceptance</div>
              <div className="flex items-center gap-2 text-[#a7b2c4]">
                <CircleAlert size={14} className="text-[#d6a84f]" /> Human
                decision required
              </div>
            </div>
          </div>
        </Card>
        <Card
          eyebrow="CLEANUP"
          title="汚染除去 / Mask比較"
          className="col-span-12"
        >
          <div className="grid grid-cols-3 gap-3">
            <EmptyMedia label="Original" />
            <EmptyMedia label="Mask" />
            <EmptyMedia label="Clean Preview" accent />
          </div>
        </Card>
      </div>
      <div className="mt-4 flex justify-end">
        <button
          onClick={() => onNavigate("operations")}
          className="flex items-center gap-2 text-xs text-[#9da5ff] hover:text-white"
        >
          Training / Runs / Compareへ <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}

export function OperationsPage({ project, onNavigate }: Props) {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [runs, setRuns] = useState<NonNullable<RunsResult["runs"]>>([]);
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [library, setLibrary] = useState<LibraryResult>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [runEvidence, setRunEvidence] = useState<RunDetail | null>(null);
  const [loadError, setLoadError] = useState("");
  const load = () =>
    void Promise.all([
      apiGet<TrainingStatus>(`/training/status?project_id=${project.id}`),
      apiGet<RunsResult>(`/training/runs?project_id=${project.id}`),
      apiGet<Workspace>(`/basepipe/projects/${project.id}/workspace`),
      apiGet<LibraryResult>(`/library/assets?project_id=${project.id}`),
      apiGet<SystemStatus>(`/system/status`),
    ])
      .then(([nextStatus, nextRuns, nextWorkspace, nextLibrary, nextSystemStatus]) => {
        setStatus(nextStatus);
        setRuns(nextRuns.runs ?? []);
        setWorkspace(nextWorkspace);
        setLibrary(nextLibrary);
        setSystemStatus(nextSystemStatus);
      })
      .catch((e) =>
        setLoadError(
          e instanceof Error ? e.message : "Operationsデータ取得失敗",
        ),
      );
  useEffect(load, [project.id]);
  const latestRun = runs[0];
  useEffect(() => {
    if (!latestRun) {
      setRunEvidence(null);
      return;
    }
    void apiGet<RunDetail>(`/training/runs/${latestRun.id}`)
      .then(setRunEvidence)
      .catch(() => setRunEvidence(null));
  }, [latestRun?.id]);
  const checkpointCount = runEvidence?.evidence.checkpoints?.length ?? 0;
  const preview = runEvidence?.evidence.preview;
  return (
    <div className="mx-auto max-w-[1180px] pb-10">
      <ShellHeader
        number="03"
        title="Training / Runs / Compare"
        subtitle={`${project.name} / Runを中心に学習・比較・Exportを追跡`}
      />
      <div className="mb-3 flex justify-end">
        <button
          onClick={load}
          className="flex items-center gap-1 rounded border border-[#34445b] px-2 py-1 text-[11px] text-[#dbe3ef]"
        >
          <RefreshCw size={12} />
          実データ更新
        </button>
      </div>
      {loadError && (
        <div className="mb-3 text-xs text-[#ef5b68]">{loadError}</div>
      )}
      <div className="grid grid-cols-12 gap-4">
        <Card
          eyebrow="TRAINING"
          title="Training Configuration"
          className="col-span-7"
        >
          <div className="grid grid-cols-3 gap-3">
            {[
              ["Model Family", status?.mode ?? "Anima"],
              [
                "GPU",
                systemStatus?.gpu?.target_device
                  ? `GPU${systemStatus.gpu.target_physical_index ?? "—"} / ${systemStatus.gpu.target_device}`
                  : "未検出",
              ],
              [
                "Snapshot",
                workspace?.snapshots?.[0]
                  ? `#${workspace.snapshots[0].id}`
                  : "未選択",
              ],
            ].map(([a, b]) => (
              <div
                key={a}
                className="rounded-lg border border-[#263348] bg-[#0c1320] p-3"
              >
                <div className="text-[10px] text-[#7f8da3]">{a}</div>
                <div className="mt-1 text-xs text-[#f4f7fb]">{b}</div>
              </div>
            ))}
          </div>
          <div className="mt-4 rounded-lg border border-[#d6a84f]/40 bg-[#d6a84f]/10 p-3 text-xs text-[#d8c28a]">
            Preflight: GPU{systemStatus?.gpu?.target_physical_index ?? "—"}の空きVRAM・占有プロセス・Dataset
            Snapshotを確認してから開始します。現在: {systemStatus?.gpu?.target_state ?? "unknown"}
          </div>
          <button
            onClick={() => onNavigate("training")}
            className="mt-4 flex items-center gap-2 rounded-lg border border-[#34445b] px-4 py-2 text-xs text-[#a7b2c4]"
          >
            <Play size={14} />
            既存Training画面でRunを作成
          </button>
        </Card>
        <Card eyebrow="RUNS" title="Run Evidence" className="col-span-5">
          <div className="overflow-hidden rounded-lg border border-[#263348]">
            <div className="grid grid-cols-4 bg-[#141e2c] px-3 py-2 text-[10px] text-[#7f8da3]">
              <span>ID</span>
              <span>Status</span>
              <span>Stage</span>
              <span>Checkpoint</span>
            </div>
            <div className="grid grid-cols-4 px-3 py-3 text-xs text-[#a7b2c4]">
              <span>{latestRun ? `#${latestRun.id}` : "—"}</span>
              <span>{latestRun?.status ?? "待機"}</span>
              <span>
                {latestRun?.current_stage ?? status?.current_stage ?? "—"}
              </span>
              <span>
                {checkpointCount > 0 ? `${checkpointCount}件` : "—"}
              </span>
            </div>
          </div>
          <div className="mt-3 flex items-center gap-2 text-xs text-[#7f8da3]">
            <CheckCircle2 size={14} className="text-[#2fd38a]" /> {runs.length}{" "}
            Runs / Requested・Resolved・Observedは既存Runsで確認
          </div>
          <div className="mt-2 text-xs text-[#a7b2c4]">
            Preview: {preview ? `${preview.succeeded}/${preview.expected} succeeded` : "未取得"}
          </div>
          <button
            onClick={() => onNavigate("runs")}
            className="mt-3 text-xs text-[#9da5ff]"
          >
            既存Runsを開く →
          </button>
        </Card>
        <Card eyebrow="COMPARE" title="Compare Matrix" className="col-span-8">
          <div className="grid grid-cols-4 gap-2">
            {["Checkpoint", "Epoch 1", "Epoch 2", "Epoch 3"].map((x) => (
              <div
                key={x}
                className="rounded border border-[#263348] bg-[#0c1320] p-3 text-center text-xs text-[#a7b2c4]"
              >
                {x}
              </div>
            ))}
            {["同一Prompt", "同一Seed", "同一Settings"].map((x) => (
              <div key={x} className="contents">
                <div className="p-2 text-xs text-[#7f8da3]">{x}</div>
                <div className="col-span-3 rounded border border-dashed border-[#34445b] p-3 text-center text-[11px] text-[#71819a]">
                  {preview
                    ? `実Preview ${preview.succeeded}/${preview.expected} succeeded`
                    : "Preview Evidence未取得"}
                </div>
              </div>
            ))}
          </div>
          <button
            onClick={() => onNavigate("compare")}
            className="mt-3 text-xs text-[#9da5ff]"
          >
            既存Compareを開く →
          </button>
        </Card>
        <Card
          eyebrow="EXPORT & LIBRARY"
          title="採用・Export・Library"
          className="col-span-4"
        >
          <div className="space-y-3 text-xs text-[#a7b2c4]">
            <div className="flex items-center gap-2">
              <Table2 size={14} /> {library.length} Library assets / Source
              Runへ逆引き
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 size={14} /> Select as Finalは人間決定
            </div>
            <button
              onClick={() => onNavigate("library")}
              className="w-full rounded-lg border border-[#34445b] px-3 py-2 text-[#9da5ff]"
            >
              既存Libraryを開く
            </button>
          </div>
        </Card>
      </div>
      <div className="mt-4 flex gap-4 text-xs">
        <button
          onClick={() => onNavigate("characterFactory")}
          className="text-[#9da5ff]"
        >
          ← Character Factory
        </button>
        <button
          onClick={() => onNavigate("styleCurator")}
          className="text-[#9da5ff]"
        >
          Style Curator →
        </button>
      </div>
    </div>
  );
}
