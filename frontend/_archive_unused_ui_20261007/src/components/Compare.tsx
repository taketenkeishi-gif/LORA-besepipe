import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { apiGet, apiPost } from "../lib/api";
import type {
  PreviewJob,
  PreviewProfile,
  PreviewSample,
  Project,
} from "../types";

type Props = { project: Project | null; showError: (message: string) => void };
type FrozenProfile = {
  id: number;
  name: string;
  snapshot_hash: string;
  payload: {
    seed?: number;
    prompt?: string;
    negative_prompt?: string;
    resolution?: number;
    steps?: number;
    cfg?: number;
    sampler?: string;
    scheduler?: string;
    model_family?: string;
  };
};
type Evaluation = {
  checkpoint_id: number;
  preview_slot: string;
  preview_profile_snapshot_id?: number | null;
  identity?: number | null;
  outfit_separation?: number | null;
  style_durability?: number | null;
  style_quality?: number | null;
  prompt_flexibility?: number | null;
  artifact?: number | null;
  accepted: number;
  note: string;
};
type EvaluationKey = `${number}:${string}`;
const RATING_FIELDS = [
  ["identity", "Identity"],
  ["outfit_separation", "Outfit separation"],
  ["style_durability", "Style durability"],
  ["style_quality", "Style quality"],
  ["prompt_flexibility", "Prompt flexibility"],
  ["artifact", "Artifact"],
] as const;

export default function Compare({ project, showError }: Props) {
  const [samples, setSamples] = useState<PreviewSample[]>([]);
  const [loading, setLoading] = useState(false);
  const [profiles, setProfiles] = useState<PreviewProfile[]>([]);
  const [profileId, setProfileId] = useState<number | "">("");
  const [frozenSnapshots, setFrozenSnapshots] = useState<FrozenProfile[]>([]);
  const [evaluations, setEvaluations] = useState<Record<string, Evaluation>>(
    {},
  );
  const [jobsByCheckpoint, setJobsByCheckpoint] = useState<
    Record<number, PreviewJob[]>
  >({});
  const [freezing, setFreezing] = useState(false);
  const [refreshingCheckpoint, setRefreshingCheckpoint] = useState<
    number | null
  >(null);
  const [comfyuiState, setComfyuiState] = useState("unknown");
  const [jobLoadWarning, setJobLoadWarning] = useState(false);
  const [loadError, setLoadError] = useState("");

  async function load() {
    if (!project) return;
    setLoading(true);
    setLoadError("");
    try {
      const [preview, profileResponse, frozenResponse, evaluationResponse, systemResponse] =
        await Promise.all([
          apiGet<{ timeline: PreviewSample[] }>(
            `/previews/${project.id}?include_images=false`,
          ),
          apiGet<{ profiles: PreviewProfile[] }>(
            `/preview-profiles?project_id=${project.id}`,
          ),
          apiGet<{ snapshots: FrozenProfile[] }>(
            `/basepipe/projects/${project.id}/preview-profile-snapshots`,
          ),
          apiGet<{ evaluations: Evaluation[] }>(
            `/basepipe/projects/${project.id}/evaluations`,
          ),
          apiGet<{ services?: Record<string, { state: string }> }>(
            "/system/status",
          ).catch(() => ({ services: {} as Record<string, { state: string }> })),
        ]);
      const timeline = preview.timeline ?? [];
      setSamples(timeline);
      const nextProfiles = profileResponse.profiles ?? [];
      setProfiles(nextProfiles);
      setFrozenSnapshots(frozenResponse.snapshots ?? []);
      setEvaluations(
        Object.fromEntries(
          (evaluationResponse.evaluations ?? []).map((evaluation) => [
            `${evaluation.checkpoint_id}:${evaluation.preview_slot}`,
            evaluation,
          ]),
        ),
      );
      setComfyuiState(systemResponse.services?.comfyui?.state ?? "unknown");
      setProfileId((current) => current || nextProfiles[0]?.id || "");

      const jobResponses = await Promise.allSettled(
        timeline.map((sample) =>
          apiGet<{ jobs: PreviewJob[] }>(
            `/previews/jobs/${sample.checkpoint_id}`,
          ),
        ),
      );
      const failedJobRequest = jobResponses.some(
        (result) => result.status === "rejected",
      );
      setJobLoadWarning(failedJobRequest);
      setJobsByCheckpoint(
        Object.fromEntries(
          timeline.map((sample, index) => {
            const result = jobResponses[index];
            return [
              sample.checkpoint_id,
              result?.status === "fulfilled" ? result.value.jobs ?? [] : [],
            ];
          }),
        ),
      );
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Preview比較の取得に失敗しました");
      showError(
        e instanceof Error ? e.message : "Preview比較の取得に失敗しました",
      );
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    void load();
    // App起動直後はproject read modelの更新とCompareの初回取得が競合する
    // ことがあるため、空のままなら一度だけ短い遅延再取得を行う。
    const retryTimer = window.setTimeout(() => {
      void load();
    }, 900);
    return () => window.clearTimeout(retryTimer);
  }, [project?.id]);

  const selectedProfile = profiles.find((profile) => profile.id === profileId);
  async function freezeSelectedProfile() {
    if (!project || !selectedProfile) return;
    setFreezing(true);
    try {
      await apiPost(
        `/basepipe/projects/${project.id}/preview-profile-snapshots/${selectedProfile.id}`,
        {},
      );
      await load();
    } catch (e) {
      showError(
        e instanceof Error
          ? e.message
          : "Preview Profileのfreezeに失敗しました",
      );
    } finally {
      setFreezing(false);
    }
  }
  async function refreshWithFrozenProfile(checkpointId: number) {
    const snapshot = frozenSnapshots[0];
    if (!snapshot)
      return showError("先にFrozen Preview Profileを作成してください");
    setRefreshingCheckpoint(checkpointId);
    try {
      await apiPost(
        `/previews/refresh-profile/${checkpointId}/${snapshot.id}`,
        {},
      );
      await load();
    } catch (e) {
      showError(
        e instanceof Error
          ? e.message
          : "Frozen条件Previewの再生成に失敗しました",
      );
    } finally {
      setRefreshingCheckpoint(null);
    }
  }
  function evaluationFor(checkpointId: number, slot: string): Evaluation {
    return (
      evaluations[`${checkpointId}:${slot}`] ?? {
        checkpoint_id: checkpointId,
        preview_slot: slot,
        accepted: 0,
        note: "",
      }
    );
  }
  function updateEvaluation(
    key: EvaluationKey,
    field: string,
    value: string | number,
  ) {
    const separator = key.indexOf(":");
    const checkpointId = Number(key.slice(0, separator));
    const previewSlot = key.slice(separator + 1);
    setEvaluations((current) => {
      const existing =
        current[key] ??
        ({
          checkpoint_id: checkpointId,
          preview_slot: previewSlot,
          accepted: 0,
          note: "",
        } as Evaluation);
      return {
        ...current,
        [key]: {
          ...existing,
          [field]: value === "" ? null : value,
        } as Evaluation,
      };
    });
  }
  async function saveEvaluation(evaluation: Evaluation) {
    if (!project) return;
    if (evaluation.accepted && !confirm("実Previewを確認したあなたの判断として、この候補をUSER_APPROVEDにしますか？")) return;
    try {
      const saved = await apiPost<Evaluation>(
        `/basepipe/projects/${project.id}/evaluations`,
        {
          ...evaluation,
          preview_profile_snapshot_id: frozenSnapshots[0]?.id ?? null,
          accepted: Boolean(evaluation.accepted),
          human_confirmed: Boolean(evaluation.accepted),
        },
      );
      setEvaluations((current) => ({
        ...current,
        [`${saved.checkpoint_id}:${saved.preview_slot}`]: saved,
      }));
    } catch (e) {
      showError(
        e instanceof Error ? e.message : "人間評価の保存に失敗しました",
      );
    }
  }
  const comparisonCards = samples.flatMap((sample) => {
    const entries = Object.entries(
      sample.sample_previews ?? sample.samples ?? {},
    );
    const jobEntries = entries.filter(([slot]) => slot.startsWith("p"));
    const selectedEntries = jobEntries.length > 0 ? jobEntries : entries;
    return selectedEntries.map(([slot, image]) => ({ sample, slot, image }));
  });
  const firstCheckpointId = comparisonCards[0]?.sample.checkpoint_id;

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-100">Compare</h2>
          <p className="text-sm text-gray-500">
            Checkpoint / EpochのPreview比較。最良判定は人間が行います。
          </p>
        </div>
        <button
          onClick={() => void load()}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border border-gray-700 px-3 py-2 text-xs text-gray-300"
        >
          <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          更新
        </button>
      </div>
      <section className="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-xs text-gray-400">
            Preview Profile
            <select
              value={profileId}
              onChange={(e) =>
                setProfileId(e.target.value ? Number(e.target.value) : "")
              }
              className="ml-2 rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-xs text-gray-200"
            >
              <option value="">未選択</option>
              {profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name}
                </option>
              ))}
            </select>
          </label>
          {selectedProfile && (
            <span className="text-[11px] text-gray-500">
              seed {selectedProfile.seed} · {selectedProfile.resolution}px ·{" "}
              {selectedProfile.steps} steps ·{" "}
              {selectedProfile.model_family || "auto"}
            </span>
          )}
          <button
            disabled={!selectedProfile || freezing}
            onClick={() => void freezeSelectedProfile()}
            className="rounded bg-indigo-600 px-3 py-2 text-xs text-white disabled:opacity-50"
          >
            {freezing ? "固定中…" : "この条件をfreeze"}
          </button>
          <span className="text-[10px] text-gray-600">
            freeze後の条件はCompare用に不変保存
          </span>
        </div>
        {frozenSnapshots.length > 0 && (
          <div className="mt-3 border-t border-gray-800 pt-3">
            <p className="text-[11px] font-semibold text-emerald-300">
              Frozen Preview Profiles
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              {frozenSnapshots.map((snapshot) => (
                <span
                  key={snapshot.id}
                  className="rounded border border-emerald-900 bg-emerald-950/30 px-2 py-1 text-[10px] text-emerald-200"
                >
                  {snapshot.name} · seed {snapshot.payload.seed} ·{" "}
                  {snapshot.snapshot_hash.slice(0, 8)}
                </span>
              ))}
            </div>
            <button
              disabled={
                firstCheckpointId == null ||
                refreshingCheckpoint != null ||
                comfyuiState !== "ready"
              }
              onClick={() =>
                firstCheckpointId != null &&
                void refreshWithFrozenProfile(firstCheckpointId)
              }
              className="mt-3 rounded border border-emerald-700 px-3 py-1.5 text-[10px] text-emerald-200 disabled:opacity-50"
            >
              {refreshingCheckpoint != null
                ? "Frozen条件で生成中…"
                : "最初のCheckpointをFrozen条件で再生成"}
            </button>
            <span className="ml-2 text-[10px] text-gray-500">
              ComfyUI: {comfyuiState === "ready" ? "接続済み" : "未接続（再生成停止）"}
            </span>
            <p className="mt-1 text-[10px] text-gray-600">
              旧Previewは履歴へ退避し、学習Runは再実行しません。
            </p>
          </div>
        )}
      </section>
      {loadError && (
        <div className="rounded-lg border border-red-900/60 bg-red-950/20 px-3 py-2 text-xs text-red-300">
          Compareデータ取得エラー: {loadError}
        </div>
      )}
      {jobLoadWarning && (
        <div className="rounded-lg border border-amber-900/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-300">
          Preview本体は表示しています。条件詳細の一部取得に失敗したため、未取得値は推測していません。
        </div>
      )}
      {comparisonCards.length === 0 ? (
        <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-8 text-center text-sm text-gray-600">
          比較可能なPreviewはまだありません。
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {comparisonCards.map(({ sample, slot, image }) => {
            const jobs = sample.preview_job_summary;
            const job = jobsByCheckpoint[sample.checkpoint_id]?.find(
              (candidate) =>
                slot === String(candidate.prompt_index) ||
                `p${candidate.prompt_index}_i${candidate.instance_index}` ===
                  slot,
            );
            const condition = job?.conditions;
            const frozen = frozenSnapshots[0]?.payload;
            const snapshot = job?.preview_snapshot;
            const actual = {
              model_family: condition?.model_family ?? job?.preview_model_family,
              resolution:
                condition?.resolution ??
                (snapshot?.width === snapshot?.height
                  ? snapshot?.width
                  : undefined),
              steps: condition?.steps ?? snapshot?.steps,
              cfg: condition?.cfg ?? snapshot?.cfg,
              sampler: condition?.sampler ?? snapshot?.sampler,
              scheduler: condition?.scheduler ?? snapshot?.scheduler,
              seed: snapshot?.seed ?? job?.seed,
              prompt: snapshot?.prompt,
              negative_prompt: snapshot?.negative_prompt,
            };
            const conditionChecks = frozen
              ? [
                  {
                    label: "model",
                    expected: frozen.model_family,
                    actual: actual.model_family,
                  },
                  {
                    label: "prompt",
                    expected: frozen.prompt,
                    actual: actual.prompt,
                  },
                  {
                    label: "negative",
                    expected: frozen.negative_prompt,
                    actual: actual.negative_prompt,
                  },
                  { label: "seed", expected: frozen.seed, actual: actual.seed },
                  {
                    label: "resolution",
                    expected: frozen.resolution,
                    actual: actual.resolution,
                  },
                  {
                    label: "steps",
                    expected: frozen.steps,
                    actual: actual.steps,
                  },
                  { label: "cfg", expected: frozen.cfg, actual: actual.cfg },
                  {
                    label: "sampler",
                    expected: frozen.sampler,
                    actual: actual.sampler,
                  },
                  {
                    label: "scheduler",
                    expected: frozen.scheduler,
                    actual: actual.scheduler,
                  },
                ]
              : [];
            const conditionComplete =
              conditionChecks.length > 0 &&
              conditionChecks.every((check) => check.actual !== undefined);
            const comparableChecks = conditionChecks.filter(
              (check) => check.expected !== undefined,
            );
            const conditionMatches =
              conditionComplete &&
              comparableChecks.length > 0 &&
              comparableChecks.every(
                (check) =>
                  check.expected === check.actual ||
                  (check.label === "sampler" &&
                    check.actual === "auto" &&
                    check.expected === "euler"),
              );
            const conditionDiffs = conditionChecks
              .filter(
                (check) =>
                  check.expected !== undefined &&
                  check.actual !== undefined &&
                  check.expected !== check.actual,
              )
              .map((check) => check.label);
            const key = `${sample.checkpoint_id}:${slot}` as EvaluationKey;
            const evaluation = evaluationFor(sample.checkpoint_id, slot);
            return (
              <article
                key={`${sample.checkpoint_id}-${sample.epoch}-${slot}`}
                className="overflow-hidden rounded-xl border border-gray-800 bg-gray-900/50"
              >
                <div className="aspect-square bg-gray-950">
                  {image && (
                    <img
                      src={image}
                      alt={`Epoch ${sample.epoch} ${slot}`}
                      className="h-full w-full object-contain"
                    />
                  )}
                </div>
                <div className="p-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-semibold text-gray-300">
                      Epoch {sample.epoch} · {slot}
                    </p>
                    <span className="text-[10px] text-gray-500">
                      {sample.validation_status ?? "unknown"}
                    </span>
                  </div>
                  <p className="text-[11px] text-gray-500">
                    Checkpoint #{sample.checkpoint_id} · step {sample.step}
                  </p>
                  {jobs && (
                    <p className="mt-1 text-[10px] text-gray-600">
                      Preview: {jobs.succeeded}/{jobs.expected} succeeded ·
                      条件はRun Snapshot
                    </p>
                  )}
                  {snapshot?.profile_snapshot_id && (
                    <p className="mt-1 text-[10px] text-indigo-300">
                      Preview Snapshot #{snapshot.profile_snapshot_id} ·{" "}
                      {snapshot.profile_snapshot_hash?.slice(0, 8)}
                    </p>
                  )}
                  {condition && (
                    <p
                      className={`mt-1 text-[10px] ${condition.source === "legacy_import" ? "text-amber-300" : conditionMatches ? "text-emerald-300" : "text-red-300"}`}
                    >
                      {condition.source === "legacy_import"
                        ? "条件証拠: 旧Preview取込（再現性は未証明）"
                        : conditionMatches
                          ? `条件一致: Prompt / Negative / Seed / ${condition.resolution}px · ${condition.steps} steps · cfg ${condition.cfg} · ${condition.sampler} / ${condition.scheduler}`
                          : conditionComplete
                            ? `条件不一致: ${conditionDiffs.join(", ") || "Frozen条件とPreview条件が不一致"}`
                            : "条件証拠不足（未取得値を推測していません）"}
                    </p>
                  )}
                  <div className="mt-3 border-t border-gray-800 pt-2">
                    <p className="text-[10px] font-semibold text-indigo-200">
                      Human Evaluation
                    </p>
                    <div className="mt-2 grid grid-cols-2 gap-1">
                      {RATING_FIELDS.map(([field, label]) => (
                        <label key={field} className="text-[9px] text-gray-500">
                          {label}
                          <select
                            value={evaluation[field] ?? ""}
                            onChange={(e) =>
                              updateEvaluation(
                                key,
                                field,
                                e.target.value ? Number(e.target.value) : "",
                              )
                            }
                            className="mt-0.5 w-full rounded border border-gray-800 bg-gray-950 px-1 py-1 text-[10px] text-gray-300"
                          >
                            <option value="">—</option>
                            {[1, 2, 3, 4, 5].map((score) => (
                              <option key={score} value={score}>
                                {score}
                              </option>
                            ))}
                          </select>
                        </label>
                      ))}
                    </div>
                    <label className="mt-2 flex items-center gap-2 text-[10px] text-gray-400">
                      <input
                        type="checkbox"
                        checked={Boolean(evaluation.accepted)}
                        onChange={(e) =>
                          updateEvaluation(
                            key,
                            "accepted",
                            e.target.checked ? 1 : 0,
                          )
                        }
                      />
                      採用候補として記録
                    </label>
                    <textarea
                      value={evaluation.note}
                      onChange={(e) =>
                        updateEvaluation(key, "note", e.target.value)
                      }
                      placeholder="評価メモ"
                      className="mt-2 h-12 w-full resize-none rounded border border-gray-800 bg-gray-950 px-2 py-1 text-[10px] text-gray-300"
                    />
                    <button
                      onClick={() => void saveEvaluation(evaluation)}
                      className="mt-2 w-full rounded bg-indigo-700 px-2 py-1.5 text-[10px] text-white hover:bg-indigo-600"
                    >
                      評価を保存
                    </button>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}
