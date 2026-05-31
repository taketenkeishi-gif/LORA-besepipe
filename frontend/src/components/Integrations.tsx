import { CheckCircle2, XCircle, HelpCircle, Wand2, Save, RefreshCw } from "lucide-react";
import { apiPost, apiGet } from "../lib/api";
import type { ToolPaths, IntegrationStatus } from "../types";

type Props = {
  toolPaths: ToolPaths;
  onToolPathsChange: (paths: ToolPaths) => void;
  integrationStatus: IntegrationStatus | null;
  showError: (msg: string) => void;
  showNotice: (msg: string) => void;
  onReload: () => Promise<void>;
};

const PATH_FIELDS: { key: keyof ToolPaths; label: string; placeholder: string }[] = [
  { key: "python_exe", label: "Python 実行ファイル", placeholder: "C:\\Python310\\python.exe" },
  { key: "kohya_root", label: "kohya_ss ルート", placeholder: "C:\\kohya_ss" },
  { key: "comfyui_root", label: "ComfyUI ルート", placeholder: "C:\\ComfyUI" },
  { key: "wd14_script", label: "WD14 スクリプト", placeholder: "C:\\tagger\\wd14.py" },
  { key: "temp_dir", label: "一時ファイルディレクトリ", placeholder: "C:\\tmp\\lora_workbench" },
  {
    key: "dataset_base_dir",
    label: "データセット補完ベースディレクトリ",
    placeholder: "C:\\SDXL\\LoRA_Training\\dataset",
  },
];

function StatusIcon({ ok, reason }: { ok?: boolean; reason?: string }) {
  if (ok === undefined)
    return (
      <span title="未確認">
        <HelpCircle size={15} className="text-gray-500" />
      </span>
    );
  return ok ? (
    <span title={reason}>
      <CheckCircle2 size={15} className="text-emerald-400" />
    </span>
  ) : (
    <span title={reason}>
      <XCircle size={15} className="text-red-400" />
    </span>
  );
}

export default function Integrations({
  toolPaths,
  onToolPathsChange,
  integrationStatus,
  showError,
  showNotice,
  onReload,
}: Props) {
  function set(key: keyof ToolPaths, value: string) {
    onToolPathsChange({ ...toolPaths, [key]: value });
  }

  async function handleSave() {
    try {
      await apiPost("/settings/tool-paths", toolPaths, "PUT");
      showNotice("連携設定を保存しました");
      await onReload();
    } catch (e) {
      showError(`保存失敗: ${String(e)}`);
    }
  }

  async function handleAutoDetect() {
    try {
      const d = await apiPost<ToolPaths>("/settings/tool-paths/autodetect", {});
      onToolPathsChange(d);
      showNotice("自動検出を実行しました");
      await onReload();
    } catch (e) {
      showError(`自動検出失敗: ${String(e)}`);
    }
  }

  const checks = integrationStatus?.checks ?? {};
  const okCount = Object.values(checks).filter((c) => c.ok).length;
  const totalCount = Object.keys(checks).length || PATH_FIELDS.length;

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h2 className="text-xl font-bold text-gray-100 mb-1">外部連携設定</h2>
        <p className="text-sm text-gray-400">
          Python, kohya_ss, ComfyUI, WD14 のパスを設定します
        </p>
      </div>

      {/* Status summary */}
      <div
        className={[
          "flex items-center gap-3 px-4 py-3 rounded-xl border text-sm",
          okCount === totalCount && totalCount > 0
            ? "bg-emerald-950 border-emerald-800 text-emerald-300"
            : okCount > 0
              ? "bg-amber-950 border-amber-800 text-amber-300"
              : "bg-gray-800 border-gray-700 text-gray-400",
        ].join(" ")}
      >
        {okCount === totalCount && totalCount > 0 ? (
          <CheckCircle2 size={18} />
        ) : (
          <HelpCircle size={18} />
        )}
        <span>
          {totalCount > 0
            ? `${okCount}/${totalCount} ツール接続OK`
            : "未確認 — 自動検出または手動でパスを設定してください"}
        </span>
      </div>

      {/* Path Form */}
      <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-700">
          <h3 className="text-sm font-semibold text-gray-200">パス設定</h3>
        </div>
        <div className="p-5 space-y-4">
          {PATH_FIELDS.map(({ key, label, placeholder }) => {
            const check = checks[key];
            return (
              <div key={key}>
                <label className="flex items-center justify-between mb-1">
                  <span className="text-xs text-gray-400">{label}</span>
                  <div className="flex items-center gap-1.5">
                    <StatusIcon ok={check?.ok} reason={check?.reason} />
                    {check?.reason && (
                      <span
                        className={`text-xs ${check.ok ? "text-emerald-500" : "text-red-500"}`}
                      >
                        {check.reason}
                      </span>
                    )}
                  </div>
                </label>
                <input
                  value={toolPaths[key]}
                  onChange={(e) => set(key, e.target.value)}
                  placeholder={placeholder}
                  className="w-full bg-gray-700 border border-gray-600 text-gray-100 placeholder-gray-500 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30"
                />
              </div>
            );
          })}
        </div>

        {/* Actions */}
        <div className="px-5 pb-5 flex gap-3">
          <button
            onClick={() => void handleAutoDetect()}
            className="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg px-4 py-2 text-sm font-medium transition-colors"
          >
            <Wand2 size={15} />
            自動検出
          </button>
          <button
            onClick={() => void handleSave()}
            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
          >
            <Save size={15} />
            保存
          </button>
          <button
            onClick={() => void onReload()}
            className="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg px-3 py-2 text-sm transition-colors"
            title="再確認"
          >
            <RefreshCw size={15} />
          </button>
        </div>
      </div>

      {/* Detailed status */}
      {integrationStatus && (
        <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-700">
            <h3 className="text-sm font-semibold text-gray-200">
              接続状態詳細
            </h3>
          </div>
          <div className="divide-y divide-gray-700">
            {PATH_FIELDS.map(({ key, label }) => {
              const c = checks[key];
              return (
                <div key={key} className="px-5 py-3 flex items-center gap-3">
                  <StatusIcon ok={c?.ok} reason={c?.reason} />
                  <span className="text-sm text-gray-300 flex-1">{label}</span>
                  <span
                    className={`text-xs ${c?.ok ? "text-emerald-400" : "text-gray-500"}`}
                  >
                    {c?.reason ?? "未確認"}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
