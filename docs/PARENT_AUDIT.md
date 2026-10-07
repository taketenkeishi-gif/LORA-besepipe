# 親タスク：LoRA学習パイプライン監査

状態：**未完了・継続対象**。UI刷新、容量整理、デスクトップ化はこの親タスクの一部であり、全工程の完了ではない。

## 目的と現在の制約

既存ComfyUIのモデルを直接参照し、データセット編集から学習・成果物確認まで、実ファイルが追える安定した運用を検証する。現在の画面構造とプロジェクトタブを保つ。ANIMAが中心で、Illustrious派生とKREA2 RedMixの対応範囲も区別する。
実行GPUの既定はGPU1。3060を使う明示指示で行った短時間テストは完了済みだが、通常UI開始ボタンのGPU0対応や本番品質まで証明していない。

## 2026-09-20 追加復旧（全工程完了ではない）
右クリック、一括自動タグ→TXT、ComfyUI本体/実インスタンスのバインドとモデル一覧、保存ワークフローの手動プレビューを復旧・独立UI検証。接続先は稼働8188へ修正し、Anima、学習済みLoRA、ワークフロー、WAI Illustriousで実画像受信。学習LoRAは共有リンクで重量データを複製しない。詳細と残る制限はRESTORED_DATASET_COMFY_ROUTES.md。学習開始→各Epochの自動Preview→評価→Exportの一貫試験は未達で、手動プレビュー成功へ置き換えない。

## 工程別の達成範囲

| 工程 | 確認できたこと | 未完・限界 |
|---|---|---|
| 起動・実体 | デスクトップ実行、モデル共有、旧環境除去、容量回収 | 今回の新backendコードを稼働5175へ反映した証拠なし |
| データセット編集 | 実UIの追加、タグ編集、外部TXT反映、並べ替え、タブ分離 | 全OS/全入力装置の網羅試験ではない |
| 入力の確定 | 今回の新コードで登録→Snapshot→Caption編集保持を実API確認 | 本番ロードは未確認 |
| 学習開始検証 | 破損Snapshotを開始・待機とも拒否、モデル依存確認 | UI開始から完走までの一貫試験は未完 |
| 実学習 | 3060でANIMA2ステップ、実LoRA出力・有限値/非ゼロ更新を確認 | エンジン単体。長時間・品質・全モデル系統の試験ではない |
| 再開 | 現在Run優先と同一モデル/確定データ/Rank等の選択を実API/SQL確認 | GPUで中断再開の通し試験は未実施。optimizer/epochの完全復元ではない |
| 候補管理 | 実学習出力の候補登録、二重登録抑止、Run/Checkpoint/SHA整合確認 | 本番候補の採用作業は行っていない |
| Preview/Final/Export | 未採用Exportと人間確認なしFinalを拒否する実API確認 | 実Preview生成→比較→ユーザー評価→Exportの一貫検証は未完 |
| 旧API | 空画像成功、再import上書き、誤重複分類を修正・隔離検証 | 現UIには未接続。Mixerは保存だけで学習への適用なし |

## 今回修正・隔離検証した問題

1. Snapshot参照中Assetの登録削除：HTTP409で保全。参照のない登録削除は実ファイルを保持。
2. 孤児Snapshot entryのINNER JOINによる黙った欠落：LEFT JOINで残して欠落を検出。確定件数との不一致も開始前に拒否。
3. 確定画像のrename/move/exclude：原本保全。Caption編集と同じ画像の新Snapshot登録は許可し、不要な複製を要求しない。
4. 同名の別学習からのResume：現在Run優先。過去Runはモデル、エンジン、確定データ、Rank/Alpha等が一致する場合だけ選択。実行/待機中の二重Resumeも拒否。
5. AnimagineをAnimaと誤判定：バックエンド側も修正。
6. 空入力→架空候補→1x1画像の成功扱い：実候補なしは0件、実ファイルのない候補は拒否。
7. 旧importの上書き・DB重複：排他的ファイル作成で既存を保持し、実画像寸法を記録。
8. 取り込み途中の障害：新規所有ファイルとDB更新をrollback。独立レビューで再現した2件目複製失敗を、修正後に故障注入して回収・再試行成功を確認。
9. 類似連結成分全体をduplicate判定：削除候補のduplicateは同一バイト列の画像に限定。類似画像はsimilarに残す。
10. Dataset編集APIのDB接続がcontext終了後も残る：明示closeを保証するcontext managerへ変更。
11. SDXLエンジン未接続でもジョブを作っていた経路：他系統同様に作成前にHTTP400。
12. `/instance`がディスク上の新コードを読んで、旧稼働コードまで更新済みに見せる問題：起動時の固定backend_buildへ変更。

## 本番反映の扱い

今回の新コードは別プロセス5184・隔離DB/画像で実UIとAPIを検証した。稼働中5175は新しい固定backend_buildをまだ返していない。従って、本番反映済みと報告しない。
以前拒否された本番再起動の操作を、別表現や別APIで通す対応は行っていない。ソース変更・隔離実行・本番ロードを別の状態として扱う。

## 次に再開する位置

1. 正規の許可されたライフサイクルで本番backendを更新し、固定backend_buildと修正ファイル群を照合する。未確認のまま本番で修正済みと扱わない。
2. 実UI開始→キュー→実学習→Checkpoint→Preview→評価→Exportを、一つのRunと成果物IDで追う。エンジン単体テストを代理にしない。
3. モデル系統別の実運用差、再開と失敗復帰、Preview条件一致を確認する。人間の審美承認はテストで捏造しない。
4. 現UI未接続の旧Collector/Mixer/Recipe/CLIP仕様は、現行機能として宣伝しない。今回の利用範囲に必要かを区別して扱う。

証拠：このタスクの `work/parent-audit/` 内の integrity-repairs.json、snapshot-workflow.json、import-repairs.json、import-rollback.json、output-pipeline.json、engine-unavailable.json、ui-guard.json、deployment-check.json。テスト中に本番設定DB・原画像を変更せず、新しいGPU学習も開始していない。

## 学習中UI照合の追加
TRAINING_TRANSPARENCY_AUDIT.md参照。旧版のEpoch/Step/ETA/loss・保存間隔・epoch別プレビュー・区切り停止・再開・キャッシュ設定が現行Dockへ未移植。手動プレビュー成功を同等性の証拠にしない。

## 2026-09-20 学習中の透明性復旧と本番反映
TRAINING_TRANSPARENCY_RESTORED.md参照。未移植だった進捗、保存間隔/キャッシュ、現在RunのCheckpoint/Preview表示、区切り停止/重み継続入口を復旧。区切り停止の実monitor未接続も修正し隔離所有CPUプロセスで保存後停止を実測。本番backend11ファイル起動hash一致とfrontend読込を確認したため、上記本番未反映という過去状態は解消。実GPU中断再開と学習開始→自動Preview→評価→Exportの通し試験はこの証拠の対象外。

## 2026-09-20 実学習→各Epochプレビューの実運用証拠
GPU_TRIGGER_RUNTIME_REPAIR.md参照。実Wasabi Run4は75画像・4epochs・300steps完走。4つの保存LoRAと各epochのnative ComfyUI previewが成功し実画面で確認。これにより「エンジン単体しか学習未確認」「学習開始からepoch preview未接続」という過去状態は今回のAnima/この設定に限り更新。人間による評価/採用/Export承認、GPU中断再開、全family、旧前処理39項目の全runtime検証は別範囲で未証明。
