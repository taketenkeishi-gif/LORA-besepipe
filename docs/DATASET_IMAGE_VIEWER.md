# データセット画像の拡大表示

- 一覧の画像をダブルクリック、または選択してEnter。
- 詳細パネルの画像をクリック、または右クリックメニューの「画像を拡大表示」。
- アプリの表示領域全体で原画像を表示。画面に合わせる／原寸100%／拡大・縮小。
- 左右キーで前後の画像、Escまたは右上の×で閉じる。

実ブラウザー画面で2048×1536の原画像を読み込み、画面内1224×918、原寸幅2048、125%幅2560を実測。一覧のスクロール位置、選択、未保存タグの保持を確認。幅390pxでも操作欄が横にはみ出さず、画像消失時はエラーを表示して閉じられました。

従来の詳細画像リンクは別ウィンドウを開く方式でしたが、デスクトップ版は新規ウィンドウを禁止していました。今回の表示はアプリ内のRadix Dialogで、別ウィンドウを使用しません。

ビルド済み・サーバー配信済み。起動済みネイティブアプリが新しい画面を読み込んだことと、独立した初見評価者による操作は未検証です。前ターンの検査用ネイティブ起動拒否に対し、同じ起動の迂回は行っていません。

Source: frontend/src/features/studio/DatasetImageViewer.tsx, DatasetFiles.tsx, dataset-files.css. Built asset index-1SCAElKD.js. Evidence: thread work/image-viewer-trial/result.json; original fit test invalidated in before-fit-failure.json and corrected to assert actual viewport fit. Native frontend loading remains unverified. Test project7 record removed, fixture files preserved. Parent full comparison audit remains active.
