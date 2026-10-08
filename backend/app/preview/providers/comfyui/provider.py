"""ComfyUI を使ったプレビュー生成 Provider。

musubi 系モデル（Flux/Qwen-Image/KREA2/Z-Image 等）は kohya_ss と異なり
サンプル生成機能を内蔵しないため、学習中に出力された LoRA チェックポイントを
ComfyUI へステージングし、汎用グラフでプレビュー画像を生成する。

KREA2 でこの生成が失敗する（重すぎる/非互換等）ことがあるが、それは
この Provider 単位の問題であり、学習 run の成否には影響しない
（呼び出し元は app.preview.registry.generate_preview_safe 経由で必ず
例外を握りつぶす）。
"""
from __future__ import annotations

from ...base import PreviewProvider, PreviewRequest, PreviewResult


class ComfyUIPreviewProvider(PreviewProvider):
    display_name = "ComfyUI"

    def is_ready(self, settings: dict[str, str]) -> bool:
        comfyui_root = (settings.get("comfyui_root") or "").strip()
        return bool(comfyui_root)

    def generate_preview(self, request: PreviewRequest) -> list[PreviewResult]:
        from pathlib import Path

        from ....services.comfyui_client import ComfyUIClient, ComfyUIError
        from ....trainers import get_spec

        def _skip(reason: str) -> list[PreviewResult]:
            # NOTE: 空リストをそのまま返すと、generate_preview_safe()のerror detail
            # 伝播機構(_failure_results)を経由せずにmonitor.py側のforループが
            # 一度も回らず last_error="" のまま記録される実バグを再発させる
            # (以前 provider未登録/未接続/例外時に空リストを返していたのと同種の
            # バグが、providerの内部スキップ経路でも起きていた。実コード監査で発見)。
            n = max(1, len(request.prompt_items))
            return [PreviewResult(prompt_index=i, seed=0, error=reason) for i in range(n)]

        spec = get_spec(request.model_family)
        if spec is None:
            return _skip(f"model_family '{request.model_family}' のModelSpecが見つかりません")

        comfyui_root = (request.settings.get("comfyui_root") or "").strip()
        if not comfyui_root:
            return _skip("comfyui_root が設定されていません(Integrations設定を確認してください)")

        if request.model_family in ('anima','sdxl'):
            # Reuse the bound native Comfy queue, shared LoRA inode and websocket output collector.
            import json,time,uuid
            from .... import desktop_comfy
            from ....db import get_conn
            from ....desktop_jobs import write_state
            url=request.extra.get('comfy_url')  # set only for a run switched to the RTX 3060 preview instance
            if url:
                from ....training.runtime import preview_gpu
                preview_gpu.touch()  # any use counts: the idle stop must never cut a preview in progress
            catalog=desktop_comfy.inspect({'include_schema':True,**({'url':url} if url else {})})
            gpu='RTX 3060' if url else 'RTX 3090 Ti'
            if not any(gpu in d.get('name','') for d in catalog['devices']):return _skip(f'プレビューの接続先が{gpu}ではありません')
            conn=get_conn()
            try:
                row=conn.execute('SELECT config_json FROM training_runs WHERE id=?',(request.run_id,)).fetchone()
                run_config=json.loads(row['config_json']);base=Path(request.extra.get('preview_base_checkpoint_path') or run_config.get('preview_base_checkpoint_path') or run_config['base_checkpoint_path'])
            finally:conn.close()
            model=next((m for m in catalog['models'] if m.get('path') and Path(m['path']).is_file() and Path(m['path']).samefile(base)),None)
            if not model:return _skip('接続先に指定した生成用ベースモデルが見つかりません')
            lora=desktop_comfy._share_checkpoint(catalog['root'],request.run_id,Path(request.checkpoint_path))
            # This entry was just created and samefile-verified by the shared-link operation.
            if lora not in catalog['loras']:catalog['loras'].append(lora)
            results=[]
            for i,item in enumerate(request.prompt_items):
                from ....services.preview_prompts import compose_positive,DEFAULT_NEGATIVE
                seed=int(item.get('seed',42));resolution=request.resolution
                payload={'family':request.model_family,'model':model['id'],'lora':lora,'positive':compose_positive(item),'negative':item.get('negative') or DEFAULT_NEGATIVE,'width':int(request.extra.get('width') or resolution),'height':int(request.extra.get('height') or resolution),'steps':request.steps or spec.preview_steps,'cfg':request.extra.get('cfg',request.cfg),'seed':seed,'sampler':request.sampler or spec.preview_sampler,'scheduler':request.extra.get('scheduler','simple'),'strength':float(request.extra.get('lora_strength',1)),'extra_loras':request.extra.get('extra_loras',[])}
                graph,outputs=desktop_comfy._graph(payload,catalog)
                folder=Path(request.checkpoint_path).parent/'sample'/('native-'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=True)
                graph_path=folder/'graph.json';graph_path.write_text(json.dumps(graph),encoding='utf8')
                job={'id':folder.name,'run_id':request.run_id,'created':time.time(),'url':catalog['url'],'graph_path':str(graph_path),'output_nodes':outputs,'expected_artifact':str(folder/'preview.png'),'status':'starting','collection_required':True,'collection_state':'PENDING'}
                def persist():write_state(folder/'job.json',json.dumps(job))
                desktop_comfy.collect_graph(job,persist)
                if job['status']=='done':results.append(PreviewResult(prompt_index=i,seed=seed,image_bytes=Path(job['image_path']).read_bytes()))
                else:results.append(PreviewResult(prompt_index=i,seed=seed,error=job.get('message','生成失敗')))
            return results

        client = ComfyUIClient(base_url=request.settings.get("comfyui_url") or None)
        # preview_checkpoint_patterns未設定なら従来通りcheckpoint_patterns(Training用)を
        # 引き継ぐ(後方互換)。設定されていれば、Training Base(例: Krea2 RAW)とは独立に
        # Preview専用のBase Model(例: Krea2 Turbo)を検索する。
        preview_patterns = spec.preview_checkpoint_patterns or spec.checkpoint_patterns
        base_models = client.resolve_preview_base_model(request.model_family, preview_patterns)
        if not base_models:
            return _skip(
                f"ComfyUI側にPreview用ベースモデル({', '.join(preview_patterns)})が見つかりません"
            )

        lora_name = client.stage_lora_for_comfyui(comfyui_root, request.run_id, Path(request.checkpoint_path))
        if not lora_name:
            return [
                PreviewResult(prompt_index=i, seed=42, error="LoRA ステージング失敗")
                for i in range(len(request.prompt_items))
            ]

        sampler = request.sampler or spec.preview_sampler
        cfg = request.cfg or spec.preview_cfg
        steps = request.steps or spec.preview_steps
        # preview_max_resolution は「学習解像度」ではなく「Preview生成がこの環境の
        # VRAM/時間内で現実的に完了する上限」を表す（実測に基づくArchitecture固有の
        # 上限。model_familyによる分岐ではなくModelSpecのフィールドとして表現する）。
        resolution = (
            min(request.resolution, spec.preview_max_resolution)
            if spec.preview_max_resolution
            else request.resolution
        )

        results: list[PreviewResult] = []
        for idx, item in enumerate(request.prompt_items):
            from ....services.preview_prompts import DEFAULT_NEGATIVE, compose_positive

            positive = compose_positive(item)
            negative = (str(item.get("negative") or "").strip()) or DEFAULT_NEGATIVE
            # prompt_itemsが明示的にseedを指定していればそれを使う(Preview Job単位で
            # 一意なseedを割り当てるため)。指定なしは既存動作(42固定)を維持。
            try:
                seed = int(item.get("seed", 42))
            except (TypeError, ValueError):
                seed = 42
            try:
                img_bytes = client.generate_musubi_preview(
                    model_family=request.model_family,
                    base_models=base_models,
                    lora_name=lora_name,
                    positive=positive,
                    negative=negative,
                    width=resolution,
                    height=resolution,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler,
                    seed=seed,
                    unet_weight_dtype=spec.preview_unet_weight_dtype,
                )
                results.append(PreviewResult(prompt_index=idx, seed=seed, image_bytes=img_bytes))
            except ComfyUIError as exc:
                results.append(PreviewResult(prompt_index=idx, seed=seed, error=str(exc)))
        return results
