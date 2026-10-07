"""One resolution rule for requested and native body learning rates."""
import math

def resolve_learning_rates(config, *, reject_conflict=False):
    main=config.get('learning_rate')
    advanced=config.get('advanced') or {}
    overrides=[v for v in (config.get('unet_lr'),advanced.get('unet_lr')) if v is not None]
    body=overrides[-1] if overrides else main
    conflict=main is not None and any(float(v)!=float(main) for v in overrides)
    if reject_conflict and conflict:
        raise ValueError('学習率と旧設定の本体学習率が不一致です。画面を再読み込みして学習率を指定し直してください（学習は開始していません）')
    if reject_conflict and (main is None or isinstance(main,bool) or not math.isfinite(float(main)) or not 0<float(main)<=1):
        raise ValueError('学習率は0より大きく1以下の有限数を指定してください')
    optimizer=str(config.get('optimizer',config.get('optimizer_type',''))).lower()
    if reject_conflict and optimizer.startswith(('prodigy','dadapt')) and float(main)!=1.0:
        raise ValueError('この自動調整Optimizerの学習率は1を指定してください。入力値を黙って変更せず学習開始を中止しました')
    return {'requested':main,'body':body,'conflict':conflict,'source':'unet_lr' if overrides else 'learning_rate'}
