"""Validated native trainer settings; no arbitrary command or path overrides."""
from __future__ import annotations

import json
import math
from pathlib import Path

CATALOG = json.loads(Path(__file__).with_name('advanced_catalog.json').read_text(encoding='utf-8'))
FIELDS = {field['key']: field for field in CATALOG}


def catalog(family: str, config: dict | None = None) -> list[dict]:
    config = config or {}
    result = []
    for field in CATALOG:
        if family not in field['families']:
            continue
        default = field.get('family_defaults', {}).get(family, field['default'])
        if field['key'] in config:
            default = config[field['key']]
        result.append({**field, 'default': default})
    return result


def validate(values: dict, family: str, config: dict | None = None) -> dict:
    if not isinstance(values, dict):
        raise ValueError('詳細設定は項目ごとのオブジェクトで指定してください')
    clean = {}
    for key, value in values.items():
        field = FIELDS.get(key)
        if not field or family not in field['families']:
            raise ValueError(f'{family}では未対応の詳細設定です: {key}')
        kind = field['type']
        valid = True
        if kind == 'bool':
            valid = isinstance(value, bool)
        elif kind in ('int', 'float'):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            valid = valid and (kind != 'int' or value == int(value))
            valid = valid and field['min'] <= value <= field['max']
        elif kind == 'choice':
            valid = value in field['choices']
        elif kind == 'lines':
            valid = isinstance(value, list) and len(value) <= 32 and all(isinstance(v, str) and '=' in v and '\n' not in v and len(v) <= 512 for v in value)
        if not valid:
            raise ValueError(f'{field["label"]}: {field["min"]}〜{field["max"]}の{"整数" if kind == "int" else "数値"}を指定してください' if kind in ('int', 'float') else f'{field["label"]}: 値の形式を確認してください')
        clean[key] = int(value) if kind == 'int' else value
    if config and 'learning_rate' in config:
        from .learning_rates import resolve_learning_rates
        resolve_learning_rates({**config, 'advanced':clean}, reject_conflict=True)
    effective = {f['key']: f['default'] for f in catalog(family, config)}
    effective.update(config or {})
    effective.update(clean)
    effective.setdefault("scheduler", "constant" if family == "anima" else "cosine_with_restarts")
    if effective.get('scheduler') in ('constant', 'adafactor') and effective.get('lr_warmup_steps', 0):
        raise ValueError('このスケジューラはウォームアップを使えません。0にするかconstant_with_warmupなどに変更してください')
    if effective.get('cache_latents', True) and (effective.get('color_aug') or effective.get('random_crop')):
        raise ValueError('色変化・ランダム切り抜きを使う場合は、画像の潜在表現キャッシュを無効にしてください')
    if effective.get('cache_text_encoder_outputs'):
        if not effective.get('network_train_unet_only', True):
            raise ValueError('Text Encoderを学習する場合は、Text Encoder出力キャッシュを無効にしてください')
        if effective.get('shuffle_caption') or effective.get('caption_tag_dropout_rate') or effective.get('caption_dropout_every_n_epochs') or (family == 'sdxl' and effective.get('caption_dropout_rate')):
            raise ValueError('Text Encoderキャッシュとタグ変化の設定が併用できません。キャッシュかタグ変化を無効にしてください')
    if effective.get('cache_text_encoder_outputs_to_disk') and not effective.get('cache_text_encoder_outputs'):
        raise ValueError('Text Encoderのディスク保存には出力キャッシュも有効にしてください')
    if effective.get('persistent_data_loader_workers') and effective.get('max_data_loader_n_workers') == 0:
        raise ValueError('worker数0ではworker維持をオフにしてください')
    if effective.get('min_bucket_reso', 256) > effective.get('max_bucket_reso', 2048):
        raise ValueError('バケットの最小辺は最大辺以下にしてください')
    if effective.get('bucket_reso_steps', 64) % (16 if family == 'anima' else 32):
        raise ValueError('バケットの刻みはAnimaで16、SDXLで32の倍数にしてください')
    if family == 'anima':
        if effective.get('attn_mode') == 'xformers' and not effective.get('split_attn'):
            raise ValueError('AnimaのxFormersにはSplit attentionが必要です')
        if effective.get('loss_type') in ('huber', 'smooth_l1') and effective.get('huber_schedule') == 'snr':
            raise ValueError('AnimaのHuberスケジュールはconstantまたはexponentialを選んでください')
    return clean


def apply_to_toml(lines: list[str], config: dict, family: str) -> list[str]:
    if 'advanced' not in config:
        return lines  # Legacy Run resume retains the previously recorded configuration.
    defaults = {f['key']: f['default'] for f in catalog(family, config)}
    defaults['unet_lr'] = config.get('learning_rate', 0.0001)
    defaults['text_encoder_lr'] = config.get('learning_rate', 0.0001)
    overrides = validate({**defaults, **config.get('advanced', {})}, family, config)
    # Remove native defaults before writing explicit values: TOML forbids duplicate keys.
    result = [line for line in lines if line.split('=', 1)[0].strip() not in overrides]
    for key, value in overrides.items():
        result.append(f'{key} = {json.dumps(value, ensure_ascii=False)}')
    return result
