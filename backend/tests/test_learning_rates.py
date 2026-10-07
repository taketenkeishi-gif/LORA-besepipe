"""Regression: a visible main LR must never be silently shadowed before launch."""
import sys,tomllib,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.training.learning_rates import resolve_learning_rates
from app.training.advanced import validate,apply_to_toml
from app.training.backends.anima.backend import AnimaBackend
from app.training.backends.sdxl.backend import SDXLBackend

class LearningRateContract(unittest.TestCase):
    def test_visible_value_reaches_native_config(self):
        for family in ('anima','sdxl'):
            for lr in (.0001,.0002):
                with self.subTest(family=family,lr=lr):
                    cfg={'learning_rate':lr,'advanced':{},'optimizer':'AdamW8bit'}
                    native=tomllib.loads('\n'.join(apply_to_toml([f'learning_rate = {lr}'],cfg,family)))
                    self.assertEqual(native['learning_rate'],lr);self.assertEqual(native['unet_lr'],lr)
    def test_legacy_conflict_rejected(self):
        for family in ('anima','sdxl'):
            with self.assertRaises(ValueError):validate({'unet_lr':.00002},family,{'learning_rate':.0002})
        with self.assertRaises(ValueError):resolve_learning_rates({'learning_rate':.0002,'unet_lr':.00002},reject_conflict=True)
    def test_prepare_cannot_bypass_api_validation(self):
        ctx=SimpleNamespace(project_id=0,run_id=0,settings={},cfg={'learning_rate':.0002,'advanced':{'unet_lr':.00002}})
        for backend in (AnimaBackend(),SDXLBackend()):
            with self.assertRaises(ValueError):backend.prepare(ctx)
    def test_optimizer_cannot_silently_replace_lr(self):
        for optimizer in ('Prodigy','DAdaptAdam'):
            with self.assertRaises(ValueError):resolve_learning_rates({'optimizer':optimizer,'learning_rate':.0002},reject_conflict=True)
            self.assertEqual(resolve_learning_rates({'optimizer':optimizer,'learning_rate':1},reject_conflict=True)['body'],1)
    def test_historical_requested_and_actual_remain_distinct(self):
        v=resolve_learning_rates({'learning_rate':.0002,'unet_lr':.00002})
        self.assertEqual(v['requested'],.0002);self.assertEqual(v['body'],.00002);self.assertTrue(v['conflict'])
    def test_text_encoder_rate_remains_independent(self):
        cfg={'learning_rate':.0002,'advanced':{'text_encoder_lr':.00001}}
        self.assertEqual(resolve_learning_rates(cfg,reject_conflict=True)['body'],.0002)

if __name__=='__main__':unittest.main()
