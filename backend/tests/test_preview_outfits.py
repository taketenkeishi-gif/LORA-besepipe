import unittest

from app.training.runtime.preview_jobs import outfit_signature_tags, prompt_for_outfit

TRIG = ["ex_a", "ex_b"]
CAPS = (["EXCEEDS, ex_a, 1girl, solo, red dress, blue eyes"] * 4 + ["EXCEEDS, ex_b, 1girl, solo, green coat, blue eyes"] * 4)


class OutfitPreviewTest(unittest.TestCase):
    def test_signature_keeps_only_distinguishing_tags(self):
        sig = outfit_signature_tags(CAPS, TRIG)
        self.assertEqual(sig["ex_a"], ["red dress"])
        self.assertEqual(sig["ex_b"], ["green coat"])  # blue eyes is shared, 1girl/solo are generic

    def test_prompt_swaps_only_outfit_parts(self):
        sig = outfit_signature_tags(CAPS, TRIG)
        base = "EXCEEDS, ex_a, red dress, 1girl, standing"
        out = prompt_for_outfit(base, TRIG, "ex_b", sig)
        self.assertEqual(out, "EXCEEDS, ex_b, green coat, 1girl, standing")

    def test_user_tags_override(self):
        out = prompt_for_outfit("EXCEEDS, 1girl", TRIG, "ex_a", {"ex_a": ["school uniform"]})
        self.assertEqual(out, "EXCEEDS, ex_a, school uniform, 1girl")


if __name__ == "__main__":
    unittest.main()
