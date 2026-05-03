import unittest

import cv2
import numpy as np

from upscaler_worker.caption_removal import CaptionMaskSettings, detect_hard_caption_mask, smooth_caption_masks


class CaptionRemovalTests(unittest.TestCase):
    def test_detect_hard_caption_mask_finds_bottom_caption_text(self) -> None:
        frame = np.full((180, 320, 3), 48, dtype=np.uint8)
        cv2.putText(frame, "HELLO WORLD", (54, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        settings = CaptionMaskSettings(bottom_region_fraction=0.45, light_threshold=180, mask_dilate_pixels=4)

        mask = detect_hard_caption_mask(frame, settings)

        self.assertGreater(int(np.count_nonzero(mask[100:, :])), 0)
        self.assertEqual(int(np.count_nonzero(mask[:70, :])), 0)

    def test_detect_hard_caption_mask_ignores_upper_text_outside_region(self) -> None:
        frame = np.full((180, 320, 3), 48, dtype=np.uint8)
        cv2.putText(frame, "TITLE", (88, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        mask = detect_hard_caption_mask(frame, CaptionMaskSettings(bottom_region_fraction=0.35))

        self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_detect_hard_caption_mask_finds_colored_midframe_caption_without_red_object(self) -> None:
        frame = np.full((180, 320, 3), 96, dtype=np.uint8)
        cv2.rectangle(frame, (12, 118), (78, 170), (40, 40, 220), -1)
        cv2.putText(frame, "and I get home", (72, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, "and I get home", (72, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 60), 2, cv2.LINE_AA)

        mask = detect_hard_caption_mask(frame, CaptionMaskSettings(bottom_region_fraction=0.75, mask_dilate_pixels=4))

        self.assertGreater(int(np.count_nonzero(mask[45:90, 65:260])), 0)
        self.assertLess(int(np.count_nonzero(mask[118:170, 12:78])), 24)

    def test_line_box_padding_covers_spaces_between_caption_words(self) -> None:
        frame = np.full((180, 320, 3), 48, dtype=np.uint8)
        cv2.putText(frame, "HI HI", (92, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        mask = detect_hard_caption_mask(
            frame,
            CaptionMaskSettings(bottom_region_fraction=0.75, light_threshold=180, mask_dilate_pixels=4, line_box_padding_pixels=8),
        )

        self.assertGreater(int(np.count_nonzero(mask[62:102, 128:160])), 0)

    def test_line_box_padding_does_not_bridge_distant_same_row_text(self) -> None:
        frame = np.full((180, 480, 3), 48, dtype=np.uint8)
        cv2.putText(frame, "NO", (24, 94), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "TITLE HERE", (270, 94), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        mask = detect_hard_caption_mask(
            frame,
            CaptionMaskSettings(bottom_region_fraction=0.75, light_threshold=180, mask_dilate_pixels=4, line_box_padding_pixels=8),
        )

        self.assertEqual(int(np.count_nonzero(mask[64:104, 120:240])), 0)
        self.assertGreater(int(np.count_nonzero(mask[64:104, 300:430])), 0)

    def test_smooth_caption_masks_expands_neighbors_within_radius(self) -> None:
        masks = [np.zeros((16, 16), dtype=np.uint8) for _ in range(3)]
        masks[1][8, 8] = 255

        smoothed = smooth_caption_masks(masks, temporal_radius=1)

        self.assertEqual(int(smoothed[0][8, 8]), 255)
        self.assertEqual(int(smoothed[1][8, 8]), 255)
        self.assertEqual(int(smoothed[2][8, 8]), 255)


if __name__ == "__main__":
    unittest.main()