"""ONNX çıkarım kodunun ön ve son işleme adımlarını sahte model çıktılarıyla test eder.

Gerçek model gerekmez:  python -m pytest tests/
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "edge"))
from yolo_onnx import letterbox, load_class_names, postprocess  # noqa: E402

NC = 19
ROOT = Path(__file__).resolve().parent.parent


def fake_output(candidates: list[tuple[float, float, float, float, int, float]]) -> np.ndarray:
    """(cx, cy, w, h, sınıf, skor) listesinden YOLOv8 biçiminde (1, 4+NC, N) çıktı üretir."""
    out = np.zeros((1, 4 + NC, len(candidates)), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(candidates):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cls, i] = score
    return out


def test_letterbox_keeps_aspect_ratio_and_centers():
    image = np.zeros((480, 640, 3), dtype=np.uint8)   # 4:3 görüntü
    boxed, scale, (pad_x, pad_y) = letterbox(image, 640)
    assert boxed.shape == (640, 640, 3)
    assert scale == 1.0
    assert (pad_x, pad_y) == (0, 80)
    assert boxed[0, 0, 0] == 114                       # üst dolgu gri


def test_postprocess_maps_box_back_to_original_image():
    # 1280x720 görüntü 640'a sığdırılınca ölçek 0.5, dikey dolgu 140 px olur.
    scale, pad = 0.5, (0, 140)
    # Orijinalde (200,100)-(400,300) kutusu, modelin koordinatında merkez (150,240), boyut 100x100.
    out = fake_output([(150, 240, 100, 100, 3, 0.9)])
    dets = postprocess(out, conf=0.5, iou=0.45, scale=scale, pad=pad, orig_shape=(720, 1280))
    assert len(dets) == 1
    assert dets[0].class_id == 3
    assert dets[0].box == (200, 100, 400, 300)


def test_low_confidence_is_filtered():
    out = fake_output([(100, 100, 50, 50, 0, 0.6)])
    assert postprocess(out, 0.70, 0.45, 1.0, (0, 0), (640, 640)) == []


def test_nms_removes_duplicates_of_same_class_only():
    out = fake_output([
        (100, 100, 50, 50, 5, 0.95),   # aynı levha
        (102, 101, 50, 50, 5, 0.80),   # neredeyse aynı kutu, aynı sınıf: elenmeli
        (101, 100, 50, 50, 9, 0.85),   # aynı yerde farklı sınıf: kalmalı
    ])
    dets = postprocess(out, 0.5, 0.45, 1.0, (0, 0), (640, 640))
    assert sorted(d.class_id for d in dets) == [5, 9]
    assert max(d.score for d in dets if d.class_id == 5) > 0.9


def test_class_names_match_data_yaml():
    names = load_class_names(ROOT / "data.yaml")
    assert len(names) == NC
    assert names[9] == "lamba_kirmizi"
