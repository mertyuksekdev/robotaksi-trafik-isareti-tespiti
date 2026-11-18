"""PyTorch ve Ultralytics olmadan, yalnızca ONNX Runtime + NumPy + OpenCV ile
YOLOv8 çıkarımı. Raspberry Pi gibi kaynakları sınırlı cihazlar için yazıldı.

YOLOv8 tespit modelinin ham çıktısı (1, 4 + sınıf_sayısı, aday_sayısı)
biçimindedir: her aday için kutu merkezi (cx, cy), genişlik, yükseklik ve
her sınıfın skoru. Bu modül görüntüyü modele hazırlar (letterbox), çıktıyı
çözer ve çakışan kutuları NMS ile eler.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 (orijinal görüntü pikseli)
    score: float
    class_id: int


def load_class_names(data_yaml: str | Path) -> list[str]:
    with open(data_yaml, encoding="utf-8") as f:
        return list(yaml.safe_load(f)["names"])


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Görüntüyü en-boy oranını bozmadan size x size kareye sığdırır, kenarları gri doldurur."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def postprocess(output: np.ndarray, conf: float, iou: float, scale: float,
                pad: tuple[int, int], orig_shape: tuple[int, int]) -> list[Detection]:
    """Ham model çıktısını orijinal görüntü koordinatlarında tespitlere çevirir."""
    preds = output[0].T                      # (aday_sayısı, 4 + sınıf_sayısı)
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(preds)), class_ids]
    keep = scores >= conf
    if not keep.any():
        return []
    preds, scores, class_ids = preds[keep], scores[keep], class_ids[keep]

    # Merkez biçiminden köşe biçimine, letterbox dolgusunu ve ölçeği geri al.
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = (cx - bw / 2 - pad[0]) / scale
    y1 = (cy - bh / 2 - pad[1]) / scale
    x2 = (cx + bw / 2 - pad[0]) / scale
    y2 = (cy + bh / 2 - pad[1]) / scale
    h, w = orig_shape
    x1, x2 = x1.clip(0, w), x2.clip(0, w)
    y1, y2 = y1.clip(0, h), y2.clip(0, h)

    # Sınıf bazında NMS: kutuları sınıf numarasına göre kaydırarak farklı
    # sınıfların birbirini elemesini engelle.
    offset = class_ids[:, None] * 4096.0
    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1) + np.concatenate(
        [offset, offset, np.zeros_like(offset), np.zeros_like(offset)], axis=1)
    idx = cv2.dnn.NMSBoxes(nms_boxes.tolist(), scores.tolist(), conf, iou)
    idx = np.array(idx).flatten() if len(idx) else []

    return [Detection((int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                      float(scores[i]), int(class_ids[i])) for i in idx]


class YoloOnnx:
    def __init__(self, model_path: str | Path, threads: int | None = None):
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), options,
                                            providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])        # sabit giriş boyutu (ör. 640 ya da 320)

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        boxed, scale, pad = letterbox(image, self.size)
        blob = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(blob, dtype=np.float32) / 255.0, scale, pad

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: blob})[0]

    def __call__(self, image: np.ndarray, conf: float = 0.70, iou: float = 0.45) -> list[Detection]:
        blob, scale, pad = self.preprocess(image)
        return postprocess(self.infer(blob), conf, iou, scale, pad, image.shape[:2])


def draw(image: np.ndarray, detections: list[Detection], names: list[str]) -> np.ndarray:
    for d in detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"{names[d.class_id]} {d.score:.2f}"
        cv2.putText(image, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return image
