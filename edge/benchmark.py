"""ONNX modelinin hızını ölçer: ön işleme, çıkarım ve son işleme sürelerini ayrı ayrı verir.

Aynı betik PC'de ve Raspberry Pi'da çalıştırılır. Sonuçlar --csv ile bir
dosyaya eklenir, böylece farklı cihaz ve giriş boyutları yan yana karşılaştırılır.

Kullanım:
    python edge/benchmark.py --model best.onnx --images datasets/parkur/valid/images --device "PC i5-12400"
    python edge/benchmark.py --model best_320.onnx --images ornekler/ --device "Raspberry Pi 5" --csv results/edge_benchmark.csv
"""
import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from yolo_onnx import YoloOnnx, postprocess

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_images(folder: str, limit: int) -> list[np.ndarray]:
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)[:limit]
    images = [img for p in paths if (img := cv2.imread(str(p))) is not None]
    if not images:
        raise SystemExit(f"Klasörde görüntü bulunamadı: {folder}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX model hız ölçümü")
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True, help="Ölçümde kullanılacak görüntü klasörü")
    parser.add_argument("--limit", type=int, default=200, help="En fazla kaç görüntü kullanılacak")
    parser.add_argument("--warmup", type=int, default=10, help="Ölçüme katılmayan ısınma turu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--device", default=platform.node(), help="Tabloda görünecek cihaz adı")
    parser.add_argument("--csv", default=None, help="Sonucu bu CSV dosyasına ekle")
    args = parser.parse_args()

    model = YoloOnnx(args.model, threads=args.threads)
    images = load_images(args.images, args.limit)

    for img in images[:args.warmup]:
        model(img)

    pre, inf, post = [], [], []
    for img in images:
        t0 = time.perf_counter()
        blob, scale, pad = model.preprocess(img)
        t1 = time.perf_counter()
        out = model.infer(blob)
        t2 = time.perf_counter()
        postprocess(out, 0.70, 0.45, scale, pad, img.shape[:2])
        t3 = time.perf_counter()
        pre.append(t1 - t0)
        inf.append(t2 - t1)
        post.append(t3 - t2)

    def ms(values: list[float]) -> float:
        return statistics.mean(values) * 1000

    total = [a + b + c for a, b, c in zip(pre, inf, post)]
    row = {
        "cihaz": args.device,
        "model": Path(args.model).name,
        "giris_boyutu": model.size,
        "goruntu_sayisi": len(images),
        "on_isleme_ms": round(ms(pre), 1),
        "cikarim_ms": round(ms(inf), 1),
        "son_isleme_ms": round(ms(post), 1),
        "toplam_ms": round(ms(total), 1),
        "fps": round(1000 / ms(total), 1),
        "p95_ms": round(float(np.percentile(total, 95)) * 1000, 1),
    }

    for key, value in row.items():
        print(f"{key:15}: {value}")

    if args.csv:
        path = Path(args.csv)
        new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(f"Sonuç eklendi: {path}")


if __name__ == "__main__":
    main()
