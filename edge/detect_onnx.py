"""ONNX modelini video dosyası ya da kamera üzerinde çalıştırır (Ultralytics gerekmez).

Kullanım:
    python edge/detect_onnx.py --model best.onnx --source parkur.mp4
    python edge/detect_onnx.py --model best.onnx --source 0
    python edge/detect_onnx.py --model best.onnx --source parkur.mp4 --headless --save cikti.mp4

Ekranı olmayan bir Pi'da --headless kullanın; FPS terminale yazılır.
"""
import argparse
import time
from pathlib import Path

import cv2

from yolo_onnx import YoloOnnx, draw, load_class_names

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX ile trafik işareti tespiti")
    parser.add_argument("--model", required=True, help="Aktarılmış model (best.onnx)")
    parser.add_argument("--source", default="0", help="Video dosyası yolu ya da kamera numarası")
    parser.add_argument("--conf", type=float, default=0.70, help="Güven eşiği")
    parser.add_argument("--threads", type=int, default=None, help="İşlemci iş parçacığı sayısı")
    parser.add_argument("--data", default=str(ROOT / "data.yaml"), help="Sınıf adları")
    parser.add_argument("--headless", action="store_true", help="Pencere açma")
    parser.add_argument("--save", default=None, help="İşaretlenmiş videoyu bu dosyaya kaydet")
    args = parser.parse_args()

    names = load_class_names(args.data)
    model = YoloOnnx(args.model, threads=args.threads)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Kaynak açılamadı: {args.source}")

    writer = None
    frames, start = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        detections = model(frame, conf=args.conf)
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        frames += 1

        annotated = draw(frame, detections, names)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if args.save:
            if writer is None:
                h, w = annotated.shape[:2]
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(annotated)

        if args.headless:
            if frames % 30 == 0:
                print(f"{frames} kare, anlık {fps:.1f} FPS")
        else:
            cv2.imshow("Robotaksi ONNX - cikis icin Q", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.perf_counter() - start
    cap.release()
    if writer:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()
    if frames:
        print(f"{frames} kare işlendi, ortalama {frames / elapsed:.1f} FPS")


if __name__ == "__main__":
    main()
