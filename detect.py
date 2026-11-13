"""Eğitilmiş modeli bir video dosyası ya da kamera üzerinde çalıştırır.

Ekranın sol üstünde anlık FPS gösterilir; çıkışta ortalama FPS yazdırılır.

Kullanım:
    python detect.py --weights runs/robotaksi/weights/best.pt --source parkur.mp4
    python detect.py --weights best.pt --source 0          # varsayılan kamera
Çıkmak için Q tuşuna basın.
"""
import argparse
import time

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Robotaksi trafik işareti tespiti")
    parser.add_argument("--weights", required=True, help="Eğitilmiş model (best.pt)")
    parser.add_argument("--source", default="0", help="Video dosyası yolu ya da kamera numarası")
    parser.add_argument("--conf", type=float, default=0.70,
                        help="Güven eşiği; yanlış tespitleri azaltmak için 0.70 kullanıldı")
    args = parser.parse_args()

    model = YOLO(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Kaynak açılamadı: {args.source}")

    frames, start = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        results = model(frame, conf=args.conf, verbose=False)
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        frames += 1

        annotated = results[0].plot()
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Robotaksi - cikis icin Q", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.perf_counter() - start
    cap.release()
    cv2.destroyAllWindows()
    if frames:
        print(f"{frames} kare işlendi, ortalama {frames / elapsed:.1f} FPS")


if __name__ == "__main__":
    main()
