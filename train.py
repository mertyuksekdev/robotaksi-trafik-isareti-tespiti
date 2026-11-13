"""YOLOv8s modelini Robotaksi trafik işareti veri setiyle eğitir.

Varsayılan değerler, README'deki sonuçları veren eğitimin ayarlarıdır
(Pro_Model_v1: 50 epoch, 640 px, batch 32, RTX 4060).

Kullanım:
    python train.py
    python train.py --epochs 100 --batch 16
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Robotaksi trafik işareti modeli eğitimi")
    parser.add_argument("--data", default="data.yaml", help="Veri seti yapılandırma dosyası")
    parser.add_argument("--model", default="yolov8s.pt", help="Başlangıç ağırlıkları")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32, help="Bellek hatası alırsanız 16'ya düşürün")
    parser.add_argument("--patience", type=int, default=10, help="Erken durdurma sabrı (epoch)")
    parser.add_argument("--device", default="0", help="GPU numarası ya da 'cpu'")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=4,
        cache=True,
        project="runs",
        name="robotaksi",
    )


if __name__ == "__main__":
    main()
