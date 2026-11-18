"""Eğitilmiş modeli uç cihazlar için ONNX (isteğe bağlı NCNN) biçimine aktarır.

Aktarılan modelin doğruluğu bozulmasın diye, istenirse doğrulama setinde
mAP değerlerini yeniden ölçer. Bu adım PC'de çalıştırılır; Raspberry Pi'a
yalnızca çıkan .onnx dosyası kopyalanır.

Kullanım:
    python export.py --weights best.pt
    python export.py --weights best.pt --imgsz 320 --val
    python export.py --weights best.pt --format ncnn
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Modeli ONNX / NCNN biçimine aktarır")
    parser.add_argument("--weights", required=True, help="Eğitilmiş model (best.pt)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Giriş boyutu; Pi'da hız için 320 denenebilir")
    parser.add_argument("--format", choices=["onnx", "ncnn"], default="onnx")
    parser.add_argument("--val", action="store_true",
                        help="Aktarılan modelin mAP değerlerini doğrulama setinde ölç")
    parser.add_argument("--data", default="data.yaml", help="--val için veri seti yapılandırması")
    args = parser.parse_args()

    model = YOLO(args.weights)
    # opset 12 ve sabit giriş boyutu, ARM üzerindeki ONNX Runtime ile en sorunsuz çalışan ayarlar.
    kwargs = {"opset": 12, "simplify": True} if args.format == "onnx" else {}
    exported = model.export(format=args.format, imgsz=args.imgsz, dynamic=False, **kwargs)
    print(f"Aktarılan model: {exported}")

    if args.val:
        metrics = YOLO(exported, task="detect").val(data=args.data, imgsz=args.imgsz,
                                                    device="cpu", verbose=False)
        print(f"mAP@0.5      : {metrics.box.map50:.3f}")
        print(f"mAP@0.5:0.95 : {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
