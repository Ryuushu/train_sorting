import cv2
from ultralytics import YOLO

# ==============================
# SETTING DROIDCAM
# ==============================

# 1) Kalau DroidCam via WiFi:
SOURCE = "http://192.168.1.4:4747/video"   # GANTI IP MU !!!

# 2) Jika DroidCam sebagai webcam driver:
# SOURCE = 1  # bisa 0 / 1 / 2 tergantung urutan kamera

# Set model
model = YOLO("best.pt")   # Ganti dengan model kamu

# ==============================
# BUKA DROIDCAM
# ==============================

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("GAGAL membuka DroidCam! Cek IP atau webcam index.")
    exit()

print("Tracking YOLO dimulai... (Tekan Q untuk keluar)")

# ==============================
# LOOP TRACKING
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 🔥 TRACK (bukan predict)
    results = model.track(
        frame,
        persist=True,      # mempertahankan ID object
        conf=0.25,
        device="cpu",      # ganti ke "0" jika pakai GPU
        verbose=False
    )

    res = results[0]

    if res.boxes is not None:
        boxes = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id
        ids = ids.cpu().numpy() if ids is not None else None
        clss = res.boxes.cls.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            track_id = int(ids[i]) if ids is not None else -1
            cls = int(clss[i])
            conf = confs[i]

            label = f"ID {track_id} | {model.names[cls]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

    cv2.imshow("YOLO TRACK - DroidCam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
