import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import os
import re
import easyocr

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
reader = easyocr.Reader(['en'], gpu=False)  # buat OCR EasyOCR

# --- Fungsi luruskan dokumen ---
def straighten_document(img, cnt, debug=True):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    rect_pts = order_points(box)
    (tl, tr, br, bl) = rect_pts

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect_pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    if debug:
        cv2.imshow("Dokumen Lurus", warped)
    
    return warped

# --- Koreksi orientasi teks ---
def correct_text_orientation(img):
    try:
        osd = pytesseract.image_to_osd(img)
        rotate_angle = int(re.search('Rotate: (\d+)', osd).group(1))
        if rotate_angle != 0:
            if rotate_angle == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotate_angle == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rotate_angle == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    except:
        pass
    return img

# --- Segmentasi karakter ---
def segment_characters(thresh_img, base_filename="noname"):
    if np.mean(thresh_img) > 127:
        thresh_img = cv2.bitwise_not(thresh_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = clean.shape
    char_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 1 < w < w_img * 0.9 and 8 < h < h_img * 0.95:
            char_boxes.append((x, y, w, h))
    if not char_boxes:
        print("Tidak ada karakter terdeteksi.")
        return []

    char_boxes = sorted(char_boxes, key=lambda b: b[0])
    target_size = (80, 120)
    chars = []
    for (x, y, w, h) in char_boxes:
        char_crop = clean[y:y+h, x:x+w]
        pad_x = max(0, (h - w)//2)
        padded = cv2.copyMakeBorder(char_crop, 5,5,pad_x+2,pad_x+2, cv2.BORDER_CONSTANT, value=(0,0,0))
        resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
        chars.append(resized)

    # Plot hasil segmentasi
    fig, axs = plt.subplots(1, len(chars), figsize=(15,4))
    if len(chars) == 1: axs=[axs]
    for i, char in enumerate(chars):
        axs[i].imshow(char, cmap="gray")
        axs[i].set_title(f"Char {i+1}")
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()
    return chars

# --- OCR tiap karakter ---
def ocr_characters(chars):
    results = []
    for c in chars:
        if len(c.shape) == 2:
            c_rgb = cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)
        else:
            c_rgb = c
        out = reader.readtext(c_rgb, detail=0)
        text = out[0] if len(out)>0 else ""
        results.append(text[0] if len(text)>0 else "")
    return results

# --- Proses frame ---
def process_frame(frame, debug=True):
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    if debug: cv2.imshow("Edges", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Dokumen tidak ditemukan!")
        return None
    cnt = max(contours, key=cv2.contourArea)
    crop_straight = straighten_document(img, cnt, debug)
    if debug:
        cv2.imshow("Crop Dokumen", crop_straight)

    gray_paper = cv2.cvtColor(crop_straight, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced2 = clahe.apply(gray_paper)
    _, text_thresh = cv2.threshold(gray_enhanced2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug: cv2.imshow("Threshold", text_thresh)

    chars = segment_characters(text_thresh)
    ocr_results = ocr_characters(chars)
    print("OCR Result:", "".join(ocr_results))
    return crop_straight

# --- Live Camera ---
# ========================= LIVE CAMERA =========================
url = "http://192.168.100.222:4747/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Camera gagal dibuka!")
    exit()

print("[INFO] Tekan 'c' untuk CAPTURE dan PROSES dokumen")
print("[INFO] Tekan 'q' untuk KELUAR")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        print("Processing image...")
        process_frame(frame)

        # if result is None:
        #     print(status)
        # else:
        #     print(status)
        #     cv2.imshow("Detected ROI", result)
        #     cv2.imwrite("hasil_bondowoso.jpg", result)
        #     print("Hasil disimpan: hasil_bondowoso.jpg")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
