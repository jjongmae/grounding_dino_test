import cv2, torch, numpy as np
from PIL import Image
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T

# ────────────────── 설정 ────────────────── #
CFG  = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
CKPT = "weights/groundingdino_swinb_cogcoor.pth"
PROMPT = (
    "a prefabricated house on a farm."
    "a tall gray utility pole beside."
    "a bridge across the highway."
)
VIDEO_PATH = r"D:\data\여주시험도로_20250610\카메라1_202506101340.mp4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────── 모델 로드 ────────────────── #
model = load_model(CFG, CKPT).to(device).eval()

# ────────────────── 전처리 ────────────────── #
def preprocess(img_np):
    img_pil = Image.fromarray(img_np)
    tf = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    return tf(img_pil, None)[0].unsqueeze(0)          # (1, C, H, W)

# ────────────────── 시각화 ────────────────── #
def draw(frame, boxes, labels):
    for b, l in zip(boxes, labels):
        x0, y0, x1, y1 = map(int, b)
        cv2.rectangle(frame, (x0, y0), (x1, y1),
                      (0, 255, 0), 2)
        cv2.putText(frame, l, (x0, max(y0 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2)
    return frame

# ────────────────── 비디오 루프 ────────────────── #
cap = cv2.VideoCapture(VIDEO_PATH)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(img_rgb).to(device)

    with torch.no_grad():
        raw_boxes, _, phrases = predict(
            model=model,
            image=img_tensor[0],          # (C, H, W)
            caption=PROMPT,
            box_threshold=0.35,
            text_threshold=0.25
        )

    if raw_boxes.numel() == 0:
        cv2.imshow("GroundingDINO", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    # ──────── ① 텐서 → NumPy ──────── #
    boxes = raw_boxes.cpu().numpy()                    # [cx,cy,w,h] norm

    # ──────── ② 정규화 → 픽셀 스케일 ──────── #
    if boxes.max() <= 1.0:
        boxes *= np.array([w, h, w, h])

    # ──────── ③ 중심 → 모서리 변환 ──────── #
    boxes_xyxy = boxes.copy()
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2   # x0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2   # y0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2   # x1
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2   # y1

    # ──────── ④ 클리핑 ──────── #
    boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0, w)
    boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0, h)

    # ──────── 디버그: 변환 후 출력 ──────── #
    for i, b in enumerate(boxes_xyxy):
        bw, bh = b[2] - b[0], b[3] - b[1]

    # ──────── 시각화 & 종료키 ──────── #
    frame = draw(frame, boxes_xyxy, phrases)
    cv2.imshow("GroundingDINO", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
