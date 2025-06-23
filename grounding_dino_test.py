import cv2, torch, numpy as np
from PIL import Image
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops

# ────────────────── 설정 ────────────────── #
CFG  = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
CKPT = "weights/groundingdino_swinb_cogcoor.pth"
PROMPT_LINES = [
    "debris on road not car",
    "trash on road not car",
    "road work zone with traffic cones",
]
PROMPT = "\n".join(PROMPT_LINES)
BOX_THRESHOLD  = 0.35
TEXT_THRESHOLD = 0.25
VIDEO_PATH = r"D:\data\도로_비정형객체\6.avi"

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

# ────────────────── 박스 변환 ────────────────── #
def convert_boxes(raw_boxes, w, h):
    # 1) 정규화 → 픽셀
    if raw_boxes.max() <= 1:          # 최대값이 1 이하면 정규화로 간주
        raw_boxes = raw_boxes * torch.tensor([w, h, w, h],
                                             dtype=raw_boxes.dtype,
                                             device=raw_boxes.device)

    # 2) cxcywh → xyxy (torch 버전 함수 사용)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(raw_boxes)

    # 3) 클리핑 + numpy 변환
    boxes_xyxy[:, 0::2].clamp_(0, w)
    boxes_xyxy[:, 1::2].clamp_(0, h)

    return boxes_xyxy.cpu().numpy().astype(int)

# ────────────────── 비디오 루프 ────────────────── #
cap = cv2.VideoCapture(VIDEO_PATH)
while cap.isOpened():
    # ──────── 프레임 읽기 ──────── #
    ret, frame = cap.read()
    if not ret:
        break

    # ──────── 프레임 전처리 ──────── #
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(img_rgb).to(device)

    # ──────── 모델 예측 ──────── #
    with torch.no_grad():
        raw_boxes, _, phrases = predict(
            model=model,
            image=img_tensor[0],          # (C, H, W)
            caption=PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

    # ──────── 박스가 없으면 건너뛰기 ──────── #
    if raw_boxes.numel() == 0:
        cv2.imshow("GroundingDINO", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    # ──────── 박스 변환 ──────── #
    boxes_xyxy = convert_boxes(raw_boxes, w, h)

    # ──────── 시각화 & 종료키 ──────── #
    frame = draw(frame, boxes_xyxy, phrases)
    cv2.imshow("GroundingDINO", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
