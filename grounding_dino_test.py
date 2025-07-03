import cv2, torch, numpy as np
from PIL import Image
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops

# ────────────────── 설정 ────────────────── #
CFG  = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
CKPT = "weights/groundingdino_swinb_cogcoor.pth"
PROMPT_LINES = [
    "highway",
    "road debris",    
    "tiny black object",    
    "standing traffic cones",
]
PROMPT = ". ".join(PROMPT_LINES) + "."
BOX_THRESHOLD  = 0.30
TEXT_THRESHOLD = 0.20
MAX_AREA_RATIO = 0.5   # 전체 프레임의 50 % 이상이면 거대 박스로 간주
IOU_THRESHOLD = 0.8
VIDEO_PATH = r"D:\data\도로_비정형객체\3.avi"

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

# ────────────────── 필터링 ────────────────── #
def filter_preds(boxes, labels, scores, label_blocklist=("the highway",)):
    keep_idx = [
        i for i, l in enumerate(labels)
        if l.strip().lower() not in label_blocklist
    ]
    return boxes[keep_idx], [labels[i] for i in keep_idx], scores[keep_idx]

# ────────────────── 중복 박스 제거 ────────────────── #
def deduplicate(boxes, labels, scores, iou_thr=IOU_THRESHOLD):
    if len(boxes) == 0:
        return boxes, labels, scores

    keep = []
    for i, box_i in enumerate(boxes):
        duplicate_with = None
        for j in keep:
            # box_ops.box_iou → (iou_tensor, union_tensor)
            iou_val = box_ops.box_iou(
                torch.as_tensor(box_i[None, :], dtype=torch.float32),
                torch.as_tensor(boxes[j][None, :], dtype=torch.float32)
            )[0][0, 0].item()          # <- 첫 번째 텐서를 선택

            if iou_val > iou_thr:
                duplicate_with = j
                break

        if duplicate_with is None:          # 새 박스
            keep.append(i)
        else:
            # 이미 있는 박스보다 점수가 높으면 교체
            if scores[i] > scores[duplicate_with]:
                keep.remove(duplicate_with)
                keep.append(i)

    # 최종 결과만 반환
    boxes_out  = boxes[keep]
    labels_out = [labels[k] for k in keep]
    scores_out = scores[keep]
    return boxes_out, labels_out, scores_out

# ────────────────── 거대 박스 필터링 ────────────────── #
def area_filter(boxes, labels, scores, w, h, max_ratio=MAX_AREA_RATIO):
    keep = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        box_area = (x1 - x0) * (y1 - y0)
        if box_area / (w * h) < max_ratio:   # 작으면 keep
            keep.append(i)
    return boxes[keep], [labels[i] for i in keep], scores[keep]

# ────────────────── 비디오 루프 ────────────────── #
cap = cv2.VideoCapture(VIDEO_PATH)
frame_skip = 5
frame_count = 0

while cap.isOpened():
    # ──────── 프레임 읽기 ──────── #
    ret, frame = cap.read()
    if not ret:
        break

    # ──────── 프레임 스킵 ──────── #
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # ──────── 프레임 전처리 ──────── #
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(img_rgb).to(device)

    # ──────── 모델 예측 ──────── #
    with torch.no_grad():
        raw_boxes, scores, phrases = predict(
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
    print("-- Detected Objects --")
    for box, score, label in zip(boxes_xyxy, scores, phrases):
        print(f"Label: {label}, Box: {box.tolist()}, Score: {score:.2f}")

    # ──────── 필터링 ──────── #
    # boxes_xyxy, phrases, scores = filter_preds(boxes_xyxy, phrases, scores)

    # ──────── 거대 박스 필터링 ──────── #
    # boxes_xyxy, phrases, scores = area_filter(boxes_xyxy, phrases, scores, w, h)

    # ──────── 중복 박스 제거 ──────── #
    boxes_xyxy, phrases, scores = deduplicate(boxes_xyxy, phrases, scores)
    
    # ──────── 결과 출력 ──────── #
    print("-- Filtered Objects --")
    for box, score, label in zip(boxes_xyxy, scores, phrases):
        print(f"Label: {label}, Box: {box.tolist()}, Score: {score:.2f}")

    # ──────── 시각화 & 종료키 ──────── #
    frame = draw(frame, boxes_xyxy, phrases)
    cv2.imshow("GroundingDINO", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
