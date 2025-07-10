import cv2, torch, numpy as np
from PIL import Image
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops

# ────────────────── 설정 ────────────────── #
CFG  = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
CKPT = "weights/groundingdino_swinb_cogcoor.pth"
PROMPT_LINES = [    
    "traffic cone",
    # "debris on road",
]
PROMPT = ". ".join(PROMPT_LINES) + "."
BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.25
IOU_THRESHOLD = 0.8
VIDEO_PATH = r"D:\data\도로_비정형객체1\5.avi"

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
def _label_color(label: str):
    """라벨 문자열을 → 고정된 BGR 색상 (OpenCV용)"""
    seed = abs(hash(label)) % (2**32)
    rng  = np.random.default_rng(seed)
    return tuple(int(c) for c in rng.integers(0, 256, size=3))

def draw_combined_traffic_cone_area(frame, boxes, labels):
    """traffic cone들을 모두 포함하는 하나의 사각형 영역을 그림"""
    # traffic cone 박스들만 필터링
    cone_boxes = []
    for box, label in zip(boxes, labels):
        if "traffic cone" in label.lower():
            cone_boxes.append(box)
    
    if len(cone_boxes) == 0:
        return frame
    
    # 모든 traffic cone 박스들을 포함하는 최소/최대 좌표 계산
    cone_boxes = np.array(cone_boxes)
    min_x = cone_boxes[:, 0].min()
    min_y = cone_boxes[:, 1].min()
    max_x = cone_boxes[:, 2].max()
    max_y = cone_boxes[:, 3].max()
    
    # 통합 영역 그리기 (두꺼운 빨간색 테두리)
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), thickness=5)
    
    # 라벨 추가
    text = f"Work Area ({len(cone_boxes)} cones)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2)
    
    # 글자 배경
    bg_tl = (min_x, max(min_y - th - 10, 0))
    bg_br = (min_x + tw + 10, min_y)
    cv2.rectangle(frame, bg_tl, bg_br, (0, 0, 255), -1)
    
    # 글자
    cv2.putText(frame, text, (min_x + 5, min_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                lineType=cv2.LINE_AA)
    
    return frame

def draw(frame, boxes, labels, scores):
    # 기존 개별 박스들 그리기
    for (x0, y0, x1, y1), lab, sc in zip(boxes, labels, scores):
        color = _label_color(lab)

        # ① 바운딩 박스 (3 px)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness=3)

        # ② 라벨 + 점수 문자열
        text = f"{lab} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=0.7, thickness=2)

        # ③ 글자 배경 (반투명 검정)
        bg_tl = (x0, max(y0 - th - 6, 0))
        bg_br = (x0 + tw + 10, y0)
        cv2.rectangle(frame, bg_tl, bg_br, (0, 0, 0), -1)  # 채우기
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_tl, bg_br, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)  # 투명도 조절

        # ④ 글자
        cv2.putText(frame, text, (x0 + 5, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    lineType=cv2.LINE_AA)
    
    # traffic cone 통합 영역 그리기
    # frame = draw_combined_traffic_cone_area(frame, boxes, labels)
    
    return frame

# ────────────────── 박스 변환 ────────────────── #
def convert_boxes(raw_boxes, w, h):
    # 1) 정규화 → 픽셀
    if raw_boxes.max() <= 1:
        raw_boxes = raw_boxes * torch.tensor([w, h, w, h],
                                             dtype=raw_boxes.dtype,
                                             device=raw_boxes.device)
    # 2) cxcywh → xyxy
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(raw_boxes)
    # 3) 클리핑 + numpy 변환
    boxes_xyxy[:, 0::2].clamp_(0, w)
    boxes_xyxy[:, 1::2].clamp_(0, h)
    return boxes_xyxy.cpu().numpy().astype(int)

# ────────────────── (선택) 필터링 함수 ────────────────── #
def deduplicate(boxes, labels, scores, iou_thr=IOU_THRESHOLD):
    if len(boxes) == 0:
        return boxes, labels, scores
    keep = []
    for i, box_i in enumerate(boxes):
        dup = None
        for j in keep:
            iou_val = box_ops.box_iou(
                torch.as_tensor(box_i[None], dtype=torch.float32),
                torch.as_tensor(boxes[j][None], dtype=torch.float32)
            )[0][0, 0].item()
            if iou_val > iou_thr:
                dup = j
                break
        if dup is None:
            keep.append(i)
        elif scores[i] > scores[dup]:
            keep.remove(dup)
            keep.append(i)
    return boxes[keep], [labels[k] for k in keep], scores[keep]

# ────────────────── 비디오 루프 ────────────────── #
cap = cv2.VideoCapture(VIDEO_PATH)
frame_skip = 5         # 매 5프레임마다 추론
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(img_rgb).to(device)

    with torch.no_grad():
        raw_boxes, scores, phrases = predict(
            model=model,
            image=img_tensor[0],          # (C, H, W)
            caption=PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

    # 박스가 없으면 그대로 출력
    if raw_boxes.numel() == 0:
        cv2.imshow("GroundingDINO", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    # ──────── 후처리 ──────── #
    boxes_xyxy = convert_boxes(raw_boxes, w, h)
    boxes_xyxy, phrases, scores = deduplicate(boxes_xyxy, phrases, scores)

    # ──────── 시각화 ──────── #
    frame_vis = draw(frame, boxes_xyxy, phrases, scores)
    cv2.imshow("GroundingDINO", frame_vis)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
