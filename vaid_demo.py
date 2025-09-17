import cv2, torch, numpy as np
from PIL import Image
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
import torchvision.ops as ops
from deep_sort_realtime.deepsort_tracker import DeepSort

# ────────────────── 설정 ────────────────── #
CFG  = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
CKPT = "weights/groundingdino_swinb_cogcoor.pth"
PROMPT_LINES = [    
    "traffic cone",
    # "debris on road",
]
PROMPT = ". ".join(PROMPT_LINES) + "."
BOX_THRESHOLD  = 0.20
TEXT_THRESHOLD = 0.20
IOU_THRESHOLD = 0.5
VIDEO_PATH = r"video\CCTV099_20250702_132042507_3_1_정지작업차2_라바콘2.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────── 모델 로드 ────────────────── #
model = load_model(CFG, CKPT).to(device).eval()
tracker = DeepSort(max_age=5, n_init=2)

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

def draw_tracks(frame, tracks):
    """DeepSORT 트랙 시각화"""
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        confidence = getattr(track, 'det_conf', 0.0)
        if confidence is None:
            confidence = 0.0
        
        if confidence <= 0.4:
            continue

        track_id = track.track_id
        ltrb = track.to_tlbr()
        label = track.get_det_class()
        
        x0, y0, x1, y1 = map(int, ltrb)
        
        # ID별로 고유 색상
        color = _label_color(str(track_id))

        # ① 바운딩 박스 (3 px)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness=3)

        text = f"ID:{track_id} {label} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=0.7, thickness=2)

        # ③ 글자 배경 (반투명 검정)
        bg_tl = (x0, max(y0 - th - 6, 0))
        bg_br = (x0 + tw + 10, y0)
        cv2.rectangle(frame, bg_tl, bg_br, (0, 0, 0), -1)
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_tl, bg_br, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # ④ 글자
        cv2.putText(frame, text, (x0 + 5, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    lineType=cv2.LINE_AA)
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

def nms_filter(boxes, scores, iou_threshold=0.8):
    """
    IOU 기반 중복 박스 제거 (NMS)
    """
    if len(boxes) == 0:
        return [], [], []

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
    return keep_indices.numpy()


# ────────────────── 비디오 루프 ────────────────── #
cap = cv2.VideoCapture(VIDEO_PATH)
frame_skip = 1   # 1: 모든 프레임, 2: 2프레임마다, 3: 3프레임마다 ...
frame_count = 0

# 비디오 저장 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

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

    # 박스가 없으면 추적기만 업데이트하고 출력
    if raw_boxes.numel() == 0:
        tracker.update_tracks([], frame=frame) # 빈 목록으로 업데이트
        out.write(frame)  # 원본 프레임 저장
        cv2.imshow("GroundingDINO + DeepSORT", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    # ──────── 후처리 ──────── #
    boxes_xyxy = convert_boxes(raw_boxes, w, h)

    # NMS 필터링
    keep_idxs = nms_filter(boxes_xyxy, scores.tolist(), iou_threshold=IOU_THRESHOLD)
    boxes_xyxy = boxes_xyxy[keep_idxs]
    scores = [scores[i] for i in keep_idxs]
    phrases = [phrases[i] for i in keep_idxs]

    # ──────── DeepSORT 연동 ──────── #
    # DeepSORT 입력 형식: ([x, y, w, h], score, class_name)
    detections = []
    for (x1, y1, x2, y2), score, label in zip(boxes_xyxy, scores, phrases):
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, label))

    # 트랙 업데이트
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # ──────── 시각화 ──────── #
    frame_vis = draw_tracks(frame, tracks)
    out.write(frame_vis)  # 검지 결과가 그려진 프레임 저장
    cv2.imshow("GroundingDINO + DeepSORT", frame_vis)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()