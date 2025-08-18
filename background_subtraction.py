import cv2, torch, numpy as np
import os, time
from PIL import Image
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops

# ────────────────── 설정 ────────────────── #
CFG  = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
CKPT = "weights/groundingdino_swinb_cogcoor.pth"
PROMPT_LINES = [    
    # "traffic cone",
    # "debris on road",
    "debris on road",
]
PROMPT = ". ".join(PROMPT_LINES) + "."
BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.25
IOU_THRESHOLD = 0.8
VIDEO_PATH = r"video/output_with_debris_1.mp4"
ROI = (2, 557, 1247, 1075) # (x1, y1, x2, y2)

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

# ────────────────── 배경 제거기 ────────────────── #
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# ────────────────── 비디오 루프 ────────────────── #
cap = cv2.VideoCapture(VIDEO_PATH)
frame_skip = 5
frame_count = 0

# 'image' 폴더가 없으면 생성
if not os.path.exists('image'):
    os.makedirs('image')

saved_background_path = os.path.join('image', 'background_img.png')
saved_background_img = None
if os.path.exists(saved_background_path):
    saved_background_img = cv2.imread(saved_background_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 배경 모델 업데이트를 위해 매 프레임 `apply` 호출
    t_start = time.time()
    fg_mask = backSub.apply(frame)
    t_apply = time.time() - t_start

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # ─── 추론 프레임 처리 ───
    
    # 1. 마스크 노이즈 제거
    t_start = time.time()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    t_morph = time.time() - t_start

    # 2. Contour 탐지 및 디버그 시각화
    t_start = time.time()
    contours, _ = cv2.findContours(fg_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t_contour = time.time() - t_start
    
    # 디버그용 시각화 이미지 생성
    mask_vis_debug = cv2.cvtColor(fg_mask_cleaned, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 최소 면적 기준을 낮춰서 작은 낙하물도 보이도록 함 (노이즈도 함께 보일 수 있음)
        if area > 10: 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(mask_vis_debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(mask_vis_debug, f"{int(area)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 3. 배경 추출 (모델 입력용)
    t_start = time.time()
    background_img = backSub.getBackgroundImage()
    t_get_bg = time.time() - t_start

    # 4. 모델 추론
    h, w = frame.shape[:2]
    raw_boxes, scores, phrases = torch.empty(0, 4), torch.empty(0), [] # 기본값
    bg_diff = None # bg_diff 초기화
    t_absdiff = 0 # 초기화

    # 4-1. bg_diff 계산 (가능한 경우)
    if background_img is not None and saved_background_img is not None and background_img.shape == saved_background_img.shape:
        t_start = time.time()
        bg_diff = cv2.absdiff(background_img, saved_background_img)
        t_absdiff = time.time() - t_start

    # 4-2. bg_diff를 사용하여 추론 수행 (bg_diff가 있을 때만)
    if bg_diff is not None:
        # ROI 영역 자르기
        x1, y1, x2, y2 = ROI
        inference_img_roi = bg_diff[y1:y2, x1:x2]
        h_roi, w_roi = inference_img_roi.shape[:2]

        # ROI 이미지로 추론 수행
        img_rgb = cv2.cvtColor(inference_img_roi, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess(img_rgb).to(device)

        with torch.no_grad():
            raw_boxes, scores, phrases = predict(
                model=model,
                image=img_tensor[0],
                caption=PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

    # 5. 시각화 (원본 프레임에)
    frame_vis = frame.copy()
    # ROI 영역 표시 (디버깅용)
    cv2.rectangle(frame_vis, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 0, 255), 2)

    if raw_boxes.numel() > 0:
        # 5-1. 박스 좌표를 ROI 내부 기준으로 변환
        boxes_xyxy = convert_boxes(raw_boxes, w_roi, h_roi)
        
        # 5-2. 박스 좌표를 전체 프레임 기준으로 변환 (오프셋 추가)
        boxes_xyxy[:, 0::2] += ROI[0] # x 좌표에 x1 더하기
        boxes_xyxy[:, 1::2] += ROI[1] # y 좌표에 y1 더하기

        # 5-3. 중복 제거 및 최종 시각화
        boxes_xyxy, phrases, scores = deduplicate(boxes_xyxy, phrases, scores)
        frame_vis = draw(frame_vis, boxes_xyxy, phrases, scores)

    # 6. 화면 출력
    display_scale = 0.8
    mask_display_scale = 0.8
    
    frame_display = cv2.resize(frame_vis, (int(w * display_scale), int(h * display_scale)))
    # 마스크 창에 디버그 시각화 이미지를 표시
    mask_display = cv2.resize(mask_vis_debug, (int(w * mask_display_scale), int(h * mask_display_scale)))

    # 배경 이미지 출력    
    if background_img is not None:
        background_display = cv2.resize(background_img, (int(w * display_scale), int(h * display_scale)))
        cv2.imshow("Background (Moving Objects Removed)", background_display)

    # 수행시간 출력
    print(f"--- Frame {frame_count} Timings ---")
    print(f"  - backSub.apply():      {t_apply*1000:.2f} ms")
    print(f"  - cv2.morphologyEx():   {t_morph*1000:.2f} ms")
    print(f"  - cv2.findContours():   {t_contour*1000:.2f} ms")
    print(f"  - backSub.getBG():      {t_get_bg*1000:.2f} ms")
    if bg_diff is not None:
        print(f"  - cv2.absdiff():        {t_absdiff*1000:.2f} ms")

    cv2.imshow("Result", frame_display)
    cv2.imshow("Foreground Mask", mask_display)

    # 저장된 배경과의 차이(bg_diff)가 계산된 경우 화면에 표시
    if bg_diff is not None:
        bg_diff_display = cv2.resize(bg_diff, (int(w * display_scale), int(h * display_scale)))
        cv2.imshow("Background Difference", bg_diff_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if background_img is not None:
            save_path = os.path.join('image', 'background_img.png')
            cv2.imwrite(save_path, background_img)
            saved_background_img = cv2.imread(save_path)
            print(f"배경 이미지가 '{save_path}'에 저장 및 리로드되었습니다.")
        else:
            print("배경 이미지가 아직 생성되지 않았습니다.")

cap.release()
cv2.destroyAllWindows()