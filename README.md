# 🏎️ Real-Time Racing Video Analysis System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-11x-green.svg)](https://docs.ultralytics.com/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-red.svg)](https://flask.palletsprojects.com/)

**레이싱 헤드캠 영상에서 차량을 실시간으로 검출·추적하고, AI 기반 추월 가능성을 분석하는 웹 기반 시스템**

---

## 프로젝트 개요

본 프로젝트는 모터스포츠 온보드 카메라 영상을 분석하여 다음을 제공합니다:

- **실시간 차량 검출** (YOLO11x 기반)
- **다중 객체 추적** (Norfair 기반)
- **속도 및 거리 추정** (픽셀 기반 계산)
- **추월 가능성 점수** (0-100점, 5단계 상태)
- **텔레메트리 데이터 로깅** (CSV 내보내기)
- **웹 기반 실시간 스트리밍 UI**

---

## 주요 기능

### 1. 차량 검출 및 추적
- **YOLO11x** 모델로 차량 자동 검출 (신뢰도 0.2 이상)
- **Norfair** 트래커로 프레임 간 일관된 ID 유지
- 최소 박스 크기 및 종횡비 필터링으로 오검출 제거

### 2. 추월 분석 시스템
- 상대 거리, 속도 차, 차선 폭 기반 종합 점수 계산
- 5단계 상태 분류:
  - 🔴 **DANGEROUS** (< 35점)
  - 🟡 **RISKY** (35-55점)
  - 🟠 **CAUTION** (55-75점)
  - 🟢 **POSSIBLE** (75점 이상)
  - ✅ **OVERTAKING** (이미 추월 중)

### 3. 차선 검출
- Canny 엣지 + Hough 변환으로 트랙 경계 인식
- 좌우 여유 공간 계산하여 추월 안전도 평가

### 4. 텔레메트리 데이터
- 프레임별 속도, 거리, 추월 점수 기록
- CSV 형식으로 다운로드 가능

### 5. 웹 인터페이스
- 비디오 업로드 (드래그 앤 드롭 지원)
- 실시간 분석 결과 스트리밍
- 진행률 표시 및 상태 모니터링

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                       Flask Web Server                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Video Upload│  │ Processing   │  │ Telemetry Export │  │
│  │   (POST)    │  │  Thread      │  │      (CSV)       │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Video Processing Pipeline                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ OpenCV   │→ │ YOLO11x  │→ │ Norfair  │→ │ Overtake │   │
│  │ Frame    │  │ Detection│  │ Tracking │  │  Score   │   │
│  │ Read     │  │          │  │          │  │ Algorithm│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Real-Time Streaming UI                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Video Stream │  │ Score/Status │  │ Progress Bar    │  │
│  │   (MJPEG)    │  │   Display    │  │   & Controls    │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

### 딥러닝 및 컴퓨터 비전
- **Ultralytics YOLO11** (x / l) - 최신 객체 검출 모델
- **Norfair 2.x** - Kalman 필터 기반 객체 추적
- **OpenCV 4.x** - 영상 처리 및 전처리
- **PyTorch 2.x + CUDA** - GPU 가속 추론

### 웹 프레임워크
- **Flask 2.x** - 백엔드 서버
- **HTML/CSS/JavaScript** - 프론트엔드 UI

### 개발 환경
- **Python 3.8+**
- **NVIDIA RTX 4070** (12GB VRAM)
- **32GB RAM**

---

## 🔬 알고리즘 상세

### 차량 검출 (YOLO11x)

```python
model = YOLO('yolo11x.pt')
model.to('cuda:0')
model.conf = 0.2  # 신뢰도 임계값
model.iou = 0.4   # NMS IoU 임계값
```

- **사용 클래스**: COCO 클래스 {2, 3, 5, 7} (차량 관련)
- **필터링**: 최소 박스 크기 8×8, 종횡비 0.2~5.0

### 객체 추적 (Norfair)

```python
tracker = Tracker(
    distance_function="euclidean",
    distance_threshold=30
)
```

- 중심점 기반 유클리드 거리로 매칭
- 최근 60프레임 궤적 저장 (30fps 기준 2초)

### 추월 점수 계산

**입력 요소**:
- 상대 거리 (화면 비율 기반)
- 거리 변화 추세
- 속도 차이
- 차선 폭 기반 공간 여유

**가중치**:
- 상대 거리: 45%
- 거리 추세: 25%
- 속도 차: 20%
- 공간 여유: 10%

### 속도 추정

```python
speed_kmh = (pixel_distance / pixels_per_meter) / time_delta * 3.6
```

- `pixels_per_meter = 15` 가정
- 최근 10프레임 기준 계산

---

## 📁 프로젝트 구조

```
racing-ai-analysis/
├── racing_ai.py           # Flask 서버 및 메인 처리 파이프라인
├── labeling_tool.py       # 커스텀 라벨링 도구
├── train_f1_mix.py        # YOLO 학습 스크립트
├── index.html             # 웹 UI
├── requirements.txt       # 패키지 의존성
├── uploads/               # 업로드된 비디오 저장 폴더
├── frames/                # 라벨링용 프레임 폴더
├── labels/                # YOLO 포맷 라벨 폴더
└── datasets/
    └── my_f1.yaml         # 커스텀 데이터셋 설정
```
---

## ⚡ 성능 및 최적화

### 하드웨어 사양
- **GPU**: NVIDIA RTX 4070 (12GB VRAM)
- **RAM**: 32GB
- **처리 속도**: 실시간 (30fps)

### 최적화 전략

#### 1. 해상도 다운스케일
```python
frame = cv2.resize(frame, (960, 540))
```
- 원본 1920×1080 → 960×540 (픽셀 수 1/4)
- 메모리 사용량 및 추론 시간 대폭 감소

#### 2. 배치 크기 조정
- **추론**: `batch=1` (실시간 처리)
- **학습**: `batch=12` (메모리 최적화)

#### 3. 신뢰도 임계값 최적화
- `conf=0.2`: 멀리 있는 작은 차량도 검출
- `iou=0.4`: 중복 박스 효과적 제거

---

## 🚧 한계 및 개선 방향

### 현재 한계

#### 1. 하드웨어 제약
- **RTX 4070 (12GB VRAM)** 으로 인한 제한
  - 더 큰 모델(YOLO11-XXL, 앙상블) 사용 불가
  - 학습 해상도 960×960으로 제한
  - 배치 크기 12 이하로 제한

#### 2. 정확도 한계
- 픽셀 기반 속도·거리 추정의 오차
- 카메라 각도 및 트랙 레이아웃 변화에 민감
- 다양한 환경(날씨, 조명, 트랙)에 대한 일반화 부족

#### 3. 데이터 제약
- 제한된 학습 데이터셋
- 멀리 있는 작은 차량 검출 정확도 낮음
- 가림(occlusion) 상황 처리 미흡

### 개선 방향

#### 하드웨어 업그레이드 시
- **RTX 4090 (24GB)** 또는 **RTX 3090 (24GB)** 사용
  - 고해상도 학습 (1920×1080)
  - 더 큰 배치 크기 (32+)
  - 복잡한 모델 및 앙상블 적용 가능
  - **더 정확한 판단 모델 구축 가능**

#### 알고리즘 개선
- 카메라 캘리브레이션 및 3D 재구성
- GPS/IMU 센서 데이터 융합
- ReID 기능 추가 (장시간 가림 대응)
- 트랙별 맞춤형 매핑 테이블

#### 데이터 확장
- 공개 데이터셋 활용 (Roboflow Universe 등)
- 데이터 증강 (회전, 밝기, 노이즈)
- 다양한 환경 데이터 수집

---

## 📊 예시 결과

### 입력 영상
- **형식**: MP4, AVI 등
- **해상도**: 960×540 이상 권장
- **FPS**: 30fps 권장

### 출력 데이터

#### CSV 텔레메트리 예시
```csv
frame,timestamp,ego_speed,lead_speed,distance,rel_dist,overtake_score,overtake_status
0,0.00,120.5,115.3,25.4,0.18,65.2,CAUTION
1,0.03,121.2,115.1,24.8,0.17,68.7,CAUTION
2,0.07,122.0,114.9,24.1,0.16,72.3,POSSIBLE
```

---

## 감사의 말

이 프로젝트는 다음 오픈소스 라이브러리를 활용했습니다:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Norfair](https://github.com/tryolabs/norfair)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/)

---

## 라이선스

이 프로젝트는 개인 학습 및 연구 목적으로 제작되었습니다.

---
