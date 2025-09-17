# 🎵 치료용 음악 생성 AI (Therapeutic Music Generation AI)

개인의 감정과 스토리를 분석하여 예술치료 및 심리안정을 위한 맞춤형 음악을 생성하는 AI 시스템입니다.

## ✨ 주요 기능

- **감정 분석**: 사용자의 텍스트에서 감정 상태를 분석 (valence, arousal, confidence)
- **치료적 음악 브리프 생성**: 감정 상태에 맞는 음악 파라미터 설계
- **AI 음악 생성**: Stable Audio 2.5를 사용한 실제 음악 파일 생성
- **치료적 조절 전략**: soothe, uplift, sustain, ground 모드 지원
- **사용자 친화적 인터페이스**: Gradio 기반 웹 인터페이스
- **다운로드 기능**: 생성된 음악 파일 다운로드 지원

## 🚀 빠른 시작

### 1. 환경 설정

\`\`\`bash
# 의존성 설치
pip install -r scripts/requirements.txt

# 환경 변수 설정
python scripts/setup_env.py
\`\`\`

### 2. API 키 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

\`\`\`env
# OpenAI API Key (필수 - 감정 분석용)
OPENAI_API_KEY=your_openai_api_key_here

# Replicate API Token (음악 생성용)
REPLICATE_API_TOKEN=your_replicate_api_token_here

# 음악 생성 활성화 (1: 활성화, 0: 분석만)
USE_REPLICATE=0
\`\`\`

### 3. 앱 실행

\`\`\`bash
python scripts/run_app.py
\`\`\`

앱이 실행되면 다음 주소에서 접근할 수 있습니다:
- 로컬: http://localhost:7860
- 공개 링크: 자동 생성됨

## 🎯 사용법

### 기본 사용법

1. **이야기 입력**: 감정이나 상황을 자유롭게 텍스트로 입력
2. **분석 선택**:
   - **감정 분석하기**: 무료로 감정 분석과 음악 브리프만 생성
   - **음악 생성하기**: 실제 음악 파일 생성 (Replicate 크레딧 사용)
3. **결과 확인**: 감정 분석 결과, 음악 브리프, 생성된 음악 확인
4. **다운로드**: 생성된 음악 파일 다운로드

### 입력 예시

\`\`\`
오늘 하루 종일 마음이 무거웠다. 일정을 정리하다가 페이지를 넘기는 손이 자주 멈췄다. 
시간이 흐르는 게 잘 느껴지지 않았다.
\`\`\`

## 🧠 AI 시스템 구조

### 1. 감정 분석 (Emotion Analysis)
- **모델**: GPT-4o-mini
- **출력**: 주요 감정, valence(-1~1), arousal(0~1), 신뢰도
- **목적**: 사용자의 정서 상태 정확한 파악

### 2. 음악 브리프 생성 (Music Brief Generation)
- **치료적 조절 전략**:
  - `soothe`: 불안·고각성 완화
  - `uplift`: 우울·저각성 부드럽게 상승
  - `sustain`: 편안한 긍정 유지
  - `ground`: 과도한 긍정/흥분을 안정적으로 접지
- **파라미터**: BPM(50-140), 조성, 악기, 스타일, 길이(60-90초)

### 3. 음악 생성 (Music Generation)
- **모델**: Stability AI Stable Audio 2.5
- **플랫폼**: Replicate API
- **출력**: 고품질 오디오 파일 (WAV/MP3)

## 💡 치료적 특징

- **개인화**: 각 사용자의 감정 상태에 맞춤형 음악
- **안전성**: 과자극 방지, 부드러운 다이내믹 범위
- **치료적 목적**: 정서 조절 및 심리적 안정 지원
- **과학적 근거**: Valence-Arousal 모델 기반 감정 분석

## ⚠️ 주의사항

- 이 도구는 전문적인 심리치료를 대체하지 않습니다
- 심각한 정신건강 문제가 있다면 전문가의 도움을 받으시기 바랍니다
- 생성된 음악은 개인적 용도로만 사용해주세요
- Replicate API 사용 시 크레딧이 소모됩니다

## 🔧 기술 스택

- **Backend**: Python, LangChain, LangGraph
- **AI Models**: OpenAI GPT-4o-mini, Stability AI Stable Audio 2.5
- **Frontend**: Gradio
- **APIs**: OpenAI API, Replicate API
- **Data**: Pydantic, JSON

## 📁 프로젝트 구조

\`\`\`
scripts/
├── music_generator_core.py    # 핵심 음악 생성 로직
├── gradio_app.py             # Gradio 웹 인터페이스
├── run_app.py                # 메인 실행 스크립트
├── setup_env.py              # 환경 설정 도우미
├── requirements.txt          # Python 의존성
└── README.md                 # 프로젝트 문서

outputs/                      # 생성된 음악 파일 저장
.env                         # 환경 변수 (사용자가 생성)
\`\`\`

## 🤝 기여하기

이 프로젝트는 헤커톤 프로젝트로 시작되었습니다. 개선 사항이나 버그 리포트는 언제든 환영합니다!

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.
