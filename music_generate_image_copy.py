# %%
# 커널과 동일한 파이썬에 설치됨 (권장)
# %pip install --upgrade pip
# %pip install --index-url https://download.pytorch.org/whl/cpu torch
# %pip install transformers pillow


# 커널과 동일한 파이썬에 설치 (권장)
# %pip install --upgrade pip
# %pip install transformers pillow


# %%
# %%
# === 0) 환경 로드 ===
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

import os
print("OPENAI ok:", bool(os.getenv("OPENAI_API_KEY")))

# %%
# %%
# === 1) 기본 임포트/스키마 ===
from typing import List, Dict, Literal, TypedDict, Optional
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

Valence = Annotated[float, Field(ge=-1.0, le=1.0)]
Arousal = Annotated[float, Field(ge=0.0, le=1.0)]
BPM     = Annotated[int,   Field(ge=50,  le=140)]
DurSec  = Annotated[int,   Field(ge=120, le=180)]

class EmotionResult(BaseModel):
    primary: str
    valence: Valence = 0.0
    arousal: Arousal = 0.5
    confidence: Arousal = 0.7
    reasons: str = "—"

class MusicBrief(BaseModel):
    mood: str
    bpm: BPM = 90
    key: str
    duration_sec: DurSec = 120
    instruments: List[str] = []
    style_tags: List[str] = []
    prompt: str  # 영어 프롬프트

class GraphState(TypedDict, total=False):
    image_path: str
    emotion: EmotionResult
    brief: MusicBrief
    meta: Dict  # 필요 시 디버그 정보

def get_llm(temp=0.2):
    return ChatOpenAI(model="gpt-4o-mini", temperature=temp)

def dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


# %%
# %%
# === 2) CLIP 로드 & 분석 ===
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

_CLIP_MODEL: Optional[CLIPModel] = None
_CLIP_PROC: Optional[CLIPProcessor] = None

def load_clip_once():
    global _CLIP_MODEL, _CLIP_PROC
    if _CLIP_MODEL is None or _CLIP_PROC is None:
        print("CLIP 모델 로드 중... (최초 1회)")
        _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_PROC  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ CLIP 로딩 완료")
    return _CLIP_MODEL, _CLIP_PROC

# 후보 문장과 valence/arousal 힌트
CLIP_MOODS = [
    ("A happy and joyful photo",          dict(valence=+0.7, arousal=0.6,  primary="joy")),
    ("A calm and peaceful photo",         dict(valence=+0.4, arousal=0.35, primary="calm")),
    ("An energetic and vibrant photo",    dict(valence=+0.5, arousal=0.75, primary="energy")),
    ("A mysterious and dreamy photo",     dict(valence=+0.2, arousal=0.5,  primary="mystery")),
    ("A sad and melancholic photo",       dict(valence=-0.6, arousal=0.45, primary="sadness")),
    ("A lonely and gloomy photo",         dict(valence=-0.5, arousal=0.4,  primary="lonely")),
]

# 영어 라벨 → 한국어 표현
KO_LABEL = {
    "A happy and joyful photo":       "행복하고 기쁜",
    "A calm and peaceful photo":      "차분하고 평화로운",
    "An energetic and vibrant photo": "활기차고 생동감 있는",
    "A mysterious and dreamy photo":  "신비롭고 몽환적인",
    "A sad and melancholic photo":    "슬프고 우울한",
    "A lonely and gloomy photo":      "외롭고 침울한",
}

# primary → 한국어 감정명
KO_PRIMARY = {
    "joy": "기쁨",
    "calm": "차분",
    "energy": "활기",
    "mystery": "신비",
    "sadness": "슬픔",
    "lonely": "외로움",
}

def _r2(x: float) -> float:
    """소수점 2자리로 고정된 float"""
    return float(f"{x:.2f}")

def _pct(p: float) -> int:
    """확률(0~1)을 % 정수로"""
    return int(round(p * 100))

def _make_korean_reason(ranked, primary: str, valence: float, arousal: float, confidence: float) -> str:
    """상위 후보들을 한국어로 풀어서 자연스러운 이유문 생성"""
    top = ranked[0]
    top_ko = KO_LABEL.get(top["label"], top["label"])
    if len(ranked) >= 2:
        second = ranked[1]
        sec_ko = KO_LABEL.get(second["label"], second["label"])
        return (
            f"이미지에서 {top_ko} 분위기가 {_pct(top['score'])}%로 가장 강하고, "
            f"{sec_ko} 분위기가 {_pct(second['score'])}%로 뒤따랐습니다. "
            f"두 특징을 반영해 정서를 ‘{KO_PRIMARY.get(primary, primary)}’ 쪽으로 추정했으며, "
            f"쾌-불쾌(valence) {_r2(valence)}, 각성도(arousal) {_r2(arousal)}, 신뢰도 {_r2(confidence)}로 산정했습니다."
        )
    else:
        return (
            f"이미지에서 {top_ko} 분위기가 두드러졌습니다({_pct(top['score'])}%). "
            f"이에 따라 정서를 ‘{KO_PRIMARY.get(primary, primary)}’로 보고, "
            f"valence {_r2(valence)}, arousal {_r2(arousal)}, 신뢰도 {_r2(confidence)}로 산정했습니다."
        )


KEYWORD_MAP = {
    "happy and joyful": ["기쁨","행복"],
    "calm and peaceful": ["차분","평화"],
    "energetic and vibrant": ["활기","생동감"],
    "mysterious and dreamy": ["신비","몽환"],
    "sad and melancholic": ["슬픔","우울"],
    "lonely and gloomy": ["외로움","침울"],
}

def analyze_image_clip(image_path: str):
    model, proc = load_clip_once()
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    texts = [t for (t, _) in CLIP_MOODS]
    inputs = proc(text=texts, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
        probs = out.logits_per_image.softmax(dim=1)[0]  # shape [len(texts)]

    results = []
    for i, (label, hints) in enumerate(CLIP_MOODS):
        p = float(probs[i].item())
        results.append({
            "label": label,
            "score": p,
            "hints": hints,  # valence/arousal/primary
            "ko_keywords": next((v for k, v in KEYWORD_MAP.items() if k in label.lower()), []),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# %%
# %%
# === 3) 이미지 감정 분석 노드 ===
def _r2(x: float) -> float:
    return float(f"{x:.2f}")

def analyze_emotion_from_image_node(state: GraphState) -> GraphState:
    ranked = analyze_image_clip(state["image_path"])
    top = ranked[0]

    # 수치 계산
    if len(ranked) >= 2:
        second = ranked[1]
        w1, w2 = top["score"], second["score"]
        s = (w1 + w2) or 1.0
        valence = (top["hints"]["valence"] * w1 + second["hints"]["valence"] * w2) / s
        arousal = (top["hints"]["arousal"] * w1 + second["hints"]["arousal"] * w2) / s
        primary = top["hints"]["primary"]
        confidence = min(0.9, max(0.5, w1))
    else:
        valence = top["hints"]["valence"]
        arousal = top["hints"]["arousal"]
        primary = top["hints"]["primary"]
        confidence = min(0.9, max(0.5, top["score"]))

    # ✅ 이유문을 먼저 완성(변수로) — 괄호/쉼표 이슈 방지
    reasons = _make_korean_reason(
        ranked=ranked,
        primary=primary,
        valence=valence,
        arousal=arousal,
        confidence=confidence,
    )

    # ✅ 소수점 2자리 고정 후 저장
    state["emotion"] = EmotionResult(
        primary=primary,
        valence=_r2(max(-1.0, min(1.0, valence))),
        arousal=_r2(max(0.0, min(1.0, arousal))),
        confidence=_r2(max(0.0, min(1.0, confidence))),
        reasons=reasons,
    )

    state["meta"] = {"clip_top5": ranked[:5]}
    return state


# %%
# %%
# === 4) 음악 브리프 노드 ===
def compose_brief_node(state: GraphState) -> GraphState:
    llm = get_llm(0.6)
    emo: EmotionResult = state["emotion"]
    sys = (
        "너는 음악 감독이다. 아래 감정 분석을 참고해 "
        "예술치료/심리 안정 목적의 짧은 BGM을 위한 Music Brief를 JSON으로 만들어라.\n\n"
        "## 치료적 목표(필수)\n"
        "- 사용자의 현재 상태를 '공조절(co-regulation)' 관점에서 보정한다.\n"
        "- 정서 조절 전략(regulation_mode)을 다음에서 고른다: "
        "  soothe(불안·고각성 완화), uplift(우울·저각성 부드럽게 상승), "
        "  sustain(편안한 긍정 유지), ground(과도한 긍정/흥분을 안정적으로 접지).\n"
        "- 선택한 전략은 style_tags에 'regulate:<mode>' 형태로 반드시 포함한다.\n\n"
        "## 파라미터 규칙\n"
        "1) bpm: 50~140, duration_sec: 120~180.\n"
        "   - arousal↑ → bpm↑. 단 soothe/ground는 70~100.\n"
        "   - uplift(저각성·우울)는 72~90.\n"
        "2) duration_sec: 120~180. 불안(arousal>0.6) 또는 우울(valence<-0.2)은 150~180 우선.\n"
        "3) key: valence>=0.2 → 메이저 / valence<=-0.2 → 마이너 / 중립 혼합 가능.\n"
        "4) instruments: 2~4개. warm piano, soft pad 기본.\n"
        "5) style_tags: 3~6개. 예: calming, minimal, warm, ambient, breathing, focus, regulate:<mode>.\n"
        "6) 구조(권장): intro(8bar) → body(16~24bar) → outro(4~6bar), 루프 안전.\n"
        "7) 안전 가드: 과도한 트랜지언트/왜곡/금속성 심벌 금지. 다이내믹 soft~medium.\n"
        "8) prompt: 영어 한 문장, 18~25단어. 숫자(BPM/key/duration) 금지. 이미지 분위기 단서를 뉘앙스로만 반영.\n"
        "9) 오직 JSON만 출력."
    )
    usr = (
        f"# Emotion from Image\n"
        f"primary={emo.primary}, valence={emo.valence}, arousal={emo.arousal}, confidence={emo.confidence}\n"
        f"reason={emo.reasons}\n"
    )
    structured = llm.with_structured_output(MusicBrief)
    brief = structured.invoke([
        {"role":"system","content":sys},
        {"role":"user","content":usr}
    ])

    # 길이 보정
    if brief.duration_sec < 120:
        brief = brief.model_copy(update={"duration_sec": 120})
    elif brief.duration_sec > 180:
        brief = brief.model_copy(update={"duration_sec": 180})

    state["brief"] = brief
    return state


# %%
# %%
# === 5) 그래프 구성 ===
workflow = StateGraph(GraphState)
workflow.add_node("analyze_emotion_from_image", analyze_emotion_from_image_node)
workflow.add_node("compose_brief",  compose_brief_node)

workflow.add_edge(START, "analyze_emotion_from_image")
workflow.add_edge("analyze_emotion_from_image", "compose_brief")
workflow.add_edge("compose_brief", END)

graph = workflow.compile()

# %%
# %%
# === 6) 실행 예시 (단일 이미지) ===
def dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj

# state = {
#     "image_path": "images/local_stitch_terrarosa.jpg",  # ← 네 이미지 경로로 교체
# }
# final = graph.invoke(state)

# print("=== Emotion ===")
# print(dump(final["emotion"]))
# print("\n=== Music Brief ===")
# print(dump(final["brief"]))
# print("\n=== CLIP Top-5 (디버그) ===")
# for r in final["meta"]["clip_top5"]:
#     print(f"- {r['label']} : {r['score']:.3f}")