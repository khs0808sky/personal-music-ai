# %%
# === 0) 환경 로드 ===
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

import os, time, requests
print("OPENAI ok:", bool(os.getenv("OPENAI_API_KEY")))
print("REPLICATE ok:", bool(os.getenv("REPLICATE_API_TOKEN")))
print("USE_REPLICATE:", os.getenv("USE_REPLICATE"))  # "1"일 때만 실제 생성

# %%
# === 1) 기본 임포트/스키마 ===
from typing import List, Dict, Literal, TypedDict
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from urllib.parse import urlparse

# replicate는 "선택 임포트"
try:
    import replicate  # 있으면 사용
    _HAVE_REPLICATE = True
except ModuleNotFoundError:
    replicate = None
    _HAVE_REPLICATE = False

# %%
Valence = Annotated[float, Field(ge=-1.0, le=1.0)]
Arousal = Annotated[float, Field(ge=0.0, le=1.0)]
BPM     = Annotated[int,   Field(ge=50,  le=140)]
DurSec  = Annotated[int,   Field(ge=120, le=180)]  # 120~180초

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
    user_text: str
    emotion: EmotionResult
    brief: MusicBrief
    audio_path: str
    provider_used: Literal["replicate","rest","skipped"]
    meta: Dict

# %%
# === 2) LLM 헬퍼 ===
def get_llm(temp=0.2):
    return ChatOpenAI(model="gpt-4o-mini", temperature=temp)

# %%
# === 3) 감정 분석 노드 ===
def analyze_emotion_node(state: GraphState) -> GraphState:
    llm = get_llm(0.2)
    sys = (
        "당신은 심리 정서를 요약하는 분석가입니다. "
        "사용자 텍스트에서 주요 감정을 한 단어(또는 짧은 구)로 도출하고, "
        "valence(-1~1), arousal(0~1), confidence(0~1)을 추정하세요. "
        "사용자 텍스트는 서술형일 수 있으며 직접적 요청이 없을 수 있다. "
        "장면·행동·몸의 단서만으로 valence/arousal을 추정하라.\n"
        "반드시 EmotionResult(JSON 스키마)에 맞춰 응답하세요."
    )
    structured = llm.with_structured_output(EmotionResult)
    result = structured.invoke([
        {"role":"system","content":sys},
        {"role":"user","content":state["user_text"]}
    ])
    state["emotion"] = result
    return state

# %%
# === 4) 음악 브리프 노드 ===
def compose_brief_node(state: GraphState) -> GraphState:
    llm = get_llm(0.6)  # 브리프 다양성 조금 ↑
    emo: EmotionResult = state["emotion"]
    sys = (
        "너는 음악 감독이다. 아래 감정 분석과 사용자 텍스트를 참고해 "
        "개인의 감정과 스토리를 반영한, 예술치료/심리 안정 목적의 짧은 BGM을 위한 "
        "Music Brief를 JSON으로 만들어라.\n\n"
        "## 치료적 목표(필수)\n"
        "- 사용자의 현재 상태를 '공조절(co-regulation)' 관점에서 보정한다.\n"
        "- 정서 조절 전략(regulation_mode)을 다음에서 고른다: "
        "  soothe(불안·고각성 완화), uplift(우울·저각성 부드럽게 상승), "
        "  sustain(편안한 긍정 유지), ground(과도한 긍정/흥분을 안정적으로 접지).\n"
        "- 선택한 전략은 style_tags에 'regulate:<mode>' 형태로 반드시 포함한다.\n\n"
        "## 파라미터 규칙\n"
        "1) bpm: 50~140 중 선택하되, duration_sec은 60~90으로 제한한다.\n"
        "   - arousal↑ → bpm↑ 경향. 단, soothe/ground 전략일 때는 중간 템포(70~100)로 과자극 방지.\n"
        "   - uplift 전략(저각성·우울)일 땐 72~90 범위에서 부드럽게 추진.\n"
        "2) duration_sec: 120~180. 불안(arousal>0.6) 또는 우울(valence<-0.2)은 150~180을 우선 고려.\n"
        "3) key: valence>=0.2 → 메이저(C/G/F/D 등), valence<=-0.2 → 마이너(A/D/E/B 등), "
        "   중립은 혼합 가능. 동일 키('C major')만 반복 사용 금지.\n"
        "4) instruments: 2~4개. 기본은 warm piano, soft pad.\n"
        "   - soothe/ground: light percussion는 있어도 아주 절제(brush, soft tick 등), 하이햇/킥 과도 금지.\n"
        "   - uplift: strings(legato)나 gentle pulse로 미세한 전진감.\n"
        "5) style_tags: 3~6개. 예: calming, minimal, warm, ambient, breathing, focus, regulate:<mode>.\n"
        "6) 구조(권장): 120~180초 안에 intro(짧은 페이드인, 8bar) → body(점진적 레이어, 16~24bar) → "
        "   outro(4~6bar, 3~4초 페이드아웃). 루프 안전(loop-safe) 문장감 유지.\n"
        "7) 안전 가드: 과도한 트랜지언트/왜곡/사이드체인 펌핑/금속성 심벌/저역 과출력 금지. "
        "   다이내믹은 soft~medium.\n"
        "8) prompt: 영어 한 문장, 18~25단어. 악기·무드·질감·다이내믹을 서술하되 숫자(BPM/key/duration) 금지. "
        "   사용자의 텍스트에서 핵심 단어 1~2개를 분위기 단서로 녹여라(직역 금지, 뉘앙스만 반영).\n"
        "9) JSON만 출력. 추가 설명 금지.\n"
        "사용자 텍스트에 요청이 없어도 valence/arousal로 regulation_mode(soothe/uplift/sustain/ground)를 결정하고 "
        "style_tags에 'regulate:<mode>'를 포함하라."
    )

    usr = (
        f"# Emotion\nprimary={emo.primary}, valence={emo.valence}, "
        f"arousal={emo.arousal}, confidence={emo.confidence}\n\n"
        f"# Text\n{state['user_text']}\n"
    )
    structured = llm.with_structured_output(MusicBrief)
    brief = structured.invoke([
        {"role":"system","content":sys},
        {"role":"user","content":usr}
    ])

    # duration 보정 (120~180초로 강제)
    if brief.duration_sec < 120:
        brief = brief.model_copy(update={"duration_sec": 120})
    elif brief.duration_sec > 180:
        brief = brief.model_copy(update={"duration_sec": 180})

    state["brief"] = brief
    return state

# %%
# === 5) Replicate 호출 유틸 (클라/REST 자동 폴백) ===
MODEL_ID = "stability-ai/stable-audio-2.5:46a2601577d0e31aa99b03c9d7fd2142fa3b96a282338758f794b620e35c75b7"
_MODEL_VERSION = MODEL_ID.split(":")[1]  # REST용 버전 해시

def _replicate_run(input_payload: dict, token: str):
    # 패키지 있으면 우선 사용
    if _HAVE_REPLICATE:
        return replicate.run(MODEL_ID, input=input_payload)
    # 없으면 REST 폴백 (설치 불필요, 크레딧 정상 차감)
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    create = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers,
        json={"version": _MODEL_VERSION, "input": input_payload},
        timeout=30,
    )
    create.raise_for_status()
    pred = create.json()
    pid = pred["id"]
    while pred["status"] not in ("succeeded", "failed", "canceled"):
        time.sleep(2)
        r = requests.get(f"https://api.replicate.com/v1/predictions/{pid}", headers=headers, timeout=30)
        r.raise_for_status()
        pred = r.json()
    if pred["status"] != "succeeded":
        raise RuntimeError(f"Replicate failed: {pred['status']} / {pred.get('error')}")
    return pred["output"]  # 보통 URL 리스트

def _save_first_output_to_file(out) -> str:
    """Replicate 출력(any) → 파일 저장하고 경로 반환"""
    first = out[0] if isinstance(out, (list, tuple)) else out
    os.makedirs("outputs", exist_ok=True)
    ts = int(time.time())

    # FileOutput (replicate 패키지 경로)
    if hasattr(first, "read"):
        ext = ".bin"
        url_attr = getattr(first, "url", None)
        if isinstance(url_attr, str):
            ext_candidate = os.path.splitext(urlparse(url_attr).path)[1].lower()
            if ext_candidate:
                ext = ext_candidate
        out_path = f"outputs/stableaudio_{ts}{ext}"
        with open(out_path, "wb") as f:
            f.write(first.read())
        return out_path

    # URL 문자열
    if isinstance(first, str):
        r = requests.get(first, timeout=120); r.raise_for_status()
        ct = (r.headers.get("Content-Type") or "").lower()
        if "wav" in ct:
            ext = ".wav"
        elif "mpeg" in ct or "mp3" in ct:
            ext = ".mp3"
        else:
            ext = os.path.splitext(urlparse(first).path)[1] or ".bin"
        out_path = f"outputs/stableaudio_{ts}{ext}"
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path

    # dict 형태 (드묾)
    if isinstance(first, dict):
        url = first.get("url") or first.get("audio") or first.get("output")
        if isinstance(url, str):
            r = requests.get(url, timeout=120); r.raise_for_status()
            ext = os.path.splitext(urlparse(url).path)[1] or ".bin"
            out_path = f"outputs/stableaudio_{ts}{ext}"
            with open(out_path, "wb") as f:
                f.write(r.content)
            return out_path

    raise RuntimeError(f"Unexpected replicate output type: {type(first)}")

def generate_with_replicate_strict(prompt: str, duration: int) -> str:
    tok = os.getenv("REPLICATE_API_TOKEN")
    assert tok, "REPLICATE_API_TOKEN이 없습니다 (.env 확인)"
    assert 120 <= int(duration) <= 180, f"duration(초)은 120~180 범위여야 합니다: {duration}"
    out = _replicate_run({"prompt": prompt, "duration": int(duration)}, tok)
    return _save_first_output_to_file(out)

# %%
# === 6) 생성 노드 ===
def generate_music_node(state: GraphState) -> GraphState:
    brief: MusicBrief = state["brief"]
    path = generate_with_replicate_strict(brief.prompt, int(brief.duration_sec))
    state["audio_path"] = path
    state["provider_used"] = "replicate" if _HAVE_REPLICATE else "rest"
    state["meta"] = {
        "emotion": state["emotion"].model_dump(),
        "brief": state["brief"].model_dump(),
        "provider": state["provider_used"],
        "path": path,
    }
    return state

# %%
# === 7) 토글 분기: 기본은 스킵(크레딧 0), 원할 때만 생성 ===
def should_generate(state: GraphState) -> str:
    use_flag = os.getenv("USE_REPLICATE", "0") == "1" or state.get("force_generate") is True
    has_token = bool(os.getenv("REPLICATE_API_TOKEN"))
    return "go" if (use_flag and has_token) else "skip"

def mark_skipped(state: GraphState) -> GraphState:
    state["provider_used"] = "skipped"
    return state

# %%
# === 8) 그래프 구성 ===
workflow = StateGraph(GraphState)
workflow.add_node("analyze_emotion", analyze_emotion_node)
workflow.add_node("compose_brief",  compose_brief_node)
workflow.add_node("generate_music", generate_music_node)
workflow.add_node("mark_skipped",   mark_skipped)

workflow.add_edge(START, "analyze_emotion")
workflow.add_edge("analyze_emotion", "compose_brief")
workflow.add_conditional_edges(
    "compose_brief",
    should_generate,
    {"go": "generate_music", "skip": "mark_skipped"}
)
workflow.add_edge("generate_music", END)
workflow.add_edge("mark_skipped", END)

graph = workflow.compile()

# %%
# === 9) 실행 예시 ===
# state = {
#     "user_text": "일정을 정리하다가 페이지를 넘기는 손이 자주 멈췄다. 시간이 흐르는 게 잘 느껴지지 않았다."
#     # "force_generate": True  # ← 정말 생성할 때만 켜기 (크레딧 사용)
# }
# final = graph.invoke(state)

def dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj

# print("=== Emotion ===")
# print(dump(final["emotion"]))
# print("\n=== Music Brief ===")
# print(dump(final["brief"]))
# print("\n=== Provider Used ===")
# print(final.get("provider_used", "skipped"))
# print("\n=== Audio Path ===")
# print(final.get("audio_path"))  # 생성 시에만 경로가 생김
