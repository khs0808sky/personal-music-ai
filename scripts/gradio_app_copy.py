import gradio as gr
import os
import json
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import re

# Import the core music generation functionality
from music_generator_core import (
    graph, GraphState, EmotionResult, MusicBrief,
    dump, generate_with_replicate_strict,
    analyze_emotion_node, compose_brief_node  # Import new nodes
)

from music_generate_image import (
    graph as image_graph, 
    GraphState as ImageGraphState,
    analyze_emotion_from_image_node,
    compose_brief_node as image_compose_brief_node
)

def _safe_filename(s: str, max_len: int = 80) -> str:
    # 한글/영문/숫자/공백/일부 기호만 허용 → 나머지는 _
    s = s.strip()
    s = re.sub(r"[^0-9A-Za-z가-힣 _\-\(\)\[\]\.]", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]

def _label_with_ext(prefix: str, path: str) -> str:
    return f"{prefix} — {os.path.basename(path)}"


def _title_from_brief(brief: MusicBrief) -> str:
    # 예: "Joyful - C major - 130bpm - regulate:uplift"
    mode = next((t for t in (brief.style_tags or []) if t.startswith("regulate:")), None)
    parts = [brief.mood, brief.key, f"{brief.bpm}bpm"]
    if mode: parts.append(mode)
    return " - ".join([p for p in parts if p])

def _rename_generated_file(src_path: str, brief: MusicBrief) -> tuple[str, str]:
    """
    생성된 파일을 '제목_YYYYMMDD_HHMMSS.ext'로 rename.
    - 파일명: OS 호환 위해 안전한 문자만 사용 (':' 같은 건 '_'로)
    - 라벨(표시용 제목): 콜론 그대로 유지 + 뒤에 '_YYYYMMDD_HHMMSS'
    반환: (새 경로, 표시용 제목)
    """
    base_title = _title_from_brief(brief) or "Therapeutic Music"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 파일명용(안전 문자만)
    safe_base = _safe_filename(base_title)   # ':' → '_'로 치환됨
    root, ext = os.path.splitext(src_path)
    dst_name = f"{safe_base}_{ts}{ext or '.wav'}"
    dst = os.path.join(os.path.dirname(src_path), dst_name)
    shutil.move(src_path, dst)

    # 표시용 제목(콜론 유지) → "…regulate:uplift_20250918_174045"
    display_title = f"{base_title}_{ts}"
    return dst, display_title

# === 상단에 유틸 추가 ===
def _norm(s: str) -> str:
    return " ".join((s or "").split()).strip()


def get_usage_md():
    return """### 📘 사용법
- **🔍 감정 분석**: 감정 상태만 분석
- **📋 감정 분석 + 음악 설계**: 감정 분석 + 음악 브리프 생성
- **🎵 음악 생성**: 현재 설계안 그대로 실제 음악 생성
"""

def get_tips_md():
    return """### 💡 팁
- 설계안이 마음에 들면 **음악 생성**을 누르면 그 설계안으로 음악이 만들어져요.
- **같은 스토리/이미지**면 이전 감정분석을 재사용해요.
- 스토리나 이미지를 바꾸면 이전 결과는 초기화돼요.
"""


def on_text_tab_select():
    # 지금부터 텍스트 모드
    app_state.current_mode = "text"
    # 반대편 입력(이미지)만 비우고, 결과 초기화
    app_state.image_path = None
    app_state.emotion_result = None
    app_state.music_brief = None
    return (
        gr.update(),             # story_input: 유지 (바꾸지 않음)
        gr.update(value=None),   # image_input: 비우기
        "",                      # emotion_output
        "",                      # brief_output
        None,                    # audio_output
        "📝 텍스트 탭으로 전환: 이전 이미지 입력과 결과를 초기화했습니다."
    )

def on_image_tab_select():
    # 지금부터 이미지 모드
    app_state.current_mode = "image"
    # 반대편 입력(텍스트)만 비우고, 결과 초기화
    app_state.user_story = None
    app_state.emotion_result = None
    app_state.music_brief = None
    return (
        gr.update(value=""),     # story_input: 비우기
        gr.update(),             # image_input: 유지 (바꾸지 않음)
        "",                      # emotion_output
        "",                      # brief_output
        None,                    # audio_output
        "🖼️ 이미지 탭으로 전환: 이전 텍스트 입력과 결과를 초기화했습니다."
    )



class AppState:
    def __init__(self):
        self.emotion_result = None
        self.music_brief = None
        self.user_story = None
        self.image_path = None
        self.current_mode = "text"  # "text" or "image"

    def clear(self):
        self.emotion_result = None
        self.music_brief = None
        self.user_story = None
        self.image_path = None
        self.current_mode = "text"

    def set_story(self, s: str):
        self.user_story = _norm(s)
        self.current_mode = "text"

    def set_image(self, img_path: str):
        self.image_path = img_path
        self.current_mode = "image"

    def has_emotion_analysis(self, current_story=None, current_image=None):
        if self.current_mode == "text":
            return (self.emotion_result is not None and 
                   self.user_story == _norm(current_story or ""))
        else:  # image mode
            return (self.emotion_result is not None and 
                   self.image_path == current_image)

    def has_music_brief(self, current_story=None, current_image=None):
        if self.current_mode == "text":
            return (self.music_brief is not None and 
                   self.user_story == _norm(current_story or ""))
        else:  # image mode
            return (self.music_brief is not None and 
                   self.image_path == current_image)

# 반복되는 코드 함수로 정의
def _join(items):
    return ", ".join(items) if items else "—"

def md_emotion(emo: EmotionResult) -> str:
    return f"""**🎭 주요 감정**: {emo.primary}

**📊 감정 강도 (Valence)**: {emo.valence:.2f}
*(-1: 매우 부정적 ↔ +1: 매우 긍정적)*

**⚡ 각성도 (Arousal)**: {emo.arousal:.2f}
*(0: 차분함 ↔ 1: 흥분됨)*

**🎯 신뢰도**: {emo.confidence:.2f}

**💭 분석 근거**:
{emo.reasons}"""

def md_brief(brief: MusicBrief) -> str:
    return f"""**🎵 음악 분위기**: {brief.mood}

**🥁 BPM**: {brief.bpm}

**🎼 조성**: {brief.key}

**⏱️ 길이**: {brief.duration_sec}초

**🎹 악기**: {_join(brief.instruments)}

**🏷️ 스타일 태그**: {_join(brief.style_tags)}

**📝 생성 프롬프트**:
{brief.prompt}"""



# Global app state instance
app_state = AppState()

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    def analyze_emotion_only(user_story, image_input):
        """
        Only analyze emotion without generating music
        """
        try:
            if image_input is not None:
                # Image mode
                if app_state.image_path != image_input:
                    app_state.clear()
                    app_state.set_image(image_input)
                
                if app_state.has_emotion_analysis(current_image=image_input):
                    emo = app_state.emotion_result
                    emotion_text = md_emotion(emo)
                    return emotion_text, "", None, "♻️ 이전 감정 분석 결과 재사용 (이미지)"
                
                # Run image emotion analysis
                state = {"image_path": image_input}
                state = analyze_emotion_from_image_node(state)
                app_state.emotion_result = state.get("emotion")
                
                emotion_text = md_emotion(app_state.emotion_result)
                status = "✅ 이미지 감정 분석 완료!"
                
            else:
                # Text mode
                if not _norm(user_story):
                    return "입력이 비어 있습니다. 텍스트를 입력하거나 이미지를 업로드해 주세요.", "", None, "⚠️ 입력이 비어 있습니다."

                if app_state.user_story != _norm(user_story):
                    app_state.clear()
                    app_state.set_story(user_story)

                if app_state.has_emotion_analysis(current_story=user_story):
                    emo = app_state.emotion_result
                    emotion_text = md_emotion(emo)
                    return emotion_text, "", None, "♻️ 이전 감정 분석 결과 재사용 (텍스트)"
                
                # Run text emotion analysis
                state = {"user_text": user_story, "force_generate": False}
                state = analyze_emotion_node(state)
                app_state.emotion_result = state.get("emotion")
                
                emotion_text = md_emotion(app_state.emotion_result)
                status = "✅ 텍스트 감정 분석 완료!"
            
            return emotion_text, "", None, status
            
        except Exception as e:
            error_msg = f"❌ 감정 분석 중 오류 발생: {str(e)}"
            return "오류 발생", "", None, error_msg
    
    def generate_music_brief_only(user_story, image_input):
        """
        Analyze emotion and generate music brief without actual music generation
        """
        try:
            if image_input is not None:
                # Image mode
                if app_state.image_path != image_input:
                    app_state.clear()
                    app_state.set_image(image_input)
                
                state = {"image_path": image_input}
                
                if app_state.has_emotion_analysis(current_image=image_input):
                    state["emotion"] = app_state.emotion_result
                else:
                    state = analyze_emotion_from_image_node(state)
                    app_state.emotion_result = state.get("emotion")
                
                state = image_compose_brief_node(state)
                app_state.music_brief = state.get("brief")
                
                status = "✅ 이미지 감정 분석 및 음악 브리프 생성 완료!"
                
            else:
                # Text mode
                if not _norm(user_story):
                    return "입력이 비어 있습니다. 텍스트를 입력하거나 이미지를 업로드해 주세요.", "", None, "⚠️ 입력이 비어 있습니다."
                
                if app_state.user_story != _norm(user_story):
                    app_state.clear()
                    app_state.set_story(user_story)

                state = {"user_text": user_story, "force_generate": False}
                
                if app_state.has_emotion_analysis(current_story=user_story):
                    state["emotion"] = app_state.emotion_result
                else:
                    state = analyze_emotion_node(state)
                    app_state.emotion_result = state.get("emotion")
                
                state = compose_brief_node(state)
                app_state.music_brief = state.get("brief")
                
                status = "✅ 텍스트 감정 분석 및 음악 브리프 생성 완료!"
            
            emotion_text = md_emotion(app_state.emotion_result)
            brief_text = md_brief(app_state.music_brief)
            
            return emotion_text, brief_text, None, status
            
        except Exception as e:
            error_msg = f"❌ 처리 중 오류 발생: {str(e)}"
            return "오류 발생", "오류 발생", None, error_msg
    
    def generate_full_music(user_story, image_input):
        """
        Generate full music from text or image input
        """
        try:
            # 탭 업데이트 기본값(변경 없음)
            tabs_update = gr.update()

            # ★ guard: 텍스트/이미지 둘 다 없는 경우
            if (image_input is None) and (not _norm(user_story)):
                return ("입력이 비어 있습니다. 텍스트를 입력하거나 이미지를 업로드해 주세요.",
                        "", None, "⚠️ 입력이 비어 있습니다.", tabs_update)

            if image_input is not None:
                # Image mode
                if app_state.image_path != image_input:
                    app_state.clear()
                    app_state.set_image(image_input)
            else:
                # Text mode
                if not _norm(user_story):
                    return ("입력이 비어 있습니다. 텍스트를 입력하거나 이미지를 업로드해 주세요.", "", None, "⚠️ 입력이 비어 있습니다.", tabs_update)
                if app_state.user_story != _norm(user_story):
                    app_state.clear()
                    app_state.set_story(user_story)

            # 환경 체크
            openai_ok = bool(os.getenv("OPENAI_API_KEY"))
            repl_ok   = bool(os.getenv("REPLICATE_API_TOKEN"))
            use_repl  = os.getenv("USE_REPLICATE", "0") == "1"
            if not openai_ok:
                return "OpenAI 키가 없습니다.", "", None, "❌ OPENAI_API_KEY 필요", tabs_update
            if not (repl_ok and use_repl):
                # 음악 합성 건너뛰고 분석/브리프만
                if app_state.current_mode == "image":
                    if not app_state.has_emotion_analysis(current_image=image_input):
                        st = {"image_path": image_input}
                        st = analyze_emotion_from_image_node(st)
                        app_state.emotion_result = st["emotion"]
                    if not app_state.has_music_brief(current_image=image_input):
                        st = {"image_path": image_input, "emotion": app_state.emotion_result}
                        st = image_compose_brief_node(st)
                        app_state.music_brief = st["brief"]
                else:
                    if not app_state.has_emotion_analysis(current_story=user_story):
                        st = {"user_text": user_story}
                        st = analyze_emotion_node(st)
                        app_state.emotion_result = st["emotion"]
                    if not app_state.has_music_brief(current_story=user_story):
                        st = {"user_text": user_story, "emotion": app_state.emotion_result}
                        st = compose_brief_node(st)
                        app_state.music_brief = st["brief"]

                emo, brief = app_state.emotion_result, app_state.music_brief
                emotion_text = md_emotion(emo)
                brief_text   = md_brief(brief)
                return emotion_text, brief_text, None, "⚠️ USE_REPLICATE=1 또는 REPLICATE 토큰 없음 → 음악 생성 건너뜀", tabs_update

            # ── 분기 ───────────────────────────────────────────────────────────
            if (app_state.has_emotion_analysis(current_story=user_story, current_image=image_input) and 
                app_state.has_music_brief(current_story=user_story, current_image=image_input)):
                # ▶ 분석/브리프 재사용 → 합성만
                brief = app_state.music_brief
                audio_path = generate_with_replicate_strict(brief.prompt, int(brief.duration_sec))
                audio_path, _ = _rename_generated_file(audio_path, brief)
                audio_ui = gr.update(value=audio_path, label=_label_with_ext("치료용 음악", audio_path))
                status = f"🎵 기존 분석/브리프 재사용하여 음악 생성 ({app_state.current_mode})"
                tabs_update = gr.update(selected=0)

            elif (app_state.has_emotion_analysis(current_story=user_story, current_image=image_input) and 
                not app_state.has_music_brief(current_story=user_story, current_image=image_input)):
                # ▶ 감정만 있음 → 브리프 생성 → 합성
                if app_state.current_mode == "image":
                    st = {"image_path": image_input, "emotion": app_state.emotion_result}
                    st = image_compose_brief_node(st)
                else:
                    st = {"user_text": user_story, "emotion": app_state.emotion_result}
                    st = compose_brief_node(st)
                app_state.music_brief = st["brief"]
                brief = app_state.music_brief
                audio_path = generate_with_replicate_strict(brief.prompt, int(brief.duration_sec))
                audio_path, _ = _rename_generated_file(audio_path, brief)
                audio_ui = gr.update(value=audio_path, label=_label_with_ext("치료용 음악", audio_path))
                status = f"🧩 기존 감정 분석 재사용 → 브리프 생성 → 음악 생성 ({app_state.current_mode})"
                tabs_update = gr.update(selected=0)

            else:
                # ▶ 원샷
                if app_state.current_mode == "image":
                    st = {"image_path": image_input}
                    final = image_graph.invoke(st)
                    emo = final["emotion"]; brief = final["brief"]
                    app_state.emotion_result = emo
                    app_state.music_brief = brief
                    audio_path = generate_with_replicate_strict(brief.prompt, int(brief.duration_sec))
                    audio_path, _ = _rename_generated_file(audio_path, brief)
                    audio_ui = gr.update(value=audio_path, label=_label_with_ext("치료용 음악", audio_path))
                else:
                    st = {"user_text": user_story, "force_generate": True}
                    final = graph.invoke(st)
                    emo = final["emotion"]; brief = final["brief"]
                    app_state.emotion_result = emo
                    app_state.music_brief = brief
                    audio_path = final.get("audio_path")
                    audio_path, _ = _rename_generated_file(audio_path, brief)
                    audio_ui = gr.update(value=audio_path, label=_label_with_ext("치료용 음악", audio_path))
                status = f"🚀 원샷 생성 ({app_state.current_mode} 모드)"
                tabs_update = gr.update(selected=0)

            # 출력 구성
            emo = app_state.emotion_result
            brief = app_state.music_brief
            emotion_text = md_emotion(emo)
            brief_text   = md_brief(brief)
            return emotion_text, brief_text, audio_ui, status, tabs_update

        except Exception as e:
            return "오류 발생", "오류 발생", None, f"❌ 처리 중 오류: {str(e)}", gr.update()

    
    # Create the Gradio interface
    with gr.Blocks(
        title="치료용 음악 생성 AI",
        theme=gr.themes.Soft(),
        css="""
        .main-container { max-width: 1200px; margin: 0 auto; }
        .story-input { min-height: 150px; }
        .result-box { border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin: 10px 0; }
        .step-button { margin: 5px; }

        /* 사용법/팁 카드 */
        .info-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #ffffff;
        padding: 14px 16px;
        transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
        }

        /* 제목/본문 가독성 강화 */
        .info-card h3 {
        margin: 0 0 8px 0;
        font-size: 16px;
        font-weight: 800;
        color: #111827;
        }
        .info-card p, .info-card li {
        color: #111827;
        line-height: 1.55;
        }

        /* ───────── Gradio 버튼 색상(변수) 오버라이드: elem_id 범위 내에서만 적용 ───────── */

        /* 감정 분석(초록 틴트) — secondary 변형 */
        #btn-emotion {
        --button-secondary-background-fill: #F0FDF4;          /* 기본 배경 */
        --button-secondary-text-color:       #065F46;          /* 텍스트 */
        --button-secondary-border-color:     #4ADE80;          /* 테두리 */
        --button-secondary-background-fill-hover: #DCFCE7;     /* 호버 배경 */
        --button-secondary-border-color-hover:     #22C55E;    /* 호버 테두리 */
        --button-shadow: 0 1px 0 rgba(0,0,0,.03);
        }

        /* 감정 분석 + 음악 설계(오렌지 틴트) — secondary 변형 */
        #btn-brief {
        --button-secondary-background-fill: #FFF7ED;
        --button-secondary-text-color:       #9A3412;
        --button-secondary-border-color:     #FB923C;
        --button-secondary-background-fill-hover: #FFEDD5;
        --button-secondary-border-color-hover:     #F97316;
        --button-shadow: 0 1px 0 rgba(0,0,0,.03);
        }

        /* 메인 CTA: 음악 생성(파랑) — primary 변형 */
        #btn-generate {
        --button-primary-background-fill: #2563EB;
        --button-primary-text-color:      #FFFFFF;
        --button-primary-border-color:    #1D4ED8;
        --button-primary-background-fill-hover: #1D4ED8;
        --button-shadow: 0 8px 18px rgba(37,99,235,0.28);
        }

        /* 크기/윤곽/전환 효과(버튼처럼 보이게) — 구조 변화에도 잘 먹히도록 넓은 선택자 */
        #btn-emotion button, #btn-emotion .gr-button, #btn-emotion [role="button"],
        #btn-brief   button, #btn-brief   .gr-button, #btn-brief   [role="button"],
        #btn-generate button, #btn-generate .gr-button, #btn-generate [role="button"] {
        height: 44px;
        min-width: 200px;
        padding: 0 16px;
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: .01em;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        transition: background .12s ease, border-color .12s ease,
                    box-shadow .12s ease, transform .06s ease;
        }

        /* 포커스 접근성 링 */
        #btn-emotion button:focus-visible,
        #btn-brief button:focus-visible,
        #btn-generate button:focus-visible {
        outline: 3px solid rgba(59,130,246,.45);
        outline-offset: 2px;
        }

        /* 호버 시 살짝 상승 효과(CTA만 더 강조) */
        #btn-generate button:hover,
        #btn-generate .gr-button:hover,
        #btn-generate [role="button"]:hover {
        transform: translateY(-1px);
        }
        #btn-generate button:active,
        #btn-generate .gr-button:active,
        #btn-generate [role="button"]:active {
        transform: translateY(0);
        }

        /* 모바일: 꽉 차게 */
        @media (max-width: 768px) {
        #btn-emotion button, #btn-emotion .gr-button, #btn-emotion [role="button"],
        #btn-brief   button, #btn-brief   .gr-button, #btn-brief   [role="button"],
        #btn-generate button, #btn-generate .gr-button, #btn-generate [role="button"] {
            width: 100%;
            min-width: 0;
        }
        }

        """
    ) as demo:
        
        gr.Markdown("""
        # 🎵 개인 감정 스토리 기반 치료용 음악 생성 AI
        
        **당신의 이야기와 감정을 분석하여 맞춤형 치료 음악을 생성합니다**
        
        이 AI는 예술치료 및 심리안정 지원을 목적으로 개발되었습니다. 
        당신의 감정 상태를 분석하고, 그에 맞는 치료적 음악을 생성합니다.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # 👇 탭을 전체 너비로 올려서 밑줄이 좌우 끝까지 보이게
                gr.Markdown("## 🔽 입력 방식을 선택하세요")

                with gr.Tabs() as input_tabs:
                    with gr.Tab("📝 텍스트") as text_tab:   # ← 변수로 받기
                        with gr.Row():
                            with gr.Column(scale=2):
                                story_input = gr.Textbox(
                                    label="감정이나 상황을 자유롭게 써주세요",
                                    placeholder="예: 오늘 하루 종일 마음이 무거웠다...",
                                    lines=6,
                                    elem_classes=["story-input"]
                                )
                            with gr.Column(scale=1):
                                gr.Markdown(get_usage_md(), elem_classes=["info-card"])
                                gr.Markdown(get_tips_md(),  elem_classes=["info-card"])

                    with gr.Tab("🖼️ 이미지") as image_tab: # ← 변수로 받기
                        with gr.Row():
                            with gr.Column(scale=2):
                                image_input = gr.Image(
                                    label="감정을 표현하는 이미지를 업로드하세요",
                                    type="filepath",
                                    sources=["upload", "clipboard"],
                                    height=300
                                )
                            with gr.Column(scale=1):
                                gr.Markdown(get_usage_md(), elem_classes=["info-card"])
                                gr.Markdown(get_tips_md(),  elem_classes=["info-card"])

                # 👇 버튼들은 탭(밑줄) 아래에 그대로 놓임 (전체 너비)
                gr.Markdown("## 🎯 원하는 작업을 선택하세요")
                with gr.Row():
                    emotion_only_btn = gr.Button(
                        "🔍 감정 분석",
                        variant="secondary",           # ← 하얀/아웃라인
                        scale=1,
                        elem_classes=["step-button"],
                        elem_id="btn-emotion"          # ← CSS 타겟팅용
                    )
                    brief_only_btn   = gr.Button(
                        "📋 감정 분석 + 음악 설계",
                        variant="secondary",           # ← 하얀/아웃라인 (변경 포인트)
                        scale=1,
                        elem_classes=["step-button"],
                        elem_id="btn-brief"            # ← CSS 타겟팅용
                    )
                    full_generate_btn= gr.Button(
                        "🎵 음악 생성",
                        variant="primary",             # ← 파란색 유지
                        scale=1,
                        elem_classes=["step-button"],
                        elem_id="btn-generate"         # ← CSS 타겟팅용
                    )

        
        # Results section
        gr.Markdown("## 📊 분석 결과")
        
        with gr.Row():
            with gr.Column():
                emotion_output = gr.Markdown(
                    label="감정 분석 결과",
                    elem_classes=["result-box"]
                )
            
            with gr.Column():
                brief_output = gr.Markdown(
                    label="음악 브리프",
                    elem_classes=["result-box"]
                )
        
        # Audio output and download
        gr.Markdown("## 🎵 생성/재생")

        with gr.Tabs() as audio_tabs:
            with gr.Tab("🎵 생성된 음악"):
                audio_output = gr.Audio(
                    label="치료용 음악",
                    type="filepath",
                    autoplay=False,
                    loop=True,              # 끝나면 다시 처음부터
                    interactive=False,      # 재생만, 편집 불가
                    editable=False,         # 자르기/되돌리기 도구 숨김
                    show_download_button=True,
                    elem_id="music_player"   # ← 커스텀 컨트롤이 찾을 ID
                )

            with gr.Tab("📁 내 파일 재생"):
                user_audio = gr.Audio(
                    label="내 파일 업로드(드래그&드롭)",
                    sources=["upload"],      # 파일 선택 + 드래그&드롭
                    type="filepath",         # 편집 후에도 파일 경로로 반환
                    autoplay=False,
                    loop=True,               # 업로드 파일도 반복 재생
                    interactive=True,        # 사용자 조작 허용
                    editable=True,           # ✂️ 자르기 / ↩ 되돌리기 / ↪ 다시하기 보이기
                    show_download_button=True,
                    elem_id="user_player"    # 커스텀 컨트롤용 ID
                )
        
        status_output = gr.Markdown("")

        def _on_user_audio_change(path):
            if not path:
                return gr.update()
            name = os.path.basename(path)
            return gr.update(label=f"내 파일 업로드 — {name}")
        
        user_audio.change(
            fn=_on_user_audio_change,
            inputs=[user_audio],
            outputs=[user_audio]
        )

        
        emotion_only_btn.click(
            fn=analyze_emotion_only,
            inputs=[story_input, image_input],
            outputs=[emotion_output, brief_output, audio_output, status_output]
        )
        
        brief_only_btn.click(
            fn=generate_music_brief_only,
            inputs=[story_input, image_input],
            outputs=[emotion_output, brief_output, audio_output, status_output]
        )
        
        full_generate_btn.click(
            fn=generate_full_music,
            inputs=[story_input, image_input],
            outputs=[emotion_output, brief_output, audio_output, status_output, audio_tabs],
        )

        text_tab.select(
            fn=on_text_tab_select,
            inputs=[],
            outputs=[story_input, image_input, emotion_output, brief_output, audio_output, status_output]
        )

        image_tab.select(
            fn=on_image_tab_select,
            inputs=[],
            outputs=[story_input, image_input, emotion_output, brief_output, audio_output, status_output]
        )


        
        # Footer
        gr.Markdown("""
        ---
        **개발 정보**: 이 AI는 개인의 감정과 스토리를 담은 퍼스널 뮤직 생성을 통해 예술치료 및 심리안정을 지원합니다.
        
        **주의사항**: 
        - 이 도구는 전문적인 심리치료를 대체하지 않습니다
        - 심각한 정신건강 문제가 있다면 전문가의 도움을 받으시기 바랍니다
        - 생성된 음악은 개인적 용도로만 사용해주세요
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch with sharing enabled for easy access
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,             # Create public link
        debug=True              # Enable debug mode
    )
