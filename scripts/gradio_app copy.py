import gradio as gr
import os
import json
from pathlib import Path
import tempfile
import shutil

# Import the core music generation functionality
from music_generator_core import (
    graph, GraphState, EmotionResult, MusicBrief,
    dump, generate_with_replicate_strict,
    analyze_emotion_node, compose_brief_node  # Import new nodes
)

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
- **같은 스토리**면 이전 감정분석을 재사용해요.
- 스토리를 바꾸면 이전 결과는 초기화돼요.
"""



class AppState:
    def __init__(self):
        self.emotion_result = None
        self.music_brief = None
        self.user_story = None

    def clear(self):
        self.emotion_result = None
        self.music_brief = None
        self.user_story = None

    def set_story(self, s: str):
        self.user_story = _norm(s)

    def has_emotion_analysis(self, current_story):
        return (self.emotion_result is not None and self.user_story == _norm(current_story))

    def has_music_brief(self, current_story):
        return (self.music_brief is not None and self.user_story == _norm(current_story))

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
    
    def analyze_emotion_only(user_story):
        """
        Only analyze emotion without generating music
        
        Args:
            user_story (str): User's emotional story/text
        
        Returns:
            tuple: (emotion_analysis, status_message)
        """
        try:
            if not _norm(user_story):
                return "스토리를 입력해 주세요.", "", None, "⚠️ 입력이 비어 있습니다."

            if app_state.user_story != _norm(user_story):
                app_state.clear()
                app_state.set_story(user_story)

            # ⬇️ 추가: 같은 스토리로 이미 분석한 결과가 있으면 재사용
            if app_state.has_emotion_analysis(user_story):
                emo = app_state.emotion_result
                emotion_text = md_emotion(emo)  # 이미 만들어둔 헬퍼 사용 권장
                return emotion_text, "", None, "♻️ 이전 감정 분석 결과 재사용"

            
            # Prepare the state for emotion analysis only
            state = {
                "user_text": user_story,
                "force_generate": False  # Never generate music
            }
            
            # Run only the emotion analysis node directly
            state = analyze_emotion_node(state)
            
            # Extract emotion result
            emotion = state.get("emotion")
            app_state.emotion_result = emotion
            
            emotion_text = md_emotion(emotion)
            
            status = "✅ 감정 분석 완료! (크레딧 사용 안함)"
            
            return emotion_text, "", None, status
            
        except Exception as e:
            error_msg = f"❌ 감정 분석 중 오류 발생: {str(e)}"
            return "오류 발생", "", None, error_msg
    
    def generate_music_brief_only(user_story):
        """
        Analyze emotion and generate music brief without actual music generation
        
        Args:
            user_story (str): User's emotional story/text
        
        Returns:
            tuple: (emotion_analysis, music_brief, status_message)
        """
        try:
            if not _norm(user_story):
                return "스토리를 입력해 주세요.", "", None, "⚠️ 입력이 비어 있습니다."
            
            if app_state.user_story != _norm(user_story):
                app_state.clear()
                app_state.set_story(user_story)

            
            # Prepare the state
            state = {
                "user_text": user_story,
                "force_generate": False  # Explicitly prevent music generation
            }
            
            if app_state.has_emotion_analysis(user_story):
                state["emotion"] = app_state.emotion_result
            else:
                state = analyze_emotion_node(state)
                app_state.emotion_result = state.get("emotion")
            
            # Run only the brief composition node  
            state = compose_brief_node(state)
            
            # Extract results without running the full graph
            emotion = state.get("emotion")
            brief = state.get("brief")
            app_state.music_brief = brief
            
            emotion_text = md_emotion(emotion)
            brief_text   = md_brief(brief)
            
            status = "✅ 감정 분석 및 음악 브리프 생성 완료! (크레딧 사용 안함)"
            
            return emotion_text, brief_text, None, status
            
        except Exception as e:
            error_msg = f"❌ 처리 중 오류 발생: {str(e)}"
            return "오류 발생", "오류 발생", None, error_msg
    
    def generate_full_music(user_story):
        """
        두 모드 지원:
        1) 원샷(분석 X) → graph.invoke 사용
        2) 분석/브리프 끝난 후 음악만 → 캐시 재사용, 노드 직접 호출(필요시 브리프만 생성), 마지막에만 합성
        """
        try:
            if not _norm(user_story):
                return "스토리를 입력해 주세요.", "", None, "⚠️ 입력이 비어 있습니다."

            # 스토리 바뀌면 캐시 초기화
            if app_state.user_story != _norm(user_story):
                app_state.clear()
                app_state.set_story(user_story)


            # 환경 체크
            openai_ok = bool(os.getenv("OPENAI_API_KEY"))
            repl_ok   = bool(os.getenv("REPLICATE_API_TOKEN"))
            use_repl  = os.getenv("USE_REPLICATE", "0") == "1"
            if not openai_ok:
                return "OpenAI 키가 없습니다.", "", None, "❌ OPENAI_API_KEY 필요"
            if not (repl_ok and use_repl):
                # 음악 합성은 건너뛰고, 분석/브리프만 보여주기
                # 필요 시 여기서 분석/브리프 생성할 수도 있음
                if not app_state.has_emotion_analysis(user_story):
                    st = {"user_text": user_story}
                    st = analyze_emotion_node(st)
                    app_state.emotion_result = st["emotion"]
                if not app_state.has_music_brief(user_story):
                    st = {"user_text": user_story, "emotion": app_state.emotion_result}
                    st = compose_brief_node(st)
                    app_state.music_brief = st["brief"]
                emo, brief = app_state.emotion_result, app_state.music_brief

                emotion_text = md_emotion(emo)
                brief_text   = md_brief(brief)

                return emotion_text, brief_text, None, "⚠️ USE_REPLICATE=1 또는 REPLICATE 토큰 없음 → 음악 생성 건너뜀"

            # ── 분기 ───────────────────────────────────────────────────────────
            if app_state.has_emotion_analysis(user_story) and app_state.has_music_brief(user_story):
                # ▶ 이미 분석/브리프가 있음 → 그대로 재사용해서 합성만
                brief = app_state.music_brief
                audio_path = generate_with_replicate_strict(brief.prompt, int(brief.duration_sec))
                status = "🎵 기존 분석/브리프 재사용하여 음악 생성"
            elif app_state.has_emotion_analysis(user_story) and not app_state.has_music_brief(user_story):
                # ▶ 감정만 있음 → 브리프만 생성 후 합성
                st = {"user_text": user_story, "emotion": app_state.emotion_result}
                st = compose_brief_node(st)
                app_state.music_brief = st["brief"]
                brief = app_state.music_brief
                audio_path = generate_with_replicate_strict(brief.prompt, int(brief.duration_sec))
                status = "🧩 기존 감정 분석 재사용 → 브리프 생성 → 음악 생성"
            else:
                # ▶ 원샷 생성(분석 없이 바로) → LangGraph 전체 파이프라인 실행
                state = {"user_text": user_story, "force_generate": True}
                final = graph.invoke(state)  # analyze → brief → generate
                emo = final["emotion"]; brief = final["brief"]
                app_state.emotion_result = emo
                app_state.music_brief = brief
                audio_path = final.get("audio_path")
                status = f"🚀 원샷 생성 (graph.invoke 사용; provider={final.get('provider_used','?')})"

            # 출력 메시지 구성 (캐시 기준)
            emo = app_state.emotion_result
            brief = app_state.music_brief
            
            emotion_text = md_emotion(emo)
            brief_text   = md_brief(brief)

            return emotion_text, brief_text, audio_path, status

        except Exception as e:
            return "오류 발생", "오류 발생", None, f"❌ 처리 중 오류: {str(e)}"
    
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
        .info-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
        border-color: #d1d5db;
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
        /* (원하면) 가운데 정렬: 주석 해제
        .info-card { text-align: center; }
        .info-card ul { display: inline-block; text-align: left; }
        */

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
                # Input section
                gr.Markdown("## 📝 당신의 이야기를 들려주세요")
                
                story_input = gr.Textbox(
                    label="감정이나 상황을 자유롭게 써주세요",
                    placeholder="예: 오늘 하루 종일 마음이 무거웠다. 일정을 정리하다가 페이지를 넘기는 손이 자주 멈췄다. 시간이 흐르는 게 잘 느껴지지 않았다...",
                    lines=6,
                    elem_classes=["story-input"]
                )
                
                gr.Markdown("## 🎯 원하는 작업을 선택하세요")
                
                with gr.Row():
                    emotion_only_btn = gr.Button(
                        "🔍 감정 분석", 
                        variant="secondary", 
                        scale=1,
                        elem_classes=["step-button"]
                    )
                    brief_only_btn = gr.Button(
                        "📋 감정 분석 + 음악 설계", 
                        variant="primary", 
                        scale=1,
                        elem_classes=["step-button"]
                    )
                    full_generate_btn = gr.Button(
                        "🎵 음악 생성", 
                        variant="primary", 
                        scale=1,
                        elem_classes=["step-button"]
                    )
            # 오른쪽 정보 패널 부분
            with gr.Column(scale=1):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(get_usage_md(), elem_classes=["info-card"])  # ← 변경
                    with gr.Column():
                        gr.Markdown(get_tips_md(), elem_classes=["info-card"])   # ← 변경

        
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

        with gr.Tabs():
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
        
        emotion_only_btn.click(
            fn=analyze_emotion_only,
            inputs=[story_input],
            outputs=[emotion_output, brief_output, audio_output, status_output]
        )
        
        brief_only_btn.click(
            fn=generate_music_brief_only,
            inputs=[story_input],
            outputs=[emotion_output, brief_output, audio_output, status_output]
        )
        
        full_generate_btn.click(
            fn=generate_full_music,
            inputs=[story_input],
            outputs=[emotion_output, brief_output, audio_output, status_output]
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
