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

class AppState:
    def __init__(self):
        self.emotion_result = None
        self.music_brief = None
        self.user_story = None
    
    def clear(self):
        self.emotion_result = None
        self.music_brief = None
        self.user_story = None
    
    def has_emotion_analysis(self, current_story):
        return (self.emotion_result is not None and 
                self.user_story == current_story)
    
    def has_music_brief(self, current_story):
        return (self.music_brief is not None and 
                self.user_story == current_story)

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
            if app_state.user_story != user_story:
                app_state.clear()
                app_state.user_story = user_story
            
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
            
            emotion_text = f"""**🎭 주요 감정**: {emotion.primary}

**📊 감정 강도 (Valence)**: {emotion.valence:.2f}
*(-1: 매우 부정적 ↔ +1: 매우 긍정적)*

**⚡ 각성도 (Arousal)**: {emotion.arousal:.2f}
*(0: 차분함 ↔ 1: 흥분됨)*

**🎯 신뢰도**: {emotion.confidence:.2f}

**💭 분석 근거**: 
{emotion.reasons}"""
            
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
            if app_state.user_story != user_story:
                app_state.clear()
                app_state.user_story = user_story
            
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
            
            emotion_text = f"""**🎭 주요 감정**: {emotion.primary}

**📊 감정 강도 (Valence)**: {emotion.valence:.2f}
*(-1: 매우 부정적 ↔ +1: 매우 긍정적)*

**⚡ 각성도 (Arousal)**: {emotion.arousal:.2f}
*(0: 차분함 ↔ 1: 흥분됨)*

**🎯 신뢰도**: {emotion.confidence:.2f}

**💭 분석 근거**: 
{emotion.reasons}"""
            
            brief_text = f"""**🎵 음악 분위기**: {brief.mood}

**🥁 BPM**: {brief.bpm}

**🎼 조성**: {brief.key}

**⏱️ 길이**: {brief.duration_sec}초

**🎹 악기**: {', '.join(brief.instruments)}

**🏷️ 스타일 태그**: {', '.join(brief.style_tags)}

**📝 생성 프롬프트**: 
{brief.prompt}"""
            
            status = "✅ 감정 분석 및 음악 브리프 생성 완료! (크레딧 사용 안함)"
            
            return emotion_text, brief_text, None, status
            
        except Exception as e:
            error_msg = f"❌ 처리 중 오류 발생: {str(e)}"
            return "오류 발생", "오류 발생", None, error_msg
    
    def generate_full_music(user_story):
        """
        Full pipeline: analyze emotion, generate brief, and create actual music
        Uses cached results if available for the same story
        
        Args:
            user_story (str): User's emotional story/text
        
        Returns:
            tuple: (emotion_analysis, music_brief, audio_file, status_message)
        """
        try:
            if app_state.user_story != user_story:
                app_state.clear()
                app_state.user_story = user_story
            
            # Prepare the state
            state = {
                "user_text": user_story,
                "force_generate": True  # Force music generation
            }
            
            if app_state.has_emotion_analysis(user_story):
                state["emotion"] = app_state.emotion_result
                status_msg = "📋 이전 감정 분석 결과 재사용"
            
            if app_state.has_music_brief(user_story):
                state["brief"] = app_state.music_brief
                status_msg = "📋 이전 분석 결과 재사용하여 음악 생성"
            
            # Run the workflow (will skip already completed steps)
            final_state = graph.invoke(state)
            
            # Extract results
            emotion = final_state.get("emotion")
            brief = final_state.get("brief")
            audio_path = final_state.get("audio_path")
            provider_used = final_state.get("provider_used", "skipped")
            
            emotion_text = f"""**🎭 주요 감정**: {emotion.primary}

**📊 감정 강도 (Valence)**: {emotion.valence:.2f}
*(-1: 매우 부정적 ↔ +1: 매우 긍정적)*

**⚡ 각성도 (Arousal)**: {emotion.arousal:.2f}
*(0: 차분함 ↔ 1: 흥분됨)*

**🎯 신뢰도**: {emotion.confidence:.2f}

**💭 분석 근거**: 
{emotion.reasons}"""
            
            brief_text = f"""**🎵 음악 분위기**: {brief.mood}

**🥁 BPM**: {brief.bpm}

**🎼 조성**: {brief.key}

**⏱️ 길이**: {brief.duration_sec}초

**🎹 악기**: {', '.join(brief.instruments)}

**🏷️ 스타일 태그**: {', '.join(brief.style_tags)}

**📝 생성 프롬프트**: 
{brief.prompt}"""
            
            # Status message
            if provider_used == "skipped":
                status = "⚠️ 음악 생성이 건너뛰어졌습니다. API 키를 확인해주세요."
                audio_file = None
            elif provider_used in ["replicate", "rest"]:
                base_status = f"🎵 음악 생성 완료! ({provider_used} 사용)"
                if 'status_msg' in locals():
                    status = f"{status_msg} → {base_status}"
                else:
                    status = base_status
                audio_file = audio_path if audio_path and os.path.exists(audio_path) else None
            else:
                status = "❌ 음악 생성 중 오류 발생"
                audio_file = None
            
            return emotion_text, brief_text, audio_file, status
            
        except Exception as e:
            error_msg = f"❌ 처리 중 오류 발생: {str(e)}"
            return "오류 발생", "오류 발생", None, error_msg
    
    def check_environment():
        """Check if required environment variables are set"""
        openai_ok = bool(os.getenv("OPENAI_API_KEY"))
        replicate_ok = bool(os.getenv("REPLICATE_API_TOKEN"))
        use_replicate = os.getenv("USE_REPLICATE", "0") == "1"
        
        status = f"""**⚙️ 환경 설정 상태**:

**OpenAI API**: {'✅ 설정됨' if openai_ok else '❌ 미설정'}

**Replicate API**: {'✅ 설정됨' if replicate_ok else '❌ 미설정'}

**USE_REPLICATE**: {'✅ 활성화됨' if use_replicate else '⚠️ 비활성화됨 (0으로 설정)'}

{'✅ 모든 설정이 완료되어 음악 생성이 가능합니다!' if (openai_ok and replicate_ok and use_replicate) else '⚠️ 실제 음악 생성을 위해서는 모든 API 키가 필요하고 USE_REPLICATE=1로 설정해야 합니다.'}"""
        
        return status
    
    def translate_prompt_to_korean(english_prompt):
        """
        Translate English music generation prompt to Korean
        This is a simple translation helper for better understanding
        """
        pass
    
    # Create the Gradio interface
    with gr.Blocks(
        title="치료용 음악 생성 AI",
        theme=gr.themes.Soft(),
        css="""
        .main-container { max-width: 1200px; margin: 0 auto; }
        .story-input { min-height: 150px; }
        .result-box { border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin: 10px 0; }
        .step-button { margin: 5px; }
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
                        "🔍 감정 분석만", 
                        variant="secondary", 
                        scale=1,
                        elem_classes=["step-button"]
                    )
                    brief_only_btn = gr.Button(
                        "📋 감정분석 + 음악설계", 
                        variant="primary", 
                        scale=1,
                        elem_classes=["step-button"]
                    )
                    full_generate_btn = gr.Button(
                        "🎵 전체 생성 (크레딧 사용)", 
                        variant="primary", 
                        scale=1,
                        elem_classes=["step-button"]
                    )
                
                gr.Markdown("""
                **💡 사용법**:
                - **🔍 감정 분석만**: 무료로 감정 상태만 분석
                - **📋 감정분석 + 음악설계**: 감정 분석 + 음악 브리프 생성 (무료)
                - **🎵 전체 생성**: 실제 음악 파일까지 생성 (Replicate 크레딧 사용)
                """)
            
            with gr.Column(scale=1):
                # Environment check
                gr.Markdown("## ⚙️ 시스템 상태")
                env_status = gr.Markdown(check_environment())
                refresh_btn = gr.Button("🔄 상태 새로고침", size="sm")
        
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
        gr.Markdown("## 🎵 생성된 음악")
        audio_output = gr.Audio(
            label="치료용 음악",
            type="filepath",
            interactive=False
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
        
        refresh_btn.click(
            fn=check_environment,
            outputs=[env_status]
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
        share=True,             # Create public link
        debug=True              # Enable debug mode
    )
