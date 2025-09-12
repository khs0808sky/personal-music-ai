import gradio as gr
import numpy as np
import json
from datetime import datetime
import re

class TextAnalyzer:
    """텍스트를 분석하여 음악적 특성을 추출하는 클래스"""
    
    def __init__(self):
        # 감정 키워드 매핑
        self.emotion_keywords = {
            'happy': ['기쁘', '행복', '즐거', '신나', '웃음', '밝', '좋', '사랑'],
            'sad': ['슬프', '우울', '눈물', '아프', '힘들', '외로', '그리워'],
            'calm': ['평온', '조용', '차분', '안정', '편안', '휴식', '명상'],
            'energetic': ['에너지', '활기', '역동', '강렬', '파워', '열정', '도전'],
            'romantic': ['로맨틱', '사랑', '달콤', '부드러', '따뜻', '포근'],
            'mysterious': ['신비', '어둠', '미스터리', '깊', '복잡', '숨겨진']
        }
        
        # 음악적 특성 매핑
        self.music_characteristics = {
            'happy': {'tempo': 'fast', 'key': 'major', 'instruments': ['piano', 'guitar', 'drums']},
            'sad': {'tempo': 'slow', 'key': 'minor', 'instruments': ['piano', 'strings', 'cello']},
            'calm': {'tempo': 'slow', 'key': 'major', 'instruments': ['piano', 'flute', 'ambient']},
            'energetic': {'tempo': 'fast', 'key': 'major', 'instruments': ['drums', 'electric_guitar', 'bass']},
            'romantic': {'tempo': 'medium', 'key': 'major', 'instruments': ['piano', 'violin', 'soft_synth']},
            'mysterious': {'tempo': 'medium', 'key': 'minor', 'instruments': ['synth', 'ambient', 'dark_piano']}
        }
    
    def analyze_emotion(self, text):
        """텍스트에서 감정을 분석"""
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                score += text.count(keyword)
            emotion_scores[emotion] = score
        
        # 가장 높은 점수의 감정 반환
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        return primary_emotion, emotion_scores
    
    def extract_music_params(self, text, emotion):
        """텍스트와 감정을 바탕으로 음악 파라미터 추출"""
        characteristics = self.music_characteristics.get(emotion, self.music_characteristics['calm'])
        
        # 텍스트 길이에 따른 음악 길이 결정
        text_length = len(text)
        if text_length < 50:
            duration = 30  # 30초
        elif text_length < 200:
            duration = 60  # 1분
        else:
            duration = 120  # 2분
        
        return {
            'emotion': emotion,
            'tempo': characteristics['tempo'],
            'key': characteristics['key'],
            'instruments': characteristics['instruments'],
            'duration': duration,
            'text_length': text_length
        }

class MusicGenerator:
    """음악 생성을 담당하는 클래스 (현재는 mock 구현)"""
    
    def __init__(self):
        self.analyzer = TextAnalyzer()
    
    def generate_music_metadata(self, text):
        """텍스트를 분석하여 음악 메타데이터 생성"""
        # 감정 분석
        primary_emotion, emotion_scores = self.analyzer.analyze_emotion(text)
        
        # 음악 파라미터 추출
        music_params = self.analyzer.extract_music_params(text, primary_emotion)
        
        # 생성 시간 기록
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            'input_text': text,
            'primary_emotion': primary_emotion,
            'emotion_scores': emotion_scores,
            'music_parameters': music_params,
            'generated_at': generation_time,
            'status': 'metadata_generated'
        }
    
    def create_audio_placeholder(self, duration=30):
        """실제 음악 생성 전 플레이스홀더 오디오 생성"""
        # 간단한 사인파 생성 (실제로는 AI 모델로 대체될 부분)
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 다양한 주파수의 사인파 조합으로 간단한 멜로디 생성
        frequencies = [261.63, 293.66, 329.63, 349.23, 392.00]  # C, D, E, F, G
        audio = np.zeros_like(t)
        
        for i, freq in enumerate(frequencies):
            start_time = i * (duration / len(frequencies))
            end_time = (i + 1) * (duration / len(frequencies))
            mask = (t >= start_time) & (t < end_time)
            audio[mask] = 0.3 * np.sin(2 * np.pi * freq * t[mask])
        
        return (sample_rate, audio.astype(np.float32))

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    generator = MusicGenerator()
    
    def process_text_and_generate_music(input_text, therapy_mode, music_style):
        """메인 처리 함수"""
        if not input_text.strip():
            return "텍스트를 입력해주세요.", None, None
        
        try:
            # 텍스트 분석 및 메타데이터 생성
            metadata = generator.generate_music_metadata(input_text)
            
            # 치료 모드에 따른 조정
            if therapy_mode == "심리 안정":
                metadata['music_parameters']['tempo'] = 'slow'
                metadata['music_parameters']['key'] = 'major'
            elif therapy_mode == "에너지 충전":
                metadata['music_parameters']['tempo'] = 'fast'
                metadata['music_parameters']['instruments'].append('upbeat_drums')
            
            # 음악 스타일 반영
            if music_style != "자동 선택":
                metadata['music_parameters']['style_override'] = music_style
            
            # 분석 결과 텍스트 생성
            analysis_text = f"""
🎵 **음악 생성 분석 결과**

📝 **입력 텍스트**: {input_text[:100]}{'...' if len(input_text) > 100 else ''}

🎭 **감정 분석**:
- 주요 감정: {metadata['primary_emotion']} 
- 감정 점수: {', '.join([f"{k}: {v}" for k, v in metadata['emotion_scores'].items() if v > 0])}

🎼 **음악 특성**:
- 템포: {metadata['music_parameters']['tempo']}
- 조성: {metadata['music_parameters']['key']}
- 악기: {', '.join(metadata['music_parameters']['instruments'])}
- 예상 길이: {metadata['music_parameters']['duration']}초

🕐 **생성 시간**: {metadata['generated_at']}

💡 **치료 모드**: {therapy_mode}
🎨 **음악 스타일**: {music_style}

⚠️ *현재는 프로토타입 버전으로, 실제 AI 음악 생성 모델 연동 예정*
            """
            
            # 플레이스홀더 오디오 생성
            audio_data = generator.create_audio_placeholder(metadata['music_parameters']['duration'])
            
            # JSON 메타데이터
            json_metadata = json.dumps(metadata, ensure_ascii=False, indent=2)
            
            return analysis_text, audio_data, json_metadata
            
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}", None, None
    
    # Gradio 인터페이스 구성
    with gr.Blocks(
        title="🎶 퍼스널 뮤직 생성 AI",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        """
    ) as interface:
        
        # 헤더
        gr.HTML("""
        <div class="main-header">
            🎶 퍼스널 뮤직 생성 AI
        </div>
        <div style="text-align: center; margin-bottom: 30px; color: #666;">
            <p><strong>예술 치료 및 심리 안정 지원</strong></p>
            <p>당신의 감정과 생각을 음악으로 표현해보세요</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # 입력 섹션
                gr.Markdown("## 📝 텍스트 입력")
                input_text = gr.Textbox(
                    label="감정이나 상황을 자유롭게 적어주세요",
                    placeholder="예: 오늘 하루 정말 힘들었어요. 마음이 무겁고 혼자인 것 같아요...",
                    lines=5,
                    max_lines=10
                )
                
                with gr.Row():
                    therapy_mode = gr.Dropdown(
                        label="🧘 치료 모드",
                        choices=["자동 선택", "심리 안정", "에너지 충전", "감정 정화", "창의성 향상"],
                        value="자동 선택"
                    )
                    
                    music_style = gr.Dropdown(
                        label="🎨 음악 스타일",
                        choices=["자동 선택", "클래식", "앰비언트", "재즈", "포크", "일렉트로닉", "뉴에이지"],
                        value="자동 선택"
                    )
                
                generate_btn = gr.Button(
                    "🎵 음악 생성하기",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # 도움말
                gr.Markdown("""
                ## 💡 사용 가이드
                
                **1단계**: 현재 감정이나 상황을 자유롭게 텍스트로 입력하세요.
                
                **2단계**: 원하는 치료 모드와 음악 스타일을 선택하세요 (선택사항).
                
                **3단계**: '음악 생성하기' 버튼을 클릭하세요.
                
                ---
                
                ### 🎭 감정 키워드 예시
                - **기쁨**: 행복, 즐거움, 웃음
                - **슬픔**: 우울, 눈물, 그리움  
                - **평온**: 차분, 안정, 휴식
                - **에너지**: 활기, 열정, 도전
                - **로맨틱**: 사랑, 따뜻함, 부드러움
                """)
        
        # 결과 섹션
        gr.Markdown("## 🎼 생성 결과")
        
        with gr.Row():
            with gr.Column(scale=2):
                analysis_output = gr.Markdown(label="분석 결과")
                
            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="생성된 음악",
                    type="numpy"
                )
        
        # 메타데이터 (개발자용)
        with gr.Accordion("🔧 개발자 정보 (메타데이터)", open=False):
            json_output = gr.Code(
                label="JSON 메타데이터",
                language="json"
            )
        
        # 이벤트 연결
        generate_btn.click(
            fn=process_text_and_generate_music,
            inputs=[input_text, therapy_mode, music_style],
            outputs=[analysis_output, audio_output, json_output]
        )
        
        # 예시 버튼들
        gr.Markdown("## 📚 예시 텍스트")
        with gr.Row():
            example_texts = [
                "오늘 승진 소식을 들었어요! 정말 기쁘고 모든 게 완벽해 보여요.",
                "요즘 너무 스트레스가 많아서 마음이 무거워요. 조용한 곳에서 쉬고 싶어요.",
                "새로운 도전을 앞두고 있어서 설레면서도 긴장돼요. 용기가 필요해요."
            ]
            
            for i, example in enumerate(example_texts):
                gr.Button(f"예시 {i+1}").click(
                    lambda x=example: x,
                    outputs=input_text
                )
    
    return interface

# 메인 실행
if __name__ == "__main__":
    print("🎶 퍼스널 뮤직 생성 AI 시작 중...")
    
    # Gradio 인터페이스 생성 및 실행
    app = create_gradio_interface()
    
    # 서버 실행
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # 공유 링크 생성
        debug=True
    )
