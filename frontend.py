import streamlit as st
import requests
import os
import tempfile
import time
from io import BytesIO
import base64
from audiorecorder import audiorecorder
import json

# 페이지 설정
st.set_page_config(
    page_title="은행 AI 콜봇",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #4caf50;
    }
    
    .status-box {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# 설정
RUNPOD_URL = "https://mqnrjc3msujef5-8000.proxy.runpod.net/"  # 실제 Runpod URL로 변경해야 함

class BankCallbotClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self):
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except Exception as e:
            return False, str(e)
    
    def speech_to_text(self, audio_file):
        """음성을 텍스트로 변환"""
        try:
            files = {"file": audio_file}
            response = requests.post(f"{self.base_url}/stt", files=files, timeout=30)
            response.raise_for_status()
            return True, response.json()
        except Exception as e:
            return False, str(e)
    
    def chat(self, text: str):
        """RAG 기반 채팅"""
        try:
            data = {"text": text}
            response = requests.post(f"{self.base_url}/chat", json=data, timeout=60)
            response.raise_for_status()
            return True, response.json()
        except Exception as e:
            return False, str(e)
    
    def text_to_speech(self, text: str, lang: str = "ko"):
        """텍스트를 음성으로 변환"""
        try:
            data = {"text": text, "lang": lang}
            response = requests.post(f"{self.base_url}/tts", json=data, timeout=30)
            response.raise_for_status()
            return True, response.content
        except Exception as e:
            return False, str(e)

def display_status(message: str, status_type: str = "info"):
    """상태 메시지 표시"""
    status_class = f"status-{status_type}"
    st.markdown(f'<div class="status-box {status_class}">{message}</div>', unsafe_allow_html=True)

def play_audio(audio_bytes):
    """오디오 재생"""
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio controls autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🏦 은행 AI 콜봇</h1>
        <p>음성으로 은행 상품에 대해 문의하세요</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # Runpod URL 설정
        runpod_url = st.text_input(
            "Runpod 서버 URL", 
            value=st.session_state.get('runpod_url', RUNPOD_URL),
            help="Runpod에서 실행 중인 서버의 URL을 입력하세요"
        )
        
        if runpod_url != st.session_state.get('runpod_url'):
            st.session_state['runpod_url'] = runpod_url
            st.session_state['client'] = BankCallbotClient(runpod_url)
        
        # 클라이언트 초기화
        if 'client' not in st.session_state:
            st.session_state['client'] = BankCallbotClient(runpod_url)
        
        # 서버 상태 확인
        if st.button("🔄 서버 상태 확인"):
            with st.spinner("서버 연결 확인 중..."):
                is_healthy, result = st.session_state['client'].health_check()
                if is_healthy:
                    st.success("✅ 서버 연결 성공!")
                    st.json(result)
                else:
                    st.error(f"❌ 서버 연결 실패: {result}")
        
        st.divider()
        
        # 대화 기록 초기화
        if st.button("🗑️ 대화 기록 삭제"):
            st.session_state['conversation_history'] = []
            st.success("대화 기록이 삭제되었습니다.")
            st.rerun()
    
    # 메인 컨텐츠
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎤 음성 입력")
        
        # 음성 녹음 옵션
        st.subheader("실시간 녹음")
        audio = audiorecorder("🎙️ 녹음 시작", "⏹️ 녹음 중지")
        
        if len(audio) > 0:
            st.audio(audio.export().read())
            
            if st.button("🔄 녹음된 음성 처리"):
                process_audio(audio.export().read(), "recorded_audio.wav")
        
        st.divider()
        
        # 파일 업로드 옵션
        st.subheader("음성 파일 업로드")
        uploaded_file = st.file_uploader(
            "음성 파일을 선택하세요",
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
            help="지원 형식: WAV, MP3, M4A, OGG, FLAC"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("🔄 업로드된 파일 처리"):
                process_audio(uploaded_file.getvalue(), uploaded_file.name)
    
    with col2:
        st.header("💬 텍스트 입력")
        
        # 직접 텍스트 입력
        user_input = st.text_area(
            "질문을 입력하세요",
            height=100,
            placeholder="예: 청년 대출 상품에 대해 알려주세요"
        )
        
        if st.button("💭 질문하기") and user_input.strip():
            process_text_query(user_input.strip())
    
    # 대화 기록 표시
    st.header("📋 대화 기록")
    
    # 대화 기록 초기화
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    
    # 대화 기록 표시
    if st.session_state['conversation_history']:
        for i, conversation in enumerate(reversed(st.session_state['conversation_history'])):
            with st.expander(f"💬 대화 {len(st.session_state['conversation_history']) - i}", expanded=(i == 0)):
                display_conversation(conversation)
    else:
        st.info("아직 대화 기록이 없습니다. 음성 또는 텍스트로 질문해보세요!")

def process_audio(audio_data, filename):
    """음성 데이터 처리"""
    with st.spinner("🎯 음성을 분석하고 있습니다..."):
        # STT 처리
        display_status("음성을 텍스트로 변환 중...", "info")
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, 'rb') as audio_file:
                success, stt_result = st.session_state['client'].speech_to_text(audio_file)
            
            if not success:
                display_status(f"음성 인식 실패: {stt_result}", "error")
                return
            
            transcribed_text = stt_result.get('text', '')
            display_status(f"음성 인식 완료: {transcribed_text}", "success")
            
            # 텍스트 쿼리 처리
            process_text_query(transcribed_text, audio_filename=filename)
            
        finally:
            # 임시 파일 삭제
            try:
                os.unlink(tmp_file_path)
            except:
                pass

def process_text_query(query_text, audio_filename=None):
    """텍스트 쿼리 처리"""
    with st.spinner("🤖 AI가 답변을 생성하고 있습니다..."):
        # 채팅 처리
        display_status("관련 문서를 검색하고 답변을 생성 중...", "info")
        success, chat_result = st.session_state['client'].chat(query_text)
        
        if not success:
            display_status(f"채팅 처리 실패: {chat_result}", "error")
            return
        
        response_text = chat_result.get('response', '')
        sources = chat_result.get('sources', [])
        
        display_status("답변 생성 완료!", "success")
        
        # TTS 처리
        display_status("텍스트를 음성으로 변환 중...", "info")
        success, tts_result = st.session_state['client'].text_to_speech(response_text)
        
        conversation = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'query': query_text,
            'response': response_text,
            'sources': sources,
            'audio_filename': audio_filename,
            'has_audio_response': success
        }
        
        if success:
            conversation['audio_response'] = tts_result
            display_status("음성 변환 완료!", "success")
        else:
            display_status(f"음성 변환 실패: {tts_result}", "error")
        
        # 대화 기록에 추가
        st.session_state['conversation_history'].append(conversation)
        
        # 결과 표시
        display_conversation(conversation, show_latest=True)

def display_conversation(conversation, show_latest=False):
    """대화 내용 표시"""
    # 사용자 질문
    st.markdown(f"""
    <div class="user-message">
        <strong>👤 고객:</strong><br>
        {conversation['query']}
    </div>
    """, unsafe_allow_html=True)
    
    # AI 응답
    st.markdown(f"""
    <div class="assistant-message">
        <strong>🤖 AI 상담원:</strong><br>
        {conversation['response']}
    </div>
    """, unsafe_allow_html=True)
    
    # 오디오 재생 (최신 대화만)
    if show_latest and conversation.get('has_audio_response'):
        st.subheader("🔊 음성 응답")
        play_audio(conversation['audio_response'])
    
    # 참조 문서
    if conversation.get('sources'):
        st.subheader("📚 참조 문서")
        for i, source in enumerate(conversation['sources']):
            st.write(f"**{i+1}.** {source['title']} (유사도: {source['score']:.3f})")
    
    # 메타데이터
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"⏰ {conversation['timestamp']}")
    with col2:
        if conversation.get('audio_filename'):
            st.caption(f"🎵 {conversation['audio_filename']}")
    with col3:
        if conversation.get('has_audio_response'):
            st.caption("🔊 음성 응답 생성됨")

if __name__ == "__main__":
    main() 