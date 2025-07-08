# 🏦 은행 AI 콜봇 시스템

음성 인식(STT), RAG 검색 증강 생성, 대화형 AI(LLM), 그리고 음성 합성(TTS)을 활용한 은행 상품 문의 콜봇 시스템입니다.

## 📋 시스템 구성

### 기술 스택
- **STT**: OpenAI Whisper Large v3
- **LLM**: Qwen2.5-8B-Instruct  
- **임베딩**: BAAI/bge-m3
- **TTS**: Google Text-to-Speech (gTTS)
- **백엔드**: FastAPI (Runpod 배포)
- **프론트엔드**: Streamlit (로컬 실행)
- **RAG**: LangChain + FAISS 기반 검색 증강 생성

### 시스템 아키텍처
```
[Streamlit Frontend] ←→ [FastAPI Backend on Runpod]
                              ↓
                         [Whisper STT]
                              ↓
                         [RAG System]
                         (BGE-M3 임베딩)
                              ↓
                      [Qwen2.5-8B LLM]
                              ↓
                         [gTTS TTS]
```

## 🧪 RAG 시스템 테스트

새로운 LangChain 기반 RAG 시스템을 테스트하려면:

```bash
# 의존성 설치 후
python test_rag_system.py
```

이 스크립트는 다음을 테스트합니다:
- ✅ 문서 로딩 및 벡터 저장소 생성
- ✅ 문서 검색 기능
- ✅ 새 문서 추가 기능
- ✅ LangChain 체인 생성

## 🚀 설치 및 실행

### 1. Runpod 백엔드 배포

#### 1.1 필요한 파일들
```
main.py                    # FastAPI 서버
rag_system.py             # RAG 시스템 구현
requirements_runpod_simple.txt  # 간소화된 의존성
start_server.sh           # 서버 시작 스크립트
docs/                     # PDF 문서들
├── diy_co.pdf
├── kbstar.pdf  
├── super.pdf
├── Youth_a.pdf
└── Youth_b.pdf
```

#### 1.2 Runpod 설정
1. **GPU 인스턴스**: NVIDIA RTX 4090 이상 권장
2. **Python**: 3.10 (3.11+ 호환성 문제로 인해)
3. **CUDA**: 12.1 이상

#### 1.3 배포 명령어
```bash
# 1. 파일 업로드 후 
cd /workspace

# 2. 의존성 설치
pip install -r requirements_runpod_simple.txt

# 3. 서버 시작
chmod +x start_server.sh
./start_server.sh
```

### 2. 로컬 프론트엔드 실행

#### 2.1 Python 가상환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 2.2 의존성 설치
```bash
pip install streamlit requests
```

#### 2.3 프론트엔드 실행
```bash
streamlit run frontend.py
```

## 📚 사용법

### 1. 음성 대화
1. **Runpod URL 설정**: 사이드바에서 Runpod 인스턴스 URL 입력
2. **음성 녹음**: "🎙️ 녹음 시작" 버튼으로 실시간 녹음
3. **파일 업로드**: 기존 음성 파일 업로드 (.wav, .mp3, .m4a 등)
4. **처리**: "🔄 처리" 버튼으로 STT → RAG → LLM → TTS 실행

### 2. 텍스트 채팅
1. **직접 입력**: 채팅창에 질문 입력
2. **AI 답변**: RAG로 관련 문서 검색 후 답변 생성
3. **음성 재생**: 답변을 음성으로 재생 가능

### 3. 지원 질문 예시
- "청년 우대 적금에 대해 알려주세요"
- "KB Star 예금 금리가 어떻게 되나요?"
- "해외 송금 수수료는 얼마인가요?"

## 🛠️ API 엔드포인트

### FastAPI 백엔드
- `GET /health`: 서버 상태 및 모델 로딩 확인
- `POST /stt`: 음성 파일을 텍스트로 변환
- `POST /chat`: RAG 기반 질의응답
- `POST /tts`: 텍스트를 음성으로 변환
- `POST /full_conversation`: 전체 대화 플로우 (STT→RAG→LLM→TTS)

### 요청/응답 예시
```python
# /chat 엔드포인트
{
    "message": "청년 우대 적금 금리가 어떻게 되나요?",
    "conversation_history": []
}

# 응답
{
    "response": "KB청년희망적금의 기본 금리는...",
    "relevant_docs": [
        {
            "content": "문서 내용...",
            "similarity": 0.85,
            "source": "Youth_a.pdf"
        }
    ]
}
```

## 📁 프로젝트 구조

```
callbot3/
├── docs/                          # 은행 상품 PDF 문서
│   ├── diy_co.pdf                 # DIY 상품 안내
│   ├── kbstar.pdf                 # KB Star 상품
│   ├── super.pdf                  # Super 정기예금
│   ├── Youth_a.pdf                # 청년 상품 A
│   └── Youth_b.pdf                # 청년 상품 B
├── main.py                        # FastAPI 백엔드 서버
├── rag_system.py                  # RAG 시스템 (BGE-M3 임베딩)
├── frontend.py                    # Streamlit 프론트엔드
├── requirements_runpod_simple.txt # Runpod용 의존성 (간소화)
├── start_server.sh                # Runpod 서버 시작 스크립트
├── TROUBLESHOOTING.md             # 문제 해결 가이드
└── README.md                      # 프로젝트 문서
```

## ⚠️ 주요 문제 해결

### 1. PyTorch 보안 취약점 (CVE-2025-32434)
**문제**: PyTorch 2.5.1에서 모델 로딩 차단
```bash
FutureWarning: You are using `torch.load` with `weights_only=False`
```

**해결**: `requirements_runpod_simple.txt`에서 problematic 패키지 제거

### 2. CUDA/CPU 디바이스 불일치
**문제**: GPU와 CPU 텐서 간 연산 오류
```python
RuntimeError: Expected all tensors to be on the same device
```

**해결**: 모든 텐서를 같은 디바이스로 이동 처리 구현

### 3. 응답 품질 문제
**문제**: AI가 대화 시뮬레이션 생성하거나 반복적/불완전한 답변
```
고객: 안녕하세요
상담원: 안녕하세요. 무엇을 도와드릴까요?
```

**해결**:
- 명확한 프롬프트 지시사항 추가
- 생성 파라미터 조정 (temperature: 0.3, max_tokens: 100)
- 응답 후처리로 불필요한 패턴 제거

### 4. 메모리 부족
**문제**: GPU 메모리 초과로 모델 로딩 실패

**해결**: 
- `gpu_memory_utilization=0.8` 설정
- 모델별 메모리 관리 최적화

## 🔧 성능 최적화

### 1. RAG 시스템
- **임베딩 캐싱**: 문서 임베딩을 메모리에 캐시
- **청크 최적화**: 1000자 청크, 200자 오버랩
- **유사도 임계값**: 0.1 이상만 검색 결과로 반환

### 2. 모델 최적화
- **Attention Mask**: 토큰화 시 올바른 어텐션 마스크 생성
- **배치 처리**: 여러 요청 동시 처리 가능
- **메모리 관리**: GPU 메모리 효율적 사용

## 📞 지원 및 문제 해결

### 빠른 진단
1. **Health Check**: `GET /health`로 서버 상태 확인
2. **로그 확인**: 백엔드/프론트엔드 터미널 로그 확인
3. **모델 로딩**: 초기 로딩에 5-10분 소요 정상

### 일반적인 문제들
- **연결 오류**: Runpod URL 및 포트(8000) 확인
- **음성 인식 실패**: 오디오 파일 형식 및 크기(50MB 이하) 확인  
- **답변 품질 저하**: 관련 PDF 문서 업로드 여부 확인
- **메모리 오류**: GPU 메모리 사용량 모니터링

### 상세 문제 해결
전체 문제 해결 가이드는 `TROUBLESHOOTING.md`를 참조하세요.

## 🎯 향후 개선 사항

### 기능 개선
- [ ] 실시간 스트리밍 음성 인식
- [ ] 다국어 지원 (영어, 중국어)
- [ ] 사용자 인증 시스템
- [ ] 대화 히스토리 DB 저장

### 성능 개선  
- [ ] 모델 양자화 (INT8/INT4)
- [ ] 응답 시간 최적화 (< 5초)
- [ ] 동시 사용자 처리 확장
- [ ] CDN 기반 파일 서빙

## 📊 시스템 요구사항

### Runpod (백엔드)
- **GPU**: RTX 4090 24GB 권장
- **RAM**: 32GB 이상  
- **Storage**: 50GB 이상
- **Python**: 3.10

### 로컬 (프론트엔드)
- **RAM**: 4GB 이상
- **Python**: 3.8+
- **네트워크**: 안정적인 인터넷 연결

---

🏦 **은행 AI 콜봇** - STT, RAG, LLM, TTS 기술을 활용한 차세대 고객 서비스 솔루션 # CallbotPoC
