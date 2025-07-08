# 🔧 문제 해결 가이드

은행 AI 콜봇 시스템 설치 및 실행 중 발생할 수 있는 문제들과 해결 방법을 안내합니다.

## 🚨 설치 관련 문제

### 1. Python 버전 호환성 오류

**문제**: `Requires-Python >=3.11` 오류 발생

**해결방법**:
```bash
# 1. 간소화된 requirements 파일 사용
pip install -r requirements_runpod_simple.txt

# 2. 또는 Python 3.11 이상 버전 사용
python --version  # 버전 확인
```

### 2. vLLM 설치 실패

**문제**: vLLM 패키지 설치 실패 또는 CUDA 호환성 문제

**해결방법**:
```bash
# 1. vLLM 없이 설치 (코드가 자동으로 Transformers 사용)
pip install -r requirements_runpod_simple.txt

# 2. GPU 환경에서만 vLLM 설치 시도
pip install vllm
```

**참고**: vLLM이 없어도 시스템이 정상 작동합니다. Transformers를 대안으로 사용합니다.

### 3. pickle-protocol 오류

**문제**: `No matching distribution found for pickle-protocol`

**해결방법**: 이 패키지는 불필요합니다. Python 내장 pickle 모듈을 사용하므로 설치할 필요가 없습니다.

### 4. PyTorch 설치 문제

**문제**: CUDA 버전 불일치 또는 PyTorch 설치 실패

**해결방법**:
```bash
# CPU 버전 설치 (GPU 없는 환경)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 버전 (GPU 환경)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 버전 (GPU 환경)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🛠️ 실행 관련 문제

### 1. GPU 메모리 부족

**문제**: CUDA out of memory 오류

**해결방법**:
```python
# config.py에서 GPU 메모리 사용량 조정
GPU_MEMORY_UTILIZATION = 0.6  # 기본값 0.8에서 줄임
```

### 2. Whisper 모델 다운로드 실패

**문제**: 인터넷 연결 문제로 모델 다운로드 실패

**해결방법**:
```bash
# 수동으로 모델 다운로드 시도
python -c "import whisper; whisper.load_model('base')"

# 더 작은 모델 사용
python -c "import whisper; whisper.load_model('tiny')"
```

### 3. PDF 문서 처리 오류

**문제**: PDF 파일을 읽을 수 없음

**해결방법**:
1. PDF 파일이 텍스트 추출 가능한지 확인
2. 스캔된 이미지 PDF는 OCR 처리 필요
3. 파일 권한 확인

### 4. 서버 연결 실패

**문제**: 프론트엔드에서 백엔드 서버에 연결할 수 없음

**해결방법**:
1. Runpod URL이 정확한지 확인
2. 포트 8000이 열려있는지 확인
3. 방화벽 설정 확인

## 💡 최적화 팁

### 1. 메모리 사용량 줄이기

```python
# 더 작은 모델 사용
WHISPER_MODEL = "base"  # large-v3 대신
CHUNK_SIZE = 500       # 1000 대신
```

### 2. 처리 속도 개선

```python
# 배치 크기 조정
batch_size = 4  # 8 대신
```

### 3. 캐시 활용

```bash
# 임베딩 캐시 파일 유지
# embeddings_cache.pkl 파일을 삭제하지 마세요
```

## 🔍 디버깅 방법

### 1. 로그 확인

```bash
# 서버 로그 확인
tail -f logs/server.log

# Python 로그 레벨 설정
export LOG_LEVEL=DEBUG
```

### 2. 개별 컴포넌트 테스트

```python
# STT 테스트
import whisper
model = whisper.load_model("base")
result = model.transcribe("test.wav")
print(result["text"])

# RAG 시스템 테스트
from rag_system import RAGSystem
rag = RAGSystem()
# ... 테스트 코드
```

### 3. API 엔드포인트 테스트

```bash
# 헬스체크
curl http://localhost:8000/health

# 단순 텍스트 채팅 테스트
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"text": "안녕하세요"}'
```

## 📞 추가 지원

### 시스템 요구사항
- **최소**: Python 3.8+, 8GB RAM
- **권장**: Python 3.11+, 16GB RAM, GPU (4GB+ VRAM)

### 대안 설정
GPU가 없거나 메모리가 부족한 경우:
1. Whisper "tiny" 모델 사용
2. CPU 버전 PyTorch 사용
3. 청크 크기를 500으로 줄이기
4. vLLM 대신 Transformers 사용 (자동)

### 문제 보고 시 포함할 정보
1. Python 버전 (`python --version`)
2. OS 정보
3. GPU 정보 (있는 경우)
4. 전체 오류 메시지
5. 설치한 패키지 목록 (`pip list`)

---

🔧 이 가이드로 해결되지 않는 문제가 있다면, 로그와 오류 메시지를 포함하여 문의해주세요. 