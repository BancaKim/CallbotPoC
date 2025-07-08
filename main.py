from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import whisper

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("vLLM not available, will use transformers instead")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    VLLM_AVAILABLE = False
from gtts import gTTS
import os
import tempfile
import asyncio
from pydantic import BaseModel
from rag_system import RAGSystem
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="은행 콜봇 API", description="은행 상품 문의를 위한 AI 콜봇 시스템")

# CORS 설정 (로컬 프론트엔드와 통신을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로딩
logger.info("모델들을 로딩 중입니다...")

# Whisper Large v3 로드
whisper_model = whisper.load_model("large-v3")
logger.info("Whisper Large v3 모델 로딩 완료")

# Qwen3-8B 모델 로드
if VLLM_AVAILABLE:
    llm = LLM(
        model="Qwen/Qwen3-8B",  # Qwen 8B 모델 사용
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        trust_remote_code=True
    )
    sampling_params = SamplingParams(
        temperature=0.3,  # 더 보수적인 설정
        top_p=0.8,
        max_tokens=100,  # 토큰 수 줄임
        repetition_penalty=1.1,  # 반복 억제 약화
        stop=["\n\n", "고객:", "상담원:", "질문:", "답변:", "=", "&"]  # 중지 토큰 추가
    )
    logger.info("Qwen3-8B 모델 (vLLM) 로딩 완료")
else:
    # Transformers로 Qwen 8B 로딩
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", 
        trust_remote_code=True
    )
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Qwen3-8B 모델 (Transformers) 로딩 완료")

# RAG 시스템 초기화
rag_system = RAGSystem()
logger.info("RAG 시스템 초기화 완료")

# 요청 모델 정의
class ChatRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str
    lang: str = "ko"

def generate_response(prompt: str) -> str:
    """Qwen3-8B 모델로 응답 생성"""
    try:
        if VLLM_AVAILABLE:
            # vLLM 사용
            outputs = llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # 후처리: 반복 제거 및 정리
            # 이상한 문자/패턴 제거
            invalid_patterns = ["=", "&", "printvalue", "QSL", "1&", "<|", "|>", "```"]
            has_invalid = any(pattern in response for pattern in invalid_patterns)
            
            if has_invalid or len(response.strip()) < 10:
                return "죄송합니다. 해당 상품에 대한 정확한 정보를 제공하기 어려워 추가 문의를 권해드립니다."
            
            # 중지 토큰들 제거
            stop_tokens = ["\n\n", "고객:", "상담원:", "질문:", "답변:", "=", "&"]
            for stop_token in stop_tokens:
                if stop_token in response:
                    response = response.split(stop_token)[0]
            
            # 같은 문장 반복 제거
            sentences = response.split('. ')
            unique_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in unique_sentences and len(sentence) > 5:
                    # 한글이 포함된 문장만 유지
                    if any('\uac00' <= char <= '\ud7af' for char in sentence):
                        unique_sentences.append(sentence)
            
            if unique_sentences:
                response = '. '.join(unique_sentences)
                if not response.endswith('.'):
                    response += '.'
            else:
                return "죄송합니다. 해당 상품에 대한 정확한 정보를 제공하기 어려워 추가 문의를 권해드립니다."
            
            return response.strip()
        else:
            # Transformers 사용 - device 및 attention_mask 문제 해결
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True,
                return_attention_mask=True
            )
            
            # 모델과 같은 device로 이동
            device = next(llm.parameters()).device
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            with torch.no_grad():
                outputs = llm.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=100,  # 토큰 수 줄임
                    temperature=0.3,  # 더 보수적인 설정
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # 반복 억제 약화
                    no_repeat_ngram_size=2  # 2-gram으로 변경
                )
            
            # 입력 프롬프트 제거하고 응답만 추출
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # 후처리: 반복 제거 및 정리
            response = response.strip()
            
            # 이상한 문자/패턴 제거
            invalid_patterns = ["=", "&", "printvalue", "QSL", "1&", "<|", "|>", "```"]
            has_invalid = any(pattern in response for pattern in invalid_patterns)
            
            if has_invalid or len(response.strip()) < 10:
                return "죄송합니다. 해당 상품에 대한 정확한 정보를 제공하기 어려워 추가 문의를 권해드립니다."
            
            # 중지 토큰들 제거
            stop_tokens = ["\n\n", "고객:", "상담원:", "질문:", "답변:", "=", "&"]
            for stop_token in stop_tokens:
                if stop_token in response:
                    response = response.split(stop_token)[0]
            
            # 같은 문장 반복 제거
            sentences = response.split('. ')
            unique_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in unique_sentences and len(sentence) > 5:
                    # 한글이 포함된 문장만 유지
                    if any('\uac00' <= char <= '\ud7af' for char in sentence):
                        unique_sentences.append(sentence)
            
            if unique_sentences:
                response = '. '.join(unique_sentences)
                if not response.endswith('.'):
                    response += '.'
            else:
                return "죄송합니다. 해당 상품에 대한 정확한 정보를 제공하기 어려워 추가 문의를 권해드립니다."
            
            return response.strip()
    except Exception as e:
        logger.error(f"응답 생성 오류: {str(e)}")
        return "죄송합니다. 현재 시스템에 문제가 있어 답변을 생성할 수 없습니다."

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 RAG 시스템 초기화"""
    logger.info("RAG 시스템 문서 로딩을 시작합니다...")
    await rag_system.initialize_documents()
    logger.info("서버 초기화 완료")

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "message": "은행 콜봇 서버가 정상 작동 중입니다."}

@app.post("/stt")
async def speech_to_text(file: UploadFile):
    """음성을 텍스트로 변환하는 STT 엔드포인트"""
    try:
        logger.info(f"STT 요청 받음: {file.filename}")
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Whisper로 음성 인식
        result = whisper_model.transcribe(
            temp_file_path,
            language="ko",  # 한국어 지정
            task="transcribe"
        )
        
        # 임시 파일 삭제
        os.unlink(temp_file_path)
        
        transcribed_text = result["text"].strip()
        logger.info(f"STT 결과: {transcribed_text}")
        
        return {
            "success": True,
            "text": transcribed_text,
            "language": result.get("language", "ko")
        }
        
    except Exception as e:
        logger.error(f"STT 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"STT 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    """RAG 기반 채팅 엔드포인트"""
    try:
        logger.info(f"채팅 요청 받음: {request.text}")
        
        # RAG 시스템으로 관련 문서 검색
        relevant_docs = await rag_system.search_documents(request.text)
        
        # 프롬프트 구성
        context = "\n".join([doc["content"] for doc in relevant_docs])
        
        system_prompt = """당신은 친절하고 전문적인 은행 상담원입니다. 
고객의 질문에 대해 제공된 은행 상품 정보를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.
답변은 한국어로 하고, 고객이 이해하기 쉽게 설명해주세요.
만약 제공된 정보로 답변할 수 없다면, 추가 문의를 권해주세요.

중요한 지침:
1. 상담원 역할만 하고, 상담원의 답변만 생성하세요
2. 고객의 질문을 반복하거나 대화를 시뮬레이션하지 마세요
3. 같은 내용을 반복하지 마세요
4. 답변은 2-3문장으로 간결하게 완료하세요"""

        user_prompt = f"""관련 은행 상품 정보:
{context}

고객 질문: {request.text}

위 정보를 바탕으로 고객의 질문에 간결하고 정확하게 답변해주세요. 2-3문장으로 완료하세요."""

        # LLM으로 답변 생성
        if VLLM_AVAILABLE:
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"{system_prompt}\n\n{user_prompt}\n\n답변:"
        
        response_text = generate_response(full_prompt)
        
        logger.info(f"LLM 응답 생성 완료")
        
        return {
            "success": True,
            "response": response_text,
            "sources": [{"title": doc["title"], "score": doc["score"]} for doc in relevant_docs]
        }
        
    except Exception as e:
        logger.error(f"채팅 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """텍스트를 음성으로 변환하는 TTS 엔드포인트"""
    try:
        logger.info(f"TTS 요청 받음: {request.text[:50]}...")
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            output_path = temp_file.name
        
        # gTTS로 음성 생성
        tts = gTTS(text=request.text, lang=request.lang, slow=False)
        tts.save(output_path)
        
        logger.info("TTS 음성 파일 생성 완료")
        
        # 파일 반환
        def cleanup_file():
            try:
                os.unlink(output_path)
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {e}")
        
        return FileResponse(
            path=output_path,
            media_type="audio/mpeg",
            filename="response.mp3",
            background=BackgroundTask(cleanup_file)  # 전송 후 파일 삭제
        )
        
    except Exception as e:
        logger.error(f"TTS 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/full_conversation")
async def full_conversation(file: UploadFile):
    """STT -> RAG -> LLM -> TTS 전체 플로우를 한 번에 처리하는 엔드포인트"""
    try:
        logger.info(f"전체 대화 플로우 시작: {file.filename}")
        
        # 1. STT 처리
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        # 음성 인식
        stt_result = whisper_model.transcribe(temp_audio_path, language="ko")
        transcribed_text = stt_result["text"].strip()
        os.unlink(temp_audio_path)
        
        logger.info(f"STT 완료: {transcribed_text}")
        
        # 2. RAG + LLM 처리
        relevant_docs = await rag_system.search_documents(transcribed_text)
        context = "\n".join([doc["content"] for doc in relevant_docs])
        
        system_prompt = """당신은 친절하고 전문적인 은행 상담원입니다. 
고객의 질문에 대해 제공된 은행 상품 정보를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.
답변은 한국어로 하고, 고객이 이해하기 쉽게 설명해주세요.
만약 제공된 정보로 답변할 수 없다면, 추가 문의를 권해주세요.

중요한 지침:
1. 상담원 역할만 하고, 상담원의 답변만 생성하세요
2. 고객의 질문을 반복하거나 대화를 시뮬레이션하지 마세요
3. 같은 내용을 반복하지 마세요
4. 답변은 2-3문장으로 간결하게 완료하세요"""

        user_prompt = f"""관련 은행 상품 정보:
{context}

고객 질문: {transcribed_text}

위 정보를 바탕으로 고객의 질문에 간결하고 정확하게 답변해주세요. 2-3문장으로 완료하세요."""

        if VLLM_AVAILABLE:
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"{system_prompt}\n\n{user_prompt}\n\n답변:"
        
        response_text = generate_response(full_prompt)
        
        logger.info(f"LLM 응답 생성 완료")
        
        # 3. TTS 처리
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_tts:
            output_path = temp_tts.name
        
        tts = gTTS(text=response_text, lang="ko", slow=False)
        tts.save(output_path)
        
        logger.info("전체 대화 플로우 완료")
        
        return {
            "success": True,
            "transcribed_text": transcribed_text,
            "response_text": response_text,
            "audio_file": output_path,
            "sources": [{"title": doc["title"], "score": doc["score"]} for doc in relevant_docs]
        }
        
    except Exception as e:
        logger.error(f"전체 대화 플로우 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"대화 처리 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)