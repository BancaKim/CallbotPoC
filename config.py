import os
from typing import Optional

class Config:
    """은행 콜봇 시스템 설정"""
    
    # Runpod 서버 설정
    RUNPOD_URL: str = os.getenv("RUNPOD_URL", "http://your-runpod-url:8000")
    
    # API 설정
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "60"))
    MAX_AUDIO_SIZE_MB: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "50"))
    
    # 모델 설정
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen2-7B-Instruct")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    
    # RAG 설정
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    OVERLAP_SIZE: int = int(os.getenv("OVERLAP_SIZE", "200"))
    MIN_SIMILARITY_THRESHOLD: float = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.1"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    
    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "callbot.log")
    
    # TTS 설정
    TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "ko")
    TTS_SLOW: bool = os.getenv("TTS_SLOW", "false").lower() == "true"
    
    # 파일 경로
    DOCS_DIR: str = os.getenv("DOCS_DIR", "docs")
    EMBEDDINGS_CACHE_FILE: str = os.getenv("EMBEDDINGS_CACHE_FILE", "embeddings_cache.pkl")
    
    # GPU 설정
    GPU_MEMORY_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.8"))
    
    @classmethod
    def get_runpod_url(cls) -> str:
        """Runpod URL 반환 (환경변수 우선)"""
        return cls.RUNPOD_URL
    
    @classmethod
    def validate_config(cls) -> bool:
        """설정 검증"""
        if not cls.RUNPOD_URL or cls.RUNPOD_URL == "http://your-runpod-url:8000":
            return False
        return True 