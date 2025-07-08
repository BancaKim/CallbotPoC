"""
Runpod 특화 설정 파일
GPU 메모리 부족 문제 해결을 위한 최적화된 설정
"""

import os

class RunpodConfig:
    """Runpod 환경을 위한 최적화된 설정"""
    
    # GPU 메모리 설정 (현재 환경에 맞게 조정)
    GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.7"))  # 0.8에서 0.7로 감소
    
    # 모델 설정 (메모리 절약형)
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # large-v3 대신 base 사용
    LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2-7B-Instruct")
    
    # 대안 LLM 모델들 (메모리 부족시 사용)
    ALTERNATIVE_MODELS = [
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-small", 
        "distilgpt2"
    ]
    
    # RAG 설정 (메모리 절약)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))  # 1000에서 800으로 감소
    OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "150"))  # 200에서 150으로 감소
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "3"))  # 5에서 3으로 감소
    
    # vLLM 설정
    MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))  # 8192에서 4096으로 감소
    
    @classmethod
    def get_optimal_gpu_memory(cls):
        """현재 GPU 메모리 상태에 따른 최적 사용률 반환"""
        try:
            import torch
            if torch.cuda.is_available():
                # GPU 메모리 정보 가져오기
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if total_memory > 40:  # 40GB 이상
                    return 0.75
                elif total_memory > 20:  # 20-40GB
                    return 0.7  
                else:  # 20GB 미만
                    return 0.6
        except:
            pass
        return cls.GPU_MEMORY_UTILIZATION
    
    @classmethod
    def should_use_vllm(cls):
        """GPU 메모리 상태에 따라 vLLM 사용 여부 결정"""
        try:
            import torch
            if torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return available_memory >= 16  # 16GB 이상일 때만 vLLM 사용
        except:
            pass
        return False
    
    @classmethod
    def get_fallback_settings(cls):
        """메모리 부족시 대안 설정"""
        return {
            "whisper_model": "tiny",  # 가장 작은 Whisper 모델
            "gpu_memory_utilization": 0.5,
            "max_model_len": 2048,
            "chunk_size": 500,
            "overlap_size": 100,
            "use_vllm": False  # Transformers만 사용
        }

# 환경별 설정 로드
def load_runpod_config():
    """Runpod 환경 설정 로드"""
    config = RunpodConfig()
    
    print(f"🔧 Runpod 설정 로드 완료")
    print(f"   - GPU 메모리 사용률: {config.get_optimal_gpu_memory()}")
    print(f"   - Whisper 모델: {config.WHISPER_MODEL}")
    print(f"   - LLM 모델: {config.LLM_MODEL}")
    print(f"   - vLLM 사용: {config.should_use_vllm()}")
    
    return config 