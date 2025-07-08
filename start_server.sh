#!/bin/bash

# 은행 콜봇 서버 시작 스크립트

echo "🏦 은행 AI 콜봇 서버를 시작합니다..."

# 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    echo "🔄 가상환경을 활성화합니다..."
    source venv/bin/activate
fi

# 필요한 의존성 설치 확인
echo "📦 의존성을 확인합니다..."
if [ -f "requirements_runpod_simple.txt" ]; then
    echo "간소화된 requirements 파일을 사용합니다..."
    pip install -r requirements_runpod_simple.txt
else
    echo "전체 requirements 파일을 사용합니다..."
    pip install -r requirements_runpod.txt
fi

# GPU 메모리 확인
echo "🖥️  GPU 상태를 확인합니다..."
nvidia-smi

# 환경 변수 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# 로그 디렉토리 생성
mkdir -p logs

# 서버 시작
echo "🚀 FastAPI 서버를 시작합니다..."
echo "📡 서버 URL: http://0.0.0.0:8000"
echo "📋 API 문서: http://0.0.0.0:8000/docs"
echo "❤️  헬스체크: http://0.0.0.0:8000/health"
echo ""
echo "서버를 중지하려면 Ctrl+C를 누르세요."
echo "----------------------------------------"

# 서버 실행 (로그 저장)
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --access-log \
    --log-level info \
    2>&1 | tee logs/server.log 