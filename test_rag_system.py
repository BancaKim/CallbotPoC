#!/usr/bin/env python3
"""
LangChain 기반 RAG 시스템 테스트 스크립트

새로 구현된 RAG 시스템이 정상적으로 작동하는지 확인합니다.
"""

import asyncio
import logging
from rag_system import RAGSystem

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rag_system():
    """RAG 시스템을 테스트합니다."""
    
    print("🧪 LangChain 기반 RAG 시스템 테스트")
    print("=" * 50)
    
    try:
        # RAG 시스템 초기화
        print("\n1. RAG 시스템 초기화 중...")
        rag = RAGSystem()
        
        # 문서 로딩
        print("\n2. 문서 로딩 및 벡터 저장소 생성 중...")
        await rag.initialize_documents()
        
        # 문서 통계 확인
        print("\n3. 문서 통계:")
        stats = rag.get_document_stats()
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        # 테스트 쿼리들
        test_queries = [
            "청년 우대 적금에 대해 알려주세요",
            "KB Star 예금 금리는 어떻게 되나요?",
            "해외 송금 수수료는 얼마인가요?",
            "정기예금 상품 추천해주세요"
        ]
        
        print("\n4. 검색 테스트:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n   테스트 {i}: '{query}'")
            
            results = await rag.search_documents(query, top_k=3)
            
            if results:
                print(f"   ✅ {len(results)}개의 관련 문서를 찾았습니다:")
                for j, result in enumerate(results, 1):
                    print(f"      {j}. {result['title']} (점수: {result['score']:.3f})")
                    print(f"         내용: {result['content'][:100]}...")
            else:
                print("   ❌ 관련 문서를 찾지 못했습니다.")
        
        # 검색 체인 테스트
        print("\n5. LangChain 체인 생성 테스트:")
        try:
            retriever = rag.create_retrieval_chain()
            print("   ✅ 검색 체인이 성공적으로 생성되었습니다.")
            print(f"   ✅ 검색 유형: {type(retriever).__name__}")
        except Exception as e:
            print(f"   ❌ 체인 생성 실패: {str(e)}")
        
        print("\n🎉 모든 테스트가 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        logger.error(f"테스트 오류: {str(e)}", exc_info=True)

async def test_document_addition():
    """새 문서 추가 기능 테스트"""
    print("\n📄 문서 추가 기능 테스트")
    print("-" * 30)
    
    try:
        from langchain.schema import Document
        
        rag = RAGSystem()
        await rag.initialize_documents()
        
        # 테스트용 문서 생성
        test_doc = Document(
            page_content="이것은 테스트용 은행 상품입니다. 금리는 3.5%이며, 최소 예치금액은 100만원입니다.",
            metadata={
                "source_file": "test_document.txt",
                "source_path": "test_document.txt"
            }
        )
        
        print("새 문서를 추가하는 중...")
        await rag.add_documents([test_doc])
        
        # 추가된 문서 검색 테스트
        results = await rag.search_documents("테스트 상품 금리", top_k=1)
        
        if results and "테스트용" in results[0]['content']:
            print("✅ 문서 추가 및 검색이 성공했습니다!")
        else:
            print("❌ 추가된 문서를 찾을 수 없습니다.")
            
    except Exception as e:
        print(f"❌ 문서 추가 테스트 실패: {str(e)}")

async def main():
    """메인 테스트 실행"""
    await test_rag_system()
    await test_document_addition()

if __name__ == "__main__":
    asyncio.run(main()) 