import os
import asyncio
import logging
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, docs_dir: str = "docs"):
        """LangChain 기반 RAG 시스템 초기화"""
        self.docs_dir = docs_dir
        self.embeddings_model = None
        self.vectorstore = None
        self.documents = []
        self.vectorstore_cache_path = "vectorstore_cache"
        
        # 텍스트 분할 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    async def initialize_documents(self):
        """문서들을 로드하고 벡터 저장소를 초기화합니다."""
        try:
            # BGE-M3 임베딩 모델 초기화
            logger.info("BGE-M3 임베딩 모델을 로딩 중입니다...")
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name='BAAI/bge-m3',
                model_kwargs={'device': 'cuda' if self._is_cuda_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("BGE-M3 모델 로딩 완료")
            
            # 캐시된 벡터 저장소가 있는지 확인
            if self._check_vectorstore_cache():
                logger.info("캐시된 벡터 저장소를 로드합니다...")
                await self._load_vectorstore_cache()
            else:
                logger.info("PDF 문서들을 처리하고 벡터 저장소를 생성합니다...")
                await self._process_documents()
                await self._save_vectorstore_cache()
            
            if self.vectorstore:
                logger.info(f"총 {self.vectorstore.index.ntotal}개의 문서 벡터가 로드되었습니다.")
            
        except Exception as e:
            logger.error(f"문서 초기화 오류: {str(e)}")
            raise
    
    def _is_cuda_available(self) -> bool:
        """CUDA 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _check_vectorstore_cache(self) -> bool:
        """벡터 저장소 캐시 파일 존재 여부 확인"""
        cache_path = Path(self.vectorstore_cache_path)
        return cache_path.exists() and (cache_path / "index.faiss").exists()
    
    async def _process_documents(self):
        """PDF 문서들을 로드하고 처리합니다."""
        docs_path = Path(self.docs_dir)
        if not docs_path.exists():
            logger.warning(f"문서 디렉토리 '{self.docs_dir}'가 존재하지 않습니다.")
            return
        
        # PDF 파일들 로드
        pdf_files = list(docs_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning("PDF 문서를 찾을 수 없습니다.")
            return
        
        logger.info(f"{len(pdf_files)}개의 PDF 파일을 처리합니다...")
        
        all_documents = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"처리 중: {pdf_file.name}")
                
                # PyPDFLoader로 PDF 로드
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # 메타데이터 추가
                for doc in documents:
                    doc.metadata.update({
                        "source_file": pdf_file.name,
                        "source_path": str(pdf_file)
                    })
                
                # 텍스트 분할
                split_docs = self.text_splitter.split_documents(documents)
                all_documents.extend(split_docs)
                
                logger.info(f"{pdf_file.name}에서 {len(split_docs)}개의 청크 생성")
                
            except Exception as e:
                logger.error(f"{pdf_file.name} 처리 오류: {str(e)}")
                continue
        
        # 벡터 저장소 생성
        if all_documents:
            logger.info("FAISS 벡터 저장소를 생성합니다...")
            self.vectorstore = FAISS.from_documents(
                documents=all_documents,
                embedding=self.embeddings_model
            )
            self.documents = all_documents
            logger.info("벡터 저장소 생성 완료")
        else:
            logger.warning("처리할 문서가 없습니다.")
    
    async def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """쿼리와 관련된 문서들을 검색합니다."""
        if not self.vectorstore:
            logger.warning("벡터 저장소가 초기화되지 않았습니다.")
            return []
        
        try:
            # FAISS를 사용한 유사도 검색 (점수 포함)
            relevant_docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
            
            # 결과 포맷팅
            results = []
            for doc, score in relevant_docs_with_scores:
                result = {
                    "title": doc.metadata.get("source_file", "Unknown"),
                    "content": doc.page_content,
                    "source": doc.metadata.get("source_path", ""),
                    "score": float(1.0 - score),  # FAISS는 거리를 반환하므로 유사도로 변환
                    "metadata": doc.metadata
                }
                results.append(result)
            
            logger.info(f"쿼리 '{query}'에 대해 {len(results)}개의 관련 문서를 찾았습니다.")
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 오류: {str(e)}")
            return []
    

    
    async def _save_vectorstore_cache(self):
        """벡터 저장소를 캐시로 저장합니다."""
        if not self.vectorstore:
            return
        
        try:
            cache_path = Path(self.vectorstore_cache_path)
            cache_path.mkdir(exist_ok=True)
            
            # FAISS 인덱스 저장
            self.vectorstore.save_local(str(cache_path))
            
            # 문서 메타데이터 저장
            metadata_path = cache_path / "documents_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info("벡터 저장소 캐시를 저장했습니다.")
            
        except Exception as e:
            logger.error(f"벡터 저장소 캐시 저장 오류: {str(e)}")
    
    async def _load_vectorstore_cache(self):
        """캐시된 벡터 저장소를 로드합니다."""
        try:
            cache_path = Path(self.vectorstore_cache_path)
            
            # FAISS 벡터 저장소 로드
            self.vectorstore = FAISS.load_local(
                str(cache_path), 
                self.embeddings_model
            )
            
            # 문서 메타데이터 로드
            metadata_path = cache_path / "documents_metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.documents = pickle.load(f)
            
            logger.info("캐시된 벡터 저장소를 로드했습니다.")
            
        except Exception as e:
            logger.error(f"벡터 저장소 캐시 로드 오류: {str(e)}")
            # 캐시 로드 실패 시 새로 생성
            await self._process_documents()
            await self._save_vectorstore_cache()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """문서 통계 정보를 반환합니다."""
        stats = {
            "total_documents": len(self.documents) if self.documents else 0,
            "vectorstore_size": self.vectorstore.index.ntotal if self.vectorstore else 0,
            "embedding_model": "BAAI/bge-m3",
            "chunk_size": self.text_splitter._chunk_size,
            "chunk_overlap": self.text_splitter._chunk_overlap
        }
        
        # 파일별 통계
        if self.documents:
            file_stats = {}
            for doc in self.documents:
                source_file = doc.metadata.get("source_file", "Unknown")
                if source_file not in file_stats:
                    file_stats[source_file] = 0
                file_stats[source_file] += 1
            stats["files"] = file_stats
        
        return stats
    
    async def add_documents(self, documents: List[Document]):
        """새로운 문서들을 벡터 저장소에 추가합니다."""
        if not self.vectorstore:
            logger.warning("벡터 저장소가 초기화되지 않았습니다.")
            return
        
        try:
            # 텍스트 분할
            split_docs = self.text_splitter.split_documents(documents)
            
            # 벡터 저장소에 추가
            self.vectorstore.add_documents(split_docs)
            self.documents.extend(split_docs)
            
            # 캐시 업데이트
            await self._save_vectorstore_cache()
            
            logger.info(f"{len(split_docs)}개의 새로운 문서 청크가 추가되었습니다.")
            
        except Exception as e:
            logger.error(f"문서 추가 오류: {str(e)}")
    
    async def clear_cache(self):
        """캐시된 벡터 저장소를 삭제합니다."""
        try:
            cache_path = Path(self.vectorstore_cache_path)
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                logger.info("벡터 저장소 캐시가 삭제되었습니다.")
            
        except Exception as e:
            logger.error(f"캐시 삭제 오류: {str(e)}")
    
    def create_retrieval_chain(self):
        """검색 체인을 생성합니다 (LangChain 체인과 함께 사용)"""
        if not self.vectorstore:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다.")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": 5}) 