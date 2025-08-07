#!/usr/bin/env python3
"""
통합 문서 번역기 - Streamlit 버전
PDF, Word, Excel, PowerPoint 문서를 번역합니다.
"""

import streamlit as st
import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import base64
from io import BytesIO
import zipfile

# 페이지 설정
st.set_page_config(
    page_title="통합 문서 번역기",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 로깅 설정
@st.cache_resource
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# 모듈 import
from core import get_translator, DocumentTranslatorCore
from document_handler import (
    PDFHandler, WordHandler, ExcelHandler, 
    PowerPointHandler, TextHandler
)

# CSS 스타일
def load_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .file-upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f9f9f9;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
def init_session_state():
    if 'translator' not in st.session_state:
        st.session_state.translator = get_translator()
    
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    
    if 'current_files' not in st.session_state:
        st.session_state.current_files = []
    
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            'openai': '',
            'deepl': ''
        }
    
    if 'translation_in_progress' not in st.session_state:
        st.session_state.translation_in_progress = False
    
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
    
    # Streamlit secrets에서 API 키 로드
    try:
        if 'openai' in st.secrets:
            st.session_state.api_keys['openai'] = st.secrets['openai']['api_key']
            st.session_state.translator.set_api_key('openai', st.secrets['openai']['api_key'], save=False)
        
        if 'deepl' in st.secrets:
            st.session_state.api_keys['deepl'] = st.secrets['deepl']['api_key']
            st.session_state.translator.set_api_key('deepl', st.secrets['deepl']['api_key'], save=False)
    except:
        pass

# 파일 다운로드 함수
def create_download_link(file_path: str, link_text: str) -> str:
    """파일 다운로드 링크 생성"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        file_name = Path(file_path).name
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{link_text}</a>'
        return href
    except Exception as e:
        logger.error(f"다운로드 링크 생성 실패: {e}")
        return ""

def create_zip_download(file_paths: List[str]) -> bytes:
    """여러 파일을 ZIP으로 압축"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in file_paths:
            if Path(file_path).exists():
                zip_file.write(file_path, Path(file_path).name)
    return zip_buffer.getvalue()

# 메인 UI
def main():
    # CSS 로드
    load_css()
    
    # 세션 상태 초기화
    init_session_state()
    
    # 헤더
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📚 통합 문서 번역기 v2.0")
        st.markdown("**PDF, Word, Excel, PowerPoint** 문서를 번역합니다")
    
    # 사이드바 - 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # API 키 설정
        with st.expander("🔑 API 키 설정", expanded=False):
            openai_key = st.text_input(
                "OpenAI API 키",
                value=st.session_state.api_keys.get('openai', ''),
                type="password",
                help="sk-로 시작하는 API 키를 입력하세요"
            )
            
            openai_model = st.selectbox(
                "OpenAI 모델",
                options=['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
                index=0
            )
            
            deepl_key = st.text_input(
                "DeepL API 키",
                value=st.session_state.api_keys.get('deepl', ''),
                type="password",
                help="DeepL Pro API 키를 입력하세요"
            )
            
            if st.button("💾 API 키 저장", use_container_width=True):
                if openai_key:
                    st.session_state.api_keys['openai'] = openai_key
                    st.session_state.translator.set_api_key('openai', openai_key, save=False)
                    st.session_state.translator.openai_model = openai_model
                
                if deepl_key:
                    st.session_state.api_keys['deepl'] = deepl_key
                    st.session_state.translator.set_api_key('deepl', deepl_key, save=False)
                
                st.success("✅ API 키가 저장되었습니다")
        
        # API 상태 표시
        st.divider()
        st.subheader("📊 API 상태")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.translator.validate_api_key('openai'):
                st.success("✅ OpenAI")
            else:
                st.error("❌ OpenAI")
        
        with col2:
            if st.session_state.translator.validate_api_key('deepl'):
                st.success("✅ DeepL")
            else:
                st.warning("⚠️ DeepL")
        
        st.info("✅ Google 번역 (무료)")
        
        # 통계
        st.divider()
        st.subheader("📈 번역 통계")
        stats = st.session_state.translator.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("총 번역", f"{stats['total_translated']}개")
        with col2:
            avg_time = stats.get('average_time', 0)
            st.metric("평균 시간", f"{avg_time:.1f}초")
        
        # 지원 형식
        st.divider()
        st.subheader("📄 지원 형식")
        formats = st.session_state.translator.supported_formats
        for ext, desc in formats.items():
            st.text(f"{ext}: {desc}")
    
    # 메인 컨텐츠 - 탭
    tab1, tab2, tab3, tab4 = st.tabs(["📝 번역", "📋 번역 기록", "📊 대시보드", "❓ 도움말"])
    
    # 번역 탭
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📥 문서 업로드")
            
            # 파일 업로더
            uploaded_files = st.file_uploader(
                "문서 파일을 선택하세요",
                type=['pdf', 'docx', 'xlsx', 'pptx', 'txt', 'md'],
                accept_multiple_files=True,
                help="여러 파일을 동시에 선택할 수 있습니다"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)}개 파일 선택됨")
                for file in uploaded_files:
                    st.text(f"📄 {file.name} ({file.size/1024:.1f} KB)")
            
            # 번역 설정
            st.subheader("🌍 번역 설정")
            
            col_lang1, col_lang2 = st.columns(2)
            with col_lang1:
                source_lang = st.selectbox(
                    "원본 언어",
                    options=list(st.session_state.translator.supported_languages.keys()),
                    format_func=lambda x: st.session_state.translator.supported_languages[x],
                    index=0
                )
            
            with col_lang2:
                # auto를 제외한 언어 목록
                target_langs = {k: v for k, v in st.session_state.translator.supported_languages.items() if k != 'auto'}
                target_lang = st.selectbox(
                    "대상 언어",
                    options=list(target_langs.keys()),
                    format_func=lambda x: target_langs[x],
                    index=list(target_langs.keys()).index('ko') if 'ko' in target_langs else 0
                )
            
            service = st.radio(
                "번역 서비스",
                options=['openai', 'google', 'deepl'],
                format_func=lambda x: {
                    'openai': '🤖 OpenAI (고품질)',
                    'google': '🌐 Google (무료)',
                    'deepl': '🔤 DeepL (전문)'
                }[x],
                horizontal=True
            )
            
            # 고급 설정
            with st.expander("🔧 고급 설정"):
                preserve_formatting = st.checkbox(
                    "서식 보존",
                    value=True,
                    help="문서의 원본 서식을 최대한 보존합니다"
                )
                
                enable_ocr = st.checkbox(
                    "OCR 활성화",
                    value=False,
                    help="이미지 내 텍스트를 인식하여 번역합니다 (처리 시간 증가)"
                )
                
                output_formats = st.multiselect(
                    "출력 형식",
                    options=["번역본", "원문-번역 대조본", "주석 추가본"],
                    default=["번역본"]
                )
            
            # 비용 예상
            if uploaded_files and service:
                total_size = sum(file.size for file in uploaded_files) / 1024 / 1024  # MB
                
                if service == 'openai':
                    estimated_cost = total_size * 0.01  # 대략적인 추정
                    st.info(f"💰 예상 비용: ${estimated_cost:.2f} (약 ₩{estimated_cost*1300:.0f})")
                elif service == 'deepl':
                    estimated_chars = total_size * 500000
                    estimated_cost = estimated_chars / 1000000 * 20
                    st.info(f"💰 예상 비용: ${estimated_cost:.2f} (약 ₩{estimated_cost*1300:.0f})")
                else:
                    st.info("✨ 무료 서비스")
            
            # 번역 버튼
            st.divider()
            if st.button("🚀 번역 시작", type="primary", use_container_width=True, 
                        disabled=st.session_state.translation_in_progress):
                
                if not uploaded_files:
                    st.error("파일을 선택하세요")
                elif not st.session_state.translator.validate_api_key(service):
                    st.error(f"{service} 서비스를 사용하려면 API 키가 필요합니다")
                else:
                    st.session_state.translation_in_progress = True
                    st.rerun()
        
        with col2:
            st.subheader("📤 번역 결과")
            
            # 번역 진행 상황
            if st.session_state.translation_in_progress and uploaded_files:
                progress_bar = st.progress(0)
                status_text = st.empty()
                output_container = st.container()
                
                translated_files = []
                total_files = len(uploaded_files)
                
                for idx, file in enumerate(uploaded_files):
                    progress = (idx / total_files)
                    progress_bar.progress(progress)
                    status_text.text(f"📄 {file.name} 번역 중... ({idx+1}/{total_files})")
                    
                    try:
                        # 임시 파일로 저장
                        temp_input = Path(st.session_state.temp_dir) / file.name
                        with open(temp_input, 'wb') as f:
                            f.write(file.getbuffer())
                        
                        # 번역 실행
                        result = st.session_state.translator.translate_document(
                            file_path=str(temp_input),
                            source_lang=source_lang,
                            target_lang=target_lang,
                            service=service,
                            output_dir=st.session_state.temp_dir,
                            preserve_formatting=preserve_formatting,
                            enable_ocr=enable_ocr
                        )
                        
                        if result['success']:
                            translated_files.append(result)
                            with output_container:
                                st.success(f"✅ {file.name} 번역 완료 ({result['elapsed_time']:.1f}초)")
                        else:
                            with output_container:
                                st.error(f"❌ {file.name} 번역 실패: {result.get('error', '알 수 없는 오류')}")
                        
                        # 기록 추가
                        st.session_state.translation_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'filename': file.name,
                            'source_lang': source_lang,
                            'target_lang': target_lang,
                            'service': service,
                            'success': result['success'],
                            'elapsed_time': result.get('elapsed_time', 0)
                        })
                        
                    except Exception as e:
                        logger.error(f"번역 오류: {e}")
                        with output_container:
                            st.error(f"❌ {file.name} 처리 중 오류: {str(e)}")
                
                progress_bar.progress(1.0)
                status_text.text("✨ 모든 번역 완료!")
                
                # 다운로드 섹션
                if translated_files:
                    st.divider()
                    st.subheader("💾 다운로드")
                    
                    # 개별 파일 다운로드
                    for result in translated_files:
                        if result.get('output_file'):
                            file_path = result['output_file']
                            if Path(file_path).exists():
                                with open(file_path, 'rb') as f:
                                    file_data = f.read()
                                
                                st.download_button(
                                    label=f"📥 {Path(file_path).name}",
                                    data=file_data,
                                    file_name=Path(file_path).name,
                                    mime='application/octet-stream',
                                    use_container_width=True
                                )
                    
                    # ZIP 다운로드
                    if len(translated_files) > 1:
                        st.divider()
                        all_files = [r['output_file'] for r in translated_files if r.get('output_file')]
                        if all_files:
                            zip_data = create_zip_download(all_files)
                            st.download_button(
                                label="📦 모든 파일 다운로드 (ZIP)",
                                data=zip_data,
                                file_name=f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime='application/zip',
                                type="primary",
                                use_container_width=True
                            )
                
                st.session_state.translation_in_progress = False
            
            elif not st.session_state.translation_in_progress:
                st.info("📝 번역할 문서를 선택하고 '번역 시작' 버튼을 클릭하세요")
    
    # 번역 기록 탭
    with tab2:
        st.subheader("📜 번역 기록")
        
        if st.session_state.translation_history:
            # 최신순으로 정렬
            history_df = []
            for record in reversed(st.session_state.translation_history[-100:]):  # 최근 100개
                history_df.append({
                    '시간': record['timestamp'][:19],
                    '파일명': record['filename'],
                    '언어': f"{record['source_lang']} → {record['target_lang']}",
                    '서비스': record['service'],
                    '소요시간': f"{record['elapsed_time']:.1f}s",
                    '상태': '✅' if record['success'] else '❌'
                })
            
            st.dataframe(history_df, use_container_width=True, height=400)
            
            # 기록 내보내기
            if st.button("💾 기록 내보내기 (JSON)"):
                json_str = json.dumps(st.session_state.translation_history, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 다운로드",
                    data=json_str,
                    file_name=f"translation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            if st.button("🗑️ 기록 삭제", type="secondary"):
                st.session_state.translation_history = []
                st.rerun()
        else:
            st.info("번역 기록이 없습니다")
    
    # 대시보드 탭
    with tab3:
        st.subheader("📊 번역 통계 대시보드")
        
        if st.session_state.translation_history:
            # 통계 계산
            total_count = len(st.session_state.translation_history)
            success_count = sum(1 for r in st.session_state.translation_history if r['success'])
            fail_count = total_count - success_count
            total_time = sum(r.get('elapsed_time', 0) for r in st.session_state.translation_history)
            avg_time = total_time / total_count if total_count > 0 else 0
            
            # 메트릭 표시
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 번역", f"{total_count}개")
            with col2:
                st.metric("성공률", f"{(success_count/total_count*100):.1f}%")
            with col3:
                st.metric("평균 시간", f"{avg_time:.1f}초")
            with col4:
                st.metric("총 시간", f"{total_time:.0f}초")
            
            # 서비스별 통계
            st.divider()
            service_stats = {}
            for record in st.session_state.translation_history:
                service = record['service']
                if service not in service_stats:
                    service_stats[service] = {'count': 0, 'time': 0}
                service_stats[service]['count'] += 1
                service_stats[service]['time'] += record.get('elapsed_time', 0)
            
            st.subheader("📈 서비스별 사용 통계")
            for service, stats in service_stats.items():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**{service.upper()}**")
                with col2:
                    st.metric("사용 횟수", f"{stats['count']}회")
                with col3:
                    avg = stats['time'] / stats['count'] if stats['count'] > 0 else 0
                    st.metric("평균 시간", f"{avg:.1f}초")
            
            # 파일 형식별 통계
            st.divider()
            st.subheader("📁 파일 형식별 통계")
            format_stats = {}
            for record in st.session_state.translation_history:
                ext = Path(record['filename']).suffix.lower()
                format_stats[ext] = format_stats.get(ext, 0) + 1
            
            if format_stats:
                cols = st.columns(len(format_stats))
                for idx, (fmt, count) in enumerate(format_stats.items()):
                    with cols[idx]:
                        st.metric(fmt.upper(), f"{count}개")
        else:
            st.info("통계를 표시할 데이터가 없습니다")
    
    # 도움말 탭
    with tab4:
        st.markdown("""
        ## 📖 사용 가이드
        
        ### 🚀 빠른 시작
        1. **API 키 설정**: 왼쪽 사이드바에서 OpenAI 또는 DeepL API 키 입력
        2. **파일 업로드**: 번역할 문서 파일 선택 (여러 개 가능)
        3. **언어 선택**: 원본 언어와 대상 언어 선택
        4. **서비스 선택**: 번역 서비스 선택 (OpenAI, Google, DeepL)
        5. **번역 시작**: 버튼 클릭하여 번역 시작
        
        ### 📄 지원 파일 형식
        - **PDF** (.pdf): 레이아웃과 서식 보존
        - **Word** (.docx): 스타일과 표 유지
        - **Excel** (.xlsx): 셀 서식과 수식 보존
        - **PowerPoint** (.pptx): 슬라이드 레이아웃 유지
        - **텍스트** (.txt, .md): 단순 텍스트 번역
        
        ### 🌍 지원 언어
        한국어, 영어, 일본어, 중국어(간체/번체), 프랑스어, 독일어, 스페인어, 러시아어, 이탈리아어, 포르투갈어, 아랍어, 힌디어
        
        ### 💡 팁
        - **대량 번역**: 여러 파일을 한 번에 업로드하여 배치 처리
        - **서식 보존**: 고급 설정에서 '서식 보존' 옵션 활성화
        - **비용 절감**: Google 번역은 무료, OpenAI는 고품질
        - **OCR 기능**: 이미지 내 텍스트도 번역 가능 (처리 시간 증가)
        
        ### 🔑 API 키 얻는 방법
        
        **OpenAI API:**
        1. [OpenAI Platform](https://platform.openai.com/api-keys) 접속
        2. 계정 생성 또는 로그인
        3. 'Create new secret key' 클릭
        4. 생성된 키 복사
        
        **DeepL API:**
        1. [DeepL Pro](https://www.deepl.com/pro-api) 접속
        2. API 플랜 구독
        3. 계정 설정에서 인증 키 확인
        
        ### ⚠️ 주의사항
        - 대용량 파일(50MB+)은 처리 시간이 오래 걸릴 수 있습니다
        - 민감한 정보가 포함된 문서는 주의하여 처리하세요
        - API 키는 안전하게 관리하세요
        
        ### 📧 문의
        GitHub Issues를 통해 문의하세요
        """)

# 앱 실행
if __name__ == "__main__":
    main()
