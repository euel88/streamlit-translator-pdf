#!/usr/bin/env python3
"""
통합 문서 번역기 핵심 모듈
모든 문서 형식의 번역을 관리합니다.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ConfigManager:
    """설정 및 API 키 관리 클래스"""
    
    def __init__(self):
        self.config_dir = Path('config')
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / 'settings.json'
        self.key_file = self.config_dir / '.keys'
        
        # 메모리 내 API 키 저장
        self._api_keys = {}
        
        # 암호화 키 생성/로드
        self._init_encryption()
        
        # 저장된 설정 로드
        self.load_config()
    
    def _init_encryption(self):
        """암호화 초기화"""
        salt_file = self.config_dir / '.salt'
        
        if salt_file.exists():
            with open(salt_file, 'rb') as f:
                salt = f.read()
        else:
            salt = os.urandom(16)
            with open(salt_file, 'wb') as f:
                f.write(salt)
        
        # 기기별 고유 키 생성
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(b"document-translator-keys"))
        self.cipher = Fernet(key)
    
    def set_api_key(self, service: str, key: str, save: bool = True):
        """API 키 설정"""
        self._api_keys[service] = key
        
        if save:
            self.save_keys()
    
    def get_api_key(self, service: str) -> Optional[str]:
        """API 키 조회"""
        return self._api_keys.get(service)
    
    def save_keys(self):
        """API 키를 암호화하여 저장"""
        try:
            encrypted_data = self.cipher.encrypt(
                json.dumps(self._api_keys).encode()
            )
            with open(self.key_file, 'wb') as f:
                f.write(encrypted_data)
            logger.info("API 키가 안전하게 저장되었습니다.")
        except Exception as e:
            logger.error(f"API 키 저장 실패: {e}")
    
    def load_keys(self):
        """저장된 API 키 로드"""
        if self.key_file.exists():
            try:
                with open(self.key_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                self._api_keys = json.loads(decrypted_data.decode())
                logger.info("저장된 API 키를 로드했습니다.")
            except Exception as e:
                logger.warning(f"API 키 로드 실패: {e}")
                self._api_keys = {}
    
    def save_config(self, config: dict):
        """일반 설정 저장"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def load_config(self) -> dict:
        """설정 로드"""
        self.load_keys()  # API 키도 함께 로드
        
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 기본 설정
        return {
            'openai_model': 'gpt-4o-mini',
            'default_source_lang': 'en',
            'default_target_lang': 'ko',
            'threads': 4,
            'preserve_formatting': True,
            'ocr_enabled': False
        }
    
    def clear_api_keys(self):
        """모든 API 키 삭제"""
        self._api_keys = {}
        if self.key_file.exists():
            self.key_file.unlink()


class TranslationService(ABC):
    """번역 서비스 추상 클래스"""
    
    @abstractmethod
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """텍스트 번역"""
        pass
    
    @abstractmethod
    def validate_key(self) -> bool:
        """API 키 유효성 확인"""
        pass


class OpenAITranslationService(TranslationService):
    """OpenAI 번역 서비스"""
    
    def __init__(self, api_key: str, model: str = 'gpt-4o-mini'):
        self.api_key = api_key
        self.model = model
        
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """OpenAI API를 사용한 번역"""
        if not text.strip():
            return text
            
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            # 언어 매핑
            lang_map = {
                'ko': 'Korean',
                'en': 'English',
                'ja': 'Japanese',
                'zh': 'Chinese (Simplified)',
                'zh-TW': 'Chinese (Traditional)',
                'fr': 'French',
                'de': 'German',
                'es': 'Spanish',
                'ru': 'Russian',
                'it': 'Italian',
                'pt': 'Portuguese',
                'ar': 'Arabic',
                'hi': 'Hindi'
            }
            
            source_name = lang_map.get(source_lang, source_lang)
            target_name = lang_map.get(target_lang, target_lang)
            
            prompt = f"""Translate the following text from {source_name} to {target_name}.
Preserve all formatting, technical terms, and mathematical expressions.
Only provide the translation without any additional explanation.

Text to translate:
{text}"""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate accurately while preserving formatting and technical terminology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=min(len(text) * 3, 4000)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI 번역 오류: {e}")
            raise
    
    def validate_key(self) -> bool:
        """API 키 유효성 확인"""
        return bool(self.api_key and self.api_key.startswith('sk-'))


class GoogleTranslationService(TranslationService):
    """Google 번역 서비스 (무료)"""
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Google Translate API 사용"""
        if not text.strip():
            return text
            
        try:
            from googletrans import Translator
            translator = Translator()
            
            result = translator.translate(text, src=source_lang, dest=target_lang)
            return result.text
            
        except Exception as e:
            logger.error(f"Google 번역 오류: {e}")
            # 대체 방법 시도
            try:
                import requests
                url = "https://translate.googleapis.com/translate_a/single"
                params = {
                    'client': 'gtx',
                    'sl': source_lang,
                    'tl': target_lang,
                    'dt': 't',
                    'q': text
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    result = response.json()
                    return ''.join([item[0] for item in result[0] if item[0]])
            except:
                pass
            raise
    
    def validate_key(self) -> bool:
        """Google 번역은 API 키 불필요"""
        return True


class DeepLTranslationService(TranslationService):
    """DeepL 번역 서비스"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """DeepL API를 사용한 번역"""
        if not text.strip():
            return text
            
        try:
            import deepl
            
            translator = deepl.Translator(self.api_key)
            
            # DeepL 언어 코드 매핑
            deepl_lang_map = {
                'zh': 'ZH',
                'zh-TW': 'ZH-HANT'
            }
            
            target_deepl = deepl_lang_map.get(target_lang, target_lang.upper())
            
            result = translator.translate_text(
                text,
                source_lang=source_lang.upper() if source_lang != 'auto' else None,
                target_lang=target_deepl
            )
            
            return result.text
            
        except Exception as e:
            logger.error(f"DeepL 번역 오류: {e}")
            raise
    
    def validate_key(self) -> bool:
        """API 키 유효성 확인"""
        return bool(self.api_key)


class DocumentTranslatorCore:
    """통합 문서 번역 핵심 클래스"""
    
    def __init__(self):
        """초기화"""
        # 설정 관리자
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        
        # 환경 설정
        self.setup_environment()
        
        # 번역 서비스 초기화
        self.translation_services = {}
        self._init_translation_services()
        
        # 문서 핸들러는 나중에 로드 (순환 참조 방지)
        self.document_handlers = {}
        
        # 번역 통계
        self.stats = {
            'total_translated': 0,
            'total_pages': 0,
            'total_time': 0,
            'by_format': {}
        }
        
        # 지원 언어
        self.supported_languages = {
            'auto': '자동 감지',
            'ko': '한국어',
            'en': '영어',
            'ja': '일본어',
            'zh': '중국어(간체)',
            'zh-TW': '중국어(번체)',
            'fr': '프랑스어',
            'de': '독일어',
            'es': '스페인어',
            'ru': '러시아어',
            'it': '이탈리아어',
            'pt': '포르투갈어',
            'ar': '아랍어',
            'hi': '힌디어'
        }
        
        # 지원 파일 형식
        self.supported_formats = {
            '.pdf': 'PDF 문서',
            '.docx': 'Word 문서',
            '.xlsx': 'Excel 스프레드시트',
            '.pptx': 'PowerPoint 프레젠테이션',
            '.txt': '텍스트 파일',
            '.md': 'Markdown 문서'
        }
    
    def setup_environment(self):
        """환경 설정"""
        # HF 미러 설정 (중국 사용자용)
        if not os.environ.get('HF_ENDPOINT'):
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # OpenAI 모델 설정
        self.openai_model = self.config.get('openai_model', 'gpt-4o-mini')
    
    def _init_translation_services(self):
        """번역 서비스 초기화"""
        # OpenAI
        openai_key = self.get_api_key('openai')
        if openai_key:
            self.translation_services['openai'] = OpenAITranslationService(
                openai_key, self.openai_model
            )
        
        # Google (항상 사용 가능)
        self.translation_services['google'] = GoogleTranslationService()
        
        # DeepL
        deepl_key = self.get_api_key('deepl')
        if deepl_key:
            self.translation_services['deepl'] = DeepLTranslationService(deepl_key)
    
    def init_document_handlers(self):
        """문서 핸들러 초기화 (늦은 초기화)"""
        if not self.document_handlers:
            from document_handler import (
                PDFHandler, WordHandler, ExcelHandler, 
                PowerPointHandler, TextHandler
            )
            
            self.document_handlers = {
                '.pdf': PDFHandler(self),
                '.docx': WordHandler(self),
                '.xlsx': ExcelHandler(self),
                '.pptx': PowerPointHandler(self),
                '.txt': TextHandler(self),
                '.md': TextHandler(self)
            }
    
    def set_api_key(self, service: str, key: str, save: bool = True):
        """API 키 설정"""
        self.config_manager.set_api_key(service, key, save)
        
        # 번역 서비스 재초기화
        if service == 'openai':
            os.environ['OPENAI_API_KEY'] = key
            self.translation_services['openai'] = OpenAITranslationService(
                key, self.openai_model
            )
        elif service == 'deepl':
            self.translation_services['deepl'] = DeepLTranslationService(key)
    
    def get_api_key(self, service: str) -> Optional[str]:
        """API 키 조회"""
        return self.config_manager.get_api_key(service)
    
    def validate_api_key(self, service: str) -> bool:
        """API 키 유효성 확인"""
        if service in self.translation_services:
            return self.translation_services[service].validate_key()
        return False
    
    def translate_text(self, text: str, source_lang: str, target_lang: str, 
                      service: str = 'openai') -> str:
        """텍스트 번역"""
        if not text or not text.strip():
            return text
        
        # 번역 서비스 확인
        if service not in self.translation_services:
            raise ValueError(f"지원하지 않는 번역 서비스: {service}")
        
        translator = self.translation_services[service]
        if not translator.validate_key():
            raise ValueError(f"{service} 서비스를 사용하려면 API 키가 필요합니다.")
        
        # 번역 실행
        return translator.translate_text(text, source_lang, target_lang)
    
    def translate_document(self,
                          file_path: str,
                          source_lang: str = 'auto',
                          target_lang: str = 'ko',
                          service: str = 'openai',
                          output_dir: Optional[str] = None,
                          preserve_formatting: bool = True,
                          enable_ocr: bool = False,
                          callback: Optional[callable] = None) -> Dict:
        """
        문서 번역 실행
        
        Args:
            file_path: 문서 파일 경로
            source_lang: 원본 언어
            target_lang: 대상 언어
            service: 번역 서비스
            output_dir: 출력 디렉토리
            preserve_formatting: 서식 보존 여부
            enable_ocr: OCR 활성화 여부
            callback: 진행률 콜백 함수
            
        Returns:
            결과 딕셔너리
        """
        start_time = datetime.now()
        
        # 파일 확인
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # 지원 형식 확인
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
        
        # API 키 확인
        if not self.validate_api_key(service):
            raise ValueError(f"{service} 서비스를 사용하려면 API 키가 필요합니다.")
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = file_path.parent / 'output'
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 문서 핸들러 초기화
        self.init_document_handlers()
        
        # 해당 형식의 핸들러 가져오기
        handler = self.document_handlers.get(file_ext)
        if not handler:
            raise ValueError(f"처리할 수 없는 파일 형식: {file_ext}")
        
        logger.info(f"번역 시작: {file_path.name} ({file_ext})")
        logger.info(f"설정: {source_lang} → {target_lang} ({service})")
        
        try:
            # 번역 실행
            result = handler.translate(
                file_path=file_path,
                source_lang=source_lang,
                target_lang=target_lang,
                service=service,
                output_dir=output_dir,
                preserve_formatting=preserve_formatting,
                enable_ocr=enable_ocr,
                callback=callback
            )
            
            # 통계 업데이트
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.stats['total_translated'] += 1
            self.stats['total_time'] += elapsed_time
            
            if file_ext not in self.stats['by_format']:
                self.stats['by_format'][file_ext] = 0
            self.stats['by_format'][file_ext] += 1
            
            # 결과 데이터
            result_data = {
                'success': True,
                'input_file': str(file_path),
                'output_file': result.get('output_file'),
                'format': file_ext,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'service': service,
                'elapsed_time': elapsed_time,
                'timestamp': datetime.now().isoformat(),
                'details': result
            }
            
            # 번역 기록 저장
            self.save_translation_log(result_data)
            
            logger.info(f"번역 완료: {elapsed_time:.1f}초")
            
            return result_data
            
        except Exception as e:
            logger.error(f"번역 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_file': str(file_path),
                'format': file_ext,
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_translate(self,
                       file_paths: List[str],
                       source_lang: str = 'auto',
                       target_lang: str = 'ko',
                       service: str = 'openai',
                       output_dir: Optional[str] = None,
                       preserve_formatting: bool = True,
                       enable_ocr: bool = False,
                       progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        여러 문서 배치 번역
        
        Args:
            file_paths: 문서 파일 경로 리스트
            기타 파라미터는 translate_document와 동일
            
        Returns:
            결과 리스트
        """
        results = []
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(i + 1, total_files, file_path)
            
            result = self.translate_document(
                file_path=file_path,
                source_lang=source_lang,
                target_lang=target_lang,
                service=service,
                output_dir=output_dir,
                preserve_formatting=preserve_formatting,
                enable_ocr=enable_ocr
            )
            results.append(result)
        
        return results
    
    def save_translation_log(self, result: Dict):
        """번역 기록 저장"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / 'translation_history.json'
        
        # 기존 로그 읽기
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # 새 로그 추가
        logs.append(result)
        
        # 저장 (최근 100개만 유지)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs[-100:], f, ensure_ascii=False, indent=2)
    
    def get_translation_history(self, limit: int = 10) -> List[Dict]:
        """번역 기록 조회"""
        log_file = Path('logs') / 'translation_history.json'
        
        if not log_file.exists():
            return []
        
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        return logs[-limit:]
    
    def get_stats(self) -> Dict:
        """번역 통계 조회"""
        return {
            'total_translated': self.stats['total_translated'],
            'average_time': self.stats['total_time'] / max(self.stats['total_translated'], 1),
            'by_format': self.stats['by_format'],
            'supported_formats': list(self.supported_formats.keys()),
            'openai_configured': self.validate_api_key('openai'),
            'deepl_configured': self.validate_api_key('deepl')
        }
    
    def estimate_cost(self, file_path: str, service: str = 'openai') -> Dict:
        """번역 비용 예상"""
        file_size = Path(file_path).stat().st_size / 1024 / 1024  # MB
        
        if service == 'openai':
            # 대략적인 추정 (파일 크기 기반)
            estimated_tokens = file_size * 10000  # 1MB당 약 10k 토큰
            
            if self.openai_model == 'gpt-4o-mini':
                cost_per_1k = 0.00015 + 0.0006  # 입력 + 출력
            elif self.openai_model == 'gpt-4o':
                cost_per_1k = 0.005 + 0.015
            else:  # gpt-3.5-turbo
                cost_per_1k = 0.0005 + 0.0015
            
            estimated_cost = estimated_tokens / 1000 * cost_per_1k
            
            return {
                'service': f'OpenAI {self.openai_model}',
                'estimated_cost_usd': round(estimated_cost, 4),
                'estimated_cost_krw': round(estimated_cost * 1300, 0),
                'note': '실제 비용은 문서 내용에 따라 달라질 수 있습니다.'
            }
        elif service == 'deepl':
            # DeepL은 문자 수 기반
            estimated_chars = file_size * 500000  # 1MB당 약 50만 문자
            cost_per_million = 20  # $20 per million chars
            
            estimated_cost = estimated_chars / 1000000 * cost_per_million
            
            return {
                'service': 'DeepL Pro',
                'estimated_cost_usd': round(estimated_cost, 4),
                'estimated_cost_krw': round(estimated_cost * 1300, 0),
                'note': 'DeepL은 문자 수 기반 과금입니다.'
            }
        else:
            return {
                'service': service,
                'estimated_cost_usd': 0,
                'estimated_cost_krw': 0,
                'note': '무료 서비스입니다.'
            }


# 싱글톤 인스턴스
_translator_instance = None

def get_translator() -> DocumentTranslatorCore:
    """번역기 인스턴스 반환 (싱글톤)"""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = DocumentTranslatorCore()
    return _translator_instance
