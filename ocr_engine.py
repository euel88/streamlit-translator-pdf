#!/usr/bin/env python3
"""
OCR 엔진 모듈
이미지 내 텍스트를 인식하고 번역합니다.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import tempfile
import shutil
import warnings

# DLL 로딩 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class OCREngine:
    """OCR 통합 엔진 클래스"""
    
    def __init__(self):
        self.engines = {}
        self._init_engines()
        
        # 지원 언어 매핑
        self.lang_mapping = {
            'ko': ['ko', 'korean'],
            'en': ['en', 'english'],
            'ja': ['ja', 'japanese'],
            'zh': ['ch_sim', 'chinese_simplified'],
            'zh-TW': ['ch_tra', 'chinese_traditional'],
            'fr': ['fr', 'french'],
            'de': ['de', 'german'],
            'es': ['es', 'spanish'],
            'ru': ['ru', 'russian'],
            'it': ['it', 'italian'],
            'pt': ['pt', 'portuguese'],
            'ar': ['ar', 'arabic'],
            'hi': ['hi', 'hindi']
        }
    
    def _safe_import_module(self, module_name: str) -> Optional[Any]:
        """안전한 모듈 import"""
        try:
            return __import__(module_name)
        except (ImportError, OSError, RuntimeError) as e:
            logger.debug(f"{module_name} import 실패: {type(e).__name__}: {str(e)}")
            return None
        except Exception as e:
            logger.debug(f"{module_name} import 중 예외: {type(e).__name__}: {str(e)}")
            return None
    
    def _init_engines(self):
        """OCR 엔진 초기화 (안전 모드)"""
        
        # EasyOCR 시도
        try:
            # Windows에서 DLL 문제 방지
            if sys.platform == 'win32':
                os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            easyocr = self._safe_import_module('easyocr')
            if easyocr:
                self.engines['easyocr'] = {
                    'available': True,
                    'module': easyocr,
                    'reader': None,
                    'priority': 1
                }
                logger.info("EasyOCR 엔진 사용 가능")
            else:
                self.engines['easyocr'] = {'available': False}
                logger.info("EasyOCR을 사용할 수 없습니다.")
        except Exception as e:
            self.engines['easyocr'] = {'available': False}
            logger.info(f"EasyOCR 초기화 실패: {type(e).__name__}")
        
        # Tesseract 시도
        try:
            pytesseract = self._safe_import_module('pytesseract')
            if pytesseract:
                # Tesseract 실행 파일 확인
                try:
                    pytesseract.get_tesseract_version()
                    self.engines['tesseract'] = {
                        'available': True,
                        'module': pytesseract,
                        'priority': 2
                    }
                    logger.info("Tesseract OCR 엔진 사용 가능")
                except Exception:
                    self.engines['tesseract'] = {'available': False}
                    logger.info("Tesseract 실행 파일을 찾을 수 없습니다.")
            else:
                self.engines['tesseract'] = {'available': False}
        except Exception:
            self.engines['tesseract'] = {'available': False}
        
        # PaddleOCR 시도
        try:
            paddleocr_module = self._safe_import_module('paddleocr')
            if paddleocr_module:
                self.engines['paddle'] = {
                    'available': True,
                    'module': paddleocr_module,
                    'reader': None,
                    'priority': 3
                }
                logger.info("PaddleOCR 엔진 사용 가능")
            else:
                self.engines['paddle'] = {'available': False}
        except Exception:
            self.engines['paddle'] = {'available': False}
        
        # 사용 가능한 엔진 확인
        available_count = sum(1 for engine in self.engines.values() if engine.get('available', False))
        
        if available_count == 0:
            logger.warning("사용 가능한 OCR 엔진이 없습니다. OCR 기능이 비활성화됩니다.")
            logger.warning("OCR 기능을 사용하려면:")
            logger.warning("1. install_ocr.py를 실행하세요")
            if sys.platform == 'win32':
                logger.warning("2. Visual C++ Redistributable을 설치하세요")
                logger.warning("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        else:
            logger.info(f"총 {available_count}개의 OCR 엔진 사용 가능")
    
    def get_best_engine(self) -> Optional[str]:
        """사용 가능한 최적의 OCR 엔진 선택"""
        available_engines = [
            (name, info) for name, info in self.engines.items()
            if info.get('available', False)
        ]
        
        if not available_engines:
            return None
        
        # 우선순위로 정렬
        available_engines.sort(key=lambda x: x[1].get('priority', 999))
        return available_engines[0][0]
    
    def is_available(self) -> bool:
        """OCR 기능 사용 가능 여부"""
        return self.get_best_engine() is not None
    
    def extract_text(self, image_path: str, language: str = 'auto') -> List[Dict]:
        """
        이미지에서 텍스트 추출
        
        Returns:
            List[Dict]: 감지된 텍스트 정보
                - text: 추출된 텍스트
                - bbox: 경계 상자 좌표 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                - confidence: 신뢰도 (0-1)
        """
        engine = self.get_best_engine()
        if not engine:
            logger.warning("사용 가능한 OCR 엔진이 없습니다.")
            return []
        
        logger.info(f"{engine} 엔진으로 텍스트 추출 중...")
        
        try:
            # 이미지 로드
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = image_path
            
            # RGB로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 엔진별 처리
            if engine == 'easyocr':
                return self._extract_with_easyocr(image, language)
            elif engine == 'tesseract':
                return self._extract_with_tesseract(image, language)
            elif engine == 'paddle':
                return self._extract_with_paddle(image, language)
            else:
                return []
                
        except Exception as e:
            logger.error(f"OCR 텍스트 추출 실패: {type(e).__name__}: {str(e)}")
            return []
    
    def _extract_with_easyocr(self, image: Image.Image, language: str) -> List[Dict]:
        """EasyOCR로 텍스트 추출"""
        try:
            easyocr = self.engines['easyocr'].get('module')
            if not easyocr:
                return []
            
            # 언어 설정
            if language == 'auto':
                langs = ['ko', 'en']
            else:
                langs = self.lang_mapping.get(language, [language])
                if isinstance(langs, list):
                    langs = [langs[0]]
            
            # Reader 초기화 (언어별로 캐싱)
            reader_key = ','.join(langs)
            if 'readers' not in self.engines['easyocr']:
                self.engines['easyocr']['readers'] = {}
            
            if reader_key not in self.engines['easyocr']['readers']:
                logger.info(f"EasyOCR Reader 초기화 중... (언어: {langs})")
                try:
                    reader = easyocr.Reader(langs, gpu=False)
                    self.engines['easyocr']['readers'][reader_key] = reader
                except Exception as e:
                    logger.error(f"EasyOCR Reader 초기화 실패: {e}")
                    return []
            else:
                reader = self.engines['easyocr']['readers'][reader_key]
            
            # 텍스트 추출
            np_image = np.array(image)
            results = reader.readtext(np_image)
            
            # 결과 변환
            extracted = []
            for bbox, text, confidence in results:
                # bbox를 표준 형식으로 변환
                if len(bbox) == 4 and all(isinstance(point, (list, tuple)) for point in bbox):
                    bbox_normalized = bbox
                else:
                    # 다른 형식의 bbox 처리
                    bbox_normalized = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], 
                                     [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
                
                extracted.append({
                    'text': text,
                    'bbox': bbox_normalized,
                    'confidence': confidence
                })
            
            return extracted
            
        except Exception as e:
            logger.error(f"EasyOCR 처리 중 오류: {type(e).__name__}: {str(e)}")
            return []
    
    def _extract_with_tesseract(self, image: Image.Image, language: str) -> List[Dict]:
        """Tesseract로 텍스트 추출"""
        try:
            pytesseract = self.engines['tesseract'].get('module')
            if not pytesseract:
                return []
            
            # 언어 설정
            if language == 'auto':
                tess_lang = 'eng+kor'
            else:
                lang_map = {
                    'ko': 'kor',
                    'en': 'eng',
                    'ja': 'jpn',
                    'zh': 'chi_sim',
                    'zh-TW': 'chi_tra',
                    'fr': 'fra',
                    'de': 'deu',
                    'es': 'spa',
                    'ru': 'rus',
                    'it': 'ita',
                    'pt': 'por',
                    'ar': 'ara',
                    'hi': 'hin'
                }
                tess_lang = lang_map.get(language, 'eng')
            
            # 텍스트 추출 (상세 정보 포함)
            data = pytesseract.image_to_data(
                image, 
                lang=tess_lang,
                output_type=pytesseract.Output.DICT
            )
            
            # 결과 변환
            extracted = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text:  # 빈 텍스트 제외
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    conf = data['conf'][i] / 100.0  # 0-100을 0-1로 변환
                    
                    # 신뢰도가 너무 낮은 것 제외
                    if conf > 0.3:
                        bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        extracted.append({
                            'text': text,
                            'bbox': bbox,
                            'confidence': conf
                        })
            
            return extracted
            
        except Exception as e:
            logger.error(f"Tesseract 처리 중 오류: {type(e).__name__}: {str(e)}")
            return []
    
    def _extract_with_paddle(self, image: Image.Image, language: str) -> List[Dict]:
        """PaddleOCR로 텍스트 추출"""
        try:
            paddleocr_module = self.engines['paddle'].get('module')
            if not paddleocr_module:
                return []
            
            # 언어 설정
            if language == 'auto':
                lang = 'ch'
            else:
                lang_map = {
                    'ko': 'korean',
                    'en': 'en',
                    'ja': 'japan',
                    'zh': 'ch',
                    'zh-TW': 'ch',
                    'fr': 'fr',
                    'de': 'german',
                    'es': 'es',
                    'ru': 'ru',
                    'ar': 'ar'
                }
                lang = lang_map.get(language, 'en')
            
            # PaddleOCR 초기화
            if self.engines['paddle']['reader'] is None:
                try:
                    self.engines['paddle']['reader'] = paddleocr_module.PaddleOCR(
                        use_angle_cls=True,
                        lang=lang,
                        use_gpu=False
                    )
                except Exception as e:
                    logger.error(f"PaddleOCR 초기화 실패: {e}")
                    return []
            
            # 텍스트 추출
            np_image = np.array(image)
            result = self.engines['paddle']['reader'].ocr(np_image, cls=True)
            
            # 결과 변환
            extracted = []
            if result and isinstance(result, list):
                for line in result:
                    if isinstance(line, list):
                        for word_info in line:
                            if len(word_info) >= 2:
                                bbox = word_info[0]
                                text = word_info[1][0] if isinstance(word_info[1], tuple) else word_info[1]
                                confidence = word_info[1][1] if isinstance(word_info[1], tuple) and len(word_info[1]) > 1 else 0.9
                                
                                extracted.append({
                                    'text': text,
                                    'bbox': bbox,
                                    'confidence': confidence
                                })
            
            return extracted
            
        except Exception as e:
            logger.error(f"PaddleOCR 처리 중 오류: {type(e).__name__}: {str(e)}")
            return []
    
    def translate_image_text(self, image_path: str, source_lang: str, target_lang: str,
                           service: str, translator_instance) -> Tuple[Image.Image, List[Dict]]:
        """
        이미지 내 텍스트를 번역하고 오버레이
        
        Returns:
            Tuple[Image.Image, List[Dict]]: 번역된 이미지와 번역 정보
        """
        try:
            # OCR 사용 가능 확인
            if not self.is_available():
                logger.warning("OCR 엔진을 사용할 수 없습니다.")
                if isinstance(image_path, str):
                    return Image.open(image_path), []
                return image_path, []
            
            # 텍스트 추출
            extracted_texts = self.extract_text(image_path, source_lang)
            
            if not extracted_texts:
                logger.info("추출된 텍스트가 없습니다.")
                if isinstance(image_path, str):
                    return Image.open(image_path), []
                return image_path, []
            
            # 이미지 로드
            image = Image.open(image_path) if isinstance(image_path, str) else image_path
            
            # 번역된 이미지 생성
            translated_image = image.copy()
            draw = ImageDraw.Draw(translated_image)
            
            # 폰트 설정
            font = self._get_font(target_lang)
            
            translations = []
            
            for text_info in extracted_texts:
                original_text = text_info['text']
                bbox = text_info['bbox']
                
                # 번역
                try:
                    translated_text = translator_instance.translate_text(
                        original_text, source_lang, target_lang, service
                    )
                    
                    translations.append({
                        'original': original_text,
                        'translated': translated_text,
                        'bbox': bbox,
                        'confidence': text_info['confidence']
                    })
                    
                    # 배경 지우기 (흰색으로 채우기)
                    self._clear_text_area(draw, bbox)
                    
                    # 번역된 텍스트 그리기
                    self._draw_text(draw, translated_text, bbox, font, target_lang)
                    
                except Exception as e:
                    logger.warning(f"텍스트 번역 실패: {e}")
                    translations.append({
                        'original': original_text,
                        'translated': original_text,
                        'bbox': bbox,
                        'confidence': text_info['confidence'],
                        'error': str(e)
                    })
            
            return translated_image, translations
            
        except Exception as e:
            logger.error(f"이미지 텍스트 번역 중 오류: {type(e).__name__}: {str(e)}")
            if isinstance(image_path, str):
                return Image.open(image_path), []
            return image_path, []
    
    def _get_font(self, language: str) -> ImageFont.ImageFont:
        """언어별 적절한 폰트 선택"""
        font_size = 16
        
        # 시스템 폰트 경로
        font_paths = {
            'ko': [
                'C:/Windows/Fonts/malgun.ttf',  # Windows
                'C:/Windows/Fonts/NanumGothic.ttf',  # Windows Nanum
                '/System/Library/Fonts/AppleSDGothicNeo.ttc',  # macOS
                '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # Linux
            ],
            'ja': [
                'C:/Windows/Fonts/msgothic.ttc',  # Windows
                'C:/Windows/Fonts/YuGothM.ttc',  # Windows Yu Gothic
                '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',  # macOS
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # Linux
            ],
            'zh': [
                'C:/Windows/Fonts/simhei.ttf',  # Windows
                'C:/Windows/Fonts/msyh.ttc',  # Windows YaHei
                '/System/Library/Fonts/PingFang.ttc',  # macOS
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # Linux
            ],
            'default': [
                'C:/Windows/Fonts/arial.ttf',  # Windows
                'C:/Windows/Fonts/segoeui.ttf',  # Windows Segoe UI
                '/System/Library/Fonts/Helvetica.ttc',  # macOS
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'  # Linux
            ]
        }
        
        # 언어별 폰트 시도
        font_list = font_paths.get(language, font_paths['default'])
        
        for font_path in font_list:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, font_size)
                except Exception:
                    continue
        
        # fonts 디렉토리에서 찾기
        fonts_dir = Path('fonts')
        if fonts_dir.exists():
            for font_file in fonts_dir.glob('*.ttf'):
                try:
                    return ImageFont.truetype(str(font_file), font_size)
                except Exception:
                    continue
            
            for font_file in fonts_dir.glob('*.ttc'):
                try:
                    return ImageFont.truetype(str(font_file), font_size)
                except Exception:
                    continue
        
        # 기본 폰트
        try:
            return ImageFont.load_default()
        except Exception:
            # 최후의 수단
            return None
    
    def _clear_text_area(self, draw: ImageDraw.Draw, bbox: List[List[int]]):
        """텍스트 영역을 배경색으로 지우기"""
        try:
            # bbox를 사각형으로 변환
            points = [(int(p[0]), int(p[1])) for p in bbox]
            
            # 다각형 채우기 (흰색 배경)
            draw.polygon(points, fill='white', outline='white')
        except Exception as e:
            logger.debug(f"텍스트 영역 지우기 실패: {e}")
    
    def _draw_text(self, draw: ImageDraw.Draw, text: str, bbox: List[List[int]], 
                   font: Optional[ImageFont.ImageFont], language: str):
        """텍스트 그리기 (bbox에 맞춰서)"""
        try:
            # bbox의 중심점 계산
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 폰트가 없으면 기본 처리
            if font is None:
                x = x_min + 5
                y = y_min + 5
                draw.text((x, y), text, fill='black')
                return
            
            # 텍스트 크기 계산
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                # 구버전 Pillow 호환성
                text_width, text_height = draw.textsize(text, font=font)
            
            # 중앙 정렬
            x = x_min + (x_max - x_min - text_width) // 2
            y = y_min + (y_max - y_min - text_height) // 2
            
            # 텍스트 그리기
            draw.text((x, y), text, fill='black', font=font)
            
        except Exception as e:
            logger.debug(f"텍스트 그리기 실패: {e}")
            # 폴백: 기본 위치에 텍스트 그리기
            try:
                x = int(bbox[0][0]) + 5
                y = int(bbox[0][1]) + 5
                draw.text((x, y), text, fill='black')
            except:
                pass
    
    def process_pdf_with_ocr(self, pdf_path: str, source_lang: str, target_lang: str,
                           service: str, translator_instance) -> str:
        """PDF 내 이미지의 텍스트를 번역"""
        if not self.is_available():
            logger.warning("OCR 기능을 사용할 수 없습니다.")
            return pdf_path
        
        try:
            # PyMuPDF를 안전하게 import
            fitz = self._safe_import_module('fitz')
            if not fitz:
                logger.warning("PyMuPDF를 사용할 수 없습니다.")
                return pdf_path
            
            # PDF 열기
            pdf_document = fitz.open(pdf_path)
            output_pdf = fitz.open()
            
            # 페이지별 처리
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # 이미지 추출
                image_list = page.get_images()
                
                if image_list:
                    logger.info(f"페이지 {page_num + 1}에서 {len(image_list)}개 이미지 발견")
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # 이미지 추출
                            xref = img[0]
                            pix = fitz.Pixmap(pdf_document, xref)
                            
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                # PIL 이미지로 변환
                                img_data = pix.tobytes("png")
                                pil_image = Image.open(io.BytesIO(img_data))
                                
                                # OCR 및 번역
                                translated_image, _ = self.translate_image_text(
                                    pil_image, source_lang, target_lang, 
                                    service, translator_instance
                                )
                                
                                # 번역된 이미지로 교체
                                img_buffer = io.BytesIO()
                                translated_image.save(img_buffer, format='PNG')
                                img_buffer.seek(0)
                                
                                # 새 이미지 삽입
                                new_xref = output_pdf.new_xref(0)
                                output_pdf.update_stream(new_xref, img_buffer.read())
                                
                        except Exception as e:
                            logger.warning(f"이미지 처리 실패: {e}")
                
                # 페이지 복사
                output_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
            
            # 새 PDF 저장
            output_path = pdf_path.replace('.pdf', '_ocr.pdf')
            output_pdf.save(output_path)
            output_pdf.close()
            pdf_document.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"PDF OCR 처리 실패: {type(e).__name__}: {str(e)}")
            return pdf_path
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """OCR 정확도 향상을 위한 이미지 전처리"""
        try:
            cv2 = self._safe_import_module('cv2')
            if not cv2:
                return image
            
            # PIL을 numpy 배열로 변환
            img_array = np.array(image)
            
            # 그레이스케일 변환
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # 노이즈 제거
            denoised = cv2.fastNlDenoising(gray)
            
            # 대비 향상
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # 이진화
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 모폴로지 연산 (텍스트 영역 강화)
            kernel = np.ones((2, 2), np.uint8)
            morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # PIL 이미지로 변환
            return Image.fromarray(morphed)
            
        except Exception as e:
            logger.debug(f"이미지 전처리 실패: {e}")
            return image
    
    def batch_extract_text(self, image_paths: List[str], language: str = 'auto',
                          preprocess: bool = True) -> Dict[str, List[Dict]]:
        """여러 이미지에서 일괄 텍스트 추출"""
        if not self.is_available():
            logger.warning("OCR 기능을 사용할 수 없습니다.")
            return {path: [] for path in image_paths}
        
        results = {}
        
        for image_path in image_paths:
            try:
                # 이미지 로드
                image = Image.open(image_path)
                
                # 전처리 (선택사항)
                if preprocess:
                    image = self.preprocess_image(image)
                
                # 텍스트 추출
                extracted = self.extract_text(image, language)
                results[image_path] = extracted
                
            except Exception as e:
                logger.error(f"이미지 처리 실패 ({image_path}): {e}")
                results[image_path] = []
        
        return results
    
    def create_searchable_pdf(self, image_paths: List[str], output_path: str,
                            language: str = 'auto') -> bool:
        """이미지들로부터 검색 가능한 PDF 생성"""
        if not self.is_available():
            logger.warning("OCR 기능을 사용할 수 없습니다.")
            return False
        
        try:
            reportlab = self._safe_import_module('reportlab')
            if not reportlab:
                logger.warning("ReportLab을 사용할 수 없습니다.")
                return False
            
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            
            # PDF 생성
            c = canvas.Canvas(output_path, pagesize=A4)
            page_width, page_height = A4
            
            for image_path in image_paths:
                try:
                    # 이미지 로드
                    image = Image.open(image_path)
                    
                    # 텍스트 추출
                    extracted_texts = self.extract_text(image, language)
                    
                    # 이미지 추가
                    img_width, img_height = image.size
                    aspect = img_height / float(img_width)
                    
                    # 페이지에 맞게 크기 조정
                    if aspect > 1:
                        display_width = page_width * 0.8
                        display_height = display_width * aspect
                    else:
                        display_height = page_height * 0.8
                        display_width = display_height / aspect
                    
                    x = (page_width - display_width) / 2
                    y = (page_height - display_height) / 2
                    
                    # 이미지 그리기
                    c.drawInlineImage(image_path, x, y, display_width, display_height)
                    
                    # 투명 텍스트 레이어 추가 (검색 가능하게)
                    c.setFillAlpha(0)  # 투명하게
                    
                    for text_info in extracted_texts:
                        text = text_info['text']
                        bbox = text_info['bbox']
                        
                        # 좌표 변환
                        text_x = x + (bbox[0][0] / img_width) * display_width
                        text_y = y + display_height - (bbox[0][1] / img_height) * display_height
                        
                        c.drawString(text_x, text_y, text)
                    
                    c.setFillAlpha(1)  # 불투명으로 복원
                    c.showPage()
                    
                except Exception as e:
                    logger.error(f"페이지 처리 실패 ({image_path}): {e}")
            
            c.save()
            return True
            
        except Exception as e:
            logger.error(f"검색 가능한 PDF 생성 실패: {e}")
            return False


# 싱글톤 인스턴스
_ocr_instance = None

def get_ocr_engine() -> OCREngine:
    """OCR 엔진 인스턴스 반환 (싱글톤)"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = OCREngine()
    return _ocr_instance
