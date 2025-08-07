#!/usr/bin/env python3
"""
문서 형식별 처리 모듈
각 문서 형식에 대한 번역 처리를 담당합니다.
PDF 텍스트 뭉개짐 문제 해결 버전
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
import json
import re
from datetime import datetime
import tempfile
import fitz  # PyMuPDF
import io
from PIL import Image

logger = logging.getLogger(__name__)


class DocumentHandler(ABC):
    """문서 처리기 추상 클래스"""
    
    def __init__(self, translator):
        self.translator = translator
        
    @abstractmethod
    def translate(self, file_path: Path, source_lang: str, target_lang: str,
                 service: str, output_dir: Path, preserve_formatting: bool,
                 enable_ocr: bool, callback: Optional[callable] = None) -> Dict:
        """문서 번역 실행"""
        pass
    
    def _create_output_filename(self, input_path: Path, suffix: str = "") -> str:
        """출력 파일명 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = input_path.stem
        extension = input_path.suffix
        
        if suffix:
            return f"{base_name}_{suffix}_{timestamp}{extension}"
        else:
            return f"{base_name}_translated_{timestamp}{extension}"
    
    def _should_translate(self, text: str) -> bool:
        """번역이 필요한 텍스트인지 확인"""
        if not text or not text.strip():
            return False
        
        # 숫자나 특수문자만 있는 경우 제외
        if re.match(r'^[\d\s\W]+$', text):
            return False
        
        # 너무 짧은 텍스트 제외 (단, CJK 문자는 예외)
        if len(text.strip()) < 3 and not re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text):
            return False
        
        return True
    
    def _safe_get_ocr_engine(self):
        """OCR 엔진을 안전하게 가져오기"""
        try:
            from ocr_engine import get_ocr_engine
            ocr = get_ocr_engine()
            if ocr.is_available():
                return ocr
            else:
                logger.info("OCR 엔진을 사용할 수 없습니다.")
                return None
        except ImportError:
            logger.info("OCR 모듈이 설치되지 않았습니다.")
            return None
        except Exception as e:
            logger.warning(f"OCR 엔진 로드 실패: {e}")
            return None


class PDFHandler(DocumentHandler):
    """개선된 PDF 문서 처리기 - 안정적인 텍스트 렌더링"""
    
    def __init__(self, translator):
        super().__init__(translator)
        # 언어별 텍스트 확장 비율
        self.expansion_rates = {
            ('en', 'ko'): 0.7,
            ('ko', 'en'): 1.5,
            ('ja', 'ko'): 0.9,
            ('zh', 'ko'): 0.8,
            ('ko', 'ja'): 1.1,
            ('ko', 'zh'): 1.2,
        }
    
    def translate(self, file_path: Path, source_lang: str, target_lang: str,
                 service: str, output_dir: Path, preserve_formatting: bool,
                 enable_ocr: bool, callback: Optional[callable] = None) -> Dict:
        """PDF 번역 실행 - 안정적인 방식"""
        logger.info(f"PDF 번역 시작: {file_path.name}")
        
        # 메인 번역 시도
        try:
            return self._translate_with_overlay(
                file_path, source_lang, target_lang, service, 
                output_dir, preserve_formatting, callback
            )
        except Exception as e:
            logger.warning(f"오버레이 방식 실패, 대체 방식 시도: {e}")
            # 폴백: 새 PDF 생성 방식
            return self._translate_with_new_pdf(
                file_path, source_lang, target_lang, service, 
                output_dir, callback
            )
    
    def _translate_with_overlay(self, file_path: Path, source_lang: str, target_lang: str,
                                service: str, output_dir: Path, preserve_formatting: bool,
                                callback: Optional[callable] = None) -> Dict:
        """오버레이 방식 번역 (원본 위에 덮어쓰기)"""
        
        pdf_document = fitz.open(str(file_path))
        total_pages = len(pdf_document)
        
        try:
            for page_num in range(total_pages):
                if callback:
                    callback(page_num + 1, total_pages, f"페이지 {page_num + 1}/{total_pages} 번역 중...")
                
                page = pdf_document[page_num]
                
                # 텍스트 블록 추출
                blocks = page.get_text("blocks")
                
                # 텍스트 블록을 뒤에서부터 처리 (레이어 순서 유지)
                for block in reversed(blocks):
                    if block[6] == 0:  # 텍스트 블록인 경우
                        bbox = fitz.Rect(block[:4])
                        original_text = block[4].strip()
                        
                        if self._should_translate(original_text):
                            try:
                                # 텍스트 번역
                                translated_text = self.translator.translate_text(
                                    original_text, source_lang, target_lang, service
                                )
                                
                                # 기존 텍스트 영역을 흰색으로 덮기
                                page.draw_rect(bbox, color=None, fill=(1, 1, 1))
                                
                                # 번역된 텍스트 삽입
                                self._insert_text_safe(
                                    page, bbox, translated_text, 
                                    source_lang, target_lang
                                )
                                
                            except Exception as e:
                                logger.debug(f"블록 번역 실패, 건너뜀: {e}")
                                continue
            
            # 저장
            output_path = output_dir / self._create_output_filename(file_path)
            pdf_document.save(str(output_path), garbage=3, deflate=True)
            pdf_document.close()
            
            return {
                'output_file': str(output_path),
                'format': 'pdf',
                'pages': total_pages
            }
            
        except Exception as e:
            pdf_document.close()
            raise e
    
    def _translate_with_new_pdf(self, file_path: Path, source_lang: str, target_lang: str,
                               service: str, output_dir: Path, 
                               callback: Optional[callable] = None) -> Dict:
        """새 PDF 생성 방식 (더 안전하지만 레이아웃 단순)"""
        logger.info("새 PDF 생성 방식으로 번역")
        
        pdf_document = fitz.open(str(file_path))
        output_pdf = fitz.open()
        total_pages = len(pdf_document)
        
        try:
            for page_num in range(total_pages):
                if callback:
                    callback(page_num + 1, total_pages, f"페이지 {page_num + 1}/{total_pages} 처리 중...")
                
                page = pdf_document[page_num]
                
                # 페이지 이미지로 변환 (배경 유지)
                mat = fitz.Matrix(2, 2)  # 2배 해상도
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # 새 페이지 생성
                new_page = output_pdf.new_page(
                    width=page.rect.width,
                    height=page.rect.height
                )
                
                # 배경 이미지 삽입
                img_data = pix.tobytes("png")
                img_rect = new_page.rect
                new_page.insert_image(img_rect, stream=img_data)
                
                # 텍스트 추출 및 번역
                text_page = page.get_textpage()
                blocks = page.get_text("dict", textpage=text_page)
                
                # 각 텍스트 블록 처리
                for block in blocks["blocks"]:
                    if block["type"] == 0:  # 텍스트 블록
                        self._process_text_block(
                            new_page, block, source_lang, target_lang, service
                        )
            
            # 저장
            output_path = output_dir / self._create_output_filename(file_path)
            output_pdf.save(str(output_path))
            output_pdf.close()
            pdf_document.close()
            
            return {
                'output_file': str(output_path),
                'format': 'pdf',
                'pages': total_pages,
                'note': '새 PDF 생성 방식 사용'
            }
            
        except Exception as e:
            output_pdf.close()
            pdf_document.close()
            logger.error(f"새 PDF 생성 실패: {e}")
            raise e
    
    def _process_text_block(self, page, block: Dict, source_lang: str, 
                          target_lang: str, service: str):
        """텍스트 블록 처리"""
        try:
            # 블록의 전체 텍스트 수집
            block_text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"]
                block_text += " "
            
            block_text = block_text.strip()
            
            if not self._should_translate(block_text):
                return
            
            # 번역
            translated_text = self.translator.translate_text(
                block_text, source_lang, target_lang, service
            )
            
            # 원본 영역에 흰색 사각형 그리기
            bbox = block["bbox"]
            rect = fitz.Rect(bbox)
            
            # 흰색 배경
            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(color=None, fill=(1, 1, 1))
            shape.commit()
            
            # 번역된 텍스트 삽입
            self._insert_text_with_wrap(
                page, rect, translated_text, 
                block.get("lines", []), target_lang
            )
            
        except Exception as e:
            logger.debug(f"텍스트 블록 처리 실패: {e}")
    
    def _insert_text_safe(self, page, rect: fitz.Rect, text: str, 
                         source_lang: str, target_lang: str):
        """안전한 텍스트 삽입"""
        try:
            # 텍스트 확장률 계산
            expansion_key = (source_lang, target_lang)
            expansion_rate = self.expansion_rates.get(expansion_key, 1.0)
            
            # 기본 폰트 크기 계산
            base_font_size = min(12, rect.height / 2)
            
            # 확장률에 따른 폰트 크기 조정
            if expansion_rate > 1.2:
                font_size = base_font_size * 0.7
            elif expansion_rate < 0.8:
                font_size = base_font_size * 1.1
            else:
                font_size = base_font_size
            
            # 최소/최대 크기 제한
            font_size = max(6, min(font_size, 16))
            
            # 텍스트 래핑
            wrapped_lines = self._wrap_text(text, rect.width, font_size)
            
            # 텍스트 삽입 위치 계산
            x = rect.x0 + 2
            y = rect.y0 + font_size
            line_height = font_size * 1.2
            
            # 사용 가능한 폰트 찾기
            fontname = self._get_best_font(target_lang)
            
            # 각 줄 삽입
            for i, line in enumerate(wrapped_lines):
                if y + (i * line_height) > rect.y1 - 2:
                    break  # 영역을 벗어나면 중단
                
                try:
                    page.insert_text(
                        (x, y + (i * line_height)),
                        line,
                        fontsize=font_size,
                        fontname=fontname,
                        color=(0, 0, 0),
                        render_mode=0  # fill
                    )
                except:
                    # 폰트 오류 시 기본 폰트 사용
                    page.insert_text(
                        (x, y + (i * line_height)),
                        line,
                        fontsize=font_size,
                        color=(0, 0, 0)
                    )
                    
        except Exception as e:
            logger.debug(f"텍스트 삽입 실패: {e}")
            # 최후의 수단: 단순 텍스트 삽입
            try:
                page.insert_text(
                    (rect.x0, rect.y0 + 10),
                    text[:50],  # 처음 50자만
                    fontsize=8,
                    color=(0, 0, 0)
                )
            except:
                pass
    
    def _insert_text_with_wrap(self, page, rect: fitz.Rect, text: str, 
                              original_lines: List, target_lang: str):
        """텍스트 래핑과 함께 삽입"""
        try:
            # 원본 스타일 정보 추출
            font_size = 10  # 기본값
            if original_lines and len(original_lines) > 0:
                if "spans" in original_lines[0] and len(original_lines[0]["spans"]) > 0:
                    font_size = original_lines[0]["spans"][0].get("size", 10)
            
            # 텍스트 길이에 따른 폰트 크기 조정
            text_length = len(text)
            if text_length > 100:
                font_size = font_size * 0.8
            elif text_length > 200:
                font_size = font_size * 0.6
            
            font_size = max(6, min(font_size, 14))
            
            # 텍스트 래핑
            lines = self._wrap_text(text, rect.width - 4, font_size)
            
            # 폰트 선택
            fontname = self._get_best_font(target_lang)
            
            # 텍스트 삽입
            x = rect.x0 + 2
            y = rect.y0 + font_size
            line_height = font_size * 1.2
            
            for i, line in enumerate(lines):
                if y + (i * line_height) > rect.y1:
                    break
                
                try:
                    text_inst = page.insert_text(
                        (x, y + (i * line_height)),
                        line,
                        fontsize=font_size,
                        fontname=fontname,
                        color=(0, 0, 0)
                    )
                except Exception as e:
                    # 기본 폰트로 재시도
                    try:
                        page.insert_text(
                            (x, y + (i * line_height)),
                            line,
                            fontsize=font_size,
                            color=(0, 0, 0)
                        )
                    except:
                        pass
                        
        except Exception as e:
            logger.debug(f"래핑 텍스트 삽입 실패: {e}")
    
    def _wrap_text(self, text: str, max_width: float, font_size: float) -> List[str]:
        """텍스트 래핑"""
        # 대략적인 문자당 너비 계산
        char_width = font_size * 0.5  # 평균적인 문자 너비
        chars_per_line = int(max_width / char_width)
        
        if chars_per_line <= 0:
            return [text]
        
        # 단어 단위로 분할
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            
            if current_length + word_length + 1 <= chars_per_line:
                current_line.append(word)
                current_length += word_length + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]
    
    def _get_best_font(self, language: str) -> str:
        """언어에 맞는 최적 폰트 선택"""
        # PyMuPDF 기본 폰트
        font_map = {
            'ko': 'china-s',   # CJK 폰트 (한글 지원)
            'ja': 'japan',     # 일본어
            'zh': 'china-s',   # 중국어 간체
            'zh-TW': 'china-t', # 중국어 번체  
            'ar': 'arab',      # 아랍어
            'hi': 'deva',      # 힌디어
            'ru': 'cour',      # 키릴 문자
        }
        
        # 폰트가 없으면 기본 폰트 사용
        return font_map.get(language, 'helv')
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """텍스트를 청크로 분할"""
        chunks = []
        current_chunk = ""
        
        sentences = text.split('. ')
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class WordHandler(DocumentHandler):
    """Word 문서 처리기"""
    
    def translate(self, file_path: Path, source_lang: str, target_lang: str,
                 service: str, output_dir: Path, preserve_formatting: bool,
                 enable_ocr: bool, callback: Optional[callable] = None) -> Dict:
        """Word 문서 번역"""
        logger.info(f"Word 번역 시작: {file_path.name}")
        
        try:
            from docx import Document
            from docx.shared import RGBColor, Pt
            
            # 문서 열기
            doc = Document(str(file_path))
            output_doc = Document()
            
            # 문서 속성 복사
            if preserve_formatting:
                try:
                    output_doc.core_properties.author = doc.core_properties.author
                    output_doc.core_properties.title = doc.core_properties.title
                except:
                    pass
            
            total_items = len(doc.paragraphs) + sum(len(table.rows) * len(table.columns) for table in doc.tables)
            processed_items = 0
            
            # 단락 번역
            for paragraph in doc.paragraphs:
                if callback:
                    processed_items += 1
                    callback(processed_items, total_items, "단락 번역 중...")
                
                # 빈 단락 처리
                if not paragraph.text.strip():
                    output_doc.add_paragraph()
                    continue
                
                # 새 단락 생성
                new_para = output_doc.add_paragraph()
                
                # 서식 복사
                if preserve_formatting:
                    try:
                        new_para.alignment = paragraph.alignment
                        new_para.paragraph_format.space_before = paragraph.paragraph_format.space_before
                        new_para.paragraph_format.space_after = paragraph.paragraph_format.space_after
                        new_para.paragraph_format.line_spacing = paragraph.paragraph_format.line_spacing
                    except:
                        pass
                
                # Run 단위로 번역
                for run in paragraph.runs:
                    if self._should_translate(run.text):
                        try:
                            translated_text = self.translator.translate_text(
                                run.text, source_lang, target_lang, service
                            )
                        except Exception as e:
                            logger.warning(f"텍스트 번역 실패: {e}")
                            translated_text = run.text
                    else:
                        translated_text = run.text
                    
                    # 새 Run 생성
                    new_run = new_para.add_run(translated_text)
                    
                    # 서식 복사
                    if preserve_formatting:
                        try:
                            if run.bold is not None:
                                new_run.bold = run.bold
                            if run.italic is not None:
                                new_run.italic = run.italic
                            if run.underline:
                                new_run.underline = run.underline
                            if run.font.size:
                                new_run.font.size = run.font.size
                            if run.font.name:
                                new_run.font.name = run.font.name
                        except:
                            pass
            
            # 표 번역
            for table_idx, table in enumerate(doc.tables):
                if callback:
                    callback(processed_items, total_items, f"표 {table_idx + 1} 번역 중...")
                
                # 새 표 생성
                new_table = output_doc.add_table(rows=len(table.rows), cols=len(table.columns))
                
                # 표 스타일 복사
                if preserve_formatting and table.style:
                    try:
                        new_table.style = table.style
                    except:
                        pass
                
                # 셀 번역
                for row_idx, row in enumerate(table.rows):
                    for col_idx, cell in enumerate(row.cells):
                        processed_items += 1
                        
                        new_cell = new_table.rows[row_idx].cells[col_idx]
                        
                        # 셀 텍스트 번역
                        cell_text = cell.text.strip()
                        if self._should_translate(cell_text):
                            try:
                                translated_text = self.translator.translate_text(
                                    cell_text, source_lang, target_lang, service
                                )
                                new_cell.text = translated_text
                            except Exception as e:
                                logger.warning(f"셀 번역 실패: {e}")
                                new_cell.text = cell_text
                        else:
                            new_cell.text = cell_text
            
            # 파일 저장
            output_path = output_dir / self._create_output_filename(file_path)
            output_doc.save(str(output_path))
            
            return {
                'output_file': str(output_path),
                'format': 'docx',
                'paragraphs': len(doc.paragraphs),
                'tables': len(doc.tables)
            }
            
        except Exception as e:
            logger.error(f"Word 번역 실패: {e}")
            raise


class ExcelHandler(DocumentHandler):
    """Excel 문서 처리기"""
    
    def translate(self, file_path: Path, source_lang: str, target_lang: str,
                 service: str, output_dir: Path, preserve_formatting: bool,
                 enable_ocr: bool, callback: Optional[callable] = None) -> Dict:
        """Excel 문서 번역"""
        logger.info(f"Excel 번역 시작: {file_path.name}")
        
        try:
            from openpyxl import load_workbook, Workbook
            from openpyxl.styles import Font, PatternFill, Alignment
            
            # 워크북 열기
            wb = load_workbook(str(file_path), data_only=False)
            output_wb = Workbook()
            
            # 기본 시트 제거
            if 'Sheet' in output_wb.sheetnames:
                output_wb.remove(output_wb['Sheet'])
            
            total_cells = sum(ws.max_row * ws.max_column for ws in wb.worksheets)
            processed_cells = 0
            
            # 시트별 처리
            for sheet_idx, ws in enumerate(wb.worksheets):
                if callback:
                    callback(processed_cells, total_cells, f"시트 '{ws.title}' 번역 중...")
                
                # 새 시트 생성
                output_ws = output_wb.create_sheet(title=ws.title)
                
                # 열 너비 복사
                if preserve_formatting:
                    for col in ws.column_dimensions:
                        output_ws.column_dimensions[col].width = ws.column_dimensions[col].width
                
                # 행 높이 복사
                if preserve_formatting:
                    for row in ws.row_dimensions:
                        output_ws.row_dimensions[row].height = ws.row_dimensions[row].height
                
                # 셀 처리
                for row in range(1, ws.max_row + 1):
                    for col in range(1, ws.max_column + 1):
                        processed_cells += 1
                        
                        cell = ws.cell(row=row, column=col)
                        output_cell = output_ws.cell(row=row, column=col)
                        
                        # 수식 처리
                        if cell.data_type == 'f':
                            # 수식은 그대로 복사
                            output_cell.value = cell.value
                            output_cell._value = cell._value
                            output_cell.data_type = cell.data_type
                        
                        # 텍스트 처리
                        elif cell.value and isinstance(cell.value, str):
                            if self._should_translate(cell.value):
                                try:
                                    translated_text = self.translator.translate_text(
                                        cell.value, source_lang, target_lang, service
                                    )
                                    output_cell.value = translated_text
                                except Exception as e:
                                    logger.warning(f"셀 번역 실패 ({row},{col}): {e}")
                                    output_cell.value = cell.value
                            else:
                                output_cell.value = cell.value
                        else:
                            # 숫자, 날짜 등은 그대로 복사
                            output_cell.value = cell.value
                        
                        # 서식 복사
                        if preserve_formatting:
                            try:
                                # 폰트
                                if cell.font:
                                    output_cell.font = Font(
                                        name=cell.font.name,
                                        size=cell.font.size,
                                        bold=cell.font.bold,
                                        italic=cell.font.italic,
                                        color=cell.font.color
                                    )
                                
                                # 배경색
                                if cell.fill:
                                    output_cell.fill = PatternFill(
                                        fill_type=cell.fill.fill_type,
                                        start_color=cell.fill.start_color,
                                        end_color=cell.fill.end_color
                                    )
                                
                                # 정렬
                                if cell.alignment:
                                    output_cell.alignment = Alignment(
                                        horizontal=cell.alignment.horizontal,
                                        vertical=cell.alignment.vertical,
                                        wrap_text=cell.alignment.wrap_text
                                    )
                                
                                # 숫자 서식
                                if cell.number_format:
                                    output_cell.number_format = cell.number_format
                            except:
                                pass
                
                # 병합 셀 처리
                for merged_range in ws.merged_cells.ranges:
                    output_ws.merge_cells(str(merged_range))
            
            # 파일 저장
            output_path = output_dir / self._create_output_filename(file_path)
            output_wb.save(str(output_path))
            
            return {
                'output_file': str(output_path),
                'format': 'xlsx',
                'sheets': len(wb.worksheets),
                'total_cells': processed_cells
            }
            
        except Exception as e:
            logger.error(f"Excel 번역 실패: {e}")
            raise


class PowerPointHandler(DocumentHandler):
    """PowerPoint 문서 처리기"""
    
    def translate(self, file_path: Path, source_lang: str, target_lang: str,
                 service: str, output_dir: Path, preserve_formatting: bool,
                 enable_ocr: bool, callback: Optional[callable] = None) -> Dict:
        """PowerPoint 프레젠테이션 번역"""
        logger.info(f"PowerPoint 번역 시작: {file_path.name}")
        
        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt
            
            # 프레젠테이션 열기
            prs = Presentation(str(file_path))
            output_prs = Presentation()
            
            # 슬라이드 크기 복사
            output_prs.slide_width = prs.slide_width
            output_prs.slide_height = prs.slide_height
            
            total_items = sum(len(slide.shapes) for slide in prs.slides)
            processed_items = 0
            
            # 슬라이드별 처리
            for slide_idx, slide in enumerate(prs.slides):
                if callback:
                    callback(processed_items, total_items, f"슬라이드 {slide_idx + 1} 번역 중...")
                
                # 슬라이드 레이아웃 복사
                slide_layout = output_prs.slide_layouts[min(slide_idx, len(output_prs.slide_layouts) - 1)]
                output_slide = output_prs.slides.add_slide(slide_layout)
                
                # 도형별 처리
                for shape in slide.shapes:
                    processed_items += 1
                    
                    # 텍스트 프레임이 있는 경우
                    if shape.has_text_frame:
                        # 해당하는 도형 찾기 또는 생성
                        output_shape = self._find_or_create_shape(output_slide, shape)
                        
                        # 단락별 번역
                        for para_idx, paragraph in enumerate(shape.text_frame.paragraphs):
                            if self._should_translate(paragraph.text):
                                try:
                                    translated_text = self.translator.translate_text(
                                        paragraph.text, source_lang, target_lang, service
                                    )
                                    
                                    # 출력 단락 설정
                                    if para_idx < len(output_shape.text_frame.paragraphs):
                                        output_para = output_shape.text_frame.paragraphs[para_idx]
                                    else:
                                        output_para = output_shape.text_frame.add_paragraph()
                                    
                                    output_para.text = translated_text
                                    
                                    # 서식 복사
                                    if preserve_formatting and paragraph.runs:
                                        try:
                                            run = paragraph.runs[0]
                                            output_run = output_para.runs[0] if output_para.runs else output_para.add_run()
                                            
                                            if run.font.name:
                                                output_run.font.name = run.font.name
                                            if run.font.size:
                                                output_run.font.size = run.font.size
                                            if run.font.bold is not None:
                                                output_run.font.bold = run.font.bold
                                            if run.font.italic is not None:
                                                output_run.font.italic = run.font.italic
                                        except:
                                            pass
                                    
                                except Exception as e:
                                    logger.warning(f"텍스트 번역 실패: {e}")
                                    if para_idx < len(output_shape.text_frame.paragraphs):
                                        output_shape.text_frame.paragraphs[para_idx].text = paragraph.text
                    
                    # 표가 있는 경우
                    elif shape.has_table:
                        table = shape.table
                        # 표 번역 로직
                        for row in table.rows:
                            for cell in row.cells:
                                if self._should_translate(cell.text):
                                    try:
                                        translated = self.translator.translate_text(
                                            cell.text, source_lang, target_lang, service
                                        )
                                        cell.text = translated
                                    except:
                                        pass
            
            # 파일 저장
            output_path = output_dir / self._create_output_filename(file_path)
            output_prs.save(str(output_path))
            
            return {
                'output_file': str(output_path),
                'format': 'pptx',
                'slides': len(prs.slides),
                'shapes': processed_items
            }
            
        except Exception as e:
            logger.error(f"PowerPoint 번역 실패: {e}")
            raise
    
    def _find_or_create_shape(self, slide, original_shape):
        """슬라이드에서 해당하는 도형 찾기 또는 생성"""
        # 간단한 구현 - 실제로는 더 정교한 매칭 필요
        for shape in slide.shapes:
            if shape.shape_type == original_shape.shape_type:
                return shape
        
        # 새 텍스트 상자 생성
        from pptx.util import Inches
        left = top = Inches(1)
        width = height = Inches(3)
        
        return slide.shapes.add_textbox(left, top, width, height)


class TextHandler(DocumentHandler):
    """텍스트 파일 처리기 (txt, md)"""
    
    def translate(self, file_path: Path, source_lang: str, target_lang: str,
                 service: str, output_dir: Path, preserve_formatting: bool,
                 enable_ocr: bool, callback: Optional[callable] = None) -> Dict:
        """텍스트 파일 번역"""
        logger.info(f"텍스트 파일 번역 시작: {file_path.name}")
        
        try:
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Markdown 파일인 경우
            if file_path.suffix.lower() == '.md':
                translated_content = self._translate_markdown(
                    content, source_lang, target_lang, service, preserve_formatting
                )
            else:
                # 일반 텍스트 파일
                lines = content.split('\n')
                translated_lines = []
                
                total_lines = len(lines)
                for idx, line in enumerate(lines):
                    if callback:
                        callback(idx + 1, total_lines, f"줄 {idx + 1}/{total_lines} 번역 중...")
                    
                    if self._should_translate(line):
                        try:
                            translated_line = self.translator.translate_text(
                                line, source_lang, target_lang, service
                            )
                            translated_lines.append(translated_line)
                        except Exception as e:
                            logger.warning(f"줄 번역 실패: {e}")
                            translated_lines.append(line)
                    else:
                        translated_lines.append(line)
                
                translated_content = '\n'.join(translated_lines)
            
            # 파일 저장
            output_path = output_dir / self._create_output_filename(file_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            
            return {
                'output_file': str(output_path),
                'format': file_path.suffix[1:],
                'lines': len(content.split('\n')),
                'characters': len(content)
            }
            
        except Exception as e:
            logger.error(f"텍스트 파일 번역 실패: {e}")
            raise
    
    def _translate_markdown(self, content: str, source_lang: str, target_lang: str,
                          service: str, preserve_formatting: bool) -> str:
        """Markdown 문서 번역 (구조 보존)"""
        import re
        
        # 코드 블록 보호
        code_blocks = []
        code_pattern = r'```[\s\S]*?```'
        
        def save_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        content = re.sub(code_pattern, save_code_block, content)
        
        # 인라인 코드 보호
        inline_codes = []
        inline_pattern = r'`[^`]+`'
        
        def save_inline_code(match):
            inline_codes.append(match.group(0))
            return f"__INLINE_CODE_{len(inline_codes)-1}__"
        
        content = re.sub(inline_pattern, save_inline_code, content)
        
        # 줄별 번역
        lines = content.split('\n')
        translated_lines = []
        
        for line in lines:
            # 빈 줄이나 특수 마크다운 요소는 그대로
            if not line.strip() or line.strip().startswith(('---', '***', '___')):
                translated_lines.append(line)
                continue
            
            # 제목 처리
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if heading_match:
                level = heading_match.group(1)
                text = heading_match.group(2)
                
                if self._should_translate(text):
                    try:
                        translated_text = self.translator.translate_text(
                            text, source_lang, target_lang, service
                        )
                        translated_lines.append(f"{level} {translated_text}")
                    except:
                        translated_lines.append(line)
                else:
                    translated_lines.append(line)
                continue
            
            # 일반 텍스트
            if self._should_translate(line):
                try:
                    translated_line = self.translator.translate_text(
                        line, source_lang, target_lang, service
                    )
                    translated_lines.append(translated_line)
                except:
                    translated_lines.append(line)
            else:
                translated_lines.append(line)
        
        # 번역된 내용 합치기
        translated_content = '\n'.join(translated_lines)
        
        # 보호된 요소 복원
        for idx, code in enumerate(inline_codes):
            translated_content = translated_content.replace(f"__INLINE_CODE_{idx}__", code)
        
        for idx, block in enumerate(code_blocks):
            translated_content = translated_content.replace(f"__CODE_BLOCK_{idx}__", block)
        
        return translated_content
