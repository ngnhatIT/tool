"""
SQL Reviewer Tool - Công cụ Review SQL chuyên nghiệp với Gemini AI
Version: 2.0
Author: Advanced Version
"""

import sys
import os
import json
from datetime import datetime, timedelta
import random
from typing import Optional, Dict, List, Any, Union, Set, Tuple
from enum import Enum
import mysql.connector
from google.generativeai import configure, GenerativeModel  # type: ignore
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QTreeWidget, QTreeWidgetItem, QMessageBox,
    QFormLayout, QComboBox, QFileDialog, QTabWidget,
    QStatusBar, QMainWindow, QSplitter, QDialog, QProgressBar, QCheckBox,
    QGroupBox, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QRect
from PyQt6.QtGui import QCloseEvent, QAction, QFont, QColor, QPalette, QPainter
import traceback
import functools
import logging
import faulthandler
import qtawesome as qta

# -------- Native crash / low-level fault capture --------
try:
    CRASH_LOG = os.path.join(os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__), 'fatal_crash.log')
    with open(CRASH_LOG, 'w', encoding='utf-8') as _f:
        faulthandler.enable(file=_f)  # Ghi traceback native (segfault, abort)
except Exception:
    pass

# Force plugin debug to diagnose Qt crashes when running frozen
os.environ.setdefault('QT_DEBUG_PLUGINS', '1')

# ---------------- Logging Setup -----------------
# Tạo logger ghi ra file bên cạnh exe (hoặc script) để debug crash khi build .exe
LOG_FILE = os.path.join(os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__), 'app.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('SQLReviewerPro')

# Constants
MSG_WARNING = 'Cảnh báo'
MSG_SUCCESS = 'Thành công'
MSG_ERROR = 'Lỗi'

# Relationship Types Enum
class RelationshipType(Enum):
    """Các loại quan hệ giữa bảng"""
    ONE_TO_ONE = '1-1'      # FK + Unique constraint
    ONE_TO_MANY = '1-N'     # FK thông thường (parent has many children)
    MANY_TO_ONE = 'N-1'     # Ngược lại của 1-N (child belongs to parent)
    MANY_TO_MANY = 'N-N'    # Thông qua junction table

# Data Generation Configuration
class DataGenConfig:
    """Configuration cho test data generation"""
    def __init__(self):
        self.row_count: int = 10
        self.include_parents: bool = True
        self.include_children: bool = False
        self.respect_fk: bool = True
        self.use_ai_generation: bool = True
        self.relationship_multipliers: Dict[RelationshipType, Tuple[int, int]] = {
            RelationshipType.ONE_TO_ONE: (1, 1),      # Đúng 1-1
            RelationshipType.ONE_TO_MANY: (1, 5),     # 1 parent -> 1-5 children
            RelationshipType.MANY_TO_ONE: (1, 1),     # N children -> 1 parent
            RelationshipType.MANY_TO_MANY: (2, 4)     # 2-4 records mỗi bên
        }

class RelationshipInfo:
    """Thông tin chi tiết về relationship giữa 2 bảng"""
    def __init__(self, constraint: str, from_table: str, from_column: str,
                 to_table: str, to_column: str, rel_type: RelationshipType):
        self.constraint = constraint
        self.from_table = from_table
        self.from_column = from_column
        self.to_table = to_table
        self.to_column = to_column
        self.rel_type = rel_type
        self.is_manual = False

# Exception handler decorator for button clicks
def safe_execute(func):
    """Decorator xử lý exception an toàn + logging cho mọi method UI gọi từ signal.
    - Bắt mọi exception, ghi vào app.log
    - Không để exception làm sập event loop của Qt
    - Hỗ trợ PyQt signals tự động truyền thêm đối số (checked, v.v.)
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            import inspect
            sig = inspect.signature(func)
            param_count = len(sig.parameters)
            # Nếu chỉ có self -> bỏ qua mọi arg thêm
            if param_count == 1:
                return func(self)
            else:
                return func(self, *args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("UI method crashed: %s | %s\nTraceback:\n%s", func.__name__, e, tb)
            error_msg = f"❌ Lỗi ở phương thức {func.__name__}:\n{type(e).__name__}: {e}"
            try:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, 'Operation Error', error_msg)
            except Exception:
                # Fallback ghi ra stderr
                print(error_msg)
            # Không re-raise để tránh app bị đóng
            return None
    return wrapper

# Color scheme - Modern Blue Theme với contrast cao
COLORS = {
    'primary': '#2563eb',      # Blue 600
    'primary_hover': '#1d4ed8', # Blue 700
    'success': '#059669',      # Emerald 600
    'success_hover': '#047857', # Emerald 700
    'danger': '#dc2626',       # Red 600
    'danger_hover': '#b91c1c',  # Red 700
    'warning': '#f59e0b',      # Amber 500
    'warning_hover': '#b45309', # Amber 700
    'secondary': '#4f46e5',    # Indigo 600
    'secondary_hover': '#3730a3', # Indigo 800
    'text_primary': '#111827',  # Gray 900 (very dark)
    'text_secondary': '#374151',# Gray 700
    'text_white': '#ffffff',
    'bg_primary': '#f9fafb',    # Gray 50 (very light)
    'bg_secondary': '#f3f4f6',  # Gray 100
    'bg_hover': '#e0e7ef',      # Blue-tinted light
    'border': '#cbd5e1',       # Gray 300
    'border_focus': '#2563eb',  # Blue 600
    'tree_bg': '#f1f5f9',       # Gray 100
    'tree_header': '#e0e7ef',   # Blue-tinted light
}

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


class ConfigManager:
    """Quản lý cấu hình từ file JSON"""
    
    def __init__(self, config_file='config.json'):
        # For .exe builds, save config in same directory as executable
        if hasattr(sys, '_MEIPASS'):
            # Running as PyInstaller bundle
            exe_dir = os.path.dirname(sys.executable)
            self.config_file = os.path.join(exe_dir, config_file)
        else:
            # Running as script
            self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load cấu hình từ file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Lỗi load config: {e}")
                return self.get_default_config()
        else:
            return self.get_default_config()
    
    def save_config(self, config: Optional[dict] = None):
        """Lưu cấu hình vào file"""
        try:
            if config is not None:
                self.config = config
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Lỗi save config: {e}")
    
    def get_default_config(self) -> dict:
        """Trả về cấu hình mặc định"""
        return {
            "gemini_api_key": "YOUR_API_KEY_HERE",
            "gemini_model": "gemini-1.5-flash",
            "last_connection": {
                "host": "localhost",
            "port": "3306",
            "database": "",
            "user": "root",
            "auth_plugin": "caching_sha2_password"
        },
            "connection_profiles": []
        }
    
    def get_api_key(self) -> str:
        return self.config.get('gemini_api_key', 'YOUR_API_KEY_HERE')
    
    def get_model(self) -> str:
        return self.config.get('gemini_model', 'gemini-1.5-flash')
    
    def get_last_connection(self) -> dict:
        return self.config.get('last_connection', {})
    
    def save_last_connection(self, conn_info: dict):
        self.config['last_connection'] = conn_info
        self.save_config()


class GeminiWorker(QThread):
    """Worker chạy trong luồng riêng để gọi API Gemini"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, prompt: str, model_name: str):
        super().__init__()
        self.prompt = prompt
        self.model_name = model_name

    def run(self):
        """Thực thi call tới Gemini trong thread với logging và bắt lỗi an toàn."""
        try:
            logger.info("GeminiWorker started: model=%s", self.model_name)
            self.progress.emit('Đang kết nối với Gemini AI...')
            model = GenerativeModel(self.model_name)
            
            self.progress.emit('Đang phân tích SQL query...')
            response = model.generate_content(self.prompt)
            text = getattr(response, 'text', '') or ''
            self.progress.emit('Hoàn thành!')
            
            logger.info("GeminiWorker success, received %d chars", len(text))
            self.finished.emit(text)
        except Exception as e:
            logger.error("GeminiWorker error: %s\n%s", e, traceback.format_exc())
            self.error.emit(str(e))


class AIDataGeneratorWorker(QThread):
    """Worker sử dụng AI để generate test data thông minh"""
    finished = pyqtSignal(dict)  # {table: [rows]}
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, model_name: str, schema: Dict[str, List[Dict[str, Any]]],
                 table: str, row_count: int, relationships: List[RelationshipInfo],
                 existing_data: Dict[str, List[Dict[str, Any]]]):
        super().__init__()
        self.model_name = model_name
        self.schema = schema
        self.table = table
        self.row_count = row_count
        self.relationships = relationships
        self.existing_data = existing_data

    def run(self):
        """Generate data sử dụng AI để tạo dữ liệu có ngữ nghĩa"""
        try:
            self.progress.emit(f'🤖 AI đang phân tích schema cho bảng {self.table}...')
            
            # Build context cho AI
            columns_info = []
            for col in self.schema.get(self.table, []):
                col_desc = f"- {col['name']} ({col['type']})"
                if col.get('key') == 'PRI':
                    col_desc += " [PRIMARY KEY]"
                if col.get('nullable') == 'NO':
                    col_desc += " [NOT NULL]"
                columns_info.append(col_desc)
            
            # Relationships context
            rel_info = []
            for rel in self.relationships:
                rel_info.append(
                    f"- {rel.from_column} references {rel.to_table}.{rel.to_column} ({rel.rel_type.value})"
                )
            
            # Existing data context (để AI generate data liên kết đúng)
            existing_info = []
            for rel in self.relationships:
                if rel.to_table in self.existing_data:
                    existing_rows = self.existing_data[rel.to_table]
                    if existing_rows:
                        sample_values = [row.get(rel.to_column) for row in existing_rows[:3]]
                        existing_info.append(
                            f"- Bảng {rel.to_table}.{rel.to_column} có giá trị: {sample_values}"
                        )
            
            prompt = f"""Bạn là chuyên gia database testing. Hãy generate {self.row_count} dòng dữ liệu mẫu REALISTIC cho bảng `{self.table}`.

SCHEMA:
{chr(10).join(columns_info)}

RELATIONSHIPS:
{chr(10).join(rel_info) if rel_info else "- Không có foreign key"}

DỮ LIỆU HIỆN TẠI:
{chr(10).join(existing_info) if existing_info else "- Chưa có dữ liệu parent"}

YÊU CẦU:
1. Dữ liệu phải REALISTIC và có ngữ nghĩa đúng (ví dụ: email thật, tên người thật, địa chỉ hợp lý)
2. Phải tôn trọng foreign key constraints - chỉ reference đến giá trị có sẵn
3. Tuân thủ data types và constraints (NOT NULL, PRIMARY KEY, etc.)
4. Đa dạng dữ liệu, không lặp lại quá nhiều
5. Format output là JSON array thuần túy, KHÔNG thêm markdown hoặc text giải thích

OUTPUT FORMAT (chỉ trả về JSON, không có ```json hoặc text nào khác):
[
  {{"column1": "value1", "column2": "value2", ...}},
  {{"column1": "value1", "column2": "value2", ...}}
]
"""

            self.progress.emit('🤖 AI đang generate dữ liệu thông minh...')
            model = GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            text = getattr(response, 'text', '') or ''
            
            # Parse JSON response
            # Remove markdown code blocks if present
            text = text.strip()
            if text.startswith('```'):
                lines = text.split('\n')
                text = '\n'.join(lines[1:-1]) if len(lines) > 2 else text
                if text.startswith('json'):
                    text = text[4:].strip()
            
            generated_rows = json.loads(text)
            
            self.progress.emit(f'✅ AI đã generate {len(generated_rows)} dòng dữ liệu')
            self.finished.emit({self.table: generated_rows})
            
        except json.JSONDecodeError as e:
            logger.error(f"AI response không phải JSON hợp lệ: {e}")
            self.error.emit(f"AI response không đúng format JSON: {e}")
        except Exception as e:
            logger.error(f"AIDataGenerator error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))


class ChatWorker(QThread):
    """Worker thread for Gemini chat to prevent UI freezing."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, history: list, model_name: str):
        super().__init__()
        self.history = history
        self.model_name = model_name

    def run(self):
        try:
            logger.info("ChatWorker started, history length: %d", len(self.history))
            model = GenerativeModel(self.model_name)
            
            # Start a chat session with history
            chat = model.start_chat(history=self.history[:-1]) # History excluding the last user message
            last_message = self.history[-1]['parts'][0]
            
            response = chat.send_message(last_message)
            
            text = getattr(response, 'text', '') or ''
            logger.info("ChatWorker success, received %d chars", len(text))
            self.finished.emit(text)
        except Exception as e:
            logger.error("ChatWorker exception: %s\n%s", e, traceback.format_exc())
            self.error.emit(f'Lỗi khi gọi Gemini: {str(e)}')


class LoadingOverlay(QWidget):
    """Loading overlay với spinner animation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        
        self.angle = 0
        self.message = "Đang xử lý..."
        self._show_background = True  # Default to showing the background
        
        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotate)
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup loading UI"""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Loading label with enhanced styling
        self.loading_label = QLabel(self.message)
        self.loading_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['primary']};
                font-size: 16px;
                font-weight: bold;
                background-color: white;
                padding: 25px 45px;
                border-radius: 12px;
                border: 3px solid {COLORS['primary']};
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
        """)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.loading_label)
        
        self.setLayout(layout)
    
    def paintEvent(self, a0):
        """Draw spinning circle"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Semi-transparent background (optional)
        if self._show_background:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        
        # Draw spinner
        rect = self.rect()
        center_x = rect.width() // 2
        center_y = rect.height() // 2 - 60
        radius = 30
        
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Draw spinning arcs
        for i in range(8):
            alpha = int(255 * (i + 1) / 8)
            color = QColor(37, 99, 235, alpha)  # Primary color with varying alpha
            painter.setBrush(color)
            
            angle_deg = (self.angle + i * 45) % 360
            angle_rad = angle_deg * 3.14159 / 180
            
            x = center_x + int(radius * 0.8 * (i / 8) * abs(1 - (i / 4)))
            y = center_y
            
            # Draw circle
            painter.drawEllipse(
                int(center_x + radius * abs(1 - (i / 4)) * (1 if i < 4 else -1)),
                int(center_y + radius * abs(1 - (i / 4)) * (1 if (i >= 2 and i < 6) else -1)),
                8, 8
            )
    
    def rotate(self):
        """Rotate animation"""
        self.angle = (self.angle + 10) % 360
        self.update()
    
    def show_loading(self, message="Đang xử lý...", show_background=True):
        """Show loading overlay"""
        self.message = message
        self._show_background = False
        self.loading_label.setText(f"🤖 {message}")
        
        parent_widget = self.parent()
        if parent_widget and isinstance(parent_widget, QWidget):
            self.setGeometry(parent_widget.rect())
        
        self.show()
        self.raise_()
        self.timer.start(50)  # 50ms = 20fps animation
    
    def hide_loading(self):
        """Hide loading overlay"""
        self.timer.stop()
        self.hide()
    
    def set_message(self, message: str):
        """Update loading message"""
        self.message = message
        self.loading_label.setText(f"🤖 {message}")


class AIChatDialog(QDialog):
    """Dialog để chat với Gemini AI"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('🤖 Gemini AI Assistant')
        self.setGeometry(200, 200, 800, 600)
        self.config_manager = ConfigManager()
        self.chat_history = []
        self.chat_worker: Optional[ChatWorker] = None
        
        self.init_ui()
        
        # Create loading overlay for chat
        self.chat_loading = LoadingOverlay(self)
        self.chat_loading.hide()
    
    def init_ui(self):
        """Khởi tạo giao diện chat"""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel('<h2>🤖 Chat với Gemini AI Assistant</h2>')
        header.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['primary']}, stop:1 #1e40af);
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(header)
        
        # Chat history display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont('Segoe UI', 10))
        self.chat_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: #f9fafb;
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        layout.addWidget(self.chat_display)
        
        # Quick suggestions
        suggestions_layout = QHBoxLayout()
        suggestions_label = QLabel('💡 <b>Gợi ý:</b>')
        suggestions_layout.addWidget(suggestions_label)
        
        suggestions = [
            '� MySQL 8.0 mới gì?',
            '⚡ Window Functions',
            '� Common Table Expression',
            '� JSON trong MySQL',
            '🎯 Partition Tables'
        ]
        
        for suggestion in suggestions:
            btn = QPushButton(suggestion)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['secondary']};
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['primary']};
                }}
            """)
            btn.clicked.connect(lambda checked, text=suggestion: self.use_suggestion(text))
            suggestions_layout.addWidget(btn)
        
        suggestions_layout.addStretch()
        layout.addLayout(suggestions_layout)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.user_input = QTextEdit()
        self.user_input.setPlaceholderText('Nhập câu hỏi của bạn... (Shift+Enter để xuống dòng)')
        self.user_input.setMaximumHeight(80)
        self.user_input.setFont(QFont('Segoe UI', 10))
        self.user_input.setStyleSheet(f"""
            QTextEdit {{
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        input_layout.addWidget(self.user_input)
        
        # Send button
        send_btn = QPushButton('📤 Gửi')
        send_btn.clicked.connect(self.send_message)
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
            QPushButton:disabled {{
                background-color: #9ca3af;
            }}
        """)
        self.send_btn = send_btn
        input_layout.addWidget(send_btn)
        
        layout.addLayout(input_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        clear_btn = QPushButton('🗑️ Xóa lịch sử')
        clear_btn.clicked.connect(self.clear_history)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: white;
                padding: 8px 15px;
                border-radius: 6px;
            }}
        """)
        action_layout.addWidget(clear_btn)
        
        action_layout.addStretch()
        
        close_btn = QPushButton('❌ Đóng')
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
                padding: 8px 15px;
                border-radius: 6px;
            }}
        """)
        action_layout.addWidget(close_btn)
        
        layout.addLayout(action_layout)
        
        self.setLayout(layout)
        
        # Welcome message
        self.add_message('assistant', 
            '👋 Xin chào! Tôi là Gemini AI Assistant chuyên về MySQL 8.0.\n\n'
            'Tôi có thể giúp bạn:\n'
            '• Giải thích các tính năng mới của MySQL 8.0\n'
            '• Hướng dẫn sử dụng Window Functions, CTE, JSON\n'
            '• Tối ưu hóa queries và performance tuning\n'
            '• Best practices cho database design\n'
            '• Và nhiều vấn đề khác!\n\n'
            'Hãy đặt câu hỏi bất kỳ cho tôi! 😊'
        )
    
    def use_suggestion(self, suggestion: str):
        """Sử dụng gợi ý"""
        # Remove emoji from suggestion
        text = suggestion.split(' ', 1)[-1]
        questions = {
            'MySQL 8.0 mới gì?': 'MySQL 8.0 có những tính năng mới gì so với phiên bản cũ? Cho ví dụ cụ thể về Window Functions và CTE.',
            'Window Functions': 'Giải thích Window Functions trong MySQL 8.0 (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD) với ví dụ thực tế.',
            'Common Table Expression': 'Common Table Expression (CTE) trong MySQL 8.0 là gì? Khi nào nên dùng CTE thay vì subquery? Cho ví dụ Recursive CTE.',
            'JSON trong MySQL': 'Làm thế nào để làm việc với JSON trong MySQL 8.0? Giải thích các hàm JSON_EXTRACT, JSON_SET, JSON_ARRAYAGG với ví dụ.',
            'Partition Tables': 'Table Partitioning trong MySQL là gì? Các loại partition (RANGE, LIST, HASH, KEY) và khi nào nên dùng? Cho ví dụ cụ thể.'
        }
        self.user_input.setPlainText(questions.get(text, text))
        self.send_message()
    
    def add_message(self, role: str, message: str):
        """Thêm message vào chat display with improved bubble style"""
        if role == 'user':
            formatted = f"""
<div style='background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%); 
            color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px; 
            margin: 8px 0 8px 60px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 80%;'>
    <b>👤 Bạn:</b><br>{message.replace(chr(10), '<br>')}
</div>
"""
        else:
            formatted = f"""
<div style='background-color: white; color: {COLORS["text_primary"]}; 
            padding: 12px 16px; border-radius: 18px 18px 18px 4px; 
            margin: 8px 60px 8px 0; border: 2px solid {COLORS["border"]};
            box-shadow: 0 2px 4px rgba(0,0,0,0.08); max-width: 80%;'>
    <b style="color: {COLORS["primary"]};">🤖 Gemini AI:</b><br>{message.replace(chr(10), '<br>')}
</div>
"""
        
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertHtml(formatted)
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())
    
    def send_message(self):
        """Gửi message tới Gemini"""
        user_message = self.user_input.toPlainText().strip()
        if not user_message:
            return
        
        # Add user message
        self.add_message('user', user_message)
        # Use the format expected by the GenerativeModel
        self.chat_history.append({'role': 'user', 'parts': [user_message]})
        
        # Clear input
        self.user_input.clear()
        
        # Show loading and disable button
        self.chat_loading.show_loading("Gemini đang suy nghĩ...", show_background=False)
        self.send_btn.setEnabled(False)
        self.send_btn.setText('⏳ Đang xử lý...')
        
        # Call Gemini API via worker thread
        model_name = self.config_manager.get_model()
        self.chat_worker = ChatWorker(self.chat_history, model_name)
        self.chat_worker.finished.connect(self.on_chat_finished)
        self.chat_worker.error.connect(self.on_chat_error)
        self.chat_worker.start()

    def on_chat_finished(self, ai_message: str):
        """Handles successful response from ChatWorker."""
        self.chat_loading.hide_loading()
        
        self.add_message('assistant', ai_message)
        self.chat_history.append({'role': 'model', 'parts': [ai_message]})
        
        self.send_btn.setEnabled(True)
        self.send_btn.setText('📤 Gửi')
        self.chat_worker = None

    def on_chat_error(self, error_message: str):
        """Handles error response from ChatWorker."""
        self.chat_loading.hide_loading()
        
        error_msg_display = f'❌ Lỗi: {error_message}\n\n💡 Vui lòng kiểm tra API key hoặc kết nối internet.'
        self.add_message('assistant', error_msg_display)
        
        self.send_btn.setEnabled(True)
        self.send_btn.setText('📤 Gửi')
        self.chat_worker = None

    def call_gemini(self, message: str):
        """Gọi Gemini API"""
        try:
            self.chat_loading.set_message("Đang nhận phản hồi từ Gemini...")
            response = model.generate_content(context)  # type: ignore
            
            # Hide loading
            self.chat_loading.hide_loading()
            
            # Add AI response
            ai_message = response.text
            self.add_message('assistant', ai_message)
            self.chat_history.append({'role': 'assistant', 'content': ai_message})
            
        except Exception as e:
            self.chat_loading.hide_loading()
            error_msg = f'❌ Lỗi: {str(e)}\n\n💡 Vui lòng kiểm tra API key hoặc kết nối internet.'
            self.add_message('assistant', error_msg)
        
        finally:
            # Re-enable send button
            self.send_btn.setEnabled(True)
            self.send_btn.setText('📤 Gửi')
    
    def clear_history(self):
        """Xóa lịch sử chat"""
        reply = QMessageBox.question(
            self,
            '🗑️ Xóa lịch sử',
            'Bạn có chắc muốn xóa toàn bộ lịch sử chat?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.chat_display.clear()
            self.chat_history = []
            self.add_message('assistant', 
                '✅ Đã xóa lịch sử chat.\n\n'
                '💬 Bạn có thể bắt đầu cuộc hội thoại mới!'
            )


class ManualRelationshipDialog(QDialog):
    """Dialog thêm quan hệ thủ công cho Test Data"""

    def __init__(self, parent, schema: Dict[str, List[Dict[str, Any]]]):
        super().__init__(parent)
        self.setWindowTitle('Thêm quan hệ thủ công')
        self.setMinimumWidth(520)
        self.schema = schema
        self.result: Optional[Dict[str, Any]] = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        info_label = QLabel(
            'Chọn bảng cha (được tham chiếu) và bảng con (FK) để bổ sung manual relationship.'
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form = QFormLayout()

        self.child_table_combo = QComboBox()
        self.parent_table_combo = QComboBox()

        tables = sorted(self.schema.keys())
        for table in tables:
            self.child_table_combo.addItem(table)
            self.parent_table_combo.addItem(table)

        self.child_table_combo.currentTextChanged.connect(self.refresh_child_columns)
        self.parent_table_combo.currentTextChanged.connect(self.refresh_parent_columns)

        self.child_column_combo = QComboBox()
        self.parent_column_combo = QComboBox()
        self.refresh_child_columns()
        self.refresh_parent_columns()

        form.addRow('Bảng con (FK):', self.child_table_combo)
        form.addRow('Cột con (FK):', self.child_column_combo)
        form.addRow('Bảng cha (PK):', self.parent_table_combo)
        form.addRow('Cột cha (PK):', self.parent_column_combo)

        self.parent_is_pk = QCheckBox('Cột cha là Primary Key')
        self.parent_is_pk.setChecked(True)
        self.child_is_fk = QCheckBox('Cột con là Foreign Key')
        self.child_is_fk.setChecked(True)
        extra_layout = QHBoxLayout()
        extra_layout.addWidget(self.parent_is_pk)
        extra_layout.addWidget(self.child_is_fk)
        extra_layout.addStretch()
        form.addRow('Xác nhận:', extra_layout)

        layout.addLayout(form)

        button_layout = QHBoxLayout()
        add_btn = QPushButton('Thêm quan hệ')
        add_btn.clicked.connect(self.on_submit)
        cancel_btn = QPushButton('Hủy')
        cancel_btn.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(add_btn)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def refresh_child_columns(self):
        table = self.child_table_combo.currentText()
        self.child_column_combo.clear()
        for col in self.schema.get(table, []):
            self.child_column_combo.addItem(col.get('name', ''))

    def refresh_parent_columns(self):
        table = self.parent_table_combo.currentText()
        self.parent_column_combo.clear()
        for col in self.schema.get(table, []):
            self.parent_column_combo.addItem(col.get('name', ''))

    def on_submit(self):
        child_table = self.child_table_combo.currentText()
        child_column = self.child_column_combo.currentText()
        parent_table = self.parent_table_combo.currentText()
        parent_column = self.parent_column_combo.currentText()

        if not all([child_table, child_column, parent_table, parent_column]):
            QMessageBox.warning(self, 'Thiếu dữ liệu', 'Vui lòng chọn bảng và cột đầy đủ.')
            return

        self.result = {
            'child_table': child_table,
            'child_column': child_column,
            'parent_table': parent_table,
            'parent_column': parent_column,
            'parent_is_pk': self.parent_is_pk.isChecked(),
            'child_is_fk': self.child_is_fk.isChecked()
        }
        self.accept()


class SQLReviewerApp(QMainWindow):
    """Ứng dụng chính"""
    
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.db_conn: Any = None
        self.db_schema: Dict[str, List[Dict[str, Any]]] = {}
        self.db_relationships: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.current_schema: Optional[Dict[str, Any]] = None  # Schema for type validation
        self.gemini_worker: Optional[GeminiWorker] = None
        self.current_review_result = ""
        self.randomizer = random.Random(42)
        
        # Test data UI placeholders
        self.testdata_table_combo: Optional[QComboBox] = None
        self.testdata_row_spin: Optional[QSpinBox] = None
        self.testdata_include_parents_checkbox: Optional[QCheckBox] = None
        self.testdata_include_children_checkbox: Optional[QCheckBox] = None
        self.testdata_respect_fk_checkbox: Optional[QCheckBox] = None
        self.relationship_tree: Optional[QTreeWidget] = None
        self.relationship_hint_label: Optional[QLabel] = None
        self.testdata_output: Optional[QTextEdit] = None
        self.testdata_copy_btn: Optional[QPushButton] = None
        self.manual_relationships: List[Dict[str, Any]] = []
        
        # AI Data Generation
        self.ai_data_worker: Optional[AIDataGeneratorWorker] = None
        self.data_gen_config = DataGenConfig()
        self.detected_relationships: Dict[str, List[RelationshipInfo]] = {}
        
        # Khởi tạo API
        self.init_gemini_api()
        
        # Khởi tạo giao diện
        self.init_ui()
        
        # Load last connection
        self.load_last_connection()
        
        # Create loading overlay
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()
    
    def init_gemini_api(self):
        """Khởi tạo Gemini API"""
        api_key = self.config_manager.get_api_key()
        if api_key and api_key != 'YOUR_API_KEY_HERE':
            try:
                configure(api_key=api_key)
            except Exception as e:
                reply = QMessageBox.critical(
                    self, 
                    'Lỗi Cấu hình',
                    f'❌ Không thể khởi tạo Gemini API\n\n'
                    f'Chi tiết lỗi: {e}\n\n'
                    f'💡 Bạn có muốn cấu hình lại API Key?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.configure_api_key()
        else:
            reply = QMessageBox.question(
                self, 
                '🔑 Cấu hình API Key',
                '⚠️ Chưa cấu hình Gemini API Key!\n\n'
                '💡 Bạn cần API key để sử dụng tính năng AI.\n\n'
                '📝 Bạn có muốn cấu hình ngay bây giờ?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.configure_api_key()
    
    def init_ui(self):
        """Khởi tạo giao diện"""
        self.setWindowTitle('SQL Reviewer Pro - Powered by Gemini AI')
        self.setGeometry(100, 100, 1400, 900)
        
        # Tạo menu bar
        self.create_menu_bar()
        
        # Widget chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout chính
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Splitter cho left và right panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- PANEL TRÁI ---
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        # Connection form header with gradient
        connection_header = QLabel('🔌 Thông tin kết nối MySQL')
        connection_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        connection_header.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['primary']}, stop:1 {COLORS['secondary']});
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
                margin-bottom: 5px;
            }}
        """)
        left_layout.addWidget(connection_header)
        
        self.connection_form = QFormLayout()
        self.connection_form.setSpacing(12)
        self.connection_form.setContentsMargins(5, 10, 5, 10)
        
        # Style for input fields with better visibility
        input_style = f"""
            QLineEdit {{
                background-color: white;
                border: 2px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
                color: {COLORS['text_primary']};
            }}
            QLineEdit:focus {{
                border: 2px solid {COLORS['primary']};
                background-color: #fefefe;
            }}
            QLineEdit:hover {{
                border-color: {COLORS['primary']};
            }}
        """
        
        self.db_host_input = QLineEdit('localhost')
        self.db_host_input.setStyleSheet(input_style)
        self.db_port_input = QLineEdit('3306')
        self.db_port_input.setStyleSheet(input_style)
        self.db_name_input = QLineEdit('')
        self.db_name_input.setStyleSheet(input_style)
        self.db_name_input.setPlaceholderText('Nhập tên database...')
        self.db_user_input = QLineEdit('root')
        self.db_user_input.setStyleSheet(input_style)
        self.db_pass_input = QLineEdit()
        self.db_pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.db_pass_input.setStyleSheet(input_style)
        self.db_pass_input.setPlaceholderText('Nhập password...')
        self.auth_plugin_input = QComboBox()
        self.auth_plugin_input.setEditable(True)
        for plugin in ['caching_sha2_password', 'mysql_native_password', 'sha256_password', 'dialog', 'authentication_ldap_simple']:
            self.auth_plugin_input.addItem(plugin)
        self.auth_plugin_input.setCurrentText('caching_sha2_password')
        
        # Create bold labels
        label_style = f"color: {COLORS['text_primary']}; font-weight: bold; font-size: 13px;"
        host_label = QLabel('🖥️ Host:')
        host_label.setStyleSheet(label_style)
        port_label = QLabel('🔌 Port:')
        port_label.setStyleSheet(label_style)
        db_label = QLabel('💾 Database:')
        db_label.setStyleSheet(label_style)
        user_label = QLabel('👤 User:')
        user_label.setStyleSheet(label_style)
        pass_label = QLabel('🔒 Password:')
        pass_label.setStyleSheet(label_style)
        auth_label = QLabel('🔐 Auth Plugin:')
        auth_label.setStyleSheet(label_style)
        
        self.connection_form.addRow(host_label, self.db_host_input)
        self.connection_form.addRow(port_label, self.db_port_input)
        self.connection_form.addRow(db_label, self.db_name_input)
        self.connection_form.addRow(user_label, self.db_user_input)
        self.connection_form.addRow(pass_label, self.db_pass_input)
        self.connection_form.addRow(auth_label, self.auth_plugin_input)
        
        left_layout.addLayout(self.connection_form)
        
        # Connect buttons
        btn_layout = QHBoxLayout()
        self.test_conn_button = QPushButton(' Test Connection')
        self.test_conn_button.setIcon(qta.icon('fa5s.plug', color=COLORS['text_white']))
        self.test_conn_button.clicked.connect(lambda: self.test_connection())
        self.test_conn_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: {COLORS['text_white']};
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['warning_hover']};
            }}
        """)
        btn_layout.addWidget(self.test_conn_button)
        
        self.connect_button = QPushButton(' Load Schema')
        self.connect_button.setIcon(qta.icon('fa5s.database', color=COLORS['text_white']))
        self.connect_button.clicked.connect(lambda: self.load_schema())
        self.connect_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_white']};
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
        """)
        btn_layout.addWidget(self.connect_button)
        left_layout.addLayout(btn_layout)
        
        # Schema tree with improved header
        schema_header = QLabel('📊 Cấu trúc Database')
        schema_header.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_primary']};
                font-weight: bold;
                font-size: 14px;
                padding: 8px;
                background-color: {COLORS['bg_hover']};
                border-radius: 6px;
                margin-top: 10px;
            }}
        """)
        left_layout.addWidget(schema_header)
        self.schema_tree = QTreeWidget()
        self.schema_tree.setHeaderLabels(['Tên', 'Kiểu', 'Chi tiết'])
        self.schema_tree.setColumnWidth(0, 200)
        self.schema_tree.setColumnWidth(1, 100)
        self.schema_tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: white;
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                font-size: 13px;
            }}
            QTreeWidget::item {{
                padding: 6px;
                border-bottom: 1px solid {COLORS['bg_secondary']};
            }}
            QTreeWidget::item:hover {{
                background-color: {COLORS['bg_hover']};
                color: {COLORS['primary']};
            }}
            QTreeWidget::item:selected {{
                background-color: {COLORS['primary']};
                color: white;
            }}
            QHeaderView::section {{
                background-color: {COLORS['tree_header']};
                color: {COLORS['text_primary']};
                padding: 8px;
                border: none;
                border-bottom: 2px solid {COLORS['primary']};
                font-weight: bold;
            }}
        """)
        left_layout.addWidget(self.schema_tree)
        
        # --- PANEL PHẢI ---
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # SQL Input with header
        sql_header = QLabel('💻 Nhập câu lệnh SQL')
        sql_header.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_primary']};
                font-weight: bold;
                font-size: 14px;
                padding: 8px;
                background-color: {COLORS['bg_hover']};
                border-radius: 6px;
            }}
        """)
        right_layout.addWidget(sql_header)
        self.sql_input = QTextEdit()
        self.sql_input.setPlaceholderText(
            'SELECT u.id, u.name, o.order_date\n' +
            'FROM users u\n' +
            'JOIN orders o ON u.id = o.user_id\n' +
            'WHERE u.status = "active"\n' +
            'ORDER BY o.order_date DESC;'
        )
        self.sql_input.setMinimumHeight(200)
        self.sql_input.setFont(QFont('Courier New', 10))
        self.sql_input.setStyleSheet(f"""
            QTextEdit {{
                background-color: white;
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                color: {COLORS['text_primary']};
            }}
            QTextEdit:focus {{
                border-color: {COLORS['primary']};
            }}
        """)
        right_layout.addWidget(self.sql_input)
        
        # Review buttons
        review_btn_layout = QHBoxLayout()
        self.review_button = QPushButton(' Review với Gemini AI')
        self.review_button.setIcon(qta.icon('fa5s.rocket', color=COLORS['text_white']))
        self.review_button.clicked.connect(lambda: self.review_sql_with_gemini())
        self.review_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success_hover']};
            }}
            QPushButton:disabled {{
                background-color: #cbd5e1;
                color: #94a3b8;
            }}
        """)
        review_btn_layout.addWidget(self.review_button)
        
        self.export_button = QPushButton(' Export Result')
        self.export_button.setIcon(qta.icon('fa5s.file-export', color=COLORS['text_white']))
        self.export_button.clicked.connect(lambda: self.export_result())
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: {COLORS['text_white']};
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary_hover']};
            }}
            QPushButton:disabled {{
                background-color: #cbd5e1;
                color: #94a3b8;
            }}
        """)
        review_btn_layout.addWidget(self.export_button)
        
        self.clear_button = QPushButton(' Clear')
        self.clear_button.setIcon(qta.icon('fa5s.trash-alt', color=COLORS['text_white']))
        self.clear_button.clicked.connect(self.clear_results)
        self.clear_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['text_secondary']};
                color: {COLORS['text_white']};
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['text_primary']};
            }}
        """)
        review_btn_layout.addWidget(self.clear_button)
        
        right_layout.addLayout(review_btn_layout)
        
        # Result output với tabs
        self.result_tabs = QTabWidget()
        
        # Tab 1: Review Result
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setFont(QFont('Segoe UI', 10))
        self.result_tabs.addTab(self.result_output, qta.icon('fa5s.poll'), 'Kết quả Review')
        
        # Tab 2: Raw SQL
        self.raw_sql_output = QTextEdit()
        self.raw_sql_output.setReadOnly(True)
        self.raw_sql_output.setFont(QFont('Courier New', 9))
        self.result_tabs.addTab(self.raw_sql_output, qta.icon('fa5s.code'), 'SQL Query')
        
        # Tab 3: SQL Bind Params (NEW)
        bind_widget = QWidget()
        bind_layout = QVBoxLayout()
        bind_widget.setLayout(bind_layout)
        
        # SQL with placeholders
        bind_layout.addWidget(QLabel('<b>SQL Query với Placeholders (?):</b>'))
        self.bind_sql_input = QTextEdit()
        self.bind_sql_input.setPlaceholderText(
            'SELECT * FROM users\n' +
            'WHERE company_id = ?\n' +
            '  AND status = ?\n' +
            '  AND created_date > ?'
        )
        self.bind_sql_input.setFont(QFont('Courier New', 10))
        self.bind_sql_input.setMinimumHeight(150)
        bind_layout.addWidget(self.bind_sql_input)
        
        # Parameters input
        params_header_layout = QHBoxLayout()
        params_header_layout.addWidget(QLabel('<b>Parameters (JSON Array):</b>'))
        
        help_btn = QPushButton(' Hướng dẫn')
        help_btn.setIcon(qta.icon('fa5s.question-circle', color=COLORS['text_white']))
        help_btn.clicked.connect(self.show_bind_help)
        help_btn.setMaximumWidth(120)
        help_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: {COLORS['text_white']};
                padding: 4px 8px;
                font-size: 11px;
            }}
        """)
        params_header_layout.addWidget(help_btn)
        params_header_layout.addStretch()
        bind_layout.addLayout(params_header_layout)
        
        self.bind_params_input = QTextEdit()
        self.bind_params_input.setPlaceholderText(
            'Format 1 (JSON): ["COMP001", "active", 123]\n'
            'Format 2 (Param): param:[1-COMP001][2-active][3-123]'
        )
        self.bind_params_input.setFont(QFont('Courier New', 10))
        self.bind_params_input.setMaximumHeight(80)
        bind_layout.addWidget(self.bind_params_input)
        
        # Bind button
        bind_btn_layout = QHBoxLayout()
        self.bind_button = QPushButton(' Bind Parameters')
        self.bind_button.setIcon(qta.icon('fa5s.link', color=COLORS['text_white']))
        self.bind_button.clicked.connect(lambda: self.bind_sql_parameters())
        self.bind_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_white']};
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
        """)
        bind_btn_layout.addWidget(self.bind_button)
        
        self.copy_result_btn = QPushButton(' Copy Result')
        self.copy_result_btn.setIcon(qta.icon('fa5s.copy', color=COLORS['text_primary']))
        self.copy_result_btn.clicked.connect(self.copy_bind_result)
        self.copy_result_btn.setEnabled(False)
        bind_btn_layout.addWidget(self.copy_result_btn)
        bind_layout.addLayout(bind_btn_layout)
        
        # Result output
        bind_layout.addWidget(QLabel('<b>Kết quả SQL đã Bind:</b>'))
        self.bind_result_output = QTextEdit()
        self.bind_result_output.setReadOnly(True)
        self.bind_result_output.setFont(QFont('Courier New', 10))
        bind_layout.addWidget(self.bind_result_output)
        
        self.result_tabs.addTab(bind_widget, qta.icon('fa5s.plug'), 'Bind Parameters')
        
        # Tab 4: Test Data Generator
        test_data_widget = QWidget()
        test_data_layout = QVBoxLayout()
        test_data_widget.setLayout(test_data_layout)
        
        config_group = QGroupBox('⚙️ Cấu hình dữ liệu mẫu')
        config_layout = QFormLayout()
        
        self.testdata_table_combo = QComboBox()
        self.testdata_table_combo.addItem('Chưa load schema', '')
        self.testdata_table_combo.currentIndexChanged.connect(self.refresh_relationship_summary)
        config_layout.addRow('Bảng nguồn:', self.testdata_table_combo)
        
        self.testdata_row_spin = QSpinBox()
        self.testdata_row_spin.setRange(1, 500)
        self.testdata_row_spin.setValue(5)
        config_layout.addRow('Số dòng cần tạo:', self.testdata_row_spin)
        
        option_widget = QWidget()
        option_layout = QHBoxLayout()
        option_layout.setContentsMargins(0, 0, 0, 0)
        option_layout.setSpacing(12)
        self.testdata_include_parents_checkbox = QCheckBox('Bảng cha')
        self.testdata_include_parents_checkbox.setChecked(True)
        self.testdata_include_children_checkbox = QCheckBox('Bảng con')
        self.testdata_include_children_checkbox.setChecked(False)
        self.testdata_respect_fk_checkbox = QCheckBox('Giữ quan hệ FK')
        self.testdata_respect_fk_checkbox.setChecked(True)
        self.testdata_use_ai_checkbox = QCheckBox('🤖 Sử dụng AI')
        self.testdata_use_ai_checkbox.setChecked(True)
        self.testdata_use_ai_checkbox.setToolTip('Sử dụng Gemini AI để generate dữ liệu thông minh, realistic và có ngữ nghĩa đúng')
        option_layout.addWidget(self.testdata_include_parents_checkbox)
        option_layout.addWidget(self.testdata_include_children_checkbox)
        option_layout.addWidget(self.testdata_respect_fk_checkbox)
        option_layout.addWidget(self.testdata_use_ai_checkbox)
        option_layout.addStretch()
        option_widget.setLayout(option_layout)
        config_layout.addRow('Tùy chọn:', option_widget)
        
        config_group.setLayout(config_layout)
        test_data_layout.addWidget(config_group)
        
        rel_group = QGroupBox('🔗 Quan hệ liên bảng')
        rel_layout = QVBoxLayout()
        self.relationship_hint_label = QLabel('Chưa có schema để hiển thị.')
        self.relationship_hint_label.setWordWrap(True)
        rel_layout.addWidget(self.relationship_hint_label)
        
        self.relationship_tree = QTreeWidget()
        self.relationship_tree.setHeaderLabels(['Loại', 'Bảng liên quan', 'Chi tiết'])
        self.relationship_tree.setColumnWidth(0, 120)
        self.relationship_tree.setColumnWidth(1, 160)
        rel_layout.addWidget(self.relationship_tree)
        
        manual_btn = QPushButton(' Thêm quan hệ thủ công')
        manual_btn.setIcon(qta.icon('fa5s.link', color=COLORS['text_white']))
        manual_btn.clicked.connect(self.open_manual_relationship_dialog)
        manual_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_white']};
                padding: 8px 14px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
        """)
        rel_layout.addWidget(manual_btn)
        rel_group.setLayout(rel_layout)
        test_data_layout.addWidget(rel_group)
        
        action_layout = QHBoxLayout()
        self.testdata_generate_btn = QPushButton(' Generate Sample Data')
        self.testdata_generate_btn.setIcon(qta.icon('fa5s.seedling', color=COLORS['text_white']))
        self.testdata_generate_btn.clicked.connect(lambda: self.generate_test_data())
        self.testdata_generate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: {COLORS['text_white']};
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success_hover']};
            }}
        """)
        action_layout.addWidget(self.testdata_generate_btn)
        
        self.testdata_copy_btn = QPushButton(' Copy SQL')
        self.testdata_copy_btn.setIcon(qta.icon('fa5s.copy', color=COLORS['text_primary']))
        self.testdata_copy_btn.clicked.connect(self.copy_testdata_sql)
        self.testdata_copy_btn.setEnabled(False)
        action_layout.addWidget(self.testdata_copy_btn)
        action_layout.addStretch()
        test_data_layout.addLayout(action_layout)
        
        test_data_layout.addWidget(QLabel('<b>Script INSERT giả lập:</b>'))
        self.testdata_output = QTextEdit()
        self.testdata_output.setReadOnly(True)
        self.testdata_output.setFont(QFont('Courier New', 10))
        self.testdata_output.setMinimumHeight(180)
        test_data_layout.addWidget(self.testdata_output)
        
        self.result_tabs.addTab(test_data_widget, qta.icon('fa5s.database'), 'Test Data')
        
        right_layout.addWidget(self.result_tabs)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
        # Floating AI Assistant Button
        self.create_floating_ai_button()
        
        # Status bar with improved styling
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS['bg_primary']}, stop:1 {COLORS['bg_secondary']});
                color: {COLORS['text_primary']};
                border-top: 2px solid {COLORS['primary']};
                font-size: 13px;
                font-weight: 500;
                padding: 6px 12px;
            }}
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('✅ Sẵn sàng - Chào mừng đến với SQL Reviewer Pro!')
        
        # Apply stylesheet
        self.apply_stylesheet()
        self.populate_testdata_controls()
    
    def create_floating_ai_button(self):
        """Tạo floating button để chat với Gemini AI"""
        self.float_ai_btn = QPushButton('🤖', self)
        self.float_ai_btn.setToolTip('💬 Chat với Gemini AI Assistant - Click để mở')
        self.float_ai_btn.clicked.connect(lambda: self.open_ai_chat())
        self.float_ai_btn.setFixedSize(65, 65)
        self.float_ai_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {COLORS['primary']}, stop:1 #1e40af);
                color: white;
                border: 4px solid #e0e7ff;
                border-radius: 32px;
                font-size: 26px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1e40af, stop:1 {COLORS['primary']});
                border: 4px solid white;
                font-size: 28px;
            }}
            QPushButton:pressed {{
                background: #1e3a8a;
                border: 4px solid #cbd5e1;
                font-size: 24px;
            }}
        """)
        
        # Position at bottom right
        self.float_ai_btn.move(self.width() - 90, self.height() - 90)
        self.float_ai_btn.raise_()
        self.float_ai_btn.show()
    
    def resizeEvent(self, a0):
        """Handle window resize to reposition floating button and loading overlay"""
        super().resizeEvent(a0)
        if hasattr(self, 'float_ai_btn'):
            self.float_ai_btn.move(self.width() - 90, self.height() - 100)
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.setGeometry(self.rect())
    
    def create_menu_bar(self):
        """Tạo menu bar"""
        menubar = self.menuBar()
        if menubar is None:
            return
        
        # File menu
        file_menu = menubar.addMenu('&File')
        if file_menu is None:
            return
        
        export_action = QAction(qta.icon('fa5s.file-export', color=COLORS['text_primary']), ' Export Result', self)
        export_action.triggered.connect(self.export_result)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction(qta.icon('fa5s.sign-out-alt', color=COLORS['text_primary']), ' Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu('&Settings')
        if settings_menu is None:
            return
        
        api_key_action = QAction(qta.icon('fa5s.key', color=COLORS['text_primary']), ' Cấu hình API Key', self)
        api_key_action.triggered.connect(self.configure_api_key)
        settings_menu.addAction(api_key_action)
        
        settings_menu.addSeparator()
        
        config_action = QAction(qta.icon('fa5s.cog', color=COLORS['text_primary']), ' Open Config File', self)
        config_action.triggered.connect(self.open_config_file)
        settings_menu.addAction(config_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        if help_menu is None:
            return
        
        about_action = QAction(qta.icon('fa5s.info-circle', color=COLORS['text_primary']), ' About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def apply_stylesheet(self):
        """Apply improved stylesheet for high contrast and modern look"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['bg_secondary']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
                font-weight: 500;
            }}
            QLineEdit, QTextEdit {{
                background-color: {COLORS['bg_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px 12px;
                font-size: 13px;
                color: {COLORS['text_primary']};
                selection-background-color: {COLORS['primary']};
                selection-color: {COLORS['text_white']};
            }}
            QLineEdit:focus, QTextEdit:focus {{
                border-color: {COLORS['primary']};
                background-color: #fff;
                outline: none;
            }}
            QLineEdit:hover, QTextEdit:hover {{
                border-color: {COLORS['primary']};
            }}
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: {COLORS['text_white']};
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
                min-height: 18px;
                transition: background 0.2s;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary_hover']};
                color: {COLORS['text_white']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_white']};
            }}
            QPushButton:disabled {{
                background-color: #e5e7eb;
                color: #9ca3af;
            }}
            QTreeWidget {{
                background-color: {COLORS['tree_bg']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                font-size: 13px;
                color: {COLORS['text_primary']};
                alternate-background-color: {COLORS['bg_primary']};
                padding: 4px;
            }}
            QTreeWidget::item {{
                padding: 8px 4px;
                border-bottom: 1px solid {COLORS['bg_secondary']};
                margin: 1px 0;
            }}
            QTreeWidget::item:hover {{
                background-color: {COLORS['bg_hover']};
                color: {COLORS['primary']};
                border-radius: 4px;
            }}
            QTreeWidget::item:selected {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_white']};
                border-radius: 4px;
            }}
            QHeaderView::section {{
                background-color: {COLORS['tree_header']};
                color: {COLORS['text_primary']};
                padding: 10px 8px;
                border: none;
                border-bottom: 3px solid {COLORS['primary']};
                font-weight: bold;
                font-size: 13px;
            }}
            QComboBox {{
                background-color: {COLORS['bg_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px 12px;
                font-size: 13px;
                color: {COLORS['text_primary']};
                min-width: 100px;
            }}
            QComboBox:hover, QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {COLORS['text_secondary']};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                selection-background-color: {COLORS['primary']};
                selection-color: {COLORS['text_white']};
                padding: 4px;
            }}
            QTabWidget::pane {{
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                background-color: {COLORS['bg_primary']};
                top: -2px;
            }}
            QTabBar::tab {{
                background-color: {COLORS['bg_secondary']};
                padding: 12px 24px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
                font-size: 13px;
                color: {COLORS['text_secondary']};
                font-weight: 500;
                min-width: 80px;
            }}
            QTabBar::tab:hover {{
                background-color: {COLORS['bg_hover']};
                color: {COLORS['primary']};
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['bg_primary']};
                color: {COLORS['primary']};
                font-weight: bold;
                border-bottom: 3px solid {COLORS['primary']};
                margin-bottom: -2px;
            }}
            QStatusBar {{
                background-color: {COLORS['bg_primary']};
                color: {COLORS['text_primary']};
                border-top: 2px solid {COLORS['border']};
                font-size: 12px;
                padding: 4px 8px;
            }}
            QFormLayout QLabel {{
                color: {COLORS['text_secondary']};
                font-weight: 600;
                font-size: 13px;
            }}
            QMessageBox {{
                background-color: {COLORS['bg_primary']};
            }}
            QMessageBox QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
                font-weight: normal;
                background-color: transparent;
                padding: 4px;
            }}
            QMessageBox QPushButton {{
                min-width: 90px;
                padding: 10px 20px;
            }}
            QScrollBar:vertical {{
                background-color: {COLORS['bg_secondary']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {COLORS['primary']};
                border-radius: 6px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {COLORS['secondary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: {COLORS['bg_secondary']};
                height: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {COLORS['primary']};
                border-radius: 6px;
                min-width: 30px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {COLORS['secondary']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            QSplitter::handle {{
                background-color: {COLORS['border']};
                width: 2px;
                height: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {COLORS['primary']};
            }}
        """)
    
    def load_last_connection(self):
        """Load thông tin kết nối lần cuối"""
        last_conn = self.config_manager.get_last_connection()
        if last_conn:
            self.db_host_input.setText(last_conn.get('host', 'localhost'))
            self.db_port_input.setText(last_conn.get('port', '3306'))
            self.db_name_input.setText(last_conn.get('database', ''))
            self.db_user_input.setText(last_conn.get('user', 'root'))
            plugin = last_conn.get('auth_plugin', 'caching_sha2_password') or 'caching_sha2_password'
            if self.auth_plugin_input:
                self.auth_plugin_input.setCurrentText(plugin)

    def get_auth_plugin(self) -> Optional[str]:
        """Return selected authentication plugin or None"""
        if not self.auth_plugin_input:
            return None
        text = self.auth_plugin_input.currentText().strip()
        return text or None
    
    @safe_execute
    def test_connection(self):
        """Test kết nối database"""
        self.loading_overlay.show_loading('Đang test connection...')
        QApplication.processEvents()  # Force UI update
        
        try:
            # NOTE: Crash log showed access violation inside mysql.connector C extension (connection_cext.py).
            # Workaround: force pure-Python implementation with use_pure=True to avoid native DLL issues under PyInstaller.
            conn = mysql.connector.connect(
                host=self.db_host_input.text(),
                port=int(self.db_port_input.text()),
                database=self.db_name_input.text() if self.db_name_input.text() else None,
                user=self.db_user_input.text(),
                password=self.db_pass_input.text(),
                connect_timeout=5,
                use_pure=True,  # Force pure Python connector (avoids crashing C extension in frozen exe)
                auth_plugin=self.get_auth_plugin()
            )
            conn.close()
            QMessageBox.information(self, 'Kết nối Thành công', 
                '✅ Kết nối database thành công!\n\n' +
                f'Host: {self.db_host_input.text()}\n' +
                f'Database: {self.db_name_input.text()}\n' +
                f'User: {self.db_user_input.text()}')
            self.status_bar.showMessage('✅ Kết nối thành công', 3000)
        except Exception as e:
            error_msg = str(e)
            if 'Access denied' in error_msg:
                detail = '❌ Username hoặc password không đúng'
            elif 'Unknown database' in error_msg:
                detail = '❌ Database không tồn tại'
            elif "Can't connect" in error_msg:
                detail = '❌ Không thể kết nối tới server'
            else:
                detail = f'❌ {error_msg}'
            
            QMessageBox.critical(self, 'Kết nối Thất bại', 
                f'{detail}\n\n' +
                f'Thông tin kết nối:\n' +
                f'• Host: {self.db_host_input.text()}\n' +
                f'• Port: {self.db_port_input.text()}\n' +
                f'• Database: {self.db_name_input.text()}\n' +
                f'• User: {self.db_user_input.text()}\n\n' +
                '💡 Kiểm tra lại thông tin và đảm bảo MySQL server đang chạy.')
            self.status_bar.showMessage('❌ Kết nối thất bại', 3000)
        finally:
            self.loading_overlay.hide_loading()
    
    @safe_execute
    def load_schema(self):
        """Kết nối và load schema"""
        self.schema_tree.clear()
        self.db_schema = {}
        self.db_relationships = {}
        self.populate_testdata_controls()
        self.loading_overlay.show_loading('Đang kết nối và load schema...')
        QApplication.processEvents()
        
        try:
            # Validate inputs
            if not self.db_name_input.text():
                raise ValueError('⚠️ Vui lòng nhập tên database')
            
            # Close existing connection
            if self.db_conn:
                self.db_conn.close()
            
            # Connect to MySQL
            # Force pure Python connector to mitigate access violation in packaged exe
            self.db_conn = mysql.connector.connect(
                host=self.db_host_input.text(),
                port=int(self.db_port_input.text()),
                database=self.db_name_input.text(),
                user=self.db_user_input.text(),
                password=self.db_pass_input.text(),
                use_pure=True,
                auth_plugin=self.get_auth_plugin()
            )
            
            # Load schema with detailed info
            self.get_mysql_schema_detailed()
            self.get_mysql_relationships()
            self.apply_manual_relationships()
            
            # Store schema for type validation
            self.current_schema = self.db_schema
            
            # Save last connection
            self.config_manager.save_last_connection({
                'host': self.db_host_input.text(),
                'port': self.db_port_input.text(),
                'database': self.db_name_input.text(),
                'user': self.db_user_input.text(),
                'auth_plugin': self.get_auth_plugin()
            })
            
            QMessageBox.information(self, 'Load Schema Thành công',
                '✅ Đã kết nối và tải schema thành công!\n\n' +
                f'📊 Database: {self.db_name_input.text()}\n' +
                f'📋 Số bảng: {len(self.db_schema)}\n\n' +
                '💡 Bạn có thể xem chi tiết cấu trúc ở panel bên trái.')
            self.status_bar.showMessage(f'✅ Đã load {len(self.db_schema)} bảng', 3000)
            self.populate_testdata_controls()
            self.refresh_relationship_summary()
            
        except ValueError as e:
            QMessageBox.warning(self, 'Thiếu Thông tin', str(e))
            self.status_bar.showMessage('❌ Thiếu thông tin', 3000)
        except Exception as e:
            error_msg = str(e)
            if 'Access denied' in error_msg:
                detail = 'Username hoặc password không đúng'
            elif 'Unknown database' in error_msg:
                detail = f'Database "{self.db_name_input.text()}" không tồn tại'
            elif "Can't connect" in error_msg:
                detail = 'Không thể kết nối tới MySQL server'
            else:
                detail = error_msg
            
            QMessageBox.critical(self, 'Lỗi Kết nối', 
                f'❌ Không thể kết nối database\n\n' +
                f'Lỗi: {detail}\n\n' +
                '• Kiểm tra MySQL server đang chạy\n' +
                '• Kiểm tra username/password\n' +
                '• Kiểm tra tên database\n' +
                '• Thử "Test Connection" trước khi load schema')
            self.status_bar.showMessage('❌ Lỗi kết nối', 3000)
        finally:
            self.loading_overlay.hide_loading()
    
    def get_mysql_schema_detailed(self):
        """Lấy schema chi tiết từ MySQL"""
        if not self.db_conn:
            return
        
        cursor = self.db_conn.cursor()
        db_name = self.db_name_input.text()
        
        # Get all tables
        cursor.execute("""
            SELECT table_name, table_rows, table_comment
            FROM information_schema.tables 
            WHERE table_schema = %s AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """, (db_name,))
        tables = cursor.fetchall()
        
        for (table_name, table_rows, table_comment) in tables:
            # Table item with row count
            table_display = f"{table_name} ({table_rows or 0} rows)"
            table_item = QTreeWidgetItem(self.schema_tree, [table_display, 'TABLE', table_comment or ''])
            
            # Style cho table name - Bold và màu primary
            font = QFont()
            font.setBold(True)
            font.setPointSize(11)
            table_item.setFont(0, font)
            table_item.setForeground(0, QColor(COLORS['primary']))
            table_item.setForeground(1, QColor(COLORS['text_secondary']))
            
            table_name_str = str(table_name)
            self.db_schema[table_name_str] = []
            
            # Get columns with detailed info
            cursor.execute("""
                SELECT 
                    column_name, 
                    column_type,
                    is_nullable,
                    column_key,
                    column_default,
                    extra,
                    column_comment
                FROM information_schema.columns 
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position;
            """, (db_name, table_name_str))
            
            columns = cursor.fetchall()
            for (col_name, col_type, is_nullable, col_key, col_default, extra, col_comment) in columns:
                # Build detailed info
                details = []
                if col_key == 'PRI':
                    details.append('🔑 PRIMARY KEY')
                elif col_key == 'UNI':
                    details.append('🔒 UNIQUE')
                elif col_key == 'MUL':
                    details.append('🔗 INDEX')
                
                if is_nullable == 'NO':
                    details.append('NOT NULL')
                
                extra_str = str(extra) if extra else ''
                if 'auto_increment' in extra_str.lower():
                    details.append('AUTO_INCREMENT')
                
                if col_default is not None:
                    details.append(f'DEFAULT: {col_default}')
                
                if col_comment:
                    details.append(f'Comment: {col_comment}')
                
                detail_str = ' | '.join(details) if details else ''
                
                col_item = QTreeWidgetItem(table_item, [str(col_name), str(col_type), detail_str])
                
                # Font cho columns
                col_font = QFont()
                col_font.setPointSize(10)
                col_item.setFont(0, col_font)
                col_item.setFont(1, col_font)
                col_item.setFont(2, col_font)
                
                # Color coding với màu rõ ràng hơn
                if col_key == 'PRI':
                    col_item.setForeground(0, QColor(COLORS['danger']))  # Red for PK
                    col_font.setBold(True)
                    col_item.setFont(0, col_font)
                elif col_key in ['UNI', 'MUL']:
                    col_item.setForeground(0, QColor(COLORS['warning']))  # Orange for indexed
                else:
                    col_item.setForeground(0, QColor(COLORS['text_primary']))
                
                col_item.setForeground(1, QColor(COLORS['text_secondary']))
                col_item.setForeground(2, QColor(COLORS['text_secondary']))
                
                self.db_schema[table_name_str].append({
                    'name': str(col_name),
                    'type': str(col_type),
                    'nullable': str(is_nullable),
                    'key': str(col_key),
                    'default': col_default,
                    'extra': extra_str,
                    'comment': str(col_comment) if col_comment else ''
                })
        
        cursor.close()
        
        # Expand first table if exists
        if self.schema_tree.topLevelItemCount() > 0:
            first_item = self.schema_tree.topLevelItem(0)
            if first_item:
                first_item.setExpanded(True)
    
    def get_mysql_relationships(self):
        """Lấy thông tin quan hệ khóa ngoại giữa các bảng"""
        if not self.db_conn:
            return
        
        db_name = self.db_name_input.text()
        cursor = self.db_conn.cursor()
        # Khởi tạo cấu trúc rỗng cho tất cả bảng
        self.db_relationships = {
            table: {'references': [], 'referenced_by': []}
            for table in self.db_schema.keys()
        }
        
        cursor.execute("""
            SELECT 
                kcu.CONSTRAINT_NAME,
                kcu.TABLE_NAME,
                kcu.COLUMN_NAME,
                kcu.REFERENCED_TABLE_NAME,
                kcu.REFERENCED_COLUMN_NAME,
                rc.UPDATE_RULE,
                rc.DELETE_RULE
            FROM information_schema.KEY_COLUMN_USAGE kcu
            JOIN information_schema.REFERENTIAL_CONSTRAINTS rc
              ON kcu.CONSTRAINT_SCHEMA = rc.CONSTRAINT_SCHEMA
             AND kcu.CONSTRAINT_NAME = rc.CONSTRAINT_NAME
            WHERE 
                kcu.TABLE_SCHEMA = %s
                AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
            ORDER BY kcu.TABLE_NAME, kcu.COLUMN_NAME;
        """, (db_name,))
        rows = cursor.fetchall()
        
        for (constraint_name, table_name, column_name, ref_table, ref_column, update_rule, delete_rule) in rows:
            table = str(table_name)
            referenced_table = str(ref_table)
            column = str(column_name)
            referenced_column = str(ref_column)
            
            self.db_relationships.setdefault(table, {'references': [], 'referenced_by': []})
            self.db_relationships.setdefault(referenced_table, {'references': [], 'referenced_by': []})
            
            relation = {
                'constraint': str(constraint_name),
                'column': column,
                'referenced_table': referenced_table,
                'referenced_column': referenced_column,
                'update_rule': str(update_rule),
                'delete_rule': str(delete_rule)
            }
            self.db_relationships[table]['references'].append(relation)
            
            reverse_relation = {
                'constraint': str(constraint_name),
                'table': table,
                'column': referenced_column,
                'referencing_column': column,
                'update_rule': str(update_rule),
                'delete_rule': str(delete_rule)
            }
            self.db_relationships[referenced_table]['referenced_by'].append(reverse_relation)
        
        cursor.close()
        
        # Đảm bảo bảng nào cũng có entry
        for table in self.db_schema.keys():
            self.db_relationships.setdefault(table, {'references': [], 'referenced_by': []})
        
        # Detect relationship types
        self.detect_all_relationship_types()
    
    def detect_all_relationship_types(self):
        """Phát hiện loại quan hệ cho tất cả FK relationships"""
        self.detected_relationships.clear()
        
        for table in self.db_schema.keys():
            rels = self.db_relationships.get(table, {}).get('references', [])
            for rel in rels:
                rel_info = self.detect_relationship_type(
                    from_table=table,
                    from_column=rel['column'],
                    to_table=rel['referenced_table'],
                    to_column=rel['referenced_column'],
                    constraint=rel['constraint']
                )
                if rel_info:
                    self.detected_relationships.setdefault(table, []).append(rel_info)
    
    def detect_relationship_type(self, from_table: str, from_column: str,
                                to_table: str, to_column: str,
                                constraint: str) -> Optional[RelationshipInfo]:
        """
        Phát hiện loại quan hệ giữa 2 bảng:
        - 1-1: FK column có UNIQUE constraint
        - 1-N: FK thông thường (default)
        - N-N: Cần detect junction table (bảng trung gian)
        """
        if not from_table or not to_table:
            return None
        
        # Check xem from_column có UNIQUE constraint không
        from_columns = self.db_schema.get(from_table, [])
        from_col_info = next((c for c in from_columns if c['name'] == from_column), None)
        
        if from_col_info and from_col_info.get('key') in ['UNI', 'PRI']:
            # 1-1 relationship: FK column là UNIQUE hoặc PK
            rel_type = RelationshipType.ONE_TO_ONE
        elif self.is_junction_table(from_table):
            # N-N relationship: from_table là junction table
            rel_type = RelationshipType.MANY_TO_MANY
        else:
            # 1-N relationship: Default case
            rel_type = RelationshipType.ONE_TO_MANY
        
        return RelationshipInfo(
            constraint=constraint,
            from_table=from_table,
            from_column=from_column,
            to_table=to_table,
            to_column=to_column,
            rel_type=rel_type
        )
    
    def is_junction_table(self, table: str) -> bool:
        """
        Kiểm tra xem table có phải junction table không:
        - Có ít nhất 2 FK
        - Phần lớn columns là FK
        - Thường có composite PK
        """
        references = self.db_relationships.get(table, {}).get('references', [])
        if len(references) < 2:
            return False
        
        total_columns = len(self.db_schema.get(table, []))
        fk_count = len(references)
        
        # Junction table thường có >50% columns là FK
        return fk_count >= 2 and (fk_count / total_columns) > 0.5
    
    @safe_execute
    def review_sql_with_gemini(self):
        """Review SQL với Gemini AI"""
        sql_query = self.sql_input.toPlainText().strip()
        
        if not sql_query:
            QMessageBox.warning(self, 'SQL Trống', 
                '⚠️ Vui lòng nhập câu lệnh SQL để review\n\n' +
                '💡 Bạn có thể paste SQL query của mình vào ô input phía trên.')
            return
        
        if not self.db_schema:
            reply = QMessageBox.question(self, 'Không có Schema',
                '⚠️ Chưa load schema database\n\n' +
                'Review sẽ chính xác hơn nếu có thông tin về cấu trúc database.\n\n' +
                'Bạn có muốn tiếp tục review mà không có schema?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Show raw SQL in tab
        self.raw_sql_output.setText(sql_query)
        
        # Show loading overlay
        self.loading_overlay.show_loading("Đang kết nối với Gemini AI...")
        
        # Disable button
        self.review_button.setDisabled(True)
        self.result_output.setText('⏳ Đang liên hệ với Gemini AI... Vui lòng chờ...')
        self.status_bar.showMessage('Đang phân tích SQL...')
        
        # Build prompt and start worker
        prompt = self.build_enhanced_prompt(sql_query)
        model_name = self.config_manager.get_model()
        
        self.gemini_worker = GeminiWorker(prompt, model_name)
        self.gemini_worker.finished.connect(self.on_review_finished)
        self.gemini_worker.error.connect(self.on_review_error)
        self.gemini_worker.progress.connect(self.on_review_progress)
        self.gemini_worker.start()
    
    def on_review_progress(self, message: str):
        """Cập nhật progress"""
        self.status_bar.showMessage(message)
        self.loading_overlay.set_message(message)
    
    def on_review_finished(self, result_text: str):
        """Xử lý khi review xong"""
        self.loading_overlay.hide_loading()
        self.current_review_result = result_text
        self.result_output.setText(result_text)
        self.review_button.setDisabled(False)
        self.export_button.setEnabled(True)
        self.gemini_worker = None
        self.status_bar.showMessage('✅ Review hoàn thành!', 3000)
    
    def on_review_error(self, error_message: str):
        """Xử lý khi có lỗi"""
        self.loading_overlay.hide_loading()
        QMessageBox.critical(self, 'Lỗi Review', 
            f'❌ Không thể review SQL query\n\n' +
            f'Lỗi: {error_message}\n\n' +
            '💡' +
            '• Kiểm tra kết nối internet\n' +
            '• Kiểm tra API key trong config.json\n' +
            '• Thử lại sau vài giây')
        self.result_output.setText(f'❌ Lỗi: {error_message}')
        self.review_button.setDisabled(False)
        self.gemini_worker = None
        self.status_bar.showMessage('❌ Review thất bại', 3000)
    
    def build_enhanced_prompt(self, sql_query: str) -> str:
        """Xây dựng prompt nâng cao cho Gemini"""
        schema_string = ""
        
        if self.db_schema:
            for table, columns in self.db_schema.items():
                schema_string += f"📋 **Bảng {table}**:\n"
                for col in columns:
                    col_info = f"  - `{col['name']}` ({col['type']})"
                    if col['key'] == 'PRI':
                        col_info += " [PRIMARY KEY]"
                    if col['nullable'] == 'NO':
                        col_info += " [NOT NULL]"
                    if 'auto_increment' in col['extra'].lower():
                        col_info += " [AUTO_INCREMENT]"
                    if col['comment']:
                        col_info += f" // {col['comment']}"
                    schema_string += col_info + "\n"
                schema_string += "\n"
        else:
            schema_string = "⚠️ Không có thông tin schema database.\n"
        
        if self.db_relationships:
            relationship_lines = []
            for table, rel in self.db_relationships.items():
                parents = rel.get('references', [])
                children = rel.get('referenced_by', [])
                if not parents and not children:
                    continue
                parent_desc = ', '.join(
                    f"{r.get('column')}→{r.get('referenced_table')}.{r.get('referenced_column')}"
                    for r in parents
                ) or 'Không tham chiếu'
                child_desc = ', '.join(
                    f"{r.get('table')}.{r.get('referencing_column')}"
                    for r in children
                ) or 'Không bị tham chiếu'
                relationship_lines.append(
                    f"- **{table}** | Cha: {parent_desc} | Con: {child_desc}"
                )
            relationship_string = "\n".join(relationship_lines) if relationship_lines else "⚠️ Không có thông tin quan hệ khóa ngoại.\n"
        else:
            relationship_string = "⚠️ Không có thông tin quan hệ khóa ngoại.\n"
        
        return f"""
Bạn là một chuyên gia Senior Database Engineer và SQL Performance Tuning Expert với hơn 15 năm kinh nghiệm.

📊 **CẤU TRÚC DATABASE**:
{schema_string}

🔗 **QUAN HỆ BẢNG**:
{relationship_string}

🔍 **SQL QUERY CẦN REVIEW**:
```sql
{sql_query}
```

📋 **NHIỆM VỤ REVIEW CHI TIẾT**:
    
Hãy phân tích toàn diện câu lệnh SQL theo các tiêu chí sau:

## 1. ✅ Phát hiện Lỗi (Errors & Issues)
- Kiểm tra cú pháp SQL
- Tên bảng, cột có tồn tại và đúng không?
- Logic query có vấn đề gì không?
- Các lỗi tiềm ẩn (data type mismatch, NULL handling...)

## 2. ⚡ Tối ưu Hiệu suất (Performance Optimization)
- Đánh giá độ phức tạp query (O notation nếu có thể)
- Đề xuất indexes cần thiết (với lý do cụ thể)
- Tối ưu JOIN (type of JOIN, order, conditions)
- Subquery vs JOIN - cái nào tốt hơn?
- Sử dụng WHERE vs HAVING đúng chỗ chưa?
- Có sử dụng SELECT * không cần thiết?
- Đề xuất query hints nếu cần

## 3. 🔒 Bảo mật (Security)
- SQL Injection vulnerabilities
- Quyền truy cập dữ liệu nhạy cảm
- Đề xuất prepared statements/parameterized queries

## 4. 📖 Khả năng Đọc & Maintain (Readability)
- Code formatting và style
- Comment có đủ không?
- Naming conventions
- Đề xuất cách viết rõ ràng hơn

## 5. 💡 Best Practices
- Tuân thủ SQL standards chưa?
- Transaction handling (nếu có)
- Error handling
- Các best practices khác

## 6. ✏️ Phiên bản Tối ưu (Optimized Version)
Viết lại câu SQL đã được tối ưu (nếu cần), kèm giải thích các thay đổi.

⚠️ **LƯU Ý**: Đưa ra đánh giá khách quan, chi tiết, có ví dụ cụ thể. Sử dụng emoji và Markdown formatting để dễ đọc.

BẮT ĐẦU REVIEW:
"""
    
    @safe_execute
    def export_result(self):
        """Export kết quả review"""
        if not self.current_review_result:
            QMessageBox.warning(self, 'Chưa có Kết quả', 
                '⚠️ Chưa có kết quả review để export\n\n' +
                '💡 Hãy review SQL query trước khi export.')
            return
        
        # Get filename
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Export Review Result',
            f'sql_review_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md',
            'Markdown Files (*.md);;Text Files (*.txt);;All Files (*.*)'
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("# SQL Review Result\n\n")
                    f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("## Original SQL Query\n\n")
                    f.write(f"```sql\n{self.sql_input.toPlainText()}\n```\n\n")
                    f.write("## Review Result\n\n")
                    f.write(self.current_review_result)
                
                QMessageBox.information(self, 'Export Thành công',
                    f'✅ Đã export kết quả thành công!\n\n' +
                    f'📁 File: {os.path.basename(filename)}\n' +
                    f'📂 Thư mục: {os.path.dirname(filename)}\n\n' +
                    f'💡 Bạn có thể mở file này bằng bất kỳ text editor nào.')
                self.status_bar.showMessage(f'✅ Đã export: {os.path.basename(filename)}', 3000)
            except Exception as e:
                QMessageBox.critical(self, 'Lỗi Export', 
                    f'❌ Không thể export file\n\n' +
                    f'Lỗi: {str(e)}\n\n' +
                    '💡 Kiểm tra quyền ghi file và đường dẫn.')
    
    def clear_results(self):
        """Xóa kết quả"""
        self.result_output.clear()
        self.raw_sql_output.clear()
        self.current_review_result = ""
        self.export_button.setEnabled(False)
        if self.testdata_output:
            self.testdata_output.clear()
        if self.testdata_copy_btn:
            self.testdata_copy_btn.setEnabled(False)
        self.status_bar.showMessage('Đã xóa kết quả', 2000)
    
    def configure_api_key(self):
        """Dialog để cấu hình API key"""
        dialog = QDialog(self)
        dialog.setWindowTitle('🔑 Cấu hình Gemini API Key')
        dialog.setMinimumWidth(600)
        
        layout = QVBoxLayout()
        
        # Header
        header = QLabel('<h2>🔑 Cấu hình Gemini API Key</h2>')
        header.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['primary']}, stop:1 #1e40af);
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(header)
        
        # Info message
        info_label = QLabel(
            '💡 <b>Hướng dẫn:</b><br>'
            '1. Truy cập: <a href="https://makersuite.google.com/app/apikey" style="color: #2563eb;">Google AI Studio</a><br>'
            '2. Đăng nhập với tài khoản Google<br>'
            '3. Click "Create API Key" và copy key<br>'
            '4. Paste vào ô bên dưới và lưu<br><br>'
            '🔒 <b>Bảo mật:</b> API key sẽ được lưu trong file config.json cục bộ.'
        )
        info_label.setWordWrap(True)
        info_label.setOpenExternalLinks(True)
        info_label.setStyleSheet(f"""
            QLabel {{
                background-color: #eff6ff;
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
                margin-bottom: 10px;
            }}
        """)
        layout.addWidget(info_label)
        
        # Current API Key display
        current_key = self.config_manager.get_api_key()
        if current_key and current_key != 'YOUR_API_KEY_HERE':
            masked_key = current_key[:8] + '...' + current_key[-4:] if len(current_key) > 12 else '***'
            current_label = QLabel(f'✅ API Key hiện tại: <code>{masked_key}</code>')
        else:
            current_label = QLabel('⚠️ Chưa cấu hình API Key')
        
        current_label.setStyleSheet(f"""
            QLabel {{
                background-color: #f9fafb;
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
            }}
        """)
        layout.addWidget(current_label)
        
        # Input form
        form_layout = QFormLayout()
        
        api_key_input = QLineEdit()
        api_key_input.setPlaceholderText('Nhập Gemini API Key của bạn...')
        api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_input.setMinimumHeight(35)
        
        show_key_checkbox = QCheckBox('Hiện API Key')
        show_key_checkbox.stateChanged.connect(
            lambda state: api_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if state else QLineEdit.EchoMode.Password
            )
        )
        
        form_layout.addRow('', api_key_input)
        form_layout.addRow('', show_key_checkbox)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        test_btn = QPushButton('🧪 Test Connection')
        test_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary_hover']};
            }}
        """)
        
        save_btn = QPushButton('💾 Lưu')
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #059669;
            }}
        """)
        
        cancel_btn = QPushButton('❌ Hủy')
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #dc2626;
            }}
        """)
        
        button_layout.addWidget(test_btn)
        button_layout.addStretch()
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        
        # Event handlers
        def test_api_key():
            """Test API key"""
            api_key = api_key_input.text().strip()
            if not api_key:
                QMessageBox.warning(dialog, '⚠️ Thiếu dữ liệu', 'Vui lòng nhập API Key!')
                return
            
            try:
                # Test với Gemini API
                configure(api_key=api_key)
                model = GenerativeModel('gemini-1.5-flash')
                response = model.generate_content('Hello')
                
                QMessageBox.information(
                    dialog,
                    '✅ Kết nối thành công',
                    f'✅ API Key hợp lệ!\n\n'
                    f'🤖 Gemini AI đã phản hồi thành công.\n\n'
                    f'💡 Bạn có thể lưu API Key này.'
                )
            except Exception as e:
                QMessageBox.critical(
                    dialog,
                    '❌ Kết nối thất bại',
                    f'❌ Không thể kết nối với Gemini API\n\n'
                    f'Lỗi: {str(e)}\n\n'
                    f'💡 Kiểm tra lại API Key hoặc kết nối internet.'
                )
        
        def save_api_key():
            """Save API key to config"""
            api_key = api_key_input.text().strip()
            if not api_key:
                QMessageBox.warning(dialog, '⚠️ Thiếu dữ liệu', 'Vui lòng nhập API Key!')
                return
            
            try:
                # Save to config
                config = self.config_manager.load_config()
                config['gemini_api_key'] = api_key
                self.config_manager.save_config(config)
                
                # Reinitialize Gemini API
                configure(api_key=api_key)
                
                QMessageBox.information(
                    dialog,
                    '✅ Lưu thành công',
                    '✅ Đã lưu API Key vào config.json\n\n' +
                    '🔄 API đã được khởi tạo lại.\n\n'
                    '💡 Bạn có thể sử dụng tool ngay bây giờ!'
                )
                
                dialog.accept()
                
            except Exception as e:
                QMessageBox.critical(
                    dialog,
                    '❌ Lỗi',
                    f'❌ Không thể lưu API Key\n\n'
                    f'Lỗi: {str(e)}'
                )
        
        test_btn.clicked.connect(test_api_key)
        save_btn.clicked.connect(save_api_key)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def open_config_file(self):
        """Mở file config"""
        config_path = os.path.abspath(self.config_manager.config_file)
        if os.path.exists(config_path):
            os.startfile(config_path)
        else:
            QMessageBox.warning(self, 'File Không Tồn tại', 
                f'❌ File config không tồn tại\n\n' +
                f'📁 Path: {config_path}\n\n' +
                '💡 File config sẽ được tạo tự động khi khởi động ứng dụng.')
    
    def show_bind_help(self):
        """Hiển thị hướng dẫn sử dụng Bind Parameters"""
        help_text = """
<h3>🔗 Hướng dẫn Bind Parameters</h3>

<h4>📝 Cách sử dụng:</h4>
<ol>
<li><b>Nhập SQL Query</b> với placeholders <code>?</code> thay cho giá trị</li>
<li><b>Nhập Parameters</b> theo 1 trong 2 format bên dưới</li>
<li>Click <b>"Bind Parameters"</b> để thay thế</li>
</ol>

<h4>📋 Format 1: JSON Array</h4>
<pre>
SQL: SELECT * FROM users WHERE id = ? AND status = ?
Params: [123, "active"]
→ SELECT * FROM users WHERE id = 123 AND status = 'active'
</pre>

<h4>� Format 2: Param Format (Compact)</h4>
<pre>
SQL: INSERT INTO table VALUES (?, ?, ?, ?)
Params: param:[1-1][2-○][3-][4-JPN]
→ INSERT INTO table VALUES (1, '', '', 'JPN')

<b>Giải thích:</b>
- [1-1]: Parameter #1 = 1 (number)
- [2-○]: Parameter #2 = '' (empty string, ký tự ○/◯/〇 = empty)
- [3-]: Parameter #3 = '' (empty string)
- [4-JPN]: Parameter #4 = 'JPN' (string)
</pre>

<h4>🔍 Type Mapping:</h4>
<ul>
<li><b>INT/BIGINT</b>: Số nguyên (123, 456)</li>
<li><b>VARCHAR/TEXT</b>: Chuỗi ("text", 'text')</li>
<li><b>DATE</b>: Ngày ("2024-01-01")</li>
<li><b>DECIMAL</b>: Số thực (123.45)</li>
<li><b>Empty string</b>: '', ○, ◯, 〇</li>
</ul>

<p><b>⚠️ Lưu ý:</b> Tool sẽ kiểm tra type mapping với schema database nếu đã load schema!</p>
<p><b>💡 Tip:</b> Format param:[...] rất tiện khi copy từ log/debug output!</p>
"""
        msg = QMessageBox(self)
        msg.setWindowTitle('📖 Hướng dẫn Bind Parameters')
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    @safe_execute
    def parse_parameters(self, params_text: str) -> List[Any]:
        """
        Parse parameters từ nhiều format:
        1. JSON array: ["value1", 123, "value2"]
        2. Param format: param:[1-value1][2-123][3-value2]
        """
        params_text = params_text.strip()
        
        # Try format: param:[1-value1][2-value2]...
        if params_text.startswith('param:['):
            return self.parse_param_format(params_text)
        
        # Try JSON format
        try:
            params = json.loads(params_text)
            if not isinstance(params, list):
                raise ValueError('Parameters phải là array/list')
            return params
        except json.JSONDecodeError:
            raise ValueError(
                'Format không hợp lệ!\n\n'
                'Hỗ trợ 2 format:\n'
                '1. JSON: ["value1", 123, "value2"]\n'
                '2. Param: param:[1-value1][2-123][3-value2]'
            )
    
    def parse_param_format(self, param_str: str) -> List[Any]:
        """
        Parse format: param:[1-value1][2-123][3-value2]
        Trả về list values theo thứ tự index
        """
        import re
        
        # Remove "param:" prefix
        if param_str.startswith('param:'):
            param_str = param_str[6:]
        
        # Pattern: [index-value]
        pattern = r'\[(\d+)-(.*?)\]'
        matches = re.findall(pattern, param_str)
        
        if not matches:
            raise ValueError(
                'Format param không đúng!\n\n'
                'Cần theo format: param:[1-value1][2-value2][3-value3]...\n'
                'Ví dụ: param:[1-1][2-○][3-][4-0][5-3015][6-JPN]'
            )
        
        # Sort by index and extract values
        sorted_matches = sorted(matches, key=lambda x: int(x[0]))
        
        # Check for missing indices
        expected_indices = list(range(1, len(sorted_matches) + 1))
        actual_indices = [int(m[0]) for m in sorted_matches]
        
        if actual_indices != expected_indices:
            raise ValueError(
                f'Indices không liên tục!\n'
                f'Expected: {expected_indices}\n'
                f'Found: {actual_indices}\n\n'
                f'Indices phải bắt đầu từ 1 và liên tục.'
            )
        
        # Convert values to appropriate types
        params = []
        for idx, value in sorted_matches:
            params.append(self.convert_param_value(value))
        
        return params
    
    def convert_param_value(self, value: str) -> Any:
        """
        Convert string value to appropriate Python type:
        - Empty string "" → None (NULL)
        - "0" → 0 (integer)
        - "123" → 123 (integer)
        - "3.14" → 3.14 (float)
        - "○", "◯" → empty string ""
        - Other → string
        """
        if not value or value in ['○', '◯', '〇']:
            # Empty or circle symbols → empty string
            return ''
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Default: string
        return value
    
    @safe_execute
    def bind_sql_parameters(self):
        """Bind parameters vào SQL query và validate type (hỗ trợ nhiều format)"""
        self.loading_overlay.show_loading('Đang bind parameters...')
        QApplication.processEvents()
        
        try:
            import re
            
            sql = self.bind_sql_input.toPlainText().strip()
            params_text = self.bind_params_input.toPlainText().strip()
            
            if not sql:
                QMessageBox.warning(self, '⚠️ Thiếu dữ liệu', 'Vui lòng nhập SQL query!')
                return
            
            if not params_text:
                QMessageBox.warning(self, '⚠️ Thiếu dữ liệu', 'Vui lòng nhập parameters!')
                return
            
            # Count placeholders
            placeholder_count = sql.count('?')
            
            # Parse parameters (support multiple formats)
            try:
                params = self.parse_parameters(params_text)
            except ValueError as e:
                QMessageBox.critical(
                    self, 
                    '❌ Lỗi Parse Parameters', 
                    f'{str(e)}\n\n💡 Kiểm tra lại format của parameters.'
                )
                return
            
            # Check count match
            if len(params) != placeholder_count:
                QMessageBox.critical(
                    self,
                    '❌ Số lượng không khớp',
                    f'SQL có {placeholder_count} placeholders (?)\n' +
                    f'Nhưng bạn cung cấp {len(params)} parameters!\n\n' +
                    '⚠️ Số lượng phải bằng nhau.'
                )
                return
            
            # Extract table and column info from SQL for type checking
            validation_errors = []
            if self.current_schema:
                validation_errors = self.validate_parameter_types(sql, params)
            
            if validation_errors:
                error_msg = '⚠️ <b>Type Mismatch Warnings:</b><br><br>' + '<br>'.join(validation_errors)
                error_msg += '<br><br>💡 Bạn có muốn tiếp tục bind parameters không?'
                
                reply = QMessageBox.question(
                    self,
                    '⚠️ Cảnh báo Type',
                    error_msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.No:
                    return
            
            # Bind parameters
            result_sql = sql
            for param in params:
                # Format parameter based on type
                if param is None:
                    formatted = 'NULL'
                elif isinstance(param, bool):
                    formatted = 'TRUE' if param else 'FALSE'
                elif isinstance(param, (int, float)):
                    formatted = str(param)
                elif isinstance(param, str):
                    if param == '':
                        # Empty string → empty string literal
                        formatted = "''"
                    else:
                        # Escape single quotes
                        escaped = param.replace("'", "''")
                        formatted = f"'{escaped}'"
                else:
                    formatted = f"'{str(param)}'"
                
                # Replace first occurrence of ?
                result_sql = result_sql.replace('?', formatted, 1)
            
            # Display result
            self.bind_result_output.setPlainText(result_sql)
            self.copy_result_btn.setEnabled(True)
            
            # Show success message
            QMessageBox.information(
                self,
                '✅ Bind thành công',
                f'✅ Đã bind {len(params)} parameters vào SQL query!\n\n' +
                '📋 Bạn có thể copy kết quả bằng nút "Copy Result".'
            )
        finally:
            self.loading_overlay.hide_loading()
    
    def validate_parameter_types(self, sql: str, params: list) -> list:
        """Validate parameter types against database schema"""
        import re
        
        errors = []
        
        # Extract WHERE clause conditions
        where_match = re.search(r'where\s+(.*?)(?:group by|having|order by|limit|$)', sql, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return errors
        
        where_clause = where_match.group(1)
        
        # Find column comparisons with ?
        # Pattern: column_name = ?  or  table.column_name = ?
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>!]+\s*\?'
        matches = re.findall(pattern, where_clause, re.IGNORECASE)
        
        for i, (table_prefix, column_name) in enumerate(matches):
            if i >= len(params):
                break
            
            param = params[i]
            column_name = column_name.upper()
            
            # Find column in schema
            column_info = None
            if self.current_schema:
                for table_name, table_data in self.current_schema.items():
                    # table_data is a list of column dicts
                    if isinstance(table_data, list):
                        for col in table_data:
                            if col.get('name', '').upper() == column_name:
                                column_info = col
                                break
                    if column_info:
                        break
            
            if not column_info:
                continue
            
            # Check type compatibility - use 'type' key instead of 'data_type'
            db_type = column_info.get('type', '').upper()
            if not db_type:
                continue
                
            param_type = type(param).__name__
            
            is_valid = True
            expected_type = ""
            
            if any(t in db_type for t in ['INT', 'BIGINT', 'SMALLINT', 'TINYINT']):
                expected_type = 'số nguyên (int)'
                is_valid = isinstance(param, int) and not isinstance(param, bool)
            elif any(t in db_type for t in ['DECIMAL', 'FLOAT', 'DOUBLE', 'NUMERIC']):
                expected_type = 'số (int/float)'
                is_valid = isinstance(param, (int, float)) and not isinstance(param, bool)
            elif any(t in db_type for t in ['VARCHAR', 'TEXT', 'CHAR']):
                expected_type = 'chuỗi (str)'
                is_valid = isinstance(param, str)
            elif 'DATE' in db_type or 'TIME' in db_type:
                expected_type = 'ngày/giờ (str: YYYY-MM-DD)'
                is_valid = isinstance(param, str)
            
            if not is_valid:
                errors.append(
                    f'Parameter #{i+1} (<code>{param}</code> - type: <b>{param_type}</b>) ' +
                    f'không khớp với column <b>{column_name}</b> (type: <b>{db_type}</b>, cần: {expected_type})'
                )
        
        return errors
    
    def copy_bind_result(self):
        """Copy kết quả bind vào clipboard"""
        result = self.bind_result_output.toPlainText()
        if result:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(result)
                QMessageBox.information(
                    self,
                    '✅ Đã copy',
                    '📋 SQL query đã được copy vào clipboard!'
                )
            else:
                QMessageBox.warning(
                    self,
                    '❌ Lỗi clipboard',
                    'Không thể truy cập clipboard!'
                )
    
    def populate_testdata_controls(self):
        """Cập nhật danh sách bảng cho tab Test Data"""
        if not self.testdata_table_combo:
            return
        
        tables = sorted(self.db_schema.keys())
        self.testdata_table_combo.blockSignals(True)
        self.testdata_table_combo.clear()
        
        if not tables:
            self.testdata_table_combo.addItem('Chưa load schema', '')
        else:
            for table in tables:
                self.testdata_table_combo.addItem(table, table)
            self.testdata_table_combo.setCurrentIndex(0)
        
        self.testdata_table_combo.blockSignals(False)
        self.refresh_relationship_summary()
    
    def refresh_relationship_summary(self):
        """Hiển thị lại quan hệ bảng trong tab Test Data với relationship types"""
        if not self.relationship_tree or not self.testdata_table_combo:
            return
        
        self.relationship_tree.clear()
        current_table = self.testdata_table_combo.currentData()
        if not current_table or current_table not in self.db_relationships:
            placeholder = QTreeWidgetItem(['-', 'Chưa chọn bảng', '-'])
            self.relationship_tree.addTopLevelItem(placeholder)
            if self.relationship_hint_label:
                self.relationship_hint_label.setText('Chọn một bảng để xem quan hệ khóa ngoại.')
            return
        
        rel_info = self.db_relationships.get(current_table, {'references': [], 'referenced_by': []})
        parents = rel_info.get('references', [])
        children = rel_info.get('referenced_by', [])
        
        # Get detected relationship types
        detected_rels = self.detected_relationships.get(current_table, [])
        
        if self.relationship_hint_label:
            self.relationship_hint_label.setText(
                f'Bảng `{current_table}` có {len(parents)} quan hệ tới bảng cha và {len(children)} bảng con.'
            )
        
        if not parents and not children:
            self.relationship_tree.addTopLevelItem(QTreeWidgetItem(['-', 'Không có quan hệ FK', '-']))
            return

        for rel in parents:
            # Find detected relationship type
            rel_type_str = '1-N'  # Default
            for detected in detected_rels:
                if (detected.from_column == rel.get('column') and 
                    detected.to_table == rel.get('referenced_table')):
                    rel_type_str = detected.rel_type.value
                    break
            
            detail = f"[{rel_type_str}] {rel.get('column')} ➜ {rel.get('referenced_table')}.{rel.get('referenced_column')} (DEL {rel.get('delete_rule')})"
            item = QTreeWidgetItem(['FK ➡️ Cha', rel.get('referenced_table', ''), detail])
            
            # Color coding by relationship type
            if rel_type_str == '1-1':
                item.setForeground(0, QColor(COLORS['success']))  # Green for 1-1
            elif rel_type_str == 'N-N':
                item.setForeground(0, QColor(COLORS['warning']))  # Orange for N-N
            else:
                item.setForeground(0, QColor(COLORS['primary']))  # Blue for 1-N
            
            self.relationship_tree.addTopLevelItem(item)

        for rel in children:
            detail = f"{rel.get('table')}.{rel.get('referencing_column')} ⇐ {current_table}.{rel.get('column')} (DEL {rel.get('delete_rule')})"
            item = QTreeWidgetItem(['FK ⬅️ Con', rel.get('table', ''), detail])
            self.relationship_tree.addTopLevelItem(item)

    @safe_execute
    def open_manual_relationship_dialog(self):
        """Mở dialog để thêm quan hệ thủ công"""
        if not self.db_schema:
            QMessageBox.warning(self, 'Thiếu Schema', '⚠️ Load schema trước khi thêm quan hệ thủ công.')
            return

        dialog = ManualRelationshipDialog(self, self.db_schema)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result:
            self.add_manual_relationship(
                child_table=dialog.result['child_table'],
                child_column=dialog.result['child_column'],
                parent_table=dialog.result['parent_table'],
                parent_column=dialog.result['parent_column'],
                relationship_name=dialog.result.get('relationship_name')
            )
            self.refresh_relationship_summary()

    def add_manual_relationship(self, child_table: str, child_column: str,
                                parent_table: str, parent_column: str,
                                relationship_name: Optional[str] = None,
                                persist: bool = True):
        """Thêm quan hệ thủ công vào cấu hình chạy"""
        if not child_table or not parent_table or not child_column or not parent_column:
            return

        constraint_name = relationship_name or f'manual_fk_{child_table}_{child_column}_{parent_table}'
        relation = {
            'constraint': constraint_name,
            'column': child_column,
            'referenced_table': parent_table,
            'referenced_column': parent_column,
            'update_rule': 'NO ACTION',
            'delete_rule': 'NO ACTION',
            'manual': True
        }

        reverse_relation = {
            'constraint': constraint_name,
            'table': child_table,
            'column': parent_column,
            'referencing_column': child_column,
            'update_rule': 'NO ACTION',
            'delete_rule': 'NO ACTION',
            'manual': True
        }

        self.db_relationships.setdefault(child_table, {'references': [], 'referenced_by': []})
        self.db_relationships.setdefault(parent_table, {'references': [], 'referenced_by': []})

        existing = [
            r for r in self.db_relationships[child_table]['references']
            if r.get('constraint') == constraint_name
        ]
        if not existing:
            self.db_relationships[child_table]['references'].append(relation)
            self.db_relationships[parent_table]['referenced_by'].append(reverse_relation)

        if persist:
            entry = {
                'child_table': child_table,
                'child_column': child_column,
                'parent_table': parent_table,
                'parent_column': parent_column,
                'relationship_name': constraint_name
            }
            if entry not in self.manual_relationships:
                self.manual_relationships.append(entry)

    def apply_manual_relationships(self):
        """Áp dụng lại các quan hệ thủ công đã thêm"""
        for entry in self.manual_relationships:
            self.add_manual_relationship(
                child_table=entry['child_table'],
                child_column=entry['child_column'],
                parent_table=entry['parent_table'],
                parent_column=entry['parent_column'],
                relationship_name=entry.get('relationship_name'),
                persist=False
            )
    
    @safe_execute
    def generate_test_data(self):
        """Sinh dữ liệu test dựa trên schema + relationship"""
        if not self.db_schema:
            QMessageBox.warning(self, 'Thiếu Schema', '⚠️ Vui lòng load schema MySQL trước khi sinh dữ liệu.')
            return
        
        if not self.testdata_table_combo or not self.testdata_row_spin:
            return
        
        base_table = self.testdata_table_combo.currentData()
        if not base_table or base_table not in self.db_schema:
            QMessageBox.warning(self, 'Chưa chọn bảng', '⚠️ Vui lòng chọn bảng nguồn cần sinh dữ liệu.')
            return
        
        # Check if AI generation is enabled
        if hasattr(self, 'testdata_use_ai_checkbox') and self.testdata_use_ai_checkbox and self.testdata_use_ai_checkbox.isChecked():
            self.generate_test_data_with_ai()
        else:
            self.generate_test_data_traditional()
    
    @safe_execute
    def generate_test_data_with_ai(self):
        """Sinh dữ liệu test thông minh sử dụng AI"""
        if not self.testdata_table_combo or not self.testdata_row_spin:
            return
        
        base_table = self.testdata_table_combo.currentData()
        if not base_table or base_table not in self.db_schema:
            return
        
        row_count = self.testdata_row_spin.value()
        include_parents = self.testdata_include_parents_checkbox.isChecked() if self.testdata_include_parents_checkbox else True
        include_children = self.testdata_include_children_checkbox.isChecked() if self.testdata_include_children_checkbox else False
        
        self.loading_overlay.show_loading('🤖 AI đang phân tích schema và sinh dữ liệu thông minh...')
        QApplication.processEvents()
        
        try:
            # Determine generation order
            order, parent_set, child_set = self.determine_generation_order(
                base_table, include_parents, include_children
            )
            if base_table not in order:
                order.append(base_table)
            
            # Generate data sequentially using AI
            self.ai_generated_data: Dict[str, List[Dict[str, Any]]] = {}
            self.ai_generation_order = order
            self.ai_current_index = 0
            self.ai_base_rows = row_count
            self.ai_parent_set = parent_set
            self.ai_child_set = child_set
            
            self.generate_next_table_with_ai()
            
        except Exception as e:
            logger.error(f"AI test data generation error: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(
                self,
                'Lỗi sinh dữ liệu AI',
                f'❌ Không thể sinh dữ liệu với AI:\n{e}\n\n💡 Thử tắt AI generation và sử dụng traditional mode.'
            )
            self.loading_overlay.hide_loading()
    
    def generate_next_table_with_ai(self):
        """Generate data cho table tiếp theo trong order"""
        if self.ai_current_index >= len(self.ai_generation_order):
            # All tables done, compile script
            self.compile_ai_generated_script()
            return
        
        table = self.ai_generation_order[self.ai_current_index]
        rows_needed = self.resolve_row_count_for_table(
            table, self.testdata_table_combo.currentData(), 
            self.ai_base_rows, self.ai_parent_set, self.ai_child_set
        )
        
        if rows_needed <= 0:
            self.ai_current_index += 1
            self.generate_next_table_with_ai()
            return
        
        # Get relationships for this table
        relationships = self.detected_relationships.get(table, [])
        
        # Start AI worker
        model_name = self.config_manager.get_model()
        self.ai_data_worker = AIDataGeneratorWorker(
            model_name=model_name,
            schema=self.db_schema,
            table=table,
            row_count=rows_needed,
            relationships=relationships,
            existing_data=self.ai_generated_data
        )
        
        self.ai_data_worker.finished.connect(self.on_ai_table_generated)
        self.ai_data_worker.error.connect(self.on_ai_generation_error)
        self.ai_data_worker.progress.connect(lambda msg: self.loading_overlay.show_loading(msg))
        self.ai_data_worker.start()
    
    def on_ai_table_generated(self, result: dict):
        """Callback khi AI generate xong 1 table"""
        # Merge result vào ai_generated_data
        self.ai_generated_data.update(result)
        
        # Move to next table
        self.ai_current_index += 1
        self.generate_next_table_with_ai()
    
    def on_ai_generation_error(self, error_msg: str):
        """Callback khi AI generation gặp lỗi"""
        logger.error(f"AI generation error: {error_msg}")
        QMessageBox.critical(
            self,
            'Lỗi AI Generation',
            f'❌ AI không thể generate dữ liệu:\n{error_msg}\n\n💡 Fallback sang traditional mode.'
        )
        self.loading_overlay.hide_loading()
        # Fallback to traditional mode
        self.generate_test_data_traditional()
    
    def compile_ai_generated_script(self):
        """Compile dữ liệu từ AI thành SQL script"""
        try:
            statements: List[str] = []
            total_rows = 0
            
            for table in self.ai_generation_order:
                rows = self.ai_generated_data.get(table, [])
                if not rows:
                    continue
                
                total_rows += len(rows)
                insert_stmt = self.render_insert_statement(table, rows)
                if insert_stmt:
                    statements.append(insert_stmt)
            
            base_table = self.testdata_table_combo.currentData()
            header_lines = [
                f'-- 🤖 AI-Generated Test Data - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                f'-- Root table: {base_table} (rows: {self.ai_base_rows})',
                f'-- Tables involved: {", ".join(self.ai_generation_order)}',
                f'-- Powered by: Gemini AI ({self.config_manager.get_model()})',
                '-- Data is REALISTIC and respects all FK constraints'
            ]
            script = '\n'.join(header_lines) + '\n\n'
            script += '\n\n'.join(statements) if statements else '-- Không có dữ liệu nào được tạo.'
            
            if self.testdata_output:
                self.testdata_output.setPlainText(script)
            if self.testdata_copy_btn:
                self.testdata_copy_btn.setEnabled(bool(script.strip()))
            
            summary = f'✅ AI đã sinh {total_rows} dòng dữ liệu REALISTIC cho {len(self.ai_generation_order)} bảng'
            self.status_bar.showMessage(summary, 6000)
            
        except Exception as e:
            logger.error(f"Compile AI script error: {e}")
            QMessageBox.critical(self, 'Lỗi', f'❌ Không thể compile script: {e}')
        finally:
            self.loading_overlay.hide_loading()
    
    @safe_execute
    def generate_test_data_traditional(self):
        """Sinh dữ liệu test dựa trên schema + relationship (traditional mode)"""
        if not self.db_schema:
            QMessageBox.warning(self, 'Thiếu Schema', '⚠️ Vui lòng load schema MySQL trước khi sinh dữ liệu.')
            return
        
        if not self.testdata_table_combo or not self.testdata_row_spin:
            return
        
        base_table = self.testdata_table_combo.currentData()
        if not base_table or base_table not in self.db_schema:
            QMessageBox.warning(self, 'Chưa chọn bảng', '⚠️ Vui lòng chọn bảng nguồn cần sinh dữ liệu.')
            return
        
        row_count = self.testdata_row_spin.value()
        include_parents = self.testdata_include_parents_checkbox.isChecked() if self.testdata_include_parents_checkbox else True
        include_children = self.testdata_include_children_checkbox.isChecked() if self.testdata_include_children_checkbox else False
        respect_fk = self.testdata_respect_fk_checkbox.isChecked() if self.testdata_respect_fk_checkbox else True
        
        self.loading_overlay.show_loading('Đang sinh dữ liệu test...')
        QApplication.processEvents()
        
        try:
            script, summary = self.build_test_data_script(
                base_table,
                row_count,
                include_parents,
                include_children,
                respect_fk
            )
            if self.testdata_output:
                self.testdata_output.setPlainText(script)
            if self.testdata_copy_btn:
                self.testdata_copy_btn.setEnabled(bool(script.strip()))
            self.status_bar.showMessage(summary, 5000)
        except Exception as e:
            QMessageBox.critical(
                self,
                'Lỗi sinh dữ liệu',
                f'❌ Không thể sinh dữ liệu mẫu:\n{e}'
            )
        finally:
            self.loading_overlay.hide_loading()
    
    def build_test_data_script(self, base_table: str, base_rows: int,
                               include_parents: bool, include_children: bool,
                               respect_fk: bool) -> tuple[str, str]:
        """Xây dựng script INSERT dữ liệu mẫu"""
        order, parent_set, child_set = self.determine_generation_order(
            base_table, include_parents, include_children
        )
        if base_table not in order:
            order.append(base_table)
        
        generated_data: Dict[str, Dict[str, Any]] = {}
        statements: List[str] = []
        total_rows = 0
        
        for table in order:
            rows_needed = self.resolve_row_count_for_table(
                table, base_table, base_rows, parent_set, child_set
            )
            if rows_needed <= 0:
                continue
            
            rows = self.generate_rows_for_table(
                table, rows_needed, generated_data, respect_fk
            )
            generated_data[table] = {'rows': rows}
            total_rows += len(rows)
            insert_stmt = self.render_insert_statement(table, rows)
            if insert_stmt:
                statements.append(insert_stmt)
        
        header_lines = [
            f'-- Sample data generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'-- Root table: {base_table} (rows: {base_rows})',
            f'-- Tables involved: {", ".join(order)}',
            f'-- FK integrity mode: {"RESPECTED" if respect_fk else "IGNORED"}'
        ]
        script = '\n'.join(header_lines) + '\n\n'
        script += '\n\n'.join(statements) if statements else '-- Không có dữ liệu nào được tạo.'
        summary = f'Đã sinh {total_rows} dòng dữ liệu cho {len(order)} bảng'
        return script, summary
    
    def determine_generation_order(self, base_table: str,
                                   include_parents: bool,
                                   include_children: bool) -> Tuple[List[str], Set[str], Set[str]]:
        """Tính toán thứ tự sinh dữ liệu để đảm bảo FK"""
        parent_order: List[str] = []
        parent_seen: set = set()
        
        def dfs_parent(table: str):
            for rel in self.db_relationships.get(table, {}).get('references', []):
                parent = rel.get('referenced_table')
                if not parent or parent in parent_seen:
                    continue
                parent_seen.add(parent)
                dfs_parent(parent)
                parent_order.append(parent)
        
        if include_parents:
            dfs_parent(base_table)
        
        child_order: List[str] = []
        child_seen: set = set()
        
        def dfs_child(table: str):
            for rel in self.db_relationships.get(table, {}).get('referenced_by', []):
                child = rel.get('table')
                if not child or child in child_seen or child == base_table:
                    continue
                child_seen.add(child)
                child_order.append(child)
                if include_children:
                    dfs_child(child)
        
        if include_children:
            dfs_child(base_table)
        
        def dedup(seq: List[str]) -> List[str]:
            seen = set()
            ordered: List[str] = []
            for item in seq:
                if item and item not in seen:
                    seen.add(item)
                    ordered.append(item)
            return ordered
        
        parents = dedup(parent_order)
        children = dedup(child_order)
        
        ordered_tables = parents + [base_table]
        for child in children:
            if child not in ordered_tables:
                ordered_tables.append(child)
        
        return ordered_tables, set(parents), set(children)
    
    def resolve_row_count_for_table(self, table: str, base_table: str, base_rows: int,
                                    parent_set: Set[str], child_set: Set[str]) -> int:
        """Quy định số dòng sẽ sinh cho từng bảng"""
        if table == base_table:
            return max(1, base_rows)
        if table in parent_set:
            return max(1, min(base_rows, 3))
        if table in child_set:
            return max(1, base_rows)
        return max(1, min(base_rows, 2))
    
    def generate_rows_for_table(self, table: str, rows: int,
                                generated_data: Dict[str, Dict[str, Any]],
                                respect_fk: bool) -> List[Dict[str, Any]]:
        """Sinh dữ liệu mẫu cho từng bảng"""
        columns = self.db_schema.get(table, [])
        if not columns or rows <= 0:
            return []
        
        fk_map = {
            rel.get('column'): rel
            for rel in self.db_relationships.get(table, {}).get('references', [])
        }
        
        result_rows: List[Dict[str, Any]] = []
        for idx in range(rows):
            row_data: Dict[str, Any] = {}
            for col in columns:
                col_name = col.get('name')
                if respect_fk and col_name in fk_map:
                    rel = fk_map[col_name]
                    ref_table = rel.get('referenced_table')
                    ref_column = rel.get('referenced_column')
                    ref_rows = generated_data.get(ref_table, {}).get('rows', [])
                    if ref_rows:
                        ref_row = ref_rows[idx % len(ref_rows)]
                        row_data[col_name] = ref_row.get(ref_column)
                        continue
                row_data[col_name] = self.generate_value_for_column(col, idx, table)
            result_rows.append(row_data)
        return result_rows
    
    def generate_value_for_column(self, column: Dict[str, Any], index: int, table: str) -> Any:
        """Sinh giá trị phù hợp với kiểu dữ liệu"""
        col_name = column.get('name', '')
        lower_name = col_name.lower()
        col_type = (column.get('type') or '').lower()
        seq = index + 1
        
        if 'tinyint(1' in col_type or lower_name.startswith('is_'):
            return 1 if seq % 2 else 0
        
        if any(keyword in lower_name for keyword in ['email']):
            return f'user{seq}@example.com'
        
        if any(keyword in lower_name for keyword in ['phone', 'tel']):
            return f'090{self.randomizer.randint(1000000, 9999999)}'
        
        if 'date' in col_type and 'time' not in col_type:
            return (datetime.now().date() - timedelta(days=seq)).strftime('%Y-%m-%d')
        
        if 'time' in col_type or 'timestamp' in col_type or 'datetime' in col_type:
            return (datetime.now() - timedelta(hours=seq)).strftime('%Y-%m-%d %H:%M:%S')
        
        if any(t in col_type for t in ['int', 'decimal', 'numeric', 'float', 'double']):
            base = 1000 if column.get('key') == 'PRI' else 10
            return base + seq
        
        if 'json' in col_type:
            sample = {'sample': col_name, 'index': seq}
            return json.dumps(sample)
        
        if any(token in lower_name for token in ['name', 'title']):
            return f'{table}_{col_name}_{seq}'
        
        if 'status' in lower_name:
            return 'active' if seq % 2 else 'inactive'
        
        if 'desc' in lower_name or 'note' in lower_name or 'text' in col_type:
            return f'Mẫu dữ liệu cho {col_name} #{seq}'
        
        # Default string value
        return f'{col_name}_{seq}'
    
    def render_insert_statement(self, table: str, rows: List[Dict[str, Any]]) -> str:
        """Chuyển dữ liệu thành script INSERT"""
        if not rows:
            return ''
        columns = [col.get('name') for col in self.db_schema.get(table, [])]
        if not columns:
            return ''
        
        column_clause = ', '.join(f'`{col}`' for col in columns)
        value_lines = []
        for row in rows:
            values = ', '.join(self.format_sql_value(row.get(col)) for col in columns)
            value_lines.append(f'    ({values})')
        
        return f'INSERT INTO `{table}` ({column_clause}) VALUES\\n' + ',\\n'.join(value_lines) + ';'
    
    def format_sql_value(self, value: Any) -> str:
        """Format Python value -> SQL literal"""
        if value is None:
            return 'NULL'
        if isinstance(value, bool):
            return '1' if value else '0'
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value)
        text = text.replace("'", "''")
        return f"'{text}'"
    
    @safe_execute
    def copy_testdata_sql(self):
        """Copy script test data vào clipboard"""
        if not self.testdata_output:
            return
        script = self.testdata_output.toPlainText().strip()
        if not script:
            QMessageBox.information(self, 'Chưa có dữ liệu', '⚠️ Chưa có script nào để copy.')
            return
        clipboard = QApplication.clipboard()
        if clipboard is None:
            QMessageBox.warning(self, 'Lỗi clipboard', 'Không thể truy cập clipboard!')
            return
        clipboard.setText(script)
        QMessageBox.information(self, '✅ Đã copy', '📋 Script dữ liệu test đã được copy.')
    
    @safe_execute
    def open_ai_chat(self):
        """Mở dialog chat với Gemini AI"""
        dialog = AIChatDialog(self)
        dialog.exec()
    
    def show_about(self):
        """Hiển thị thông tin về app"""
        about_text = """
        <h2>SQL Reviewer Pro</h2>
        <p><b>Version:</b> 2.0</p>
        <p><b>Powered by:</b> Google Gemini AI</p>
        
        <p><b>Features:</b></p>
        <ul>
            <li>✅ Phân tích và review SQL queries chuyên sâu</li>
            <li>✅ Hiển thị schema database chi tiết</li>
            <li>✅ Lưu/Load connection profiles</li>
            <li>✅ Export kết quả review</li>
            <li>✅ Giao diện đẹp, dễ sử dụng</li>
        </ul>
        
        <p><b>Tech Stack:</b></p>
        <ul>
            <li>Python 3.x</li>
            <li>PyQt6</li>
            <li>MySQL Connector</li>
            <li>Google Generative AI</li>
        </ul>
        
        <p><i>© 2025 SQL Reviewer Pro. All rights reserved.</i></p>
        """
        QMessageBox.about(self, 'About SQL Reviewer Pro', about_text)
    
    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Đóng ứng dụng"""
        if self.db_conn:
            self.db_conn.close()
        if a0:
            a0.accept()


def main():
    """Entry point with proper exception handling for .exe builds"""
    try:
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle('Fusion')
        
        # Configure exception handling for PyQt6
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            error_msg = f"❌ Unexpected Error:\n\n{exc_type.__name__}: {str(exc_value)}"
            try:
                QMessageBox.critical(None, 'Application Error', error_msg)
            except:
                # Fallback if QMessageBox fails
                print(error_msg)
        
        sys.excepthook = handle_exception
        
        # Check config
        config_manager = ConfigManager()
        api_key = config_manager.get_api_key()
        
        if api_key == 'YOUR_API_KEY_HERE':
            QMessageBox.critical(None, 'Lỗi Cấu hình',
                '❌ API Key chưa được cấu hình!\n\n' +
                'Vui lòng:\n' +
                '1. Mở file config.json\n' +
                '2. Thay thế YOUR_API_KEY_HERE bằng API key của bạn\n' +
                '3. Lưu file và khởi động lại ứng dụng\n\n' +
                'Lấy API key tại: https://makersuite.google.com/app/apikey')
            sys.exit(1)
        
        # Create and show main window
        window = SQLReviewerApp()
        window.show()
        
        sys.exit(app.exec())
        
    except Exception as e:
        error_msg = f"❌ Fatal Error during startup:\n\n{type(e).__name__}: {str(e)}"
        try:
            if 'app' in locals():
                QMessageBox.critical(None, 'Startup Error', error_msg)
            else:
                # Create minimal app just for error display
                error_app = QApplication(sys.argv)
                QMessageBox.critical(None, 'Startup Error', error_msg)
        except:
            print(error_msg)
        sys.exit(1)


if __name__ == '__main__':
    main()
