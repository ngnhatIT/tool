"""
SQL Reviewer Tool - Công cụ Review SQL chuyên nghiệp với Gemini AI
Version: 2.0
Author: Advanced Version
"""

import sys
import os
import json
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
import mysql.connector
from google.generativeai import configure, GenerativeModel  # type: ignore
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QTreeWidget, QTreeWidgetItem, QMessageBox,
    QFormLayout, QComboBox, QFileDialog, QTabWidget,
    QStatusBar, QMainWindow, QSplitter, QDialog, QProgressBar, QCheckBox
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
    'primary': '#3b82f6',      # Blue 500
    'primary_hover': '#2563eb', # Blue 600
    'success': '#10b981',      # Emerald 500
    'success_hover': '#059669', # Emerald 600
    'danger': '#ef4444',       # Red 500
    'danger_hover': '#dc2626',  # Red 600
    'warning': '#f59e0b',      # Amber 500
    'warning_hover': '#d97706', # Amber 600
    'secondary': '#6366f1',    # Indigo 500
    'secondary_hover': '#4f46e5', # Indigo 600
    'text_primary': '#1f2937',  # Gray 800
    'text_secondary': '#6b7280',# Gray 500
    'text_white': '#ffffff',
    'bg_primary': '#ffffff',
    'bg_secondary': '#f3f4f6',  # Gray 100
    'bg_hover': '#e5e7eb',      # Gray 200
    'border': '#d1d5db',       # Gray 300
    'border_focus': '#3b82f6',  # Blue 500
    'tree_bg': '#f9fafb',       # Gray 50
    'tree_header': '#e5e7eb',   # Gray 200
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
                "user": "root"
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
            logger.error("GeminiWorker exception: %s\n%s", e, traceback.format_exc())
            if "API_KEY_INVALID" in str(e):
                self.error.emit('API Key không hợp lệ. Vui lòng kiểm tra lại Key trong config.json')
            else:
                self.error.emit(f'Lỗi khi gọi Gemini: {str(e)}')


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
        
        # Loading label
        self.loading_label = QLabel(self.message)
        self.loading_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['primary']};
                font-size: 16px;
                font-weight: bold;
                background-color: white;
                padding: 20px 40px;
                border-radius: 12px;
                border: 3px solid {COLORS['primary']};
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
        """Thêm message vào chat display"""
        if role == 'user':
            formatted = f"""
<div style='background-color: {COLORS["primary"]}; color: white; padding: 10px; 
            border-radius: 10px; margin: 5px 0; margin-left: 50px;'>
    <b>👤 Bạn:</b><br>{message.replace(chr(10), '<br>')}
</div>
"""
        else:
            formatted = f"""
<div style='background-color: white; color: {COLORS["text_primary"]}; padding: 10px; 
            border-radius: 10px; margin: 5px 0; margin-right: 50px; border: 2px solid {COLORS["border"]};'>
    <b>🤖 Gemini AI:</b><br>{message.replace(chr(10), '<br>')}
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


class SQLReviewerApp(QMainWindow):
    """Ứng dụng chính"""
    
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.db_conn: Any = None
        self.db_schema: Dict[str, List[Dict[str, Any]]] = {}
        self.current_schema: Optional[Dict[str, Any]] = None  # Schema for type validation
        self.gemini_worker: Optional[GeminiWorker] = None
        self.current_review_result = ""
        
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
        
        # Connection form header
        connection_header = QLabel('🔌 <b>Thông tin kết nối MySQL</b>')
        connection_header.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['bg_primary']};
                padding: 12px;
                border-radius: 6px;
                border: 2px solid {COLORS['border']};
                font-size: 14px;
            }}
        """)
        left_layout.addWidget(connection_header)
        
        self.connection_form = QFormLayout()
        self.db_host_input = QLineEdit('localhost')
        self.db_port_input = QLineEdit('3306')
        self.db_name_input = QLineEdit('')
        self.db_user_input = QLineEdit('root')
        self.db_pass_input = QLineEdit()
        self.db_pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.connection_form.addRow('Host:', self.db_host_input)
        self.connection_form.addRow('Port:', self.db_port_input)
        self.connection_form.addRow('Database / Schema:', self.db_name_input)
        self.connection_form.addRow('User:', self.db_user_input)
        self.connection_form.addRow('Password:', self.db_pass_input)
        
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
        
        # Schema tree
        left_layout.addWidget(QLabel('<b>Cấu trúc Database:</b>'))
        self.schema_tree = QTreeWidget()
        self.schema_tree.setHeaderLabels(['Tên', 'Kiểu', 'Chi tiết'])
        self.schema_tree.setColumnWidth(0, 200)
        self.schema_tree.setColumnWidth(1, 100)
        left_layout.addWidget(self.schema_tree)
        
        # --- PANEL PHẢI ---
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # SQL Input
        right_layout.addWidget(QLabel('<b>Nhập câu lệnh SQL:</b>'))
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
        self.bind_params_input.setPlaceholderText('["COMP001", "active", "2024-01-01"]')
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
        
        right_layout.addWidget(self.result_tabs)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
        # Floating AI Assistant Button
        self.create_floating_ai_button()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Sẵn sàng')
        
        # Apply stylesheet
        self.apply_stylesheet()
    
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
        """Apply custom stylesheet với màu sắc và UI/UX chuyên nghiệp"""
        self.setStyleSheet(f"""
            /* Main Window */
            QMainWindow {{
                background-color: {COLORS['bg_secondary']};
            }}
            
            /* Labels */
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
                font-weight: 500;
            }}
            
            /* Input Fields */
            QLineEdit, QTextEdit {{
                background-color: {COLORS['bg_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px 12px;
                font-size: 13px;
                color: {COLORS['text_primary']};
                selection-background-color: {COLORS['primary']};
                selection-color: white;
            }}
            QLineEdit:focus, QTextEdit:focus {{
                border-color: {COLORS['primary']};
                background-color: #ffffff;
                outline: none;
            }}
            QLineEdit:hover, QTextEdit:hover {{
                border-color: {COLORS['border_focus']};
            }}
            
            /* Buttons */
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: {COLORS['text_white']};
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
                min-height: 18px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary_hover']};
            }}
            QPushButton:pressed {{
                background-color: #475569;
                padding: 13px 23px 11px 25px;
            }}
            QPushButton:disabled {{
                background-color: #e2e8f0;
                color: #94a3b8;
            }}
            
            /* Tree Widget */
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
                border-radius: 4px;
            }}
            QTreeWidget::item:selected {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_white']};
                border-radius: 4px;
            }}
            QTreeWidget::branch:has-children:closed {{
                image: url(none);
            }}
            QTreeWidget::branch:has-children:open {{
                image: url(none);
            }}
            
            /* Header View */
            QHeaderView::section {{
                background-color: {COLORS['tree_header']};
                color: {COLORS['text_primary']};
                padding: 10px 8px;
                border: none;
                border-bottom: 3px solid {COLORS['primary']};
                font-weight: bold;
                font-size: 13px;
            }}
            
            /* ComboBox */
            QComboBox {{
                background-color: {COLORS['bg_primary']};
                border: 2px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px 12px;
                font-size: 13px;
                color: {COLORS['text_primary']};
                min-width: 100px;
            }}
            QComboBox:hover {{
                border-color: {COLORS['primary']};
            }}
            QComboBox:focus {{
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
                selection-color: white;
                padding: 4px;
            }}
            
            /* Tab Widget */
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
                color: {COLORS['text_primary']};
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['bg_primary']};
                color: {COLORS['primary']};
                font-weight: bold;
                border-bottom: 3px solid {COLORS['primary']};
                margin-bottom: -2px;
            }}
            
            /* Status Bar */
            QStatusBar {{
                background-color: {COLORS['bg_primary']};
                color: {COLORS['text_primary']};
                border-top: 2px solid {COLORS['border']};
                font-size: 12px;
                padding: 4px 8px;
            }}
            
            /* Form Labels */
            QFormLayout QLabel {{
                color: {COLORS['text_secondary']};
                font-weight: 600;
                font-size: 13px;
            }}
            
            /* Message Box */
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
            
            /* Scrollbar */
            QScrollBar:vertical {{
                background-color: {COLORS['bg_secondary']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {COLORS['border']};
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
                background-color: {COLORS['border']};
                border-radius: 6px;
                min-width: 30px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {COLORS['secondary']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            
            /* Splitter */
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
                use_pure=True  # Force pure Python connector (avoids crashing C extension in frozen exe)
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
                use_pure=True
            )
            
            # Load schema with detailed info
            self.get_mysql_schema_detailed()
            
            # Store schema for type validation
            self.current_schema = self.db_schema
            
            # Save last connection
            self.config_manager.save_last_connection({
                'host': self.db_host_input.text(),
                'port': self.db_port_input.text(),
                'database': self.db_name_input.text(),
                'user': self.db_user_input.text()
            })
            
            QMessageBox.information(self, 'Load Schema Thành công',
                '✅ Đã kết nối và tải schema thành công!\n\n' +
                f'📊 Database: {self.db_name_input.text()}\n' +
                f'📋 Số bảng: {len(self.db_schema)}\n\n' +
                '💡 Bạn có thể xem chi tiết cấu trúc ở panel bên trái.')
            self.status_bar.showMessage(f'✅ Đã load {len(self.db_schema)} bảng', 3000)
            
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
        
        return f"""
Bạn là một chuyên gia Senior Database Engineer và SQL Performance Tuning Expert với hơn 15 năm kinh nghiệm.

📊 **CẤU TRÚC DATABASE**:
{schema_string}

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
<li><b>Nhập Parameters</b> dưới dạng JSON array <code>["value1", "value2", ...]</code></li>
<li>Click <b>"Bind Parameters"</b> để thay thế</li>
</ol>

<h4>✅ Ví dụ đúng:</h4>
<pre>
SQL: SELECT * FROM users WHERE id = ? AND status = ?
Params: [123, "active"]
→ SELECT * FROM users WHERE id = 123 AND status = 'active'
</pre>

<h4>🔍 Kiểm tra Type:</h4>
<ul>
<li><b>INT/BIGINT</b>: Số nguyên (123, 456)</li>
<li><b>VARCHAR/TEXT</b>: Chuỗi ("text", 'text')</li>
<li><b>DATE</b>: Ngày ("2024-01-01")</li>
<li><b>DECIMAL</b>: Số thực (123.45)</li>
</ul>

<p><b>⚠️ Lưu ý:</b> Tool sẽ kiểm tra type mapping với schema database nếu đã load schema!</p>
"""
        msg = QMessageBox(self)
        msg.setWindowTitle('📖 Hướng dẫn Bind Parameters')
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()
    
    @safe_execute
    def bind_sql_parameters(self):
        """Bind parameters vào SQL query và validate type"""
        self.loading_overlay.show_loading('Đang bind parameters...')
        QApplication.processEvents()
        
        try:
            import json
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
            
            # Parse parameters
            try:
                params = json.loads(params_text)
                if not isinstance(params, list):
                    raise ValueError('Parameters phải là array')
            except json.JSONDecodeError as e:
                QMessageBox.critical(
                    self, 
                    '❌ Lỗi JSON', 
                    f'Không thể parse JSON:\n{str(e)}\n\nVí dụ đúng: ["value1", 123, "2024-01-01"]'
                )
                return
            except ValueError as e:
                QMessageBox.critical(self, '❌ Lỗi', str(e))
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
