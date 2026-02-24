import os
import sys
import shutil
import subprocess
import math
import numpy as np
import traceback
import re

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox,
    QHeaderView, QLabel, QFrame, QAbstractItemView, QProgressBar, 
    QMenu, QSlider, QSplitter, QDialog, QMenuBar
)
from PySide6.QtCore import (
    Qt, QRunnable, Slot, Signal, QObject, QThreadPool, 
    QUrl, QPoint, QTimer
)
from PySide6.QtGui import (
    QFont, QKeyEvent, QColor, QPainter, QPen, QFontDatabase, 
    QFontMetrics, QCursor, QPalette, QAction
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# --- FONTS SETUP ---
FONT_FILES = [
    "TitilliumWeb-Regular.ttf",
    "TitilliumWeb-SemiBold.ttf",
    "TitilliumWeb-SemiBoldItalic.ttf"
]

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# --- FFMPEG SETUP ---
def setup_ffmpeg():
    path_bin = resource_path("ffmpeg.bin")
    path_exe = resource_path("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    ffmpeg_bin = "ffmpeg"
    
    if os.path.exists(path_bin):
        ffmpeg_bin = path_bin
    elif os.path.exists(path_exe):
        ffmpeg_bin = path_exe
    elif shutil.which("ffmpeg"):
        ffmpeg_bin = shutil.which("ffmpeg")
    
    if getattr(sys, 'frozen', False) and os.path.exists(ffmpeg_bin):
        try: os.chmod(ffmpeg_bin, 0o755)
        except: pass
    return ffmpeg_bin

FFMPEG_BIN = setup_ffmpeg()

def get_clean_env():
    env = os.environ.copy()
    for var in ["DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]:
        if var in env: del env[var]
    return env

# -----------------------------
# ALGORITHM: PRECISION DETECTOR
# -----------------------------
def detect_clicks_precision(samples, threshold, window_size, gate_threshold):
    try:
        if len(samples) < 1000: return []
        
        diffs = np.abs(np.diff(samples, prepend=samples[0]))
        
        w = int(window_size)
        if w < 1: w = 1
        weights = np.ones(w) / w
        
        local_avg = np.convolve(diffs, weights, mode='same')
        
        epsilon = 1e-9
        prev_avg = np.roll(local_avg, 1)
        prev_avg[0] = epsilon
        
        ratios = diffs / (prev_avg + epsilon)
        condition = (ratios > threshold) & (np.abs(samples) > gate_threshold)
        indices = np.where(condition)[0]
        
        return indices.tolist()

    except Exception as e:
        print(f"Algo Error: {e}")
        traceback.print_exc()
        return []

# -----------------------------
# WORKERS
# -----------------------------
class Signals(QObject):
    discovery_finished = Signal(list)
    waveform_ready = Signal(int, object, float, int)
    scan_finished = Signal(int, str, int, object)

class FileDiscoveryWorker(QRunnable):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs
        self.signals = Signals()
        self.audio_extensions = ('.wav', '.aiff', '.aif', '.mp3', '.flac', '.ogg')

    @Slot()
    def run(self):
        found_files = []
        for path in self.inputs:
            if os.path.isdir(path):
                for root_dir, _, files in os.walk(path):
                    for f in files:
                        if f.lower().endswith(self.audio_extensions) and not f.startswith("._"):
                            found_files.append(os.path.join(root_dir, f))
            elif path.lower().endswith(self.audio_extensions):
                found_files.append(path)
        found_files.sort()
        self.signals.discovery_finished.emit(found_files)

class ClickAnalysisWorker(QRunnable):
    def __init__(self, row, path, threshold, window, gate, stop_func):
        super().__init__()
        self.row = row
        self.path = path
        self.threshold = threshold
        self.window = window
        self.gate = gate
        self.stop_func = stop_func
        self.signals = Signals()

    @Slot()
    def run(self):
        if not os.path.exists(self.path):
            self.signals.scan_finished.emit(self.row, "Missing", 0, [])
            return

        if self.stop_func(): return

        try:
            cmd = [
                FFMPEG_BIN, "-v", "info", "-nostats", "-i", self.path, 
                "-f", "f32le", "-ac", "1", "-"
            ]
            
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                startupinfo=startupinfo, env=get_clean_env()
            )
            
            try:
                raw_data, err_data = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                self.signals.scan_finished.emit(self.row, "Timeout", 0, [])
                return
            
            if self.stop_func(): return

            samples = np.frombuffer(raw_data, dtype=np.float32)
            
            if len(samples) < 100:
                self.signals.scan_finished.emit(self.row, "Too Short", 0, [])
                return

            sr = 44100
            err_str = err_data.decode('utf-8', errors='ignore')
            match = re.search(r"Audio:.*?, (\d+) Hz", err_str)
            if match:
                sr = int(match.group(1))

            duration = len(samples) / float(sr)

            target_width = 4000
            step = max(1, len(samples) // target_width)
            vis_waveform = samples[::step].copy()
            vis_waveform = np.clip(vis_waveform, -1.0, 1.0)
            self.signals.waveform_ready.emit(self.row, vis_waveform, duration, sr)

            if self.stop_func(): return

            peaks = detect_clicks_precision(samples, self.threshold, self.window, self.gate)

            final_count = len(peaks)
            status = "Clean"
            if final_count > 0: status = "Issues Found"
            if final_count > 100: status = "Heavy Artifacts"

            self.signals.scan_finished.emit(self.row, status, final_count, peaks)

        except Exception as e:
            print(f"Analysis Error: {e}")
            self.signals.scan_finished.emit(self.row, "Error", 0, [])

# -----------------------------
# CUSTOM WIDGETS
# -----------------------------
class ClickWaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMouseTracking(True)
        self.waveform = None      
        self.clicks = None        
        self.duration = 0.0       
        self.sr = 44100
        self.file_path = ""
        self.playhead_ms = 0
        self.is_playing = False
        self.setCursor(Qt.PointingHandCursor)
        self.hover_x = -1
        self.setStyleSheet("background-color: #202020; border-top: 1px solid #333;")

    def load_data(self, waveform, clicks, duration, path, sr=44100):
        self.waveform = waveform
        self.clicks = clicks
        self.duration = duration
        self.file_path = path
        self.sr = sr
        self.playhead_ms = 0
        self.update()

    def set_playhead(self, ms):
        self.playhead_ms = ms
        self.update()

    def clear(self):
        self.waveform = None
        self.clicks = None
        self.file_path = ""
        self.update()

    def mousePressEvent(self, event):
        if not self.duration or event.button() != Qt.LeftButton: return
        x = event.position().x()
        pct = x / self.width()
        ms = pct * (self.duration * 1000)
        if self.parent():
            self.window().seek_media(ms)

    def mouseMoveEvent(self, event):
        self.hover_x = event.position().x()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(32, 32, 32))

        if self.waveform is None:
            p.setPen(QColor(100, 100, 100))
            p.drawText(self.rect(), Qt.AlignCenter, "Select a file to visualize")
            return

        w = self.width()
        h = self.height()
        mid_y = h / 2
        
        p.setPen(QColor(88, 163, 156)) 
        path_points = []
        if len(self.waveform) > 0:
            x_step = w / len(self.waveform)
            for i, val in enumerate(self.waveform):
                px = i * x_step
                py = mid_y - (val * (mid_y * 0.9)) 
                path_points.append(QPoint(int(px), int(py)))
            p.drawPolyline(path_points)

        if self.clicks is not None and len(self.clicks) > 0:
            p.setPen(QPen(QColor(255, 107, 107, 200), 1)) 
            total_samples = self.duration * self.sr 
            scale = w / total_samples
            for c_idx in self.clicks:
                cx = c_idx * scale
                p.drawLine(int(cx), 0, int(cx), h)

        p.setPen(QPen(QColor(255, 255, 255), 2))
        play_x = (self.playhead_ms / (self.duration * 1000)) * w if self.duration > 0 else 0
        p.drawLine(int(play_x), 0, int(play_x), h)
        
        if self.hover_x >= 0 and self.duration > 0:
            pct = self.hover_x / w
            sec = pct * self.duration
            time_str = f"{int(sec // 60)}:{int(sec % 60):02d}"
            p.setPen(QColor(252, 163, 17))
            p.drawText(int(self.hover_x) + 5, h - 5, time_str)

DIALOG_STYLESHEET = """
    QDialog { background-color: #252525; border: 1px solid #444; }
    QLabel { color: #e0e0e0; font-size: 12px; background-color: transparent; }
    QLabel#DialogHeader {
        color: #DDD; font-weight: 600; font-style: italic; font-size: 18px; 
        letter-spacing: 1px; margin-bottom: 0px;
    }
    QPushButton {
        background-color: #444; color: white; border: 1px solid #555;
        padding: 6px 20px; border-radius: 4px; font-size: 12px; font-weight: 500;
        min-width: 80px;
    }
    QPushButton:hover { background-color: #505050; border-color: #666; }
    QPushButton:pressed { background-color: #383838; }
"""

class ModernPopup(QDialog):
    def __init__(self, parent, title, message, mode="info", min_width=400):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(max(min_width, 400))
        self.setStyleSheet(DIALOG_STYLESHEET)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(10)

        self.lbl_header = QLabel(title.upper())
        self.lbl_header.setObjectName("DialogHeader")
        layout.addWidget(self.lbl_header)

        self.lbl_msg = QLabel(message)
        self.lbl_msg.setWordWrap(True)
        self.lbl_msg.setTextFormat(Qt.RichText)
        self.lbl_msg.setOpenExternalLinks(True)
        layout.addWidget(self.lbl_msg)

        layout.addSpacing(10)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        if mode == "question":
            self.btn_yes = QPushButton("Yes")
            self.btn_yes.clicked.connect(self.accept)
            self.btn_no = QPushButton("No")
            self.btn_no.clicked.connect(self.reject)
            btn_layout.addWidget(self.btn_no)
            btn_layout.addWidget(self.btn_yes)
            self.btn_yes.setFocus()
        else:
            self.btn_ok = QPushButton("OK")
            self.btn_ok.clicked.connect(self.accept)
            btn_layout.addWidget(self.btn_ok)
            self.btn_ok.setFocus()
        
        layout.addLayout(btn_layout)

class ModernTable(QTableWidget):
    delete_signal = Signal()
    files_dropped = Signal(list)
    selection_changed_custom = Signal()

    def __init__(self, font_family):
        super().__init__(0, 3) 
        self.font_family = font_family
        self.setHorizontalHeaderLabels(["File Name", "Clicks", "Status"])
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus) 
        
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(1, QHeaderView.Fixed)
        h.setSectionResizeMode(2, QHeaderView.Fixed)
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 120)
        h.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        h.setFixedHeight(30)
        
        self.itemSelectionChanged.connect(self.selection_changed_custom.emit)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_signal.emit()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.rowCount() == 0:
            p = QPainter(self.viewport()); p.save()
            f = p.font(); f.setFamily(self.font_family); f.setWeight(QFont.DemiBold); f.setItalic(True); f.setPointSize(24); p.setFont(f)
            p.setPen(QColor(80, 80, 80)); p.drawText(self.viewport().rect(), Qt.AlignCenter, "DRAG & DROP FILES HERE"); p.restore()

    def dragEnterEvent(self, e): e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dragMoveEvent(self, e): e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e): self.files_dropped.emit([u.toLocalFile() for u in e.mimeData().urls()])

# -----------------------------
# MAIN WINDOW
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self, font_family):
        super().__init__()
        self.font_family = font_family
        self.setWindowTitle("Rogue Waves CLICK DETECTOR")
        self.resize(1100, 800)
        self.setAcceptDrops(True)
        self.setup_dark_theme()
        
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(max(1, (os.cpu_count() or 4) - 1))
        
        self.files = [] 
        self.file_data = {} 
        self.stop_flag = False
        self.is_scanning = False
        self.scan_queue_index = 0
        
        # Audio Player
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.mediaStatusChanged.connect(self.on_media_status)
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)
        
        self.current_playing_row = -1
        self.last_active_row = -1 

        # UI LAYOUT
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SIDEBAR ---
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(300)
        sb_layout = QVBoxLayout(self.sidebar)
        sb_layout.setContentsMargins(15, 20, 15, 20)
        sb_layout.setSpacing(8)

        sidebar_width = 300
        sidebar_margins = 30
        available_width = sidebar_width - sidebar_margins
        title_size = self.get_optimal_font_size("ROGUE WAVES", 42, available_width)
        subtitle_size = self.get_optimal_font_size("CLICK DETECTOR", 24, available_width)
        
        self.lbl_logo = QLabel()
        self.lbl_logo.setAlignment(Qt.AlignCenter)
        self.lbl_logo.setText(f"""
            <html><head/><body>
            <p align="center" style="margin-bottom:0px; margin-top:0px; line-height:0.75;">
                <span style="font-family:'{self.font_family}'; font-size:{title_size}px; font-weight:600; font-style:italic; color:#5ba49d;">ROGUE WAVES</span><br/>
                <span style="font-family:'{self.font_family}'; font-size:{subtitle_size}px; font-weight:600; font-style:italic; color:#808080;">CLICK DETECTOR</span>
            </p>
            </body></html>
        """)
        sb_layout.addWidget(self.lbl_logo)
        sb_layout.addSpacing(10)

        sb_layout.addWidget(self.create_header_label("INPUT"))
        self.btn_select = self.create_button("Select Files / Folder", self.select_input)
        sb_layout.addWidget(self.btn_select)
        sb_layout.addSpacing(10)

        # --- ANALYSIS SETTINGS ---
        sb_layout.addWidget(self.create_header_label("ANALYSIS SETTINGS"))
        
        self.analysis_frame = QFrame()
        self.analysis_frame.setObjectName("StatsFrame")
        af_layout = QVBoxLayout(self.analysis_frame)
        af_layout.setContentsMargins(8, 8, 8, 8)
        af_layout.setSpacing(12)
        
        thresh_layout = QVBoxLayout()
        thresh_layout.setSpacing(2)
        lbl_thresh = QLabel("THRESHOLD (Ratio)")
        lbl_thresh.setStyleSheet("color: #888; font-weight: bold; font-size: 11px;")
        
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(20, 200) 
        self.slider_thresh.setValue(50)      
        self.slider_thresh.valueChanged.connect(self.update_settings_labels)
        
        self.lbl_thresh_val = QLabel("5.00")
        self.lbl_thresh_val.setStyleSheet("color: #58A39C; font-weight: bold;")
        self.lbl_thresh_val.setAlignment(Qt.AlignRight)
        
        t_header = QHBoxLayout()
        t_header.addWidget(lbl_thresh)
        t_header.addWidget(self.lbl_thresh_val)
        
        thresh_layout.addLayout(t_header)
        thresh_layout.addWidget(self.slider_thresh)
        af_layout.addLayout(thresh_layout)

        win_layout = QVBoxLayout()
        win_layout.setSpacing(2)
        lbl_win = QLabel("CONTEXT WINDOW")
        lbl_win.setStyleSheet("color: #888; font-weight: bold; font-size: 11px;")
        
        self.slider_win = QSlider(Qt.Horizontal)
        self.slider_win.setRange(2, 200)
        self.slider_win.setValue(10)
        self.slider_win.valueChanged.connect(self.update_settings_labels)
        
        self.lbl_win_val = QLabel("10 samples")
        self.lbl_win_val.setStyleSheet("color: #58A39C; font-weight: bold;")
        self.lbl_win_val.setAlignment(Qt.AlignRight)
        
        w_header = QHBoxLayout()
        w_header.addWidget(lbl_win)
        w_header.addWidget(self.lbl_win_val)
        
        win_layout.addLayout(w_header)
        win_layout.addWidget(self.slider_win)
        af_layout.addLayout(win_layout)

        gate_layout = QVBoxLayout()
        gate_layout.setSpacing(2)
        lbl_gate = QLabel("NOISE GATE")
        lbl_gate.setStyleSheet("color: #888; font-weight: bold; font-size: 11px;")
        
        self.slider_gate = QSlider(Qt.Horizontal)
        self.slider_gate.setRange(0, 200) 
        self.slider_gate.setValue(10)     
        self.slider_gate.valueChanged.connect(self.update_settings_labels)
        
        self.lbl_gate_val = QLabel("0.010")
        self.lbl_gate_val.setStyleSheet("color: #58A39C; font-weight: bold;")
        self.lbl_gate_val.setAlignment(Qt.AlignRight)
        
        g_header = QHBoxLayout()
        g_header.addWidget(lbl_gate)
        g_header.addWidget(self.lbl_gate_val)
        
        gate_layout.addLayout(g_header)
        gate_layout.addWidget(self.slider_gate)
        af_layout.addLayout(gate_layout)

        sb_layout.addWidget(self.analysis_frame)
        sb_layout.addSpacing(10)
        
        # Stats
        sb_layout.addWidget(self.create_header_label("STATS"))
        self.stats_frame = QFrame()
        self.stats_frame.setObjectName("StatsFrame")
        stat_layout = QVBoxLayout(self.stats_frame)
        stat_layout.setSpacing(2)
        stat_layout.setContentsMargins(8, 8, 8, 8)
        
        self.lbl_total_files = QLabel("Files: 0")
        self.lbl_total_files.setStyleSheet("color: #BBBBBB;")
        self.lbl_issues = QLabel("Issues Found: 0")
        self.lbl_issues.setStyleSheet("color: #FF6B6B; font-weight: bold;")
        
        stat_layout.addWidget(self.lbl_total_files)
        stat_layout.addWidget(self.lbl_issues)
        sb_layout.addWidget(self.stats_frame)
        
        sb_layout.addStretch()

        # Moved Progress & Status above Action Button
        self.progress = QProgressBar()
        self.progress.setFixedHeight(5)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        sb_layout.addWidget(self.progress)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #888;") 
        sb_layout.addWidget(self.lbl_status)
        
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)
        
        self.btn_scan = QPushButton("SCAN")
        self.btn_scan.setObjectName("ActionBtn")
        self.btn_scan.setCursor(Qt.PointingHandCursor)
        self.btn_scan.clicked.connect(self.force_rescan)
        actions_layout.addWidget(self.btn_scan)
        
        sb_layout.addLayout(actions_layout)

        # IMPORTANT: Add the sidebar to the main layout!
        main_layout.addWidget(self.sidebar)

        # --- RIGHT SIDE ---
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(1)
        right_splitter.setStyleSheet("QSplitter::handle { background: #333; }")
        
        self.table = ModernTable(self.font_family)
        self.table.files_dropped.connect(self.load_files)
        self.table.delete_signal.connect(self.delete_selected)
        self.table.selection_changed_custom.connect(self.on_table_selection)
        self.table.doubleClicked.connect(self.on_table_double_click)
        right_splitter.addWidget(self.table)

        self.waveform = ClickWaveformWidget()
        right_splitter.addWidget(self.waveform)
        
        right_splitter.setStretchFactor(0, 4) 
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([600, 150])

        main_layout.addWidget(right_splitter)
        
        self.setup_menu_bar()
        self.grabKeyboard()

    def setup_menu_bar(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar { background-color: #333; color: #ddd; }
            QMenuBar::item { background: transparent; padding: 5px 10px; }
            QMenuBar::item:selected { background: #58A39C; color: white; }
            QMenu { background-color: #252525; color: #ddd; border: 1px solid #444; }
            QMenu::item { padding: 5px 20px; }
            QMenu::item:selected { background-color: #58A39C; color: white; }
        """)

        input_menu = menubar.addMenu("Input")
        act_sel = QAction("Select Files / Folder", self)
        act_sel.triggered.connect(self.select_input)
        input_menu.addAction(act_sel)
        input_menu.addSeparator()
        act_clear = QAction("Clear Files", self)
        act_clear.triggered.connect(self.clear_files)
        input_menu.addAction(act_clear)

        analysis_menu = menubar.addMenu("Analysis")
        act_rescan = QAction("Rescan Files", self)
        act_rescan.triggered.connect(self.force_rescan)
        analysis_menu.addAction(act_rescan)
        
        help_menu = menubar.addMenu("Help")
        act_about = QAction("About Click Detector", self)
        act_about.triggered.connect(self.show_about)
        help_menu.addAction(act_about)

    def create_header_label(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("SectionHeader")
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return lbl

    def create_button(self, text, func):
        btn = QPushButton(text)
        btn.clicked.connect(func)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def get_optimal_font_size(self, text, max_size, available_width):
        size = max_size
        font = QFont(self.font_family, size); font.setWeight(QFont.Bold)
        while size > 10:
            font.setPixelSize(size); metrics = QFontMetrics(font)
            if metrics.horizontalAdvance(text) < (available_width - 5): return size
            size -= 1
        return max_size 

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_playback()
        else:
            super().keyPressEvent(event)

    def update_settings_labels(self):
        t_val = self.slider_thresh.value() / 10.0
        self.lbl_thresh_val.setText(f"{t_val:.2f}")

        w_val = self.slider_win.value()
        self.lbl_win_val.setText(f"{w_val} samples")

        g_val = self.slider_gate.value() / 1000.0
        self.lbl_gate_val.setText(f"{g_val:.3f}")

    def select_input(self):
        f = QFileDialog.getOpenFileNames(self, "Select Audio", "", "Audio (*.wav *.aif *.mp3)")[0]
        if f: self.load_files(f)
    
    def dropEvent(self, event):
        self.load_files([url.toLocalFile() for url in event.mimeData().urls()])

    def stop_playback(self):
        self.player.stop()
        self.waveform.is_playing = False
        self.waveform.set_playhead(0)
        
        if self.current_playing_row != -1:
            self.set_row_visuals(self.current_playing_row, False)
            self.current_playing_row = -1

    def clear_files(self):
        self.stop_playback()
        self.player.setSource(QUrl()) 
        self.files = []
        self.file_data = {}
        self.table.setRowCount(0)
        self.lbl_total_files.setText("Files: 0")
        self.lbl_issues.setText("Issues Found: 0")
        self.waveform.clear()
        self.last_active_row = -1

    def load_files(self, paths):
        self.stop_playback()
        self.player.setSource(QUrl()) 

        self.file_data = {}
        self.files = []
        self.table.setRowCount(0)
        self.stop_flag = False
        self.btn_select.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.lbl_status.setText("Finding Files...")
        self.last_active_row = -1
        
        worker = FileDiscoveryWorker(paths)
        worker.signals.discovery_finished.connect(self.populate_and_scan)
        self.threadpool.start(worker)

    def populate_and_scan(self, files):
        self.files = files
        self.table.setRowCount(len(files))
        
        for i, f in enumerate(files):
            name = os.path.basename(f)
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(i, 1, QTableWidgetItem("-"))
            self.table.setItem(i, 2, QTableWidgetItem("Queued..."))
            self.file_data[i] = {'waveform': None, 'clicks': None, 'duration': 0, 'sr': 44100}

        self.lbl_total_files.setText(f"Files: {len(files)}")
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.btn_select.setEnabled(True)
        
        if len(files) > 0:
            self.table.selectRow(0)
            self.last_active_row = 0
            self.update_visualizer(0) 

        self.start_scan_process()

    def force_rescan(self):
        if self.is_scanning:
            self.stop_flag = True
            return
        self.stop_flag = False
        self.progress.setValue(0)
        self.start_scan_process()

    def start_scan_process(self):
        self.is_scanning = True
        self.btn_scan.setText("STOP")
        self.btn_scan.setObjectName("StopBtn")
        self.btn_scan.setStyle(self.btn_scan.style())
        self.lbl_status.setText("Scanning...")
        
        self.scan_queue_index = 0
        self.queue_scan_batch()

    def queue_scan_batch(self):
        if self.stop_flag: 
            self.finish_scan_ui()
            return

        batch_size = 5
        total = len(self.files)
        limit = min(self.scan_queue_index + batch_size, total)
        
        thresh = self.slider_thresh.value() / 10.0
        win = self.slider_win.value()
        gate = self.slider_gate.value() / 1000.0

        if self.scan_queue_index >= total:
            if self.threadpool.activeThreadCount() == 0:
                self.finish_scan_ui()
            else:
                QTimer.singleShot(100, self.queue_scan_batch)
            return

        for i in range(self.scan_queue_index, limit):
            self.table.setItem(i, 1, QTableWidgetItem("..."))
            self.table.setItem(i, 2, QTableWidgetItem("Decoding..."))
            
            worker = ClickAnalysisWorker(i, self.files[i], thresh, win, gate, lambda: self.stop_flag)
            worker.signals.waveform_ready.connect(self.on_waveform_ready)
            worker.signals.scan_finished.connect(self.on_scan_finished)
            self.threadpool.start(worker)

        self.scan_queue_index = limit
        QTimer.singleShot(50, self.queue_scan_batch)

    def finish_scan_ui(self):
        self.is_scanning = False
        self.btn_scan.setText("SCAN")
        self.btn_scan.setObjectName("ActionBtn")
        self.btn_scan.setStyle(self.btn_scan.style())
        self.progress.setVisible(False)
        self.lbl_status.setText("Ready" if not self.stop_flag else "Stopped")

    def on_waveform_ready(self, row, waveform, duration, sr):
        if self.stop_flag: return
        if row in self.file_data:
            self.file_data[row]['waveform'] = waveform
            self.file_data[row]['duration'] = duration
            self.file_data[row]['sr'] = sr
        self.table.setItem(row, 2, QTableWidgetItem("Analyzing..."))
        
        if self.last_active_row == row:
            self.update_visualizer(row)

    def on_scan_finished(self, row, status, count, clicks):
        if row >= self.table.rowCount(): return

        item_status = QTableWidgetItem(status)
        if "Clean" in status: item_status.setForeground(QColor("#58A39C")) 
        elif "Issues" in status: item_status.setForeground(QColor("#FF6B6B")) 
        self.table.setItem(row, 2, item_status)

        txt_count = str(count) if count > 0 else "0"
        item_count = QTableWidgetItem(txt_count)
        item_count.setTextAlignment(Qt.AlignCenter)
        if count > 0:
            item_count.setForeground(QColor("#FF6B6B"))
            item_count.setFont(QFont(self.font_family, 10, QFont.Bold))
        else:
            item_count.setForeground(QColor("#58A39C"))
        self.table.setItem(row, 1, item_count)

        if row in self.file_data:
            self.file_data[row]['clicks'] = clicks

        self.progress.setValue(self.progress.value() + 1)
        self.update_stats()
        
        if self.last_active_row == row:
            self.update_visualizer(row)

    def update_stats(self):
        issues = 0
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 1)
            if item and item.text() not in ["-", "0", "..."]:
                issues += 1
        self.lbl_issues.setText(f"Issues Found: {issues}")

    def delete_selected(self):
        rows = sorted([x.row() for x in self.table.selectionModel().selectedRows()], reverse=True)
        if self.current_playing_row in rows:
            self.stop_playback()
            self.player.setSource(QUrl())
        for r in rows:
            self.table.removeRow(r)
            del self.files[r]
            if r in self.file_data: del self.file_data[r]
        self.lbl_total_files.setText(f"Files: {len(self.files)}")
        self.update_stats()
        
        if self.last_active_row in rows:
            self.waveform.clear()
            self.last_active_row = -1

    def on_table_selection(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        row = rows[0].row()
        
        if self.current_playing_row != -1 and self.current_playing_row != row:
            self.stop_playback()
            
        self.last_active_row = row
        self.update_visualizer(row)

    def update_visualizer(self, row):
        if row in self.file_data:
            data = self.file_data[row]
            wf = data['waveform']
            dur = data['duration']
            clk = data['clicks']
            sr = data.get('sr', 44100)
            path = self.files[row]
            self.waveform.load_data(wf, clk, dur, path, sr)

    def on_table_double_click(self, index):
        self.start_playback(index.row())

    def start_playback(self, row):
        if self.is_scanning:
            self.lbl_status.setText("⚠️ Playback disabled during scan")
            return
        if row >= len(self.files): return
        
        self.player.stop() 
        self.player.setSource(QUrl())
        path = self.files[row]
        
        if self.current_playing_row != -1:
            self.set_row_visuals(self.current_playing_row, False)
        
        self.current_playing_row = row
        self.last_active_row = row 
        
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()
        self.waveform.is_playing = True
        self.table.selectRow(row)

    def set_row_visuals(self, row, playing):
        if row >= self.table.rowCount(): return
        item = self.table.item(row, 0)
        if not item: return
        f = item.font(); f.setBold(playing); item.setFont(f)
        txt = item.text()
        
        clean_txt = txt.replace("▶ ", "")
        
        if playing:
            item.setText(f"▶ {clean_txt}")
            item.setForeground(QColor("#58A39C"))
        else:
            item.setText(clean_txt)
            item.setForeground(QColor("#E0E0E0"))

    def toggle_playback(self):
        row_to_play = -1
        rows = self.table.selectionModel().selectedRows()
        
        if rows:
            row_to_play = rows[0].row()
        elif self.last_active_row != -1:
            row_to_play = self.last_active_row

        if row_to_play == -1: return

        if row_to_play != self.current_playing_row:
            self.start_playback(row_to_play)
            return

        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def seek_media(self, ms):
        if self.current_playing_row == -1:
            if self.last_active_row != -1:
                self.start_playback(self.last_active_row)
            else:
                return

        self.player.setPosition(int(ms))
        if self.player.playbackState() != QMediaPlayer.PlayingState:
            self.player.play()

    def on_playback_state_changed(self, state):
        if self.current_playing_row != -1:
            is_playing = (state == QMediaPlayer.PlayingState)
            self.set_row_visuals(self.current_playing_row, is_playing)

    def on_position_changed(self, ms):
        self.waveform.set_playhead(ms)

    def on_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.waveform.set_playhead(0)
            if self.current_playing_row != -1:
                self.set_row_visuals(self.current_playing_row, False)
                self.current_playing_row = -1

    def show_about(self):
        html = f"""
            <div style="line-height: 0.8;">
                <span style="font-family:'{self.font_family}'; font-size: 28px; font-weight: 600; font-style: italic; color: #5ba49d;">ROGUE WAVES</span><br>
                <span style="font-family:'{self.font_family}'; font-size: 16px; font-weight: 600; font-style: italic; color: #808080;">CLICK DETECTOR</span>
            </div>
            <p style="margin-top: 20px; font-size: 12px; color: #e0e0e0;">
                Version 1.0<br>
                Copyright Rogue Waves 2026.
            </p>
        """
        dlg = ModernPopup(self, "", "", "info")
        dlg.lbl_header.hide()
        dlg.btn_ok.hide()
        dlg.lbl_msg.setText(html)
        dlg.exec()

    def setup_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(32, 32, 32)) 
        palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.Highlight, QColor(61, 96, 93)) 
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        self.setStyleSheet(f"""
            QWidget {{ font-family: '{self.font_family}', sans-serif; font-size: 13px; }}
            QMainWindow {{ background-color: #1E1E1E; }}
            QFrame#Sidebar {{ background-color: #252525; border-right: 1px solid #333; }}
            
            QFrame#Sidebar QLabel#SectionHeader {{ 
                color: #DDD; font-weight: 600; font-style: italic; font-size: 20px; 
                letter-spacing: 1px; margin-top: 5px; margin-bottom: 0px; 
            }}
            
            QFrame#StatsFrame {{ background-color: #2A2A2A; border-radius: 4px; padding: 5px; }}
            
            QPushButton {{ 
                background-color: #333; border: 1px solid #444; border-radius: 4px; 
                padding: 5px; color: #ddd; font-weight: 400; min-height: 22px; 
            }}
            QPushButton:hover {{ background-color: #3E3E3E; border-color: #555; }}
            QPushButton:pressed {{ background-color: #222; }}
            
            QPushButton#ActionBtn {{ 
                background-color: #58A39C; color: white; border: none; 
                font-weight: 600; font-style: italic; font-size: 32px; 
                padding: 4px;
            }}
            QPushButton#ActionBtn:hover {{ background-color: #68B3AC; }}
            
            QPushButton#StopBtn {{ 
                background-color: #FF6B6B; color: white; border: none; 
                font-weight: 600; font-style: italic; font-size: 32px; 
                padding: 4px;
            }}
            QPushButton#StopBtn:hover {{ background-color: #FF5252; }}
            
            QSlider::handle:horizontal {{ background: #5ba49d; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
            QSlider::groove:horizontal {{ border: 1px solid #444; height: 4px; background: #333; border-radius: 2px; }}

            QTableWidget {{ 
                background-color: #202020; 
                alternate-background-color: #2A2A2A; 
                color: #E0E0E0; 
                border: none; 
                font-size: 12px;
                outline: 0;
                selection-background-color: #3D605D;
            }}
            
            QTableWidget::item:selected {{ background-color: #3D605D; color: white; }}
            QTableWidget::item:selected:active {{ background-color: #3D605D; color: white; }}
            QTableWidget::item:selected:!active {{ background-color: #3D605D; color: white; }}
            
            QHeaderView::section {{ background-color: #252525; color: #aaa; border: none; padding: 5px; font-weight: bold; font-size: 12px; text-transform: uppercase; }}
            
            QProgressBar {{ background: #252525; border: none; }}
            QProgressBar::chunk {{ background: #58A39C; }}
            
            /* Exact Mac-Friendly Scrollbars from Faux Stereo */
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: #555555;
                min-height: 30px;
                border-radius: 3px; 
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{ background-color: #666666; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
            
            QScrollBar:horizontal {{
                border: none;
                background: transparent;
                height: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: #555555;
                min-width: 30px;
                border-radius: 3px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal:hover {{ background-color: #666666; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
        """)

def load_custom_fonts():
    loaded_family = "Segoe UI"
    if sys.platform == "darwin": loaded_family = "SF Pro Text"
    found_family = None
    for font_file in FONT_FILES:
        path = resource_path(font_file)
        if os.path.exists(path):
            font_id = QFontDatabase.addApplicationFont(path)
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families: found_family = families[0]
    return found_family if found_family else loaded_family

if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        app.setStyle("Fusion")
        app_font_family = load_custom_fonts()
        font = QFont(app_font_family, 10)
        app.setFont(font)
        win = MainWindow(app_font_family)
        win.show()
        sys.exit(app.exec())
    except Exception as e:
        msg = traceback.format_exc()
        box = QMessageBox()
        box.setWindowTitle("Startup Error")
        box.setText("Critical error on launch:")
        box.setDetailedText(msg)
        box.exec()