import os
import sys
import shutil
import subprocess
import math
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from scipy.stats import median_abs_deviation

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox,
    QHeaderView, QLabel, QFrame, QAbstractItemView, QProgressBar, 
    QMenu, QSlider, QSplitter
)
from PySide6.QtCore import (
    Qt, QRunnable, Slot, Signal, QObject, QThreadPool, 
    QUrl, QPoint
)
from PySide6.QtGui import (
    QFont, QKeyEvent, QColor, QPainter, QPen, QFontDatabase, QFontMetrics, QCursor
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# --- FONTS SETUP ---
# Ensure these .ttf files are in the same folder or bundled
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
    if shutil.which("ffmpeg"):
        return "ffmpeg", "ffprobe"
    local_ffmpeg = resource_path("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    local_ffprobe = resource_path("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
    return (local_ffmpeg, local_ffprobe) if os.path.exists(local_ffmpeg) else ("ffmpeg", "ffprobe")

FFMPEG_BIN, FFPROBE_BIN = setup_ffmpeg()

def get_clean_env():
    env = os.environ.copy()
    for var in ["DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]:
        if var in env: del env[var]
    return env

# -----------------------------
# SIGNAL PROCESSING (DISCONTINUITY / 2nd DERIVATIVE)
# -----------------------------
class Signals(QObject):
    discovery_finished = Signal(list)
    waveform_ready = Signal(int, object, float) 
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
    def __init__(self, row, path, sensitivity, stop_func):
        super().__init__()
        self.row = row
        self.path = path
        self.sensitivity = sensitivity 
        self.stop_func = stop_func
        self.signals = Signals()

    @Slot()
    def run(self):
        if not os.path.exists(self.path):
            self.signals.scan_finished.emit(self.row, "Missing", 0, [])
            return

        if self.stop_func(): return

        try:
            # 1. Decode to Raw PCM
            cmd = [
                FFMPEG_BIN, "-v", "error", "-i", self.path, 
                "-f", "f32le", "-ac", "1", "-ar", "44100", "-"
            ]
            
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, 
                startupinfo=startupinfo, env=get_clean_env()
            )
            
            raw_data, _ = process.communicate()
            
            if self.stop_func(): return

            samples = np.frombuffer(raw_data, dtype=np.float32)
            
            if len(samples) < 100:
                self.signals.scan_finished.emit(self.row, "Too Short", 0, [])
                return

            duration = len(samples) / 44100.0

            # 2. IMMEDIATE VISUALIZATION
            target_width = 4000
            step = max(1, len(samples) // target_width)
            vis_waveform = samples[::step].copy()
            vis_waveform = np.clip(vis_waveform, -1.0, 1.0)
            self.signals.waveform_ready.emit(self.row, vis_waveform, duration)

            # 3. ANALYSIS (2nd Derivative for Discontinuities)
            if self.stop_func(): return

            # Calculate Acceleration (2nd Derivative)
            accel = np.abs(np.diff(samples, n=2, prepend=[0, 0]))
            
            # Square it to punish outliers (Slope Limiting)
            curvature = accel ** 2

            # Map Sensitivity (1-100)
            # High Sens (100) -> 0.05 threshold
            # Low Sens (1)    -> 5.0 threshold
            base_threshold = 0.05 
            sensitivity_factor = (101 - self.sensitivity) / 20.0 
            final_threshold = base_threshold * sensitivity_factor

            # Find Peaks
            peaks, _ = find_peaks(
                curvature, 
                height=final_threshold, 
                distance=441 # 10ms lockout
            )

            final_count = len(peaks)
            status = "Clean"
            if final_count > 0: status = "Issues Found"
            if final_count > 50: status = "Heavy Artifacts"

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
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self.waveform = None      
        self.clicks = None        
        self.duration = 0.0       
        self.file_path = ""
        self.playhead_ms = 0
        self.is_playing = False
        self.setCursor(Qt.PointingHandCursor)
        self.hover_x = -1
        self.setStyleSheet("background-color: #202020; border-top: 1px solid #333;")

    def load_data(self, waveform, clicks, duration, path):
        self.waveform = waveform
        self.clicks = clicks
        self.duration = duration
        self.file_path = path
        self.playhead_ms = 0
        self.update()

    def clear(self):
        self.waveform = None
        self.clicks = None
        self.file_path = ""
        self.update()

    def set_playhead(self, ms):
        self.playhead_ms = ms
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
        p.fillRect(self.rect(), QColor(32, 32, 32)) # Dark background

        if self.waveform is None:
            p.setPen(QColor(100, 100, 100))
            p.drawText(self.rect(), Qt.AlignCenter, "Select a file to visualize")
            return

        w = self.width()
        h = self.height()
        mid_y = h / 2
        
        # Draw Waveform (Teal)
        p.setPen(QColor(88, 163, 156)) 
        path_points = []
        if len(self.waveform) > 0:
            x_step = w / len(self.waveform)
            for i, val in enumerate(self.waveform):
                px = i * x_step
                py = mid_y - (val * (mid_y * 0.9)) 
                path_points.append(QPoint(int(px), int(py)))
            p.drawPolyline(path_points)

        # Draw Clicks (Red Lines)
        if self.clicks is not None and len(self.clicks) > 0:
            p.setPen(QPen(QColor(255, 107, 107, 200), 1)) 
            total_samples = self.duration * 44100 
            scale = w / total_samples
            for c_idx in self.clicks:
                cx = c_idx * scale
                p.drawLine(int(cx), 0, int(cx), h)

        # Playhead (White)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        play_x = (self.playhead_ms / (self.duration * 1000)) * w if self.duration > 0 else 0
        p.drawLine(int(play_x), 0, int(play_x), h)
        
        # Hover Time
        if self.hover_x >= 0 and self.duration > 0:
            pct = self.hover_x / w
            sec = pct * self.duration
            time_str = f"{int(sec // 60)}:{int(sec % 60):02d}"
            p.setPen(QColor(252, 163, 17)) # Orange
            p.drawText(int(self.hover_x) + 5, h - 5, time_str)

# -----------------------------
# MAIN TABLE & WINDOW
# -----------------------------
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

class MainWindow(QMainWindow):
    def __init__(self, font_family):
        super().__init__()
        self.font_family = font_family
        self.setWindowTitle("Rogue Waves CLICK DETECTOR")
        self.resize(1100, 800)
        self.setAcceptDrops(True)
        self.setup_dark_theme()
        
        self.threadpool = QThreadPool()
        self.files = [] 
        self.file_data = {} 
        self.stop_flag = False
        self.is_scanning = False
        
        # Audio Player
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.mediaStatusChanged.connect(self.on_media_status)
        self.current_playing_row = -1

        # UI LAYOUT
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SIDEBAR ---
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(320)
        sb_layout = QVBoxLayout(self.sidebar)
        sb_layout.setContentsMargins(20, 30, 20, 20)
        sb_layout.setSpacing(15)

        # Logo (Using HTML style from Loudness App)
        title_size = self.get_optimal_font_size("ROGUE WAVES", 32, 270)
        self.lbl_logo = QLabel()
        self.lbl_logo.setAlignment(Qt.AlignCenter)
        self.lbl_logo.setOpenExternalLinks(True)
        self.lbl_logo.setCursor(Qt.PointingHandCursor)
        self.lbl_logo.setText(f"""<html><body><p align="center" style="margin:0; line-height:1.0;"><a href="http://roguewaveslibrary.com" style="text-decoration:none;"><span style="font-family:'{self.font_family}'; font-size:{title_size}px; font-weight:700; letter-spacing: 1px; color:#ffffff;">ROGUE WAVES</span><br/><span style="font-family:'{self.font_family}'; font-size:12px; font-weight:400; letter-spacing: 3px; color:#58A39C;">CLICK DETECTOR</span></a></p></body></html>""")
        sb_layout.addWidget(self.lbl_logo)
        sb_layout.addSpacing(20)

        # Inputs
        sb_layout.addWidget(self.create_header_label("INPUT SOURCE"))
        self.btn_select = self.create_button("Select Files / Folder", self.select_input)
        sb_layout.addWidget(self.btn_select)

        self.lbl_stats = QLabel("Files: 0")
        self.lbl_stats.setStyleSheet("color: #666; font-size: 11px; font-weight: bold; margin-top: 5px;")
        self.lbl_stats.setAlignment(Qt.AlignCenter)
        sb_layout.addWidget(self.lbl_stats)

        sb_layout.addSpacing(10)
        
        # Sensitivity
        sb_layout.addWidget(self.create_header_label("ANALYSIS SETTINGS"))
        
        # Frame for controls
        self.analysis_container = QFrame()
        self.analysis_container.setObjectName("StatsFrame")
        ac_layout = QVBoxLayout(self.analysis_container)
        ac_layout.setContentsMargins(15, 15, 15, 15)
        
        lbl_sens = QLabel("SENSITIVITY")
        lbl_sens.setStyleSheet("color: #888; font-weight: bold; font-size: 11px;")
        ac_layout.addWidget(lbl_sens)
        
        self.slider_sens = QSlider(Qt.Horizontal)
        self.slider_sens.setRange(1, 100)
        self.slider_sens.setValue(60) 
        self.slider_sens.valueChanged.connect(self.update_sens_label)
        ac_layout.addWidget(self.slider_sens)
        
        self.lbl_sens_val = QLabel("Medium (60)")
        self.lbl_sens_val.setStyleSheet("color: #DDD; font-size: 11px; font-weight: bold;")
        self.lbl_sens_val.setAlignment(Qt.AlignCenter)
        ac_layout.addWidget(self.lbl_sens_val)
        
        sb_layout.addWidget(self.analysis_container)

        sb_layout.addStretch()

        # Scan Button
        self.btn_scan = QPushButton("SCAN")
        self.btn_scan.setObjectName("ActionBtn")
        self.btn_scan.setCursor(Qt.PointingHandCursor)
        self.btn_scan.clicked.connect(self.force_rescan)
        self.btn_scan.setFixedHeight(32) # Matching button height from Loudness App
        sb_layout.addWidget(self.btn_scan)

        # Progress
        self.progress = QProgressBar()
        self.progress.setFixedHeight(2) # Thin line like Loudness App
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        sb_layout.addWidget(self.progress)
        
        main_layout.addWidget(self.sidebar)

        # --- RIGHT SIDE ---
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(1)
        right_splitter.setStyleSheet("QSplitter::handle { background: #333; }")
        
        # Table
        self.table = ModernTable(self.font_family)
        self.table.files_dropped.connect(self.load_files)
        self.table.delete_signal.connect(self.delete_selected)
        self.table.selection_changed_custom.connect(self.on_table_selection)
        self.table.doubleClicked.connect(self.on_table_double_click)
        right_splitter.addWidget(self.table)

        # Visualizer Container
        self.viz_container = QWidget()
        viz_layout = QVBoxLayout(self.viz_container)
        viz_layout.setContentsMargins(0,0,0,0)
        viz_layout.setSpacing(0)
        
        self.viz_header = QLabel("WAVEFORM VISUALIZATION")
        self.viz_header.setStyleSheet("background: #252525; color: #888; font-size: 11px; padding: 5px; font-weight: bold; border-top: 1px solid #333;")
        viz_layout.addWidget(self.viz_header)
        
        self.waveform = ClickWaveformWidget()
        viz_layout.addWidget(self.waveform)
        
        right_splitter.addWidget(self.viz_container)
        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(right_splitter)
        
        self.grabKeyboard()

    # --- UI HELPERS ---
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

    # --- LOGIC ---
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_playback()
        else:
            super().keyPressEvent(event)

    def update_sens_label(self, val):
        txt = "Low"
        if val > 30: txt = "Medium" 
        if val > 70: txt = "High"
        self.lbl_sens_val.setText(f"{txt} ({val})")

    def select_input(self):
        f = QFileDialog.getOpenFileNames(self, "Select Audio", "", "Audio (*.wav *.aif *.mp3)")[0]
        if f: self.load_files(f)
    
    def dropEvent(self, event):
        self.load_files([url.toLocalFile() for url in event.mimeData().urls()])

    def load_files(self, paths):
        self.stop_playback()
        self.player.setSource(QUrl()) # Release locks

        self.file_data = {}
        self.files = []
        self.table.setRowCount(0)
        self.stop_flag = False
        self.btn_select.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        
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
            self.file_data[i] = {'waveform': None, 'clicks': None, 'duration': 0}

        self.lbl_stats.setText(f"Files: {len(files)}")
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.btn_select.setEnabled(True)
        
        # AUTO SELECT FIRST ROW
        if len(files) > 0:
            self.table.selectRow(0)

        # AUTO START SCAN
        self.start_batch_scan(list(range(len(files))))

    def force_rescan(self):
        if self.is_scanning:
            self.stop_flag = True
            return
            
        self.stop_flag = False
        self.progress.setValue(0)
        self.start_batch_scan(list(range(len(self.files))))

    def start_batch_scan(self, rows):
        self.is_scanning = True
        self.btn_scan.setText("STOP")
        self.btn_scan.setObjectName("StopBtn")
        self.btn_scan.setStyle(self.btn_scan.style())

        sens = self.slider_sens.value()

        for i in rows:
            self.table.setItem(i, 1, QTableWidgetItem("..."))
            self.table.setItem(i, 2, QTableWidgetItem("Decoding..."))
            
            worker = ClickAnalysisWorker(i, self.files[i], sens, lambda: self.stop_flag)
            worker.signals.waveform_ready.connect(self.on_waveform_ready)
            worker.signals.scan_finished.connect(self.on_scan_finished)
            self.threadpool.start(worker)

    def on_waveform_ready(self, row, waveform, duration):
        if self.stop_flag: return
        
        if row in self.file_data:
            self.file_data[row]['waveform'] = waveform
            self.file_data[row]['duration'] = duration
        
        self.table.setItem(row, 2, QTableWidgetItem("Analyzing..."))

        cur_rows = self.table.selectionModel().selectedRows()
        if cur_rows and cur_rows[0].row() == row:
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
        
        cur_rows = self.table.selectionModel().selectedRows()
        if cur_rows and cur_rows[0].row() == row:
            self.update_visualizer(row)
        
        if self.threadpool.activeThreadCount() == 0:
            self.is_scanning = False
            self.btn_scan.setText("SCAN")
            self.btn_scan.setObjectName("ActionBtn")
            self.btn_scan.setStyle(self.btn_scan.style())
            self.progress.setVisible(False)

    def delete_selected(self):
        rows = sorted([x.row() for x in self.table.selectionModel().selectedRows()], reverse=True)
        if self.current_playing_row in rows:
            self.stop_playback()
            self.player.setSource(QUrl())
            
        for r in rows:
            self.table.removeRow(r)
            del self.files[r]
            if r in self.file_data: del self.file_data[r]
        
        self.lbl_stats.setText(f"Files: {len(self.files)}")

    # --- VISUALIZATION & PLAYBACK ---
    def on_table_selection(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.waveform.clear()
            return
        
        row = rows[0].row()
        self.update_visualizer(row)

    def update_visualizer(self, row):
        if row in self.file_data:
            data = self.file_data[row]
            wf = data['waveform']
            dur = data['duration']
            clk = data['clicks']
            path = self.files[row]
            
            self.waveform.load_data(wf, clk, dur, path)
            self.viz_header.setText(f"VISUALIZATION: {os.path.basename(path)} ({dur:.1f}s)")
        else:
            self.waveform.clear()
            self.viz_header.setText("VISUALIZATION")

    def on_table_double_click(self, index):
        self.start_playback(index.row())

    def start_playback(self, row):
        if row >= len(self.files): return
        
        # STOP existing playback first to flush the buffer
        self.player.stop() 
        self.player.setSource(QUrl()) # Unload to ensure fresh load
        
        path = self.files[row]
        
        if self.current_playing_row != -1:
            self.set_row_visuals(self.current_playing_row, False)

        self.current_playing_row = row
        self.set_row_visuals(row, True)
        
        # Load the new source
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()
        
        self.waveform.is_playing = True
        
        # Ensure the row is visually selected
        self.table.selectRow(row)

    def set_row_visuals(self, row, playing):
        if row >= self.table.rowCount(): return
        item = self.table.item(row, 0)
        f = item.font(); f.setBold(playing); item.setFont(f)
        txt = item.text()
        if playing and not txt.startswith("▶ "): 
            item.setText(f"▶ {txt}")
            item.setForeground(QColor("#58A39C"))
        elif not playing and txt.startswith("▶ "): 
            item.setText(txt.replace("▶ ", ""))
            item.setForeground(QColor("#E0E0E0"))

    def toggle_playback(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        selected_row = rows[0].row()

        if selected_row != self.current_playing_row:
            self.start_playback(selected_row)
            return

        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def seek_media(self, ms):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        selected_row = rows[0].row()

        if selected_row != self.current_playing_row:
            self.start_playback(selected_row)
        
        self.player.setPosition(int(ms))
        if self.player.playbackState() != QMediaPlayer.PlayingState:
            self.player.play()

    def on_position_changed(self, ms):
        self.waveform.set_playhead(ms)

    def on_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.waveform.set_playhead(0)
            if self.current_playing_row != -1:
                self.set_row_visuals(self.current_playing_row, False)
                self.current_playing_row = -1

    # --- THEME (Copied from Loudness App) ---
    def setup_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.Highlight, QColor(88, 163, 156))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        self.setStyleSheet(f"""
            QWidget {{ font-family: '{self.font_family}', 'Segoe UI', 'SF Pro Text', sans-serif; font-size: 13px; }}
            QMainWindow {{ background-color: #1E1E1E; }}
            QFrame#Sidebar {{ background-color: #252525; border-right: 1px solid #333; }}
            QFrame#Sidebar QLabel {{ color: #bbb; }}
            
            QFrame#Sidebar QLabel#SectionHeader {{ 
                color: #58A39C; 
                font-weight: 700; 
                font-size: 11px; 
                letter-spacing: 1px; 
                text-transform: uppercase;
                margin-top: 10px; margin-bottom: 2px;
            }}
            
            QFrame#StatsFrame {{ background-color: #2A2A2A; border-radius: 4px; border: 1px solid #383838; }}
            
            QSlider::groove:horizontal {{ border: 1px solid #333; height: 4px; background: #202020; margin: 2px 0; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: #58A39C; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }}
            
            QPushButton {{ 
                background-color: #363636; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 6px 12px; 
                color: #eee; 
                font-weight: 500;
                min-height: 18px;
            }}
            QPushButton:hover {{ background-color: #404040; border-color: #666; }}
            QPushButton:pressed {{ background-color: #2a2a2a; }}
            QPushButton:disabled {{ background-color: #2a2a2a; color: #555; border-color: #333; }}
            
            /* Action Buttons - Now uniform with other buttons but highlighted border */
            QPushButton#ActionBtn {{ 
                background-color: #363636; 
                border: 1px solid #58A39C; 
                color: #58A39C;
                font-weight: 700;
            }}
            QPushButton#ActionBtn:hover {{ background-color: #58A39C; color: white; }}
            QPushButton#ActionBtn:disabled {{ border-color: #444; color: #555; background-color: #2a2a2a; }}
            
            /* Stop Button State */
            QPushButton#StopBtn {{ 
                background-color: #363636;
                border: 1px solid #FF6B6B;
                color: #FF6B6B;
                font-weight: 700;
            }}
            QPushButton#StopBtn:hover {{ background-color: #FF6B6B; color: white; }}
            
            QTableWidget {{ background-color: #1E1E1E; alternate-background-color: #252525; color: #E0E0E0; border: none; gridline-color: #333; }}
            QTableWidget::item:selected {{ background-color: #3D605D; color: white; }}
            QTableWidget::item {{ padding: 2px 5px; border: none; }}
            
            QHeaderView {{ background-color: #252525; border-bottom: 1px solid #333; }}
            QHeaderView::section {{ 
                background-color: #252525; 
                color: #888; 
                border: none; 
                padding: 6px; 
                font-weight: 600; 
                font-size: 11px; 
                text-transform: uppercase; 
            }}
            
            QProgressBar {{ 
                background: #252525; 
                border: none; 
                border-radius: 0px; 
            }}
            QProgressBar::chunk {{ background: #58A39C; }}
            
            QToolTip {{ color: #fff; background-color: #333; border: 1px solid #555; }}
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
    app.setStyle("Fusion")
    app_font_family = load_custom_fonts()
    font = QFont(app_font_family, 10)
    font.setWeight(QFont.Normal)
    app.setFont(font)
    win = MainWindow(app_font_family)
    win.show()
    sys.exit(app.exec())