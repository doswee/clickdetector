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
    QMenu, QSlider, QSplitter, QDialog, QMenuBar, QComboBox
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
FONT_FILES = ["TitilliumWeb-Regular.ttf", "TitilliumWeb-SemiBold.ttf", "TitilliumWeb-SemiBoldItalic.ttf"]

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

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

# --- FFMPEG SETUP ---
def setup_ffmpeg():
    path_bin = resource_path("ffmpeg.bin")
    path_exe = resource_path("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    ffmpeg_bin = "ffmpeg"
    if os.path.exists(path_bin): ffmpeg_bin = path_bin
    elif os.path.exists(path_exe): ffmpeg_bin = path_exe
    elif shutil.which("ffmpeg"): ffmpeg_bin = shutil.which("ffmpeg")
    return ffmpeg_bin

FFMPEG_BIN = setup_ffmpeg()

def get_clean_env():
    env = os.environ.copy()
    for var in ["DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]:
        if var in env: del env[var]
    return env

# -----------------------------
# REPAIR ENGINE
# -----------------------------
def repair_audio_logic(samples, indices, method="Cubic", gap_size=3):
    repaired = samples.copy()
    n_samples = len(repaired)
    for idx in indices:
        start = max(0, idx - gap_size)
        end = min(n_samples - 1, idx + gap_size + 1)
        if method == "Linear":
            x = [start, end]; y = [repaired[start], repaired[end]]
            repaired[start:end] = np.interp(np.arange(start, end), x, y)
        elif method == "Cubic":
            context = 6
            c_start, c_end = max(0, start-context), min(n_samples, end+context)
            x_ctx = np.concatenate([np.arange(c_start, start), np.arange(end, c_end)])
            if len(x_ctx) > 4:
                z = np.polyfit(x_ctx, repaired[x_ctx], 3)
                repaired[start:end] = np.poly1d(z)(np.arange(start, end))
        elif method == "Pro (AR)":
            order, context = 10, 40
            if start > order + context:
                x = repaired[start-context:start]
                R = np.correlate(x, x, mode='full')[len(x)-1:]
                if R[0] != 0:
                    w = R[1:order+1] / R[0]
                    p_win = repaired[start-order:start][::-1]
                    for i in range(start, end):
                        val = np.dot(w, p_win)
                        repaired[i] = val
                        p_win = np.insert(p_win[:-1], 0, val)
    return repaired

# -----------------------------
# WORKERS
# -----------------------------
class Signals(QObject):
    discovery_finished = Signal(list)
    waveform_ready = Signal(int, object, float, int)
    scan_finished = Signal(int, str, int, object)
    repair_finished = Signal(int, bool, str)

class FileDiscoveryWorker(QRunnable):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs
        self.signals = Signals()

    @Slot()
    def run(self):
        found = []
        exts = ('.wav', '.aiff', '.aif', '.mp3', '.flac', '.ogg')
        for path in self.inputs:
            if os.path.isdir(path):
                for r, _, files in os.walk(path):
                    for f in files:
                        if f.lower().endswith(exts) and not f.startswith("._"):
                            found.append(os.path.join(r, f))
            elif path.lower().endswith(exts):
                found.append(path)
        found.sort()
        self.signals.discovery_finished.emit(found)

class ClickAnalysisWorker(QRunnable):
    def __init__(self, row, path, threshold, window, gate, stop_func):
        super().__init__()
        self.row, self.path, self.threshold = row, path, threshold
        self.window, self.gate, self.stop_func = window, gate, stop_func
        self.signals = Signals()

    @Slot()
    def run(self):
        if not os.path.exists(self.path):
            self.signals.scan_finished.emit(self.row, "Missing", 0, [])
            return
        try:
            cmd = [FFMPEG_BIN, "-v", "info", "-nostats", "-i", self.path, "-f", "f32le", "-ac", "1", "-"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=get_clean_env())
            raw_data, err_data = proc.communicate(timeout=15)
            if self.stop_func(): return
            samples = np.frombuffer(raw_data, dtype=np.float32)
            sr = 44100
            match = re.search(r"Audio:.*?, (\d+) Hz", err_data.decode('utf-8', errors='ignore'))
            if match: sr = int(match.group(1))
            duration = len(samples) / float(sr)
            step = max(1, len(samples) // 4000)
            vis_wf = np.clip(samples[::step], -1.0, 1.0)
            self.signals.waveform_ready.emit(self.row, vis_wf, duration, sr)
            diffs = np.abs(np.diff(samples, prepend=samples[0]))
            weights = np.ones(int(self.window)) / int(self.window)
            local_avg = np.convolve(diffs, weights, mode='same')
            ratios = diffs / (local_avg + 1e-9)
            condition = (ratios > self.threshold) & (np.abs(samples) > self.gate)
            peaks = np.where(condition)[0].tolist()
            self.signals.scan_finished.emit(self.row, "Clean" if not peaks else "Issues Found", len(peaks), peaks)
        except: self.signals.scan_finished.emit(self.row, "Error", 0, [])

class RepairWorker(QRunnable):
    def __init__(self, row, path, indices, method, sr):
        super().__init__()
        self.row, self.path, self.indices, self.method, self.sr = row, path, indices, method, sr
        self.signals = Signals()

    @Slot()
    def run(self):
        try:
            cmd_in = [FFMPEG_BIN, "-i", self.path, "-f", "f32le", "-ac", "1", "-"]
            proc_in = subprocess.Popen(cmd_in, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw_data, _ = proc_in.communicate()
            samples = np.frombuffer(raw_data, dtype=np.float32)
            repaired = repair_audio_logic(samples, self.indices, self.method)
            base, _ = os.path.splitext(self.path)
            out_path = f"{base}_REPAIRED.wav"
            cmd_out = [FFMPEG_BIN, "-y", "-f", "f32le", "-ar", str(self.sr), "-ac", "1", "-i", "-", out_path]
            proc_out = subprocess.Popen(cmd_out, stdin=subprocess.PIPE)
            proc_out.communicate(input=repaired.tobytes())
            self.signals.repair_finished.emit(self.row, True, out_path)
        except Exception as e:
            self.signals.repair_finished.emit(self.row, False, str(e))

# -----------------------------
# UI COMPONENTS
# -----------------------------
class ClickWaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.waveform = None
        self.clicks = None
        self.duration = 0.0
        self.playhead_ms = 0
        self.hover_x = -1
        self.setStyleSheet("background-color: #202020; border-top: 1px solid #333;")

    def load_data(self, wf, clk, dur, sr=44100):
        self.waveform, self.clicks, self.duration, self.sr = wf, clk, dur, sr
        self.playhead_ms = 0
        self.update()

    def mousePressEvent(self, event):
        if not self.duration or event.button() != Qt.LeftButton: return
        pct = event.position().x() / self.width()
        self.window().seek_media(pct * self.duration * 1000)

    def mouseMoveEvent(self, event):
        self.hover_x = event.position().x()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        if self.waveform is None:
            p.setPen(QColor(100, 100, 100))
            p.drawText(self.rect(), Qt.AlignCenter, "Select a file to visualize")
            return
        w, h, mid = self.width(), self.height(), self.height()/2
        p.setPen(QColor(88, 163, 156))
        step = w / len(self.waveform)
        pts = [QPoint(int(i*step), int(mid - v*mid*0.8)) for i, v in enumerate(self.waveform)]
        if pts: p.drawPolyline(pts)
        if self.clicks:
            p.setPen(QPen(QColor(255, 107, 107, 150), 1))
            scale = w / (self.duration * self.sr)
            for c in self.clicks:
                cx = int(c * scale)
                p.drawLine(cx, 0, cx, h)
        px = (self.playhead_ms / (self.duration * 1000)) * w if self.duration > 0 else 0
        p.setPen(QPen(Qt.white, 2)); p.drawLine(int(px), 0, int(px), h)

class ModernTable(QTableWidget):
    files_dropped = Signal(list)
    selection_changed_custom = Signal()
    delete_signal = Signal()

    def __init__(self, font_family):
        super().__init__(0, 3)
        self.font_family = font_family
        self.setHorizontalHeaderLabels(["File Name", "Clicks", "Status"])
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAcceptDrops(True)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.setColumnWidth(1, 100); self.setColumnWidth(2, 120)
        self.itemSelectionChanged.connect(self.selection_changed_custom.emit)

    def dragEnterEvent(self, e): e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dragMoveEvent(self, e): e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e): self.files_dropped.emit([u.toLocalFile() for u in e.mimeData().urls()])

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace): self.delete_signal.emit()
        else: super().keyPressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.rowCount() == 0:
            p = QPainter(self.viewport())
            f = QFont(self.font_family, 24, QFont.DemiBold, True); p.setFont(f)
            p.setPen(QColor(80, 80, 80)); p.drawText(self.viewport().rect(), Qt.AlignCenter, "DRAG & DROP FILES HERE")

# -----------------------------
# MAIN WINDOW
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self, font_family):
        super().__init__()
        self.font_family = font_family
        self.setWindowTitle("Rogue Waves CLICK DETECTOR")
        self.resize(1100, 850)
        
        self.threadpool = QThreadPool()
        self.files, self.file_data = [], {}
        self.is_scanning, self.stop_flag, self.has_scanned = False, False, False

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.on_pos_changed)
        self.player.playbackStateChanged.connect(self.on_play_state_changed)
        
        self.current_playing_row = -1
        self.last_active_row = -1

        self.setup_ui()
        self.setup_dark_theme()

    def setup_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget); main_layout.setContentsMargins(0,0,0,0); main_layout.setSpacing(0)

        # SIDEBAR
        self.sidebar = QFrame(); self.sidebar.setFixedWidth(300); self.sidebar.setObjectName("Sidebar")
        sb = QVBoxLayout(self.sidebar); sb.setContentsMargins(15, 20, 15, 20); sb.setSpacing(10)

        # Logo
        self.lbl_logo = QLabel()
        self.lbl_logo.setAlignment(Qt.AlignCenter)
        title_size = self.get_optimal_font_size("ROGUE WAVES", 38, 270)
        self.lbl_logo.setText(f'<div align="center" style="line-height:0.8;"><span style="font-family:\'{self.font_family}\'; font-size:{title_size}px; font-weight:600; font-style:italic; color:#5ba49d;">ROGUE WAVES</span><br><span style="font-family:\'{self.font_family}\'; font-size:20px; font-weight:600; font-style:italic; color:#808080;">CLICK DETECTOR</span></div>')
        sb.addWidget(self.lbl_logo); sb.addSpacing(10)

        sb.addWidget(self.header_lbl("INPUT"))
        btn_sel = QPushButton("Select Files / Folder"); btn_sel.clicked.connect(self.select_input); sb.addWidget(btn_sel)

        sb.addWidget(self.header_lbl("SCAN SETTINGS"))
        set_f = QFrame(); set_f.setObjectName("StatsFrame"); set_l = QVBoxLayout(set_f)
        self.slider_thresh = self.add_slider(set_l, "THRESHOLD (Ratio)", 20, 200, 50, "ratio")
        self.slider_win = self.add_slider(set_l, "CONTEXT WINDOW", 2, 200, 10, "samples")
        self.slider_gate = self.add_slider(set_l, "NOISE GATE", 0, 200, 10, "noise")
        btn_defaults = QPushButton("RESTORE DEFAULTS"); btn_defaults.setStyleSheet("font-size: 10px; color:#888; border: 1px solid #444;"); btn_defaults.clicked.connect(self.restore_defaults)
        set_l.addWidget(btn_defaults); sb.addWidget(set_f)

        sb.addWidget(self.header_lbl("REPAIR"))
        rep_f = QFrame(); rep_f.setObjectName("StatsFrame"); rep_l = QVBoxLayout(rep_f)
        self.combo_method = QComboBox(); self.combo_method.addItems(["Fast (Linear)", "Smooth (Cubic)", "Pro (AR)"])
        rep_l.addWidget(QLabel("ALGORITHM")); rep_l.addWidget(self.combo_method)
        self.btn_repair = QPushButton("REPAIR SELECTED"); self.btn_repair.setObjectName("RepairBtn"); self.btn_repair.clicked.connect(self.run_repair); rep_l.addWidget(self.btn_repair)
        sb.addWidget(rep_f)

        sb.addWidget(self.header_lbl("ANALYSIS"))
        ana_f = QFrame(); ana_f.setObjectName("StatsFrame"); ana_l = QVBoxLayout(ana_f)
        self.lbl_total_files = QLabel("Files: 0"); self.lbl_issues = QLabel("Issues Found: 0")
        self.lbl_issues.setStyleSheet("color:#FF6B6B; font-weight:bold;"); ana_l.addWidget(self.lbl_total_files); ana_l.addWidget(self.lbl_issues); sb.addWidget(ana_f)

        sb.addStretch()
        self.progress = QProgressBar(); self.progress.setFixedHeight(6); self.progress.setVisible(False); sb.addWidget(self.progress)
        self.lbl_status = QLabel("Ready"); self.lbl_status.setAlignment(Qt.AlignCenter); self.lbl_status.setStyleSheet("color:#888; font-size:11px;"); sb.addWidget(self.lbl_status)
        self.btn_scan = QPushButton("SCAN"); self.btn_scan.setObjectName("ActionBtn"); self.btn_scan.clicked.connect(self.toggle_scan); sb.addWidget(self.btn_scan)

        main_layout.addWidget(self.sidebar)

        # RIGHT
        splitter = QSplitter(Qt.Vertical)
        self.table = ModernTable(self.font_family)
        self.table.files_dropped.connect(self.load_files); self.table.delete_signal.connect(self.delete_selected)
        self.table.selection_changed_custom.connect(self.on_table_select); self.table.doubleClicked.connect(lambda idx: self.start_playback(idx.row()))
        self.waveform = ClickWaveformWidget()
        splitter.addWidget(self.table); splitter.addWidget(self.waveform); splitter.setSizes([600, 200]); main_layout.addWidget(splitter)

    def header_lbl(self, txt):
        l = QLabel(txt); l.setObjectName("SectionHeader"); return l

    def add_slider(self, lay, txt, min_v, max_v, def_v, mode):
        h = QHBoxLayout(); lbl = QLabel(txt); lbl.setStyleSheet("color:#888; font-weight:bold; font-size:11px;")
        val_lbl = QLabel(); val_lbl.setStyleSheet("color:#58A39C; font-weight:bold;"); h.addWidget(lbl); h.addStretch(); h.addWidget(val_lbl); lay.addLayout(h)
        s = QSlider(Qt.Horizontal); s.setRange(min_v, max_v); s.setValue(def_v); lay.addWidget(s)
        def up():
            if mode == "ratio": val_lbl.setText(f"{s.value()/10:.2f}")
            elif mode == "samples": val_lbl.setText(f"{s.value()} samples")
            else: val_lbl.setText(f"{s.value()/1000:.3f}")
        s.valueChanged.connect(up); up(); return s

    def get_optimal_font_size(self, text, max_size, width):
        font = QFont(self.font_family, max_size, QFont.Bold)
        while max_size > 10 and QFontMetrics(font).horizontalAdvance(text) > width:
            max_size -= 1; font.setPixelSize(max_size)
        return max_size

    def restore_defaults(self):
        self.slider_thresh.setValue(50); self.slider_win.setValue(10); self.slider_gate.setValue(10)

    def select_input(self):
        f = QFileDialog.getOpenFileNames(self, "Select Audio", "", "Audio (*.wav *.mp3 *.aif)")[0]
        if f: self.load_files(f)

    def delete_selected(self):
        rows = sorted([x.row() for x in self.table.selectionModel().selectedRows()], reverse=True)
        for r in rows:
            if r == self.current_playing_row: self.player.stop()
            self.table.removeRow(r); del self.files[r]; del self.file_data[r]
        self.lbl_total_files.setText(f"Files: {len(self.files)}"); self.update_stats()

    def load_files(self, paths):
        self.files, self.file_data = [], {}; self.table.setRowCount(0); self.has_scanned = False; self.btn_scan.setText("SCAN")
        worker = FileDiscoveryWorker(paths); worker.signals.discovery_finished.connect(self.populate_table); self.threadpool.start(worker)

    def populate_table(self, files):
        self.files = files; self.table.setRowCount(len(files))
        for i, f in enumerate(files):
            self.table.setItem(i, 0, QTableWidgetItem(os.path.basename(f)))
            self.table.setItem(i, 1, QTableWidgetItem("-"))
            self.table.setItem(i, 2, QTableWidgetItem("Ready"))
            self.file_data[i] = {'wf': None, 'clk': [], 'dur': 0, 'sr': 44100}
        self.lbl_total_files.setText(f"Files: {len(files)}"); self.update_stats()
        if files: self.table.selectRow(0)

    def toggle_scan(self):
        if self.is_scanning: self.stop_flag = True
        else: self.start_scan_process()

    def start_scan_process(self):
        if not self.files: return
        self.is_scanning, self.stop_flag = True, False
        self.btn_scan.setText("STOP"); self.btn_scan.setObjectName("StopBtn"); self.btn_scan.setStyle(self.btn_scan.style())
        self.progress.setVisible(True); self.progress.setRange(0, len(self.files)); self.progress.setValue(0)
        t, w, g = self.slider_thresh.value()/10.0, self.slider_win.value(), self.slider_gate.value()/1000.0
        for i in range(len(self.files)):
            path = self.files[i]
            rep_path = path.replace(".wav", "_REPAIRED.wav").replace(".mp3", "_REPAIRED.wav").replace(".aif", "_REPAIRED.wav")
            if os.path.exists(rep_path) and self.has_scanned: path = rep_path; self.files[i] = path
            worker = ClickAnalysisWorker(i, path, t, w, g, lambda: self.stop_flag)
            worker.signals.waveform_ready.connect(self.on_waveform_ready)
            worker.signals.scan_finished.connect(self.on_scan_finished); self.threadpool.start(worker)

    def on_waveform_ready(self, row, wf, dur, sr):
        self.file_data[row].update({'wf': wf, 'dur': dur, 'sr': sr})
        if row == self.table.currentRow(): self.update_visualizer()

    def on_scan_finished(self, row, status, count, peaks):
        self.file_data[row]['clk'] = peaks
        self.table.setItem(row, 1, QTableWidgetItem(str(count)))
        it = QTableWidgetItem("Scanned"); it.setForeground(QColor("#58A39C" if status == "Clean" else "#FF6B6B"))
        self.table.setItem(row, 2, it); self.progress.setValue(self.progress.value() + 1)
        if self.progress.value() == len(self.files): 
            self.finish_scan_ui()
        self.update_stats()

    def finish_scan_ui(self):
        self.is_scanning, self.has_scanned = False, True
        self.btn_scan.setText("RE-SCAN"); self.btn_scan.setObjectName("ActionBtn"); self.btn_scan.setStyle(self.btn_scan.style())
        self.progress.setVisible(False)

    def run_repair(self):
        rows = [idx.row() for idx in self.table.selectionModel().selectedRows()]
        if not rows: return
        method = self.combo_method.currentText().split("(")[1].replace(")", "")
        self.lbl_status.setText("Repairing...")
        for r in rows:
            d = self.file_data[r]
            if not d['clk']: continue
            worker = RepairWorker(r, self.files[r], d['clk'], method, d['sr'])
            worker.signals.repair_finished.connect(self.on_repair_done); self.threadpool.start(worker)

    def on_repair_done(self, row, success, path):
        if success: self.table.setItem(row, 2, QTableWidgetItem("Repaired ✓")); self.lbl_status.setText(f"Saved: {os.path.basename(path)}")

    def update_stats(self):
        issues = sum(1 for i in range(self.table.rowCount()) if self.table.item(i, 1) and self.table.item(i, 1).text() not in ["-", "0"])
        self.lbl_issues.setText(f"Issues Found: {issues}")

    def on_table_select(self):
        self.update_visualizer()

    def update_visualizer(self):
        row = self.table.currentRow()
        if row in self.file_data:
            d = self.file_data[row]
            self.waveform.load_data(d['wf'], d['clk'], d['dur'], d['sr'])

    def start_playback(self, row):
        if row >= len(self.files): return
        if self.current_playing_row != -1: self.set_row_visuals(self.current_playing_row, False)
        self.current_playing_row = row
        self.player.setSource(QUrl.fromLocalFile(self.files[row])); self.player.play()

    def set_row_visuals(self, row, playing):
        if row >= self.table.rowCount(): return
        it = self.table.item(row, 0); txt = it.text().replace("▶ ", "")
        it.setText(f"▶ {txt}" if playing else txt)
        it.setForeground(QColor("#58A39C" if playing else "#E0E0E0"))

    def seek_media(self, ms):
        if self.current_playing_row == -1 and self.table.currentRow() != -1:
            self.start_playback(self.table.currentRow()); self.player.pause()
        self.player.setPosition(int(ms))

    def on_pos_changed(self, ms):
        self.waveform.playhead_ms = ms; self.waveform.update()

    def on_play_state_changed(self, state):
        if self.current_playing_row != -1:
            self.set_row_visuals(self.current_playing_row, state == QMediaPlayer.PlayingState)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.player.playbackState() == QMediaPlayer.PlayingState: self.player.pause()
            else: self.toggle_playback()
        else: super().keyPressEvent(event)

    def toggle_playback(self):
        row = self.table.currentRow() if self.current_playing_row == -1 else self.current_playing_row
        if row != -1:
            if self.current_playing_row != row: self.start_playback(row)
            else: self.player.play()

    def setup_dark_theme(self):
        pal = QPalette(); pal.setColor(QPalette.Window, QColor(30, 30, 30))
        pal.setColor(QPalette.Highlight, QColor(61, 96, 93)); self.setPalette(pal)
        self.setStyleSheet(f"""
            QWidget {{ font-family: '{self.font_family}'; font-size: 13px; color: #DDD; }}
            QFrame#Sidebar {{ background-color: #252525; border-right: 1px solid #333; }}
            QLabel#SectionHeader {{ color: #DDD; font-weight: 600; font-style: italic; font-size: 18px; margin-top: 5px; }}
            QFrame#StatsFrame {{ background-color: #2A2A2A; border-radius: 4px; padding: 10px; }}
            QPushButton {{ background-color: #333; border: 1px solid #444; border-radius: 4px; padding: 6px; color: #DDD; }}
            QPushButton:hover {{ background-color: #3E3E3E; }}
            QPushButton#ActionBtn {{ background-color: #58A39C; color: white; font-weight: bold; border: none; font-size: 14px; font-style: italic; }}
            QPushButton#RepairBtn {{ background-color: #3D605D; color: white; border: none; font-weight: bold; margin-top:5px; }}
            QPushButton#StopBtn {{ background-color: #FF6B6B; color: white; border: none; font-size: 14px; font-style: italic; }}
            QTableWidget {{ background-color: #181818; alternate-background-color: #222; border: none; selection-background-color: #3D605D; outline: 0; }}
            QHeaderView::section {{ background-color: #252525; border: none; padding: 5px; color: #888; font-weight: bold; text-transform: uppercase; }}
            QProgressBar::chunk {{ background-color: #58A39C; border-radius: 3px; }}
            QComboBox {{ background: #333; border: 1px solid #444; padding: 3px; color: #DDD; }}
            QScrollBar:vertical {{ border: none; background: transparent; width: 10px; }}
            QScrollBar::handle:vertical {{ background-color: #555; min-height: 30px; border-radius: 5px; margin: 2px; }}
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    win = MainWindow(load_custom_fonts()); win.show(); sys.exit(app.exec())