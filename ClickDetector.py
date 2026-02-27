import os
import sys

# --- CRITICAL FIX FOR BUS ERROR 10 & SEGFAULT 11 ---
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

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
    QSlider, QSplitter, QComboBox
)
from PySide6.QtCore import (
    Qt, QRunnable, Slot, Signal, QObject, QThreadPool, 
    QUrl, QPoint
)
from PySide6.QtGui import (
    QFont, QKeyEvent, QColor, QPainter, QPen, QFontDatabase, 
    QFontMetrics, QCursor, QPalette
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# --- FONTS SETUP ---
FONT_FILES =["TitilliumWeb-Regular.ttf", "TitilliumWeb-SemiBold.ttf", "TitilliumWeb-SemiBoldItalic.ttf"]

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
    for var in["DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]:
        if var in env: del env[var]
    return env

# -----------------------------
# REPAIR ENGINE (PRO BIDIRECTIONAL AR)
# -----------------------------
def repair_audio_logic(samples, indices, intensity=5):
    if not indices: return samples.copy()
    repaired = np.array(samples, copy=True, dtype=np.float32)
    n_samples = len(repaired)
    
    # Linear scaling for predictable results
    # look_around: 3 samples @ Int 1 -> 40 samples @ Int 10
    look_around = int(2 + (intensity * 3.8)) 
    # order: 15 @ Int 1 -> 180 @ Int 10 (Higher = more ringing/smearing)
    order = int(12 + (intensity * 17))
    context = int(300 + (intensity * 120))

    # Cluster regions
    regions = []
    current_region = [indices[0], indices[0]]
    for idx in indices[1:]:
        if idx - current_region[1] <= look_around * 2:
            current_region[1] = idx
        else:
            regions.append(current_region); current_region = [idx, idx]
    regions.append(current_region)

    def solve_ar_coeffs(data, p):
        if len(data) <= p: p = len(data) // 2
        x = data - np.mean(data)
        N = len(x); num_obs = N - p
        if num_obs <= 0: return None
        X = np.zeros((num_obs, p))
        for i in range(num_obs): X[i, :] = x[i : i + p][::-1]
        y = x[p:]
        try:
            # Stable Tikhonov regularization (prevents pops/explosions)
            reg = np.eye(p) * (np.std(x) * 0.01)
            coeffs, _, _, _ = np.linalg.lstsq(np.vstack([X, reg]), np.concatenate([y, np.zeros(p)]), rcond=None)
            return coeffs
        except: return None

    for r_start, r_end in regions:
        start = max(10, r_start - look_around)
        end = min(n_samples - 11, r_end + look_around + 1)
        gap_len = end - start
        if gap_len <= 0: continue
        
        # 1. THE BACKBONE (Cubic Hermite)
        # Matches Position and Slope (Velocity) to minimize bumps
        y0, y1 = repaired[start-1], repaired[end]
        v0, v1 = (repaired[start-1] - repaired[start-2]), (repaired[end+1] - repaired[end])
        t = np.linspace(0, 1, gap_len)
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        backbone = h00*y0 + h10*v0*gap_len*0.5 + h01*y1 + h11*v1*gap_len*0.5

        # 2. THE TEXTURE (Bidirectional AR)
        l_ctx, r_ctx = repaired[max(0, start-context):start], repaired[end:min(n_samples, end+context)]
        rms_in = (np.sqrt(np.mean(l_ctx**2)) + np.sqrt(np.mean(r_ctx**2))) / 2.0
        
        cf, cb = solve_ar_coeffs(l_ctx, order), solve_ar_coeffs(r_ctx[::-1], order)
        if cf is not None and cb is not None:
            pf, pb = np.zeros(gap_len), np.zeros(gap_len)
            curr_f, curr_b = (l_ctx - np.mean(l_ctx))[-order:].tolist(), (r_ctx[::-1] - np.mean(r_ctx))[-order:].tolist()
            for i in range(gap_len):
                vf = np.dot(curr_f[::-1], cf); pf[i] = vf
                curr_f.pop(0); curr_f.append(vf)
                vb = np.dot(curr_b[::-1], cb); pb[i] = vb
                curr_b.pop(0); curr_b.append(vb)
            
            # Crossfade Predictions (Raised Cosine)
            fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, gap_len)))
            texture = (pf * (1 - fade)) + (pb[::-1] * fade)
            
            # Volume Matching (RMS)
            rms_out = np.sqrt(np.mean(texture**2)) + 1e-9
            texture *= (rms_in / rms_out)
            
            # S-Curve Window (Ensures zero-energy transition at edges)
            window = np.sin(np.linspace(0, np.pi, gap_len))
            repaired[start:end] = backbone + (texture * window)
        else:
            repaired[start:end] = backbone
            
    return repaired

# -----------------------------
# WORKERS
# -----------------------------
class Signals(QObject):
    discovery_finished = Signal(list)
    waveform_ready = Signal(int, int, list, float, int) 
    scan_finished = Signal(int, int, str, int, object)    
    repair_finished = Signal(int, bool, str)

class ClickAnalysisWorker(QRunnable):
    def __init__(self, scan_id, row, path, threshold, window, gate, stop_func):
        super().__init__(); self.scan_id, self.row, self.path = scan_id, row, path
        self.threshold, self.window, self.gate = threshold, window, gate
        self.stop_func, self.signals = stop_func, Signals()
    @Slot()
    def run(self):
        try:
            cmd =[FFMPEG_BIN, "-v", "info", "-i", self.path, "-f", "f32le", "-ac", "1", "-"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=get_clean_env())
            raw_data, err_data = proc.communicate(timeout=20)
            if self.stop_func() or not raw_data: return
            samples = np.frombuffer(raw_data, dtype=np.float32).copy()
            sr = 44100
            m = re.search(r"Audio:.*?, (\d+) Hz", err_data.decode('utf-8', errors='ignore'))
            if m: sr = int(m.group(1))
            dur = len(samples) / float(sr); step = max(1, len(samples) // 4000)
            vis_wf = np.clip(samples[::step], -1.0, 1.0).tolist()
            self.signals.waveform_ready.emit(self.scan_id, self.row, vis_wf, dur, sr)
            diffs = np.abs(np.diff(samples, prepend=samples[0]))
            local_avg = np.convolve(diffs, np.ones(int(self.window))/int(self.window), mode='same')
            condition = (diffs / (local_avg + 1e-9) > self.threshold) & (np.abs(samples) > self.gate)
            peaks = np.where(condition)[0].tolist()
            self.signals.scan_finished.emit(self.scan_id, self.row, "Clicks Detected" if peaks else "Clean", len(peaks), peaks)
        except: self.signals.scan_finished.emit(self.scan_id, self.row, "Error", 0, [])

class RepairWorker(QRunnable):
    def __init__(self, row, path, indices, intensity):
        super().__init__()
        self.row, self.path, self.indices, self.intensity = row, path, indices, intensity
        self.signals = Signals()

    @Slot()
    def run(self):
        try:
            # Determine suffix based on intensity
            suffix = f"_REPAIRED{self.intensity:02d}"
            
            cmd_meta = [FFMPEG_BIN, "-i", self.path]
            proc_m = subprocess.Popen(cmd_meta, stderr=subprocess.PIPE, env=get_clean_env())
            _, err = proc_m.communicate()
            meta_str = err.decode('utf-8', errors='ignore')
            
            sr = 44100
            m_sr = re.search(r"(\d+) Hz", meta_str)
            if m_sr: sr = int(m_sr.group(1))
            
            s_fmt = "pcm_s16le"
            if "s24" in meta_str or "24 bit" in meta_str: s_fmt = "pcm_s24le"
            elif "s32" in meta_str or "32 bit" in meta_str: s_fmt = "pcm_s32le"
            elif "f32" in meta_str: s_fmt = "pcm_f32le"

            cmd_in = [FFMPEG_BIN, "-i", self.path, "-f", "f32le", "-ac", "1", "-"]
            proc_in = subprocess.Popen(cmd_in, stdout=subprocess.PIPE, env=get_clean_env())
            raw_data, _ = proc_in.communicate()
            samples = np.frombuffer(raw_data, dtype=np.float32).copy()

            repaired = repair_audio_logic(samples, self.indices, self.intensity)
            
            base, _ = os.path.splitext(self.path)
            out_path = f"{base}{suffix}.wav"
            
            cmd_out = [FFMPEG_BIN, "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-c:a", s_fmt, out_path]
            proc_out = subprocess.Popen(cmd_out, stdin=subprocess.PIPE, env=get_clean_env())
            proc_out.communicate(input=repaired.tobytes())
            
            self.signals.repair_finished.emit(self.row, True, out_path)
        except Exception as e: 
            self.signals.repair_finished.emit(self.row, False, str(e))

class FileDiscoveryWorker(QRunnable):
    def __init__(self, inputs):
        super().__init__(); self.inputs = inputs; self.signals = Signals()
    @Slot()
    def run(self):
        found = []; exts = ('.wav', '.aiff', '.aif', '.mp3', '.flac', '.ogg')
        for path in self.inputs:
            if os.path.isdir(path):
                for r, _, files in os.walk(path):
                    for f in files:
                        if f.lower().endswith(exts) and not f.startswith("._"): found.append(os.path.join(r, f))
            elif path.lower().endswith(exts): found.append(path)
        found.sort(); self.signals.discovery_finished.emit(found)

# -----------------------------
# UI COMPONENTS
# -----------------------------
class ClickWaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(120); self.setMouseTracking(True)
        self.waveform, self.clicks, self.duration, self.playhead_ms = None, None, 0.0, 0
        self.setStyleSheet("background-color: #202020; border-top: 1px solid #333;")
    def load_data(self, wf, clk, dur, sr=44100):
        self.waveform, self.clicks, self.duration, self.sr = wf, clk, dur, sr; self.update()
    def mousePressEvent(self, event):
        if self.duration and event.button() == Qt.LeftButton:
            ms = (event.position().x() / self.width()) * self.duration * 1000
            self.window().seek_media(ms)
    def paintEvent(self, event):
        p = QPainter(self)
        if not self.waveform:
            p.setPen(QColor(100, 100, 100)); p.drawText(self.rect(), Qt.AlignCenter, "Select a file to visualize"); return
        w, h, mid = self.width(), self.height(), self.height()/2
        p.setPen(QColor(88, 163, 156)); step = w / len(self.waveform)
        for i in range(len(self.waveform)-1):
            p.drawLine(int(i*step), int(mid - self.waveform[i]*mid*0.8), int((i+1)*step), int(mid - self.waveform[i+1]*mid*0.8))
        if self.clicks:
            p.setPen(QPen(QColor(255, 107, 107, 120), 1))
            scale = w / (self.duration * self.sr)
            for c in self.clicks:
                cx = int(c * scale); p.drawLine(cx, 0, cx, h)
        px = (self.playhead_ms / (self.duration * 1000)) * w if self.duration > 0 else 0
        p.setPen(QPen(Qt.white, 2)); p.drawLine(int(px), 0, int(px), h)

class ModernTable(QTableWidget):
    files_dropped, selection_changed_custom, delete_signal, space_pressed = Signal(list), Signal(), Signal(), Signal()
    def __init__(self, font_family):
        super().__init__(0, 3); self.font_family = font_family
        self.setHorizontalHeaderLabels(["FILE NAME", "CLICKS", "STATUS"])
        for i in range(3): self.horizontalHeaderItem(i).setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.verticalHeader().setVisible(False); self.setShowGrid(False); self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows); self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAcceptDrops(True); self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.itemSelectionChanged.connect(self.selection_changed_custom.emit)
    def dragEnterEvent(self, e): e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dragMoveEvent(self, e): e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()
    def dropEvent(self, e): self.files_dropped.emit([u.toLocalFile() for u in e.mimeData().urls()])
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space: self.space_pressed.emit()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace): self.delete_signal.emit()
        else: super().keyPressEvent(event)
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.rowCount() == 0:
            p = QPainter(self.viewport()); f = QFont(self.font_family, 24, QFont.DemiBold, True); p.setFont(f)
            p.setPen(QColor(80, 80, 80)); p.drawText(self.viewport().rect(), Qt.AlignCenter, "DRAG & DROP FILES HERE")

# -----------------------------
# MAIN WINDOW
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self, font_family):
        super().__init__(); self.font_family = font_family; self.setWindowTitle("Rogue Waves CLICK DETECTOR"); self.resize(1100, 850)
        self.threadpool = QThreadPool(); self.files, self.file_data = [], {}
        self.is_scanning, self.stop_flag, self.current_scan_id = False, False, 0
        self.player = QMediaPlayer(); self.audio_output = QAudioOutput(); self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.on_pos_changed); self.player.playbackStateChanged.connect(self.on_play_state_changed)
        self.current_playing_row = self.visualized_row = -1
        self.setup_ui(); self.setup_dark_theme()

    def setup_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        self.sidebar = QFrame(); self.sidebar.setFixedWidth(300); self.sidebar.setObjectName("Sidebar")
        sb = QVBoxLayout(self.sidebar); sb.setContentsMargins(15, 20, 15, 20); sb.setSpacing(10)
        title_size = self.get_optimal_font_size("ROGUE WAVES", 38, 270)
        self.lbl_logo = QLabel(f'<div align="center" style="line-height:0.8;"><span style="font-family:\'{self.font_family}\'; font-size:{title_size}px; font-weight:600; font-style:italic; color:#5ba49d;">ROGUE WAVES</span><br><span style="font-family:\'{self.font_family}\'; font-size:20px; font-weight:600; font-style:italic; color:#808080;">CLICK DETECTOR</span></div>')
        sb.addWidget(self.lbl_logo); sb.addSpacing(10); sb.addWidget(self.header_lbl("INPUT"))
        btn_sel = QPushButton("Select Files / Folder"); btn_sel.clicked.connect(self.select_input); sb.addWidget(btn_sel)
        sb.addWidget(self.header_lbl("SCAN SETTINGS"))
        set_f = QFrame(); set_f.setObjectName("StatsFrame"); set_l = QVBoxLayout(set_f)
        self.slider_thresh = self.add_slider(set_l, "THRESHOLD (Ratio)", 20, 200, 50, "ratio")
        self.slider_win = self.add_slider(set_l, "CONTEXT WINDOW", 2, 200, 10, "samples")
        self.slider_gate = self.add_slider(set_l, "NOISE GATE", 0, 200, 10, "noise")
        btn_def = QPushButton("RESTORE DEFAULTS"); btn_def.setFocusPolicy(Qt.NoFocus); btn_def.setStyleSheet("font-size: 10px; color:#888; border: 1px solid #444;")
        btn_def.clicked.connect(self.restore_defaults); set_l.addWidget(btn_def); sb.addWidget(set_f)
        
        sb.addWidget(self.header_lbl("REPAIR INTENSITY"))
        int_f = QFrame(); int_f.setObjectName("StatsFrame"); int_l = QVBoxLayout(int_f)
        self.slider_intense = self.add_slider(int_l, "STRENGTH", 1, 10, 5, "intensity")
        sb.addWidget(int_f)
        
        sb.addStretch()
        self.progress = QProgressBar(); self.progress.setFixedHeight(6); self.progress.setVisible(False); sb.addWidget(self.progress)
        self.lbl_status = QLabel("Ready"); self.lbl_status.setAlignment(Qt.AlignCenter); self.lbl_status.setStyleSheet("color:#888; font-size:11px;"); sb.addWidget(self.lbl_status)
        btn_scan = QPushButton("SCAN"); btn_scan.setObjectName("ActionBtn"); btn_scan.clicked.connect(self.toggle_scan)
        btn_rep = QPushButton("REPAIR"); btn_rep.setObjectName("ActionBtn"); btn_rep.clicked.connect(self.run_repair)
        h = QHBoxLayout(); h.addWidget(btn_scan); h.addWidget(btn_rep); sb.addLayout(h)
        layout.addWidget(self.sidebar)
        splitter = QSplitter(Qt.Vertical); self.table = ModernTable(self.font_family)
        self.table.files_dropped.connect(self.load_files); self.table.selection_changed_custom.connect(self.on_table_select)
        self.table.doubleClicked.connect(lambda idx: self.start_playback(idx.row())); self.table.space_pressed.connect(self.handle_space_playback)
        self.table.delete_signal.connect(self.delete_selected)
        self.waveform = ClickWaveformWidget(); splitter.addWidget(self.table); splitter.addWidget(self.waveform)
        splitter.setSizes([600, 200]); layout.addWidget(splitter)

    def header_lbl(self, txt):
        l = QLabel(txt); l.setObjectName("SectionHeader"); return l
    def add_slider(self, lay, txt, min_v, max_v, def_v, mode):
        h = QHBoxLayout(); lbl = QLabel(txt); lbl.setStyleSheet("color:#888; font-weight:bold; font-size:11px;")
        v_l = QLabel(); v_l.setStyleSheet("color:#58A39C; font-weight:bold;"); h.addWidget(lbl); h.addStretch(); h.addWidget(v_l); lay.addLayout(h)
        s = QSlider(Qt.Horizontal); s.setRange(min_v, max_v); s.setValue(def_v); lay.addWidget(s)
        def up():
            if mode == "ratio": v_l.setText(f"{s.value()/10:.2f}")
            elif mode == "samples": v_l.setText(f"{s.value()} samples")
            elif mode == "intensity": v_l.setText(f"{s.value()}")
            else: v_l.setText(f"{s.value()/1000:.3f}")
        s.valueChanged.connect(up); up(); return s
    def get_optimal_font_size(self, text, max_size, width):
        font = QFont(self.font_family, max_size, QFont.Bold)
        while max_size > 10 and QFontMetrics(font).horizontalAdvance(text) > width:
            max_size -= 1; font.setPixelSize(max_size)
        return max_size
    def restore_defaults(self): self.slider_thresh.setValue(50); self.slider_win.setValue(10); self.slider_gate.setValue(10)
    def select_input(self):
        f = QFileDialog.getOpenFileNames(self, "Select Audio", "", "Audio (*.wav *.mp3 *.aif)")[0]
        if f: self.load_files(f)
    def delete_selected(self):
        rows = sorted([x.row() for x in self.table.selectionModel().selectedRows()], reverse=True)
        for r in rows:
            if r == self.current_playing_row: self.player.stop()
            self.table.removeRow(r); del self.files[r]
    def load_files(self, paths):
        self.files, self.file_data = [], {}; self.table.setRowCount(0); self.visualized_row = -1
        worker = FileDiscoveryWorker(paths); worker.signals.discovery_finished.connect(self.populate_table); self.threadpool.start(worker)
    def populate_table(self, files):
        self.files = files; self.table.setRowCount(len(files))
        for i, f in enumerate(files):
            self.table.setItem(i, 0, QTableWidgetItem(os.path.basename(f))); self.table.setItem(i, 1, QTableWidgetItem("-"))
            self.table.setItem(i, 2, QTableWidgetItem("Ready")); self.file_data[i] = {'wf': None, 'clk': [], 'dur': 0, 'sr': 44100}
        if files: self.table.selectRow(0); self.start_scan_process()
    def toggle_scan(self):
        if self.is_scanning: self.stop_flag = True
        else: self.start_scan_process()
    def start_scan_process(self):
        if not self.files: return
        self.is_scanning, self.stop_flag, self.current_scan_id = True, False, self.current_scan_id + 1
        self.progress.setVisible(True); self.progress.setRange(0, len(self.files)); self.progress.setValue(0)
        t, w, g = self.slider_thresh.value()/10.0, self.slider_win.value(), self.slider_gate.value()/1000.0
        for i in range(len(self.files)):
            worker = ClickAnalysisWorker(self.current_scan_id, i, self.files[i], t, w, g, lambda: self.stop_flag)
            worker.signals.waveform_ready.connect(self.on_waveform_ready); worker.signals.scan_finished.connect(self.on_scan_finished); self.threadpool.start(worker)
    def on_waveform_ready(self, sid, row, wf, dur, sr):
        if sid == self.current_scan_id:
            self.file_data[row].update({'wf': wf, 'dur': dur, 'sr': sr})
            if row == self.table.currentRow(): self.update_visualizer()
    def on_scan_finished(self, sid, row, status, count, peaks):
        if sid != self.current_scan_id: return
        self.file_data[row]['clk'] = peaks
        clr = QColor("#FF6B6B") if count > 0 else QColor("#58A39C")
        it_clk = QTableWidgetItem(str(count)); it_clk.setForeground(clr); self.table.setItem(row, 1, it_clk)
        it_st = QTableWidgetItem(status); it_st.setForeground(clr); self.table.setItem(row, 2, it_st)
        if row == self.table.currentRow(): self.update_visualizer()
        self.progress.setValue(self.progress.value() + 1)
        if self.progress.value() == len(self.files): self.is_scanning = False; self.progress.setVisible(False)
    def run_repair(self):
        itns = self.slider_intense.value()
        for r in range(self.table.rowCount()):
            d = self.file_data[r]
            if d['clk']:
                worker = RepairWorker(r, self.files[r], d['clk'], itns)
                worker.signals.repair_finished.connect(self.on_repair_done); self.threadpool.start(worker)
    def on_repair_done(self, row, success, path):
        if success:
            it = QTableWidgetItem("Repaired")
            it.setForeground(QColor("#58A39C"))
            self.table.setItem(row, 2, it)
            self.add_single_file(path) # Automatically add the new file to the bottom
        else:
            QMessageBox.critical(self, "Error", path)


    def add_single_file(self, path):
        if not path or path in self.files: return
        new_row = self.table.rowCount()
        self.files.append(path)
        self.file_data[new_row] = {'wf': None, 'clk': [], 'dur': 0, 'sr': 44100}
        
        self.table.blockSignals(True)
        self.table.insertRow(new_row)
        self.table.setItem(new_row, 0, QTableWidgetItem(os.path.basename(path)))
        self.table.setItem(new_row, 1, QTableWidgetItem("-"))
        self.table.setItem(new_row, 2, QTableWidgetItem("Ready"))
        self.table.blockSignals(False)
        
        # Immediate auto-scan
        t, w, g = self.slider_thresh.value()/10.0, self.slider_win.value(), self.slider_gate.value()/1000.0
        worker = ClickAnalysisWorker(self.current_scan_id, new_row, path, t, w, g, lambda: self.stop_flag)
        worker.signals.waveform_ready.connect(self.on_waveform_ready)
        worker.signals.scan_finished.connect(self.on_scan_finished)
        self.threadpool.start(worker)

    def on_table_select(self):
        """Hard-reset playback and UI whenever a new file is clicked."""
        self.player.stop()
        self.player.setSource(QUrl("")) # Clear buffer
        self.waveform.playhead_ms = 0
        
        row = self.table.currentRow()
        if row != -1 and row in self.file_data:
            self.visualized_row = row
            d = self.file_data[row]
            self.waveform.load_data(d['wf'], d['clk'], d['dur'], d['sr'])
            
            # Re-stage for new file
            self.player.setSource(QUrl.fromLocalFile(self.files[row]))
            self.player.setPosition(0) 
            self.current_playing_row = row
        
        self.waveform.update()

    def start_playback(self, row):
        if row >= len(self.files): return
        # Set source and play from the visual playhead position
        self.player.setSource(QUrl.fromLocalFile(self.files[row]))
        self.current_playing_row = row
        self.player.setPosition(int(self.waveform.playhead_ms))
        self.player.play()

    def seek_media(self, ms):
        self.waveform.playhead_ms = ms
        self.waveform.update()
        if self.visualized_row == self.current_playing_row:
            self.player.setPosition(int(ms))
    def on_table_select(self): self.update_visualizer()
    def update_visualizer(self):
        row = self.table.currentRow()
        if row != -1 and row in self.file_data:
            d = self.file_data[row]; self.visualized_row = row
            self.waveform.load_data(d['wf'], d['clk'], d['dur'], d['sr'])
    def start_playback(self, row):
            if row >= len(self.files): return
            
            # Update source if it's a different file
            if self.current_playing_row != row:
                self.player.setSource(QUrl.fromLocalFile(self.files[row]))
                self.current_playing_row = row
            
            # Force position to the playhead shown on the waveform
            self.player.setPosition(int(self.waveform.playhead_ms))
            self.player.play()

    def seek_media(self, ms):
        """Updates internal playhead and syncs player if it's the current file."""
        self.waveform.playhead_ms = ms
        self.waveform.update()
        
        # If we are seeking the file currently loaded in the player, sync immediately
        if self.visualized_row == self.current_playing_row:
            self.player.setPosition(int(ms))
        else:
            # If user clicks a different file's waveform, prepare that file
            self.player.setSource(QUrl.fromLocalFile(self.files[self.visualized_row]))
            self.current_playing_row = self.visualized_row
            self.player.setPosition(int(ms))
    def handle_space_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState: self.player.pause()
        else: self.start_playback(self.table.currentRow())
    def seek_media(self, ms):
        self.waveform.playhead_ms = ms; self.waveform.update()
        # Even if paused, force the player to the position for the next play()
        if self.visualized_row != self.current_playing_row:
             self.player.setSource(QUrl.fromLocalFile(self.files[self.visualized_row]))
             self.current_playing_row = self.visualized_row
        self.player.setPosition(int(ms))
    def on_pos_changed(self, ms):
        if self.current_playing_row == self.visualized_row: self.waveform.playhead_ms = ms; self.waveform.update()
    def on_play_state_changed(self, state): self.set_row_visuals(self.current_playing_row, state == QMediaPlayer.PlayingState)
    def set_row_visuals(self, row, playing):
        if row < 0 or row >= self.table.rowCount(): return
        it = self.table.item(row, 0)
        if not it: return
        txt = it.text().replace("▶ ", "")
        it.setText(f"▶ {txt}" if playing else txt); it.setForeground(QColor("#58A39C" if playing else "#DDD"))
    def setup_dark_theme(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(30, 30, 30))
        pal.setColor(QPalette.Highlight, QColor(88, 163, 156)) 
        self.setPalette(pal)
        self.setStyleSheet(f"""
            QWidget {{ font-family: '{self.font_family}'; font-size: 13px; color: #DDD; }}
            QFrame#Sidebar {{ background-color: #252525; border-right: 1px solid #333; }}
            QLabel#SectionHeader {{ color: #DDD; font-weight: 600; font-style: italic; font-size: 18px; margin-top: 5px; }}
            QFrame#StatsFrame {{ background-color: #2A2A2A; border-radius: 4px; padding: 10px; }}
            QPushButton {{ background-color: #333; border: 1px solid #444; border-radius: 4px; padding: 6px; color: #DDD; }}
            QPushButton#ActionBtn {{ background-color: #58A39C; color: white; font-weight: bold; border: none; font-style: italic; font-size: 14px; }}
            QTableWidget {{ background-color: #181818; alternate-background-color: #222; border: none; outline: 0; selection-background-color: #58A39C; }}
            QTableWidget::item:selected {{ background-color: #58A39C; color: white; }}
            QHeaderView::section {{ background-color: #252525; border: none; padding: 10px; color: #888; font-weight: bold; }}
            QProgressBar::chunk {{ background-color: #58A39C; border-radius: 3px; }}
            QSlider::groove:horizontal {{ height: 4px; background: #111; border-radius: 2px; }}
            QSlider::sub-page:horizontal {{ background: #58A39C; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: #FFF; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    win = MainWindow(load_custom_fonts()); win.show(); sys.exit(app.exec())