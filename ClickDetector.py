import os
import sys
import shutil
import subprocess
import math
import numpy as np
import tempfile
import re

# --- CRITICAL FIX FOR BUS ERROR 10 & SEGFAULT 11 ---
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox,
    QHeaderView, QLabel, QFrame, QAbstractItemView, QProgressBar, 
    QSlider, QSplitter, QScrollArea
)
from PySide6.QtCore import (
    Qt, QRunnable, Slot, Signal, QObject, QThreadPool, QUrl, QTimer
)
from PySide6.QtGui import (
    QFont, QColor, QPainter, QPen, QFontDatabase, 
    QFontMetrics, QPalette, QShortcut, QKeySequence
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# --- RESOURCES & FONTS ---
def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def load_custom_fonts():
    f_family = "Segoe UI" if sys.platform == "win32" else "SF Pro Text"
    for f in ["TitilliumWeb-Regular.ttf", "TitilliumWeb-SemiBold.ttf"]:
        path = resource_path(f)
        if os.path.exists(path):
            fid = QFontDatabase.addApplicationFont(path)
            if fid != -1: 
                fams = QFontDatabase.applicationFontFamilies(fid)
                if fams: f_family = fams[0]
    return f_family

def setup_ffmpeg():
    p_bin = resource_path("ffmpeg.bin")
    p_exe = resource_path("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    if os.path.exists(p_bin): return p_bin
    if os.path.exists(p_exe): return p_exe
    return shutil.which("ffmpeg") or "ffmpeg"

FFMPEG_BIN = setup_ffmpeg()

def get_clean_env():
    env = os.environ.copy()
    for v in ["DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]:
        if v in env: del env[v]
    return env

# -----------------------------
# REPAIR ENGINE
# -----------------------------
def repair_audio_logic(samples, indices, intensity=5, lf_atten=80, thump_width_ms=15, lf_cutoff=600):
    if not indices: return samples.copy()
    repaired = np.array(samples, copy=True, dtype=np.float32)
    n_samples, fs = len(repaired), 44100
    
    look_around = int(2 + (intensity * 3.5)) 
    order, context = int(12 + (intensity * 15)), int(300 + (intensity * 100))

    regions = []
    current_region = [indices[0], indices[0]]
    for idx in indices[1:]:
        if idx - current_region[1] <= look_around * 2.5: 
            current_region[1] = idx
        else: 
            regions.append(current_region)
            current_region = [idx, idx]
    regions.append(current_region)

    def get_low_band_gentle(chunk, cutoff_hz):
        dt = 1.0 / fs
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        alpha = dt / (rc + dt)
        def filter_pass(data):
            res = np.zeros_like(data)
            if len(data) == 0: return res
            v = data[0]
            for i in range(len(data)): 
                v = v + alpha * (data[i] - v)
                res[i] = v
            v = res[-1]
            for i in range(len(res)-1, -1, -1): 
                v = v + alpha * (res[i] - v)
                res[i] = v
            return res
        return filter_pass(filter_pass(chunk))

    def solve_ar_coeffs(data, p):
        if len(data) <= p: p = len(data) // 2
        if p < 2: return None
        x = data - np.mean(data)
        try:
            X = np.zeros((len(x)-p, p))
            for i in range(len(x)-p): X[i, :] = x[i : i + p][::-1]
            y = x[p:]
            reg = np.eye(p) * (np.std(x) * 0.01 + 1e-6)
            c, _, _, _ = np.linalg.lstsq(np.vstack([X, reg]), np.concatenate([y, np.zeros(p)]), rcond=None)
            return c
        except: return None

    # Pass 1
    for r_start, r_end in regions:
        start, end = max(10, r_start - look_around), min(n_samples - 10, r_end + look_around + 1)
        gap_len = end - start
        if gap_len <= 0: continue
        pad = 500
        c_start, c_end = max(0, start-pad), min(n_samples, end+pad)
        chunk = repaired[c_start:c_end]
        l = get_low_band_gentle(chunk, lf_cutoff)
        h = chunk - l
        li, ri = start - c_start, end - c_start
        y0, y1 = l[li-1], l[ri]
        v0, v1 = (l[li-1] - l[li-10])/9.0, (l[ri+9] - l[ri])/9.0
        t = np.linspace(0, 1, gap_len)
        h00, h10, h01, h11 = 2*t**3-3*t**2+1, t**3-2*t**2+t, -2*t**3+3*t**2, t**3-t**2
        low_bridge = h00*y0 + h10*v0*gap_len + h01*y1 + h11*v1*gap_len
        l_ctx, r_ctx = h[max(0, li-context):li], h[ri:min(len(h), ri+context)]
        h_recon = np.zeros(gap_len)
        if len(l_ctx) > order and len(r_ctx) > order:
            cf, cb = solve_ar_coeffs(l_ctx, order), solve_ar_coeffs(r_ctx[::-1], order)
            if cf is not None and cb is not None:
                pf, pb = np.zeros(gap_len), np.zeros(gap_len)
                cur_f, cur_b = (l_ctx-np.mean(l_ctx))[-order:].tolist(), (r_ctx[::-1]-np.mean(r_ctx))[-order:].tolist()
                for i in range(gap_len):
                    vf = np.dot(cur_f[::-1], cf); pf[i] = vf; cur_f.pop(0); cur_f.append(vf)
                    vb = np.dot(cur_b[::-1], cb); pb[i] = vb; cur_b.pop(0); cur_b.append(vb)
                fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, gap_len)))
                h_recon = (pf * (1 - fade)) + (pb[::-1] * fade)
        repaired[start:end] = low_bridge + h_recon

    # Pass 2
    if lf_atten > 0:
        sigma_s = ((thump_width_ms / 2.0) / 1000.0) * fs
        win_h = int(sigma_s * 4)
        for ts, te in regions:
            mid = (ts + te) // 2
            es, ee = max(0, mid-win_h), min(n_samples, mid+win_h)
            chunk = repaired[es:ee]
            if len(chunk) < 10: continue
            l = get_low_band_gentle(chunk, lf_cutoff)
            h = chunk - l
            x = np.arange(len(chunk)) - (mid - es)
            mask = 1.0 - ((lf_atten / 100.0) * np.exp(-(x**2) / (2 * (sigma_s**2))))
            repaired[es:ee] = (l * mask) + h
    return repaired

# -----------------------------
# WORKERS
# -----------------------------
class Signals(QObject):
    discovery_finished = Signal(list)
    waveform_ready = Signal(int, int, list, float, int)
    scan_finished = Signal(int, int, str, int, object)
    repair_finished = Signal(int, bool, str)
    preview_finished = Signal(int, bool, str)

class ClickAnalysisWorker(QRunnable):
    def __init__(self, sid, row, path, thresh, win, gate, stop_f):
        super().__init__(); self.sid, self.row, self.path, self.thresh, self.win, self.gate, self.stop_f, self.signals = sid, row, path, thresh, win, gate, stop_f, Signals()
    @Slot()
    def run(self):
        try:
            cmd = [FFMPEG_BIN, "-v", "info", "-i", self.path, "-f", "f32le", "-ac", "1", "-"]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=get_clean_env())
            raw, err = p.communicate(timeout=20)
            if self.stop_f() or not raw: return
            samps = np.frombuffer(raw, dtype=np.float32).copy(); sr = 44100
            m = re.search(r"Audio:.*?, (\d+) Hz", err.decode('utf-8', errors='ignore'))
            if m: sr = int(m.group(1))
            self.signals.waveform_ready.emit(self.sid, self.row, samps[::max(1, len(samps)//4000)].tolist(), len(samps)/float(sr), sr)
            diffs = np.abs(np.diff(samps, prepend=samps[0]))
            avg = np.convolve(diffs, np.ones(int(self.win))/int(self.win), mode='same')
            pks = np.where((diffs/(avg+1e-9) > self.thresh) & (np.abs(samps) > self.gate))[0].tolist()
            self.signals.scan_finished.emit(self.sid, self.row, "Clicks Detected" if pks else "Clean", len(pks), pks)
        except: self.signals.scan_finished.emit(self.sid, self.row, "Error", 0, [])

class RepairWorker(QRunnable):
    def __init__(self, row, path, indices, intensity, atten, width, cutoff, is_prev=False):
        super().__init__(); self.row, self.path, self.indices, self.intensity, self.atten, self.width, self.cutoff, self.is_prev, self.signals = row, path, indices, intensity, atten, width, cutoff, is_prev, Signals()
    @Slot()
    def run(self):
        try:
            cmd_m = [FFMPEG_BIN, "-i", self.path]
            p_m = subprocess.Popen(cmd_m, stderr=subprocess.PIPE, env=get_clean_env())
            _, err = p_m.communicate(); meta = err.decode('utf-8', errors='ignore'); sr = 44100
            m_sr = re.search(r"(\d+) Hz", meta)
            if m_sr: sr = int(m_sr.group(1))
            cmd_in = [FFMPEG_BIN, "-i", self.path, "-f", "f32le", "-ac", "1", "-"]
            p_in = subprocess.Popen(cmd_in, stdout=subprocess.PIPE, env=get_clean_env())
            raw, _ = p_in.communicate(); samps = np.frombuffer(raw, dtype=np.float32).copy()
            rep = repair_audio_logic(samps, self.indices, self.intensity, self.atten, self.width, self.cutoff)
            out = os.path.join(tempfile.gettempdir(), f"rw_p_{self.row}.wav") if self.is_prev else f"{os.path.splitext(self.path)[0]}_REPAIRED.wav"
            cmd_out = [FFMPEG_BIN, "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-c:a", "pcm_s16le", out]
            p_out = subprocess.Popen(cmd_out, stdin=subprocess.PIPE, env=get_clean_env())
            p_out.communicate(input=rep.tobytes())
            if self.is_prev: self.signals.preview_finished.emit(self.row, True, out)
            else: self.signals.repair_finished.emit(self.row, True, out)
        except Exception as e:
            if self.is_prev: self.signals.preview_finished.emit(self.row, False, str(e))
            else: self.signals.repair_finished.emit(self.row, False, str(e))

class FileDiscoveryWorker(QRunnable):
    def __init__(self, inputs): super().__init__(); self.inputs, self.signals = inputs, Signals()
    @Slot()
    def run(self):
        fnd = []
        for p in self.inputs:
            if os.path.isdir(p):
                for r, _, fs in os.walk(p):
                    for f in fs:
                        if f.lower().endswith(('.wav', '.aiff', '.aif', '.mp3', '.flac', '.ogg')) and not f.startswith("._"): fnd.append(os.path.join(r, f))
            elif p.lower().endswith(('.wav', '.mp3', '.aif')): fnd.append(p)
        fnd.sort(); self.signals.discovery_finished.emit(fnd)

# -----------------------------
# UI COMPONENTS
# -----------------------------
class ClickWaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(120); self.setMouseTracking(True)
        self.waveform, self.clicks, self.duration, self.playhead_ms = None, None, 0.0, 0
        self.is_preview = self.is_processing_preview = False
    def load_data(self, wf, clk, dur, sr=44100): self.waveform, self.clicks, self.duration, self.sr = wf, clk, dur, sr; self.update()
    def mousePressEvent(self, e):
        if self.duration and e.button() == Qt.LeftButton: 
            px = e.position().x() if hasattr(e, 'position') else e.x()
            self.window().seek_media((px/self.width())*self.duration*1000)
    def paintEvent(self, e):
        p = QPainter(self)
        if not self.waveform: 
            p.setPen(QColor(80,80,80)); p.drawText(self.rect(), Qt.AlignCenter, "Select a file to visualize"); return
        w, h, mid = self.width(), self.height(), self.height()/2
        p.setPen(QColor(88, 163, 156)); step = w / len(self.waveform)
        for i in range(len(self.waveform)-1): p.drawLine(int(i*step), int(mid-self.waveform[i]*mid*0.8), int((i+1)*step), int(mid-self.waveform[i+1]*mid*0.8))
        if self.clicks:
            p.setPen(QPen(QColor(255, 107, 107, 120), 1)); scale = w/(self.duration*self.sr)
            for c in self.clicks: cx = int(c*scale); p.drawLine(cx, 0, cx, h)
        if self.is_processing_preview: p.setPen(QColor(212, 139, 48)); p.drawText(10, 20, "PROCESSING...")
        elif self.is_preview: p.setPen(QColor(88, 163, 156)); p.drawText(10, 20, "PREVIEW ACTIVE")
        px = (self.playhead_ms/(self.duration*1000))*w if self.duration > 0 else 0
        p.setPen(QPen(Qt.white, 2)); p.drawLine(int(px), 0, int(px), h)

class ModernTable(QTableWidget):
    files_dropped, selection_changed_custom, delete_signal, space_pressed = Signal(list), Signal(), Signal(), Signal()
    def __init__(self, font):
        super().__init__(0, 3); self.font_family = font; self.setHorizontalHeaderLabels(["FILE NAME", "CLICKS", "STATUS"])
        self.verticalHeader().setVisible(False); self.setShowGrid(False); self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows); self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAcceptDrops(True)
        header = self.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setFixedHeight(30)
        self.itemSelectionChanged.connect(self.selection_changed_custom.emit)
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
        else: e.ignore()
    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
        else: e.ignore()
    def dropEvent(self, e): 
        self.files_dropped.emit([u.toLocalFile() for u in e.mimeData().urls()])
    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace): self.delete_signal.emit()
        else: super().keyPressEvent(e)
    def paintEvent(self, e):
        super().paintEvent(e)
        if self.rowCount() == 0:
            p = QPainter(self.viewport()); p.setFont(QFont(self.font_family, 24, QFont.DemiBold)); p.setPen(QColor(60,60,60))
            p.drawText(self.viewport().rect(), Qt.AlignCenter, "DRAG & DROP FILES")

# -----------------------------
# MAIN WINDOW
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self, font_family):
        super().__init__(); self.font_family = font_family; self.setWindowTitle("Rogue Waves CLICK DETECTOR"); self.resize(1100, 850)
        self.threadpool, self.files, self.file_data = QThreadPool(), [], {}
        self.is_scanning, self.stop_flag, self.current_scan_id = False, False, 0
        self.player, self.audio_output = QMediaPlayer(), QAudioOutput(); self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.on_pos_changed); self.player.playbackStateChanged.connect(self.on_play_state_changed)
        self.current_playing_row = self.visualized_row = -1
        self.setup_ui(); self.setup_dark_theme()
        
        # Shortcut for Spacebar playback
        self.space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_shortcut.activated.connect(self.handle_space_playback)
        
        # Status Monitoring
        self.status_timer = QTimer(); self.status_timer.timeout.connect(self.check_thread_status); self.status_timer.start(500)

    def setup_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget); layout = QHBoxLayout(main_widget); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        
        self.scroll_area = QScrollArea(); self.scroll_area.setFixedWidth(330); self.scroll_area.setWidgetResizable(True); self.scroll_area.setObjectName("SidebarScroll")
        self.sidebar_widget = QWidget(); self.sidebar_widget.setObjectName("SidebarWidget")
        sb = QVBoxLayout(self.sidebar_widget); sb.setContentsMargins(15, 20, 15, 20); sb.setSpacing(10)
        
        title_size = self.get_optimal_font_size("ROGUE WAVES", 38, 270)
        self.lbl_logo = QLabel(f'<div align="center" style="line-height:0.8;"><span style="font-family:\'{self.font_family}\'; font-size:{title_size}px; font-weight:600; font-style:italic; color:#5ba49d;">ROGUE WAVES</span><br><span style="font-family:\'{self.font_family}\'; font-size:20px; font-weight:600; font-style:italic; color:#808080;">CLICK DETECTOR</span></div>')
        sb.addWidget(self.lbl_logo); sb.addSpacing(15)

        sb.addWidget(self.header_lbl("INPUT"))
        btn_sel = QPushButton("Select Files / Folder"); btn_sel.setFocusPolicy(Qt.NoFocus); btn_sel.clicked.connect(self.select_input); sb.addWidget(btn_sel)
        sb.addSpacing(10)

        sb.addWidget(self.header_lbl("SCAN SETTINGS"))
        scan_f = QFrame(); scan_f.setObjectName("StatsFrame"); scan_l = QVBoxLayout(scan_f)
        self.slider_thresh = self.add_slider(scan_l, "SENSITIVITY", 20, 200, 50, "ratio")
        self.slider_win = self.add_slider(scan_l, "CONTEXT WINDOW", 2, 200, 10, "samples")
        self.slider_gate = self.add_slider(scan_l, "NOISE GATE", 0, 200, 10, "noise")
        sb.addWidget(scan_f)

        sb.addSpacing(10); sb.addWidget(self.header_lbl("REPAIR SETTINGS"))
        sb.addWidget(self.header_lbl("CLICK RECONSTRUCTION", is_sub=True))
        cr_f = QFrame(); cr_f.setObjectName("StatsFrame"); cr_l = QVBoxLayout(cr_f)
        self.slider_intense = self.add_slider(cr_l, "STRENGTH", 1, 10, 5, "intensity")
        sb.addWidget(cr_f)

        sb.addWidget(self.header_lbl("LF ATTENUATION", is_sub=True))
        lf_f = QFrame(); lf_f.setObjectName("StatsFrame"); lf_l = QVBoxLayout(lf_f)
        self.slider_lf_atten = self.add_slider(lf_l, "ATTENUATION", 0, 100, 80, "atten")
        self.slider_thump_width = self.add_slider(lf_l, "WIDTH", 2, 200, 15, "width")
        self.slider_lf_cutoff = self.add_slider(lf_l, "CUTOFF", 50, 2000, 600, "hz")
        sb.addWidget(lf_f)
        
        btn_def = QPushButton("RESTORE DEFAULTS"); btn_def.setFocusPolicy(Qt.NoFocus); btn_def.setStyleSheet("font-size:10px; color:#888; border: 1px solid #444; margin-top:5px;"); btn_def.clicked.connect(self.restore_defaults)
        sb.addWidget(btn_def)

        sb.addStretch()
        self.lbl_status = QLabel("Ready"); self.lbl_status.setAlignment(Qt.AlignCenter); self.lbl_status.setStyleSheet("color:#888; font-size:11px;"); sb.addWidget(self.lbl_status)
        
        self.btn_preview = QPushButton("PREVIEW"); self.btn_preview.setObjectName("ActionBtn"); self.btn_preview.setFocusPolicy(Qt.NoFocus); self.btn_preview.setCheckable(True); self.btn_preview.clicked.connect(self.toggle_preview)
        btn_rep = QPushButton("REPAIR"); btn_rep.setObjectName("ActionBtn"); btn_rep.setFocusPolicy(Qt.NoFocus); btn_rep.clicked.connect(self.run_repair)
        h = QHBoxLayout(); h.addWidget(self.btn_preview); h.addWidget(btn_rep); sb.addLayout(h)
        
        self.scroll_area.setWidget(self.sidebar_widget); layout.addWidget(self.scroll_area)

        splitter = QSplitter(Qt.Vertical); self.table = ModernTable(self.font_family); self.waveform = ClickWaveformWidget()
        self.table.files_dropped.connect(self.load_files); self.table.selection_changed_custom.connect(self.on_table_select); self.table.doubleClicked.connect(lambda idx: self.start_playback(idx.row())); self.table.delete_signal.connect(self.delete_selected)
        splitter.addWidget(self.table); splitter.addWidget(self.waveform); splitter.setSizes([600, 200]); layout.addWidget(splitter)

    def header_lbl(self, txt, is_sub=False): 
        l = QLabel(txt); l.setObjectName("SectionHeader")
        if is_sub: l.setStyleSheet("font-size: 13px; color: #888; margin-top: 8px; font-weight: 600; font-style: normal;")
        return l

    def add_slider(self, lay, txt, min_v, max_v, def_v, mode):
        h = QHBoxLayout(); lbl = QLabel(txt); lbl.setStyleSheet("color:#888; font-weight:bold; font-size:11px;"); v_l = QLabel(); v_l.setStyleSheet("color:#58A39C; font-weight:bold;"); h.addWidget(lbl); h.addStretch(); h.addWidget(v_l); lay.addLayout(h)
        s = QSlider(Qt.Horizontal); s.setRange(min_v, max_v); s.setValue(def_v); s.setFocusPolicy(Qt.NoFocus); lay.addWidget(s)
        def up():
            if mode == "ratio": v_l.setText(f"{s.value()/10:.2f}")
            elif mode == "samples": v_l.setText(f"{s.value()} samples")
            elif mode == "intensity": v_l.setText(f"{s.value()}")
            elif mode == "width": v_l.setText(f"{s.value()}ms")
            elif mode == "hz": v_l.setText(f"{s.value()}Hz")
            elif mode == "atten": v_l.setText(f"{s.value()}%")
            else: v_l.setText(f"{s.value()/1000:.3f}")
            if hasattr(self, 'btn_preview') and self.btn_preview.isChecked(): 
                self.btn_preview.setChecked(False); self.disable_preview()
        s.valueChanged.connect(up); up(); return s

    def restore_defaults(self): 
        self.slider_thresh.setValue(50); self.slider_win.setValue(10); self.slider_gate.setValue(10); self.slider_intense.setValue(5); self.slider_lf_atten.setValue(80); self.slider_thump_width.setValue(15); self.slider_lf_cutoff.setValue(600)

    def get_optimal_font_size(self, t, ms, w):
        f = QFont(self.font_family, ms, QFont.Bold); qfm = QFontMetrics(f)
        while ms > 10 and qfm.horizontalAdvance(t) > w: ms -= 1; f.setPixelSize(ms); qfm = QFontMetrics(f)
        return ms

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
        if hasattr(self, 'btn_preview') and self.btn_preview.isChecked(): self.btn_preview.setChecked(False); self.disable_preview()
        worker = FileDiscoveryWorker(paths); worker.signals.discovery_finished.connect(self.populate_table); self.threadpool.start(worker)

    def populate_table(self, files):
        self.files = files; self.table.setRowCount(len(files))
        for i, f in enumerate(files):
            self.table.setItem(i, 0, QTableWidgetItem(os.path.basename(f)))
            self.table.setItem(i, 1, QTableWidgetItem("-"))
            self.table.setItem(i, 2, QTableWidgetItem("Ready"))
            self.file_data[i] = {'wf': None, 'clk': [], 'dur': 0, 'sr': 44100}
        if files: self.table.selectRow(0); self.start_scan_process()

    def start_scan_process(self):
        if not self.files: return
        self.is_scanning, self.stop_flag = True, False; self.current_scan_id += 1; self.lbl_status.setText("Scanning...")
        t, w, g = self.slider_thresh.value()/10.0, self.slider_win.value(), self.slider_gate.value()/1000.0
        for i in range(len(self.files)):
            worker = ClickAnalysisWorker(self.current_scan_id, i, self.files[i], t, w, g, lambda: self.stop_flag)
            worker.signals.waveform_ready.connect(self.on_waveform_ready)
            worker.signals.scan_finished.connect(self.on_scan_finished); self.threadpool.start(worker)

    def check_thread_status(self):
        if self.is_scanning and self.threadpool.activeThreadCount() == 0:
            self.is_scanning = False; self.lbl_status.setText("Ready")

    def toggle_preview(self):
        if self.btn_preview.isChecked():
            row = self.table.currentRow()
            if row == -1 or not self.file_data.get(row, {}).get('clk'): 
                self.btn_preview.setChecked(False); return
            self.waveform.is_processing_preview = True; self.waveform.update()
            worker = RepairWorker(row, self.files[row], self.file_data[row]['clk'], self.slider_intense.value(), self.slider_lf_atten.value(), self.slider_thump_width.value(), self.slider_lf_cutoff.value(), is_prev=True)
            worker.signals.preview_finished.connect(self.on_preview_done); self.threadpool.start(worker)
        else: self.disable_preview()

    def on_preview_done(self, row, success, path):
        self.waveform.is_processing_preview = False
        if not self.btn_preview.isChecked() or row != self.table.currentRow(): 
            self.waveform.update(); return
        if success:
            self.waveform.is_preview = True; self.waveform.update()
            was_playing = (self.player.playbackState() == QMediaPlayer.PlayingState)
            pos = self.player.position(); self.player.setSource(QUrl.fromLocalFile(path)); self.player.setPosition(pos)
            if was_playing: self.player.play()
        else: self.btn_preview.setChecked(False); self.waveform.update()

    def disable_preview(self):
        if self.waveform.is_preview or self.waveform.is_processing_preview:
            self.waveform.is_preview = self.waveform.is_processing_preview = False; self.waveform.update()
            row = self.table.currentRow()
            if row != -1 and row == self.current_playing_row:
                was_playing = (self.player.playbackState() == QMediaPlayer.PlayingState)
                pos = self.player.position()
                self.player.setSource(QUrl.fromLocalFile(self.files[row])); self.player.setPosition(pos)
                if was_playing: self.player.play()

    def on_waveform_ready(self, sid, row, wf, dur, sr):
        if sid == self.current_scan_id: self.file_data[row].update({'wf': wf, 'dur': dur, 'sr': sr})
        if row == self.table.currentRow(): self.update_visualizer()

    def on_scan_finished(self, sid, row, status, count, pks):
        if sid != self.current_scan_id: return
        self.file_data[row]['clk'] = pks; clr = QColor("#FF6B6B") if count > 0 else QColor("#58A39C")
        self.table.setItem(row, 1, QTableWidgetItem(str(count))); self.table.item(row, 1).setForeground(clr)
        self.table.setItem(row, 2, QTableWidgetItem(status)); self.table.item(row, 2).setForeground(clr)
        if row == self.table.currentRow(): self.update_visualizer()

    def run_repair(self):
        for r in range(self.table.rowCount()):
            d = self.file_data[r]
            if d['clk']: 
                worker = RepairWorker(r, self.files[r], d['clk'], self.slider_intense.value(), self.slider_lf_atten.value(), self.slider_thump_width.value(), self.slider_lf_cutoff.value())
                worker.signals.repair_finished.connect(self.on_repair_done); self.threadpool.start(worker)

    def on_repair_done(self, row, success, path):
        if success: 
            it = QTableWidgetItem("Repaired"); it.setForeground(QColor("#58A39C")); self.table.setItem(row, 2, it)
            self.add_single_file(path) 
        else: QMessageBox.critical(self, "Error", path)

    def add_single_file(self, path):
        if not path or path in self.files: return
        row = self.table.rowCount(); self.files.append(path); self.file_data[row] = {'wf': None, 'clk': [], 'dur': 0, 'sr': 44100}
        self.table.blockSignals(True); self.table.insertRow(row); self.table.setItem(row, 0, QTableWidgetItem(os.path.basename(path)))
        self.table.setItem(row, 1, QTableWidgetItem("-")); self.table.setItem(row, 2, QTableWidgetItem("Ready")); self.table.blockSignals(False)
        worker = ClickAnalysisWorker(self.current_scan_id, row, path, self.slider_thresh.value()/10.0, self.slider_win.value(), self.slider_gate.value()/1000.0, lambda: self.stop_flag)
        worker.signals.waveform_ready.connect(self.on_waveform_ready); worker.signals.scan_finished.connect(self.on_scan_finished); self.threadpool.start(worker)

    def on_table_select(self):
        if hasattr(self, 'btn_preview') and self.btn_preview.isChecked(): 
            self.btn_preview.setChecked(False); self.disable_preview()
        row = self.table.currentRow()
        if row != -1 and row in self.file_data:
            self.visualized_row = row; d = self.file_data[row]
            self.waveform.load_data(d['wf'], d['clk'], d['dur'], d['sr'])
            abs_p = os.path.abspath(self.files[row])
            self.player.setSource(QUrl.fromLocalFile(abs_p)); self.player.setPosition(0)
            self.current_playing_row = row; self.waveform.playhead_ms = 0
        self.waveform.update()

    def seek_media(self, ms):
        self.waveform.playhead_ms = ms; self.waveform.update()
        if self.visualized_row == self.current_playing_row: self.player.setPosition(int(ms))

    def start_playback(self, row):
        if row >= len(self.files): return
        if self.current_playing_row != row:
            if hasattr(self, 'btn_preview') and self.btn_preview.isChecked(): 
                self.btn_preview.setChecked(False); self.disable_preview()
            self.player.setSource(QUrl.fromLocalFile(os.path.abspath(self.files[row])))
            self.current_playing_row = row
        self.player.setPosition(int(self.waveform.playhead_ms)); self.player.play()

    def handle_space_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState: self.player.pause()
        else: 
            row = self.table.currentRow()
            if row != -1: self.start_playback(row)

    def on_pos_changed(self, ms):
        if self.current_playing_row == self.visualized_row: self.waveform.playhead_ms = ms; self.waveform.update()

    def on_play_state_changed(self, state):
        r = self.current_playing_row
        if r < 0 or r >= self.table.rowCount(): return
        it = self.table.item(r, 0)
        if it: 
            txt = it.text().replace("▶ ", "")
            if state == QMediaPlayer.PlayingState:
                it.setText(f"▶ {txt}"); it.setForeground(QColor("#58A39C"))
            else:
                it.setText(txt); it.setForeground(QColor("#DDD"))

    def setup_dark_theme(self):
        pal = QPalette(); pal.setColor(QPalette.Window, QColor(30, 30, 30)); pal.setColor(QPalette.Highlight, QColor(88, 163, 156)); self.setPalette(pal)
        self.setStyleSheet(f"""
            QWidget {{ font-family: '{self.font_family}'; font-size: 13px; color: #DDD; }}
            QScrollArea#SidebarScroll {{ border: none; background-color: #252525; border-right: 1px solid #333; }}
            QWidget#SidebarWidget {{ background-color: #252525; }}
            QLabel#SectionHeader {{ color: #DDD; font-weight: 600; font-style: italic; font-size: 18px; margin-top: 5px; }}
            QFrame#StatsFrame {{ background-color: #2A2A2A; border-radius: 4px; padding: 10px; }}
            QPushButton {{ background-color: #333; border: 1px solid #444; border-radius: 4px; padding: 6px; color: #DDD; }}
            QPushButton#ActionBtn {{ background-color: #58A39C; color: white; font-weight: bold; border: none; font-style: italic; font-size: 14px; }}
            QPushButton#ActionBtn:checked {{ background-color: #D48B30; color: white; border: none; }}
            QTableWidget {{ background-color: #181818; alternate-background-color: #222; border: none; selection-background-color: #58A39C; }}
            QHeaderView::section {{ background-color: #252525; border: none; padding: 4px 10px; color: #888; font-weight: bold; border-bottom: 1px solid #333; }}
            QSlider::groove:horizontal {{ height: 4px; background: #111; border-radius: 2px; }}
            QSlider::sub-page:horizontal {{ background: #58A39C; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: #FFF; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
            QScrollBar:vertical {{ border: none; background: #252525; width: 7px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: #444; min-height: 20px; border-radius: 3px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion"); win = MainWindow(load_custom_fonts()); win.show(); sys.exit(app.exec())