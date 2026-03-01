import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PrecisionDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Precision Audio Click Finder")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f4f4f4")

        # Data
        self.y = None
        self.sr = None
        self.y_view = None
        self.click_indices = []

        # --- UI TOP BAR ---
        top_frame = tk.Frame(root, bg="#2c3e50", pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top_frame, text="OPEN AUDIO FILE", command=self.load_file, 
                  bg="#27ae60", fg="white", font=("Arial", 10, "bold"), padx=20).pack(side=tk.LEFT, padx=20)

        self.file_label = tk.Label(top_frame, text="No file loaded", bg="#2c3e50", fg="white", font=("Arial", 10))
        self.file_label.pack(side=tk.LEFT)

        # --- UI SIDEBAR ---
        side_frame = tk.Frame(root, width=320, bg="#ffffff", padx=20, pady=20, highlightbackground="#dddddd", highlightthickness=1)
        side_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(side_frame, text="DETECTION CONTROLS", font=("Arial", 11, "bold"), bg="#ffffff").pack(pady=(0,20))

        # 1. Sensitivity (Threshold Ratio) with Fine-Tuning
        tk.Label(side_frame, text="Sensitivity (Ratio)", bg="#ffffff", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # Row for manual entry and buttons
        thresh_ctrl_frame = tk.Frame(side_frame, bg="#ffffff")
        thresh_ctrl_frame.pack(fill=tk.X, pady=5)
        
        self.thresh_val = tk.DoubleVar(value=5.0)
        
        # Decrease Button
        tk.Button(thresh_ctrl_frame, text="-", width=3, command=lambda: self.nudge_thresh(-0.05)).pack(side=tk.LEFT)
        
        # Manual Entry Box
        self.thresh_entry = tk.Entry(thresh_ctrl_frame, width=8, font=("Courier", 11, "bold"), justify="center")
        self.thresh_entry.insert(0, "5.00")
        self.thresh_entry.pack(side=tk.LEFT, padx=5)
        self.thresh_entry.bind("<Return>", self.manual_thresh_update)
        
        # Increase Button
        tk.Button(thresh_ctrl_frame, text="+", width=3, command=lambda: self.nudge_thresh(0.05)).pack(side=tk.LEFT)
        tk.Label(thresh_ctrl_frame, text="x", bg="#ffffff").pack(side=tk.LEFT)

        # The Slider
        self.thresh_slider = ttk.Scale(side_frame, from_=2.0, to=20.0, variable=self.thresh_val, orient=tk.HORIZONTAL, command=self.sync_entry_from_slider)
        self.thresh_slider.pack(fill=tk.X, pady=(5,20))
        self.thresh_slider.bind("<ButtonRelease-1>", self.update_analysis)

        # 2. Context Window
        tk.Label(side_frame, text="Context Window (Samples)", bg="#ffffff").pack(anchor="w")
        self.win_val = tk.IntVar(value=10)
        self.win_lab_num = tk.Label(side_frame, text="10", font=("Courier", 10, "bold"), fg="#e67e22", bg="#ffffff")
        self.win_lab_num.pack(anchor="e")
        self.win_slider = ttk.Scale(side_frame, from_=2, to=200, variable=self.win_val, orient=tk.HORIZONTAL, command=self.update_labels)
        self.win_slider.pack(fill=tk.X, pady=(0,20))
        self.win_slider.bind("<ButtonRelease-1>", self.update_analysis)

        # 3. Noise Gate
        tk.Label(side_frame, text="Noise Gate (Min Amp)", bg="#ffffff").pack(anchor="w")
        self.gate_val = tk.DoubleVar(value=0.01)
        self.gate_lab_num = tk.Label(side_frame, text="0.010", font=("Courier", 10, "bold"), fg="#e67e22", bg="#ffffff")
        self.gate_lab_num.pack(anchor="e")
        self.gate_slider = ttk.Scale(side_frame, from_=0.000, to=0.200, variable=self.gate_val, orient=tk.HORIZONTAL, command=self.update_labels)
        self.gate_slider.pack(fill=tk.X, pady=(0,20))
        self.gate_slider.bind("<ButtonRelease-1>", self.update_analysis)

        # Click List
        tk.Label(side_frame, text="Detected Click List:", bg="#ffffff", font=("Arial", 9, "bold")).pack(anchor="w", pady=(10,0))
        self.listbox = tk.Listbox(side_frame, height=15, font=("Courier", 9))
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.listbox.bind('<<ListboxSelect>>', self.zoom_to_click)

        # --- PLOT AREA ---
        self.fig, self.ax = plt.subplots(figsize=(8, 5), facecolor="#f4f4f4")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

    # --- CONTROL LOGIC ---

    def nudge_thresh(self, amount):
        new_val = round(self.thresh_val.get() + amount, 2)
        self.thresh_val.set(new_val)
        self.sync_entry_from_slider()
        self.update_analysis()

    def sync_entry_from_slider(self, event=None):
        val = self.thresh_val.get()
        self.thresh_entry.delete(0, tk.END)
        self.thresh_entry.insert(0, f"{val:.2f}")

    def manual_thresh_update(self, event=None):
        try:
            val = float(self.thresh_entry.get())
            self.thresh_val.set(val)
            self.update_analysis()
        except ValueError:
            self.sync_entry_from_slider()

    def update_labels(self, event=None):
        self.win_lab_num.config(text=f"{int(self.win_val.get())}")
        self.gate_lab_num.config(text=f"{self.gate_val.get():.3f}")

    # --- CORE LOGIC ---

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac")])
        if not path: return
        self.file_label.config(text="Processing...")
        self.root.update_idletasks()
        self.y, self.sr = librosa.load(path, sr=None)
        step = max(1, len(self.y) // 8000)
        self.y_view = self.y[::step]
        self.view_indices = np.arange(len(self.y))[::step]
        self.file_label.config(text=f"Loaded: {path.split('/')[-1]} ({self.sr}Hz)")
        self.update_analysis()

    def update_analysis(self, event=None):
        if self.y is None: return
        t = self.thresh_val.get()
        w = int(self.win_val.get())
        g = self.gate_val.get()

        diffs = np.abs(np.diff(self.y, prepend=self.y[0]))
        weights = np.ones(w) / w
        local_avg = np.convolve(diffs, weights, mode='same')
        epsilon = 1e-9
        prev_avg = np.roll(local_avg, 1)
        prev_avg[0] = epsilon
        ratios = diffs / (prev_avg + epsilon)

        condition = (ratios > t) & (np.abs(self.y) > g)
        self.click_indices = np.where(condition)[0]

        self.listbox.delete(0, tk.END)
        for idx in self.click_indices:
            self.listbox.insert(tk.END, f"{(idx/self.sr):.3f}s | Idx: {idx}")

        self.ax.clear()
        self.ax.plot(self.view_indices, self.y_view, color='#3498db', alpha=0.4)
        if len(self.click_indices) > 0:
            display_clicks = self.click_indices[:200]
            for c in display_clicks:
                self.ax.axvline(x=c, color='#e74c3c', linestyle='--', alpha=0.5)
            self.ax.scatter(display_clicks, self.y[display_clicks], color='red', s=15)
        self.ax.set_title(f"Threshold: {t:.2f}x | Found {len(self.click_indices)} clicks")
        self.ax.set_ylim(-1.1, 1.1)
        self.canvas.draw()

    def zoom_to_click(self, event):
        selection = self.listbox.curselection()
        if not selection: return
        idx = self.click_indices[selection[0]]
        start, end = max(0, idx-200), min(len(self.y), idx+200)
        self.ax.clear()
        self.ax.plot(range(start, end), self.y[start:end], marker='o', markersize=4, color='#3498db', linewidth=1)
        self.ax.axvline(x=idx, color='red', linewidth=2)
        jump = np.abs(self.y[idx] - self.y[idx-1])
        self.ax.set_title(f"Sample {idx} | Local Jump: {jump:.4f}")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PrecisionDetector(root)
    root.mainloop()