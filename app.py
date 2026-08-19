"""Openhouse AI Assistant - Enhanced with embedded camera and motion detection."""

import sys, os, threading, time, logging
from datetime import datetime
from collections import defaultdict

import cv2, numpy as np, tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk

import config, database, face, llm, tts, stt
from dashboard import VisitorDashboard

logger = logging.getLogger("openhouse")

_last_greeting_time = {}
_state_lock = threading.Lock()
_gui_update_event = threading.Event()
_dashboard = VisitorDashboard()
_tts_lock = threading.Lock()

_active_interaction = False
_active_interaction_lock = threading.Lock()


def _is_interaction_active() -> bool:
    with _active_interaction_lock:
        return _active_interaction


def _set_interaction_active(active: bool) -> None:
    global _active_interaction
    with _active_interaction_lock:
        _active_interaction = active


def _is_in_cooldown(visitor_id: int) -> bool:
    with _state_lock:
        last = _last_greeting_time.get(visitor_id, 0.0)
        return (time.time() - last) < config.COOLDOWN_SECONDS

def _mark_greeted(visitor_id: int) -> None:
    with _state_lock:
        _last_greeting_time[visitor_id] = time.time()

def _append_chat(app, who: str, text: str) -> None:
    """Add message to chat log safely on the Tkinter main thread."""
    def _do_append():
        try:
            app.chat_log.config(state=tk.NORMAL)
            app.chat_log.tag_configure("host", justify="left", lmargin1=4, lmargin2=4, foreground="#4fc3f7")
            app.chat_log.tag_configure("visitor", justify="right", rmargin=4, foreground="#81c784")

            if who == "Visitor":
                app.chat_log.insert(tk.END, f"{who}: {text}\n", "visitor")
            else:
                app.chat_log.insert(tk.END, f"{who}: {text}\n", "host")

            app.chat_log.see(tk.END)
            app.chat_log.config(state=tk.DISABLED)
        except Exception as e:
            logger.error("Error updating chat log: %s", e)

    if threading.current_thread() is threading.main_thread():
        _do_append()
    else:
        app.after(0, _do_append)


def _host_say(app, text: str) -> None:
    _append_chat(app, "Host", text)
    with _tts_lock:
        tts.speak(text)

def _greet_returning(visitor_id: int, name: str, set_status, app) -> None:
    _set_interaction_active(True)
    try:
        database.update_visitor_last_seen(visitor_id)
        database.log_event(visitor_id, "return_visit")
        set_status(f"Welcome back, {name}!", "green")
        message = llm.generate_return_greeting(name, "")
        _mark_greeted(visitor_id)
        logger.info("Returning visitor %s: %s", name, message)
        _host_say(app, message)
        _dashboard.record_visitor(visitor_id, name, "return")
        app.after(0, app._update_visitor_count)

        # Start conversation with returning visitor
        _start_conversation(visitor_id, name, set_status, app)
    finally:
        _set_interaction_active(False)

def _greet_new(encoding: np.ndarray, set_status, app) -> None:
    _set_interaction_active(True)
    try:
        set_status("New visitor - greeting and asking for name...", "orange")
        warm_greeting = "Hi there, welcome to our open house! It is really nice to meet you. What name would you like me to call you?"
        _host_say(app, warm_greeting)
        heard_name = stt.capture_name(timeout=10, phrase_time_limit=5)

        if heard_name:
            name = heard_name.strip().capitalize()
            _append_chat(app, "Visitor", heard_name)
        else:
            name = f"Visitor_{datetime.now().strftime('%Y%m%d_%H%M')}"
            logger.info("No name captured, using fallback: %s", name)
            set_status(f"Name not heard, using: {name}", "red")
            _append_chat(app, "Visitor", "(no response)")

        visitor_id = database.add_visitor(name, encoding)
        face.register_known_encoding(visitor_id, name, encoding)
        _mark_greeted(visitor_id)
        database.log_event(visitor_id, "new_visitor_registered", f"Registered as {name}")
        set_status(f"New visitor registered: {name}", "blue")

        message = llm.generate_new_visitor_greeting(name)
        logger.info("New visitor %s: %s", name, message)
        _host_say(app, message)
        _dashboard.record_visitor(visitor_id, name, "new")
        app.after(0, app._update_visitor_count)

        # Start conversation with new visitor
        _start_conversation(visitor_id, name, set_status, app)
    finally:
        _set_interaction_active(False)


def _start_conversation(visitor_id: int, name: str, set_status, app) -> None:
    """Start continuous conversation with visitor."""
    time.sleep(1)  # Wait briefly after initial greeting

    for _ in range(3):  # Allow up to 3 exchanges
        set_status(f"Listening to {name}...", "blue")

        # Listen for visitor response
        visitor_response = stt.capture_name(timeout=10, phrase_time_limit=8)

        if not visitor_response:
            set_status(f"No response from {name}", "gray")
            break

        visitor_response = visitor_response.strip()
        _append_chat(app, "Visitor", visitor_response)
        logger.info("Visitor %s said: %s", name, visitor_response)

        # Generate response from LLM
        host_response = llm.generate_chat_response(visitor_response)
        logger.info("Host response: %s", host_response)

        # Display and speak the response
        set_status(f"Host: {host_response}", "green")
        _host_say(app, host_response)

        # Ask if they want to continue
        time.sleep(1)
        continue_prompt = "Would you like to ask anything else about the house?"
        _host_say(app, continue_prompt)
        continue_response = stt.capture_name(timeout=5, phrase_time_limit=4)

        if not continue_response or any(kw in continue_response.lower() for kw in ["no", "nope", "that's all", "bye"]):
            _host_say(app, f"Thank you for visiting, {name}! Enjoy taking a look around!")
            break

        _append_chat(app, "Visitor", continue_response)

    set_status("Ready - waiting for next visitor", "green")

def _camera_thread(cap, set_frame_cb, set_status, app):
    """Background thread for fast camera feed and periodic face detection."""
    last_detection_check = 0.0
    detection_interval = 0.3  # Run face detection ~3 times per second
    consecutive_detections = 0

    while cap.isOpened() and not app._stop_event.is_set():
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            set_status("Camera disconnected", "red")
            time.sleep(0.5)
            consecutive_detections = 0
            continue

        set_frame_cb(frame)

        now = time.time()
        if now - last_detection_check < detection_interval:
            time.sleep(0.01)
            continue
        last_detection_check = now

        # If an active interaction (greeting/conversation) is underway, skip triggering new greetings
        if _is_interaction_active():
            continue

        encoding, bbox = face.capture_face_encoding(frame)
        if encoding is None:
            set_status("Waiting for visitor", "gray")
            consecutive_detections = 0
            continue

        consecutive_detections += 1

        known_ids, known_names, known_embeddings = face.load_known_encodings()
        if known_names:
            match_idx, distance = face.compare_faces(encoding, known_embeddings)
        else:
            match_idx, distance = None, float("inf")

        # Check match or new visitor
        if match_idx is not None and distance < config.FACE_THRESHOLD:
            visitor_id = known_ids[match_idx]
            visitor_name = known_names[match_idx]
            if _is_in_cooldown(visitor_id):
                set_status(f"Visitor present: {visitor_name}", "green")
            else:
                threading.Thread(
                    target=_greet_returning,
                    args=(visitor_id, visitor_name, set_status, app),
                    daemon=True,
                ).start()
        else:
            if consecutive_detections == 1:
                threading.Thread(
                    target=_greet_new,
                    args=(encoding, set_status, app),
                    daemon=True,
                ).start()

class OpenhouseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Openhouse AI Assistant")
        self.geometry("1200x700")
        self.configure(bg="#1a1a2e")
        
        self.photo = None
        self.current_frame = None
        self._cap = None
        self._stop_event = threading.Event()
        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _setup_ui(self):
        """Setup main UI with camera and visitor count."""
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        title = ttk.Label(top_frame, text="Openhouse AI Assistant", font=("Arial", 14, "bold"))
        title.pack(side=tk.LEFT)
        
        self.visitor_count_var = tk.StringVar(value="Visitors Today: 0")
        count_label = ttk.Label(top_frame, textvariable=self.visitor_count_var, font=("Arial", 11))
        count_label.pack(side=tk.RIGHT, padx=10)
        
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        left_frame = ttk.LabelFrame(main_frame, text="Live Camera Feed", padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.cam_label = tk.Label(left_frame, bg="#0f0f1a", height=25, width=50)
        self.cam_label.pack(fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(right_frame, text="Status", font=("Arial", 10, "bold")).pack()
        self.status_var = tk.StringVar(value="Starting...")
        status_label = ttk.Label(right_frame, textvariable=self.status_var, wraplength=250, justify=tk.LEFT)
        status_label.pack(fill=tk.X, pady=5)
        
        ttk.Label(right_frame, text="Conversations", font=("Arial", 10, "bold")).pack()
        self.chat_log = scrolledtext.ScrolledText(
            right_frame, height=12, width=35, font=("Courier", 9), bg="#16213e", fg="#eee", state=tk.DISABLED
        )
        self.chat_log.pack(fill=tk.BOTH, expand=True, pady=5)

        # Interactive manual input field for testing or typing questions
        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=5)
        self.msg_entry = ttk.Entry(input_frame)
        self.msg_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.msg_entry.bind("<Return>", lambda event: self._send_manual_message())
        ttk.Button(input_frame, text="Send", command=self._send_manual_message).pack(side=tk.RIGHT)
        
        ttk.Button(right_frame, text="View Dashboard", command=self._show_dashboard).pack(pady=5)
    
    def _send_manual_message(self):
        text = self.msg_entry.get().strip()
        if not text:
            return
        self.msg_entry.delete(0, tk.END)
        _append_chat(self, "Visitor", text)

        def _process():
            self._set_status("Thinking...", "blue")
            reply = llm.generate_chat_response(text)
            self._set_status(f"Host: {reply}", "green")
            _host_say(self, reply)
            self._set_status("Ready - waiting for next visitor", "green")

        threading.Thread(target=_process, daemon=True).start()

    def _set_status(self, msg, color="gray"):
        if threading.current_thread() is threading.main_thread():
            self.status_var.set(msg)
        else:
            self.after(0, lambda: self.status_var.set(msg))
    
    def _set_frame(self, frame):
        self.current_frame = frame
        _gui_update_event.set()
    
    def _draw_frame(self, frame):
        h, w = frame.shape[:2]
        win_w = self.cam_label.winfo_width()
        if win_w < 50:
            win_w = 800
        scale = (win_w - 20) / w
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.cam_label.config(image=self.photo)
    
    def _update_visitor_count(self):
        stats = _dashboard.get_daily_stats()
        self.visitor_count_var.set(f"Visitors Today: {stats['total_visits']}")
    
    def _show_dashboard(self):
        dashboard_window = tk.Toplevel(self)
        dashboard_window.title("Visitor Dashboard")
        dashboard_window.geometry("700x500")
        
        stats = _dashboard.get_daily_stats()
        hourly = _dashboard.get_hourly_stats()
        
        text_widget = scrolledtext.ScrolledText(dashboard_window, state=tk.NORMAL, height=30, width=80)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget.insert(tk.END, f"Date: {stats['date']}\n")
        text_widget.insert(tk.END, f"Total Visits: {stats['total_visits']}\n")
        text_widget.insert(tk.END, f"Unique Visitors: {stats['unique_visitors']}\n\n")
        text_widget.insert(tk.END, "=== Hourly Breakdown ===\n\n")
        
        for hour_data in hourly:
            text_widget.insert(tk.END, f"{hour_data['hour']}: {hour_data['count']} visitor(s)\n")
            for visitor in hour_data['visitors']:
                text_widget.insert(tk.END, f"  - {visitor['name']} ({visitor['type']})\n")
            text_widget.insert(tk.END, "\n")
        
        text_widget.config(state=tk.DISABLED)
    
    def _start_camera(self):
        self._set_status("Opening camera...", "blue")
        
        def _camera_init():
            try:
                self._cap = face.open_camera()
                self._set_status("Ready - watching for visitors", "green")
                
                t = threading.Thread(
                    target=_camera_thread,
                    args=(self._cap, self._set_frame, self._set_status, self),
                    daemon=True,
                )
                t.start()
            except RuntimeError as e:
                self._set_status(f"Camera failed: {e}", "red")
                logger.error("Camera open failed: %s", e)
        
        threading.Thread(target=_camera_init, daemon=True).start()
        
        def _gui_loop():
            if not self._stop_event.is_set():
                _gui_update_event.wait(timeout=0.05)
                _gui_update_event.clear()
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                    self.current_frame = None
                    self._draw_frame(frame)
            self.after(30, _gui_loop)
        
        _gui_loop()
    
    def _on_close(self):
        self._stop_event.set()
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self.destroy()

def run():
    import logging as _logging
    
    log_dir = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    _logging.basicConfig(
        level=getattr(_logging, config.LOG_LEVEL, _logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            _logging.FileHandler(log_dir / "app.log"),
            _logging.StreamHandler(),
        ],
    )
    
    logger.info("Initialising database at %s", config.DB_PATH)
    database.init_db()
    
    known_ids, known_names, _ = face.load_known_encodings()
    logger.info("Loaded %d known visitors.", len(known_names))
    
    app = OpenhouseApp()
    app.after(500, app._start_camera)
    app.mainloop()

if __name__ == "__main__":
    run()
