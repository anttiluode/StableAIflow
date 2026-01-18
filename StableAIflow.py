import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image, ImageTk, ImageFilter
# FIXED: Added StringVar, Entry to imports
from tkinter import Tk, Label, Scale, HORIZONTAL, Frame, Checkbutton, BooleanVar, Button, StringVar, Entry, VERTICAL, Canvas
from threading import Thread, Lock
import time

# --- 1. Phase Correlation Engine (The "Complex Signal" Lock) ---
class PhaseLocker:
    def __init__(self):
        self.prev_gray = None
        self.hann = None
        self.momentum_x = 0.0
        self.momentum_y = 0.0
        self.smoothness = 0.5 

    def calculate_lock_shift(self, current_image_rgb):
        gray = cv2.cvtColor(current_image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if self.hann is None or self.hann.shape != gray.shape:
            h, w = gray.shape
            self.hann = cv2.createHanningWindow((w, h), cv2.CV_32F)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0

        try:
            shift, response = cv2.phaseCorrelate(self.prev_gray, gray, window=self.hann)
            dx, dy = shift
            if abs(dx) > 50 or abs(dy) > 50: dx, dy = 0, 0
            self.momentum_x = self.momentum_x * self.smoothness + dx * (1.0 - self.smoothness)
            self.momentum_y = self.momentum_y * self.smoothness + dy * (1.0 - self.smoothness)
        except:
            self.momentum_x, self.momentum_y = 0, 0

        self.prev_gray = gray
        return self.momentum_x, self.momentum_y

    def lock_layer(self, image_to_shift, dx, dy):
        rows, cols = image_to_shift.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        # Use BORDER_REPLICATE to avoid black edges pulling into the feedback loop
        return cv2.warpAffine(image_to_shift, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

# --- 2. Face Tracker (Restored for Masking) ---
class FaceTracker:
    def __init__(self):
        try:
            self.frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        except: self.frontal = None
        self.last_rect = None 

    def get_mask(self, image_rgb, expand_factor=1.0):
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        if self.frontal is None: return np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Detect
        faces = self.frontal.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) == 0: faces = self.profile.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
            
        target = None
        if len(faces) > 0:
            target = max(faces, key=lambda r: r[2] * r[3])
            self.last_rect = target
        elif self.last_rect is not None:
            target = self.last_rect # Memory
            
        if target is not None:
            x, y, fw, fh = target
            # Apply Scale
            nw = int(fw * expand_factor)
            nh = int(fh * expand_factor * 1.3)
            cx, cy = x + fw//2, y + fh//2
            # Ellipse mask
            cv2.ellipse(mask, (cx, cy), (nw//2, nh//2), 0, 0, 360, 1.0, -1)
            # Heavy Blur for smooth blending
            mask = cv2.GaussianBlur(mask, (99, 99), 0)
            
        return np.repeat(mask[:, :, np.newaxis], 3, axis=2)

# --- 3. Fractal Viscosity (Signal Stability) ---
class FractalViscosity:
    def __init__(self):
        self.last_val = None
        self.sensitivity = 15.0 
    
    def measure(self, image_tensor):
        if image_tensor.size(1) == 3:
            gray = (0.299 * image_tensor[:, 0] + 0.587 * image_tensor[:, 1] + 0.114 * image_tensor[:, 2])
        else:
            gray = image_tensor.squeeze(1)
        b1 = torch.std(torch.nn.functional.avg_pool2d(gray.unsqueeze(1), 3, 1, 1))
        b4 = torch.std(torch.nn.functional.avg_pool2d(gray.unsqueeze(1), 9, 1, 4))
        curr = float(b1 - b4)
        if self.last_val is None:
            self.last_val = curr
            return 0.0
        delta = abs(curr - self.last_val)
        self.last_val = curr
        return max(0.0, 1.0 - min(1.0, delta * self.sensitivity))

# --- 4. Main Application ---
class PhaseLockedFlow:
    def __init__(self, master):
        self.master = master
        self.master.title("Stable AI Flow v21 (Side Layout + Masking)")
        self.master.geometry("900x700")
        self.master.configure(bg='#222222')
        
        self.cap = None
        
        # SIGNAL PROCESSING MODULES
        self.phase_locker = PhaseLocker()
        self.tracker = FaceTracker() # Restored Tracker
        self.fractal = FractalViscosity()
        
        self.frame_lock = Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Thread-safe Params
        self.params = {
            "strength": 0.50,
            "gravity": 0.70,     
            "sharpness": 1.0,
            "mask_enabled": True,
            "mask_scale": 1.0
        }
        
        self.current_webcam = None
        self.last_ai_frame = None 
        self.display_frame = None
        self.live_generation = False
        self.pipe = None
        
        self.setup_gui()
        self.update_video()
        self.master.after(100, self.start_model_loading)
    
    def start_model_loading(self):
        Thread(target=self.load_model, daemon=True).start()
    
    def load_model(self):
        try:
            self.master.after(0, lambda: self.status_var.set("📦 Loading AI..."))
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
            ).to("cuda")
            self.pipe.set_progress_bar_config(disable=True)
            self.master.after(0, lambda: self.start_button.config(state='normal', text="Start Phase Lock"))
            self.master.after(0, lambda: self.status_var.set("✓ Ready"))
        except Exception as e:
            print(f"Model Load Error: {e}")
            self.master.after(0, lambda: self.status_var.set(f"Error: {e}"))

    def update_param(self, name, value):
        self.params[name] = float(value)

    def setup_gui(self):
        # Main container splitting left (controls) and right (video)
        main_container = Frame(self.master, bg='#222222')
        main_container.pack(fill='both', expand=True)

        # --- LEFT SIDEBAR ---
        sidebar = Frame(main_container, width=300, bg='#333333', relief='sunken', bd=2)
        sidebar.pack(side='left', fill='y', padx=5, pady=5)
        sidebar.pack_propagate(False) # Don't shrink

        Label(sidebar, text="Controls", bg='#333333', fg='white', font=("Arial", 14, "bold")).pack(pady=(10, 20))

        self.start_button = Button(sidebar, text="Loading...", command=self.toggle_live, 
                                 bg='#cccccc', font=('Arial', 12, 'bold'), state='disabled', height=2)
        self.start_button.pack(fill='x', padx=10, pady=10)

        # Prompt
        Label(sidebar, text="Prompt:", bg='#333333', fg='white', anchor='w').pack(fill='x', padx=10)
        self.prompt_var = StringVar(value="charcoal sketch of a cyborg, dark, gritty, detailed")
        self.prompt_entry = Entry(sidebar, textvariable=self.prompt_var, bg='#555555', fg='white', insertbackground='white')
        self.prompt_entry.pack(fill='x', padx=10, pady=(0, 20))
        
        # Sliders helper function for cleaner code
        def create_slider(label_text, param_name, from_, to_, res, default_val):
            Label(sidebar, text=label_text, bg='#333333', fg='white', anchor='w').pack(fill='x', padx=10, pady=(10, 0))
            s = Scale(sidebar, from_=from_, to=to_, resolution=res, orient=HORIZONTAL, 
                      bg='#333333', fg='white', troughcolor='#555555', highlightthickness=0,
                      command=lambda v: self.update_param(param_name, v))
            s.set(default_val)
            s.pack(fill='x', padx=10)

        create_slider("Dream Strength", "strength", 0.1, 1.0, 0.05, 0.50)
        create_slider("Phase Lock Gravity (Stability)", "gravity", 0.0, 0.99, 0.01, 0.70)
        create_slider("Loop Sharpness (Crystallizer)", "sharpness", 0.0, 2.0, 0.1, 1.0)
        
        # Mask Controls added back
        Label(sidebar, text="--- Masking ---", bg='#333333', fg='#aaaaaa').pack(pady=(20, 5))
        create_slider("Mask Scale", "mask_scale", 0.5, 2.0, 0.1, 1.0)
        
        self.mask_var = BooleanVar(value=True)
        cb = Checkbutton(sidebar, text="Enable Face Mask", variable=self.mask_var, 
                    bg='#333333', fg='white', selectcolor='#555555', activebackground='#333333', activeforeground='white',
                    command=lambda: self.update_param("mask_enabled", self.mask_var.get()))
        cb.pack(fill='x', padx=10, pady=10)

        # --- RIGHT VIDEO AREA ---
        video_area = Frame(main_container, bg='black')
        video_area.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        self.panel = Label(video_area, bg='black')
        # Use expand=True so it centers in the available space
        self.panel.pack(expand=True)

        # Status Bar at bottom
        self.status_var = StringVar(value="Init...")
        Label(self.master, textvariable=self.status_var, relief='sunken', anchor='w', bg='#222222', fg='white').pack(side='bottom', fill='x')

    def live_loop(self):
        while self.live_generation:
            if self.current_webcam is None: 
                time.sleep(0.01)
                continue
            
            # Safe Param Read
            p_strength = self.params["strength"]
            p_gravity = self.params["gravity"]
            p_sharpness = self.params["sharpness"]
            p_mask_enabled = self.params["mask_enabled"]
            p_mask_scale = self.params["mask_scale"]
            
            with self.frame_lock:
                webcam_raw = self.current_webcam.copy()
                webcam_512 = cv2.cvtColor(cv2.resize(webcam_raw, (512, 512)), cv2.COLOR_BGR2RGB)
            
            try:
                # 1. PHASE CORRELATION: Calculate Shift (dx, dy)
                dx, dy = self.phase_locker.calculate_lock_shift(webcam_512)
                
                # 2. Fractal Viscosity
                img_tensor = torch.from_numpy(webcam_512).float().permute(2,0,1).to(self.device) / 255.0
                viscosity = self.fractal.measure(img_tensor)
                
                # 3. FEEDBACK MIXING
                input_pil = Image.fromarray(webcam_512)
                mix_amount = 0.0
                
                if self.last_ai_frame is not None:
                    # A. PHASE LOCKING THE DREAM
                    locked_dream_np = self.phase_locker.lock_layer(self.last_ai_frame, dx, dy)
                    
                    # B. CRYSTALLIZATION (Sharpening to fight blur loop)
                    if p_sharpness > 0:
                        # Simple unsharp mask in CV2 is faster than PIL
                        gaussian = cv2.GaussianBlur(locked_dream_np, (0, 0), 2.0)
                        locked_dream_np = cv2.addWeighted(locked_dream_np, 1.0 + p_sharpness, gaussian, -p_sharpness, 0)
                    
                    # C. BLEND
                    mix_amount = viscosity * p_gravity
                    locked_dream_pil = Image.fromarray(locked_dream_np)
                    input_pil = Image.blend(input_pil, locked_dream_pil, mix_amount)

                # 4. DIFFUSION
                # Safe Step Calc
                steps = max(2, min(50, int(2.0 / max(0.01, p_strength))))

                result = self.pipe(
                    prompt=self.prompt_var.get(),
                    image=input_pil,
                    strength=p_strength,
                    guidance_scale=0.0,
                    num_inference_steps=steps
                ).images[0]
                
                dream_np = np.array(result)
                self.last_ai_frame = dream_np 

                # 5. MASKING (Restored Logic)
                if p_mask_enabled:
                    # Get mask from current webcam frame
                    mask = self.tracker.get_mask(webcam_512, expand_factor=p_mask_scale)
                    # Composite: AI where mask is white, Webcam where mask is black
                    final_img = (dream_np * mask + webcam_512 * (1.0 - mask)).astype(np.uint8)
                else:
                    # Full frame AI
                    final_img = dream_np
                
                with self.frame_lock:
                    self.display_frame = final_img
                
                self.master.after(0, lambda v=viscosity, m=mix_amount, s=(dx,dy): 
                                  self.status_var.set(f"Phase Shift: ({s[0]:.1f}, {s[1]:.1f}) | Visc: {v:.2f} | Feed: {m*100:.0f}%"))
                
            except Exception as e:
                print(f"Loop Error: {e}")
                time.sleep(0.1)

    def update_video(self):
        if self.cap is None: self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if ret:
            with self.frame_lock:
                self.current_webcam = frame
            
            disp = None
            if self.live_generation and self.display_frame is not None:
                disp = self.display_frame
            else:
                disp = cv2.cvtColor(cv2.resize(frame, (512,512)), cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(disp)
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            
        self.master.after(20, self.update_video)

    def toggle_live(self):
        if self.live_generation:
            self.live_generation = False
            self.start_button.config(text="Start Phase Lock", bg='#cccccc')
        else:
            self.live_generation = True
            self.start_button.config(text="Stop Phase Lock", bg='#ffaaaa')
            Thread(target=self.live_loop, daemon=True).start()

if __name__ == "__main__":
    root = Tk()
    app = PhaseLockedFlow(root)
    root.mainloop()