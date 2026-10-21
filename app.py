"""GUI на CustomTkinter для запуска пакетной обработки видео.

Сценарий:
  1. Пользователь выбирает папку, внутри которой лежат подпапки с кадрами.
  2. Выбирает формат кадров (.tif / .png / .jpeg), по умолчанию .tif.
  3. Нажимает «Запустить» — обработка идёт в фоновом потоке.
  4. Верхний прогресс-бар показывает, что текущее видео обрабатывается
     (indeterminate), нижний — общий прогресс по видео (determinate).
  5. По завершении — сообщение об успехе и кнопка «Завершить работу».
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

from config import Config, FRAME_FORMAT_ALIASES
from pipeline import ProcessVideo, load_seg_model, load_clf_model


FRAME_FORMATS = list(FRAME_FORMAT_ALIASES.keys())   # [".tif", ".png", ".jpeg"]


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ErythroFlowAI")
        self.geometry("620x480")
        self.resizable(False, False)

        self.cfg = Config()
        self.folder = tk.StringVar(value="Папка не выбрана")
        self.status = tk.StringVar(value="")
        self.frame_ext = tk.StringVar(value=self.cfg.frame_ext)

        # --- Выбор папки ---
        ctk.CTkLabel(self, text="Выберите папку с видео",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(20, 8))
        ctk.CTkLabel(self, textvariable=self.folder, wraplength=560).pack(pady=4)

        self.btn_pick = ctk.CTkButton(self, text="Выбрать папку", command=self.pick)
        self.btn_pick.pack(pady=8)

        # --- Выбор формата кадров ---
        fmt_frame = ctk.CTkFrame(self, fg_color="transparent")
        fmt_frame.pack(pady=6)
        ctk.CTkLabel(fmt_frame, text="Формат кадров:").pack(side="left", padx=(0, 8))
        self.fmt_menu = ctk.CTkOptionMenu(
            fmt_frame, values=FRAME_FORMATS, variable=self.frame_ext, width=120,
            command=self._on_ext_change,
        )
        self.fmt_menu.pack(side="left")

        self.btn_run = ctk.CTkButton(self, text="Запустить", command=self.start,
                                     state="disabled")
        self.btn_run.pack(pady=10)

        # --- Два прогресс-бара ---
        ctk.CTkLabel(self, text="Текущее видео:").pack(anchor="w", padx=60)
        self.progress_current = ctk.CTkProgressBar(self, width=500, mode="indeterminate")
        self.progress_current.set(0)
        self.progress_current.pack(pady=(2, 8))

        ctk.CTkLabel(self, text="Общий прогресс:").pack(anchor="w", padx=60)
        self.progress_total = ctk.CTkProgressBar(self, width=500, mode="determinate")
        self.progress_total.set(0)
        self.progress_total.pack(pady=(2, 10))

        ctk.CTkLabel(self, textvariable=self.status).pack(pady=4)

        self.btn_quit = ctk.CTkButton(self, text="Завершить работу",
                                      fg_color="#a33", hover_color="#822",
                                      command=self.destroy, state="disabled")
        self.btn_quit.pack(pady=12)

    # ----------------------------- обработчики -----------------------------

    def pick(self):
        path = filedialog.askdirectory()
        if path:
            self.folder.set(path)
            self._update_run_button()

    def _on_ext_change(self, value):
        self.frame_ext.set(value)
        self._update_run_button()

    def _update_run_button(self):
        ok = (os.path.isdir(self.folder.get())
              and self.frame_ext.get() in FRAME_FORMAT_ALIASES)
        self.btn_run.configure(state="normal" if ok else "disabled")

    def start(self):
        folder = self.folder.get()
        if not os.path.isdir(folder):
            return
        self.cfg.set_frame_ext(self.frame_ext.get())
        self.btn_pick.configure(state="disabled")
        self.fmt_menu.configure(state="disabled")
        self.btn_run.configure(state="disabled")
        self.status.set("Загрузка моделей...")
        self.progress_total.configure(mode="determinate")
        self.progress_total.set(0)
        threading.Thread(target=self._worker, args=(folder,), daemon=True).start()

    # ----------------------------- воркер -----------------------------

    def _worker(self, folder):
        try:
            seg = load_seg_model(self.cfg)
            clf = load_clf_model(self.cfg)
            videos = [d for d in sorted(os.listdir(folder))
                      if os.path.isdir(os.path.join(folder, d))
                      and self._has_frames(os.path.join(folder, d))]
            total = len(videos)
            if total == 0:
                self.after(0, lambda: messagebox.showwarning(
                    "Пусто", f"В папке нет видео с кадрами {self.cfg.frame_ext}"))
                self.after(0, self._reset)
                return

            for i, name in enumerate(videos, 1):
                self.after(0, lambda n=name, i=i, t=total: self._set_current(n, i, t))
                self.after(0, self.progress_current.start)
                ProcessVideo(os.path.join(folder, name), self.cfg, seg, clf)
                self.after(0, self.progress_current.stop)
                self.after(0, self._set_total, i, total)

            self.after(0, lambda: self.status.set(
                f"Обработка завершена: {total} видео"))
            self.after(0, lambda: messagebox.showinfo(
                "Готово", "Обработка завершена успешно"))
            self.after(0, lambda: self.btn_quit.configure(state="normal"))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
            self.after(0, lambda: self.status.set("Ошибка обработки"))
        finally:
            self.after(0, self._reset)

    def _has_frames(self, path):
        exts = self.cfg.frame_exts
        return any(f.endswith(exts) for f in os.listdir(path))

    def _set_current(self, name, i, total):
        self.status.set(f"Видео {i} из {total}: {name}")

    def _set_total(self, done, total):
        self.progress_total.set(done / total)
        self.status.set(f"Обработано {done} из {total}")

    def _reset(self):
        self.btn_pick.configure(state="normal")
        self.fmt_menu.configure(state="normal")
        self._update_run_button()


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()