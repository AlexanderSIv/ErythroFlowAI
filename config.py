from dataclasses import dataclass, field


# Соответствие «человеческое имя формата → фактические расширения файлов»
FRAME_FORMAT_ALIASES: dict[str, tuple] = {
    ".tif":  (".tif", ".tiff"),
    ".png":  (".png",),
    ".jpeg": (".jpeg", ".jpg"),
}


@dataclass
class Config:
    # --- Параметры трекинга и анализа (значения под текущее оборудование) ---
    traj_sensitivity: int = 30
    capture_sensitivity: int = 50
    channels_area_coeff: float = 60 / 1.5

    # --- Сегментация ---
    seg_input_size: tuple = (1920, 160)   # (W, H), как в DataForSegment
    seg_batch_size: int = 1

    # --- Патчи траекторий и классификация ---
    patch_size: int = 31
    num_classes: int = 3
    class_names: dict = field(default_factory=lambda: {
        0: "relaxed",
        1: "not relaxed",
        2: "outlets",
    })

    # --- Формат кадров ---
    frame_ext: str = ".tif"                        # выбранный в UI
    frame_exts: tuple = FRAME_FORMAT_ALIASES[".tif"]  # фактические расширения

    # --- Пути ---
    temp_dir: str = "temp"
    outputs_dir: str = "outputs"
    weights_dir: str = "weights"
    seg_weights: str = "weights/unet.safetensors"
    clf_weights: str = "weights/cnnlstm.safetensors"

    # --- Устройство ---
    device: str = "cuda"

    def set_frame_ext(self, ext: str):
        """Устанавливает формат и связанный кортеж расширений."""
        if ext not in FRAME_FORMAT_ALIASES:
            raise ValueError(f"Unknown frame format: {ext}")
        self.frame_ext = ext
        self.frame_exts = FRAME_FORMAT_ALIASES[ext]