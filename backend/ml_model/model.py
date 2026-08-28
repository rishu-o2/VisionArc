from pathlib import Path


class VisionArcModel:
    """Placeholder model wrapper for diabetic retinopathy inference."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None

    def load(self):
        # Model loading will be implemented once training artifacts are available.
        self.model = None
        return self

    def predict(self, image_tensor):
        raise NotImplementedError("Prediction pipeline is not implemented yet.")
