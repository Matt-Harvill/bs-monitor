"""Configuration for multi-classifier training."""

from dataclasses import dataclass, field


@dataclass
class MultiClassifierConfig:
    """Configuration for multi-classifier training."""

    # Data paths
    data_dir: str = "/home/matthew/Code/bs-monitor/data"
    output_dir: str = "/home/matthew/Code/bs-monitor/outputs"

    # Model settings
    model_name: str = "facebook/hubert-xlarge-ll60k"
    num_classes: int = 4  # single, multiple, harmonic, none

    # Training settings
    batch_size: int = 16
    accumulation_steps: int = 1  # Gradient accumulation steps
    learning_rate: float = 5e-4
    num_epochs: int = 100
    max_length: float = 2.0  # seconds

    # Data settings
    sample_rate: int = 16000
    duration: int = 2  # chunk duration in seconds

    # Training behavior
    num_workers: int = 4
    eval_steps: int = float("inf")  # Evaluate every N steps
    log_every_n_steps: int = 10  # Log to wandb every N optimizer steps

    # Logging
    log_level: str = "INFO"
    wandb_project: str = "bowel-sound-multi-classifier"
    wandb_run_name: str | None = None

    # Resume training
    resume_from: str | None = None

    # Dataset selection
    datasets: list[str] = field(default_factory=lambda: ["AS_1", "23M74M"])

    # Class mapping
    class_to_idx: dict = field(
        default_factory=lambda: {"none": 0, "single": 1, "multiple": 2, "harmonic": 3}
    )

    idx_to_class: dict = field(
        default_factory=lambda: {0: "none", 1: "single", 2: "multiple", 3: "harmonic"}
    )

    # Class weights for loss calculation (to handle class imbalance)
    class_weights: list[float] | None = None

    def __post_init__(self):
        """Set default class weights based on expected distribution."""
        if self.class_weights is None:
            # Based on the label distribution we observed:
            # AS_1: 69.6% single, 19.3% none, 9.5% multiple, 1.7% harmonic
            # 23M74M: 66% multiple, 27% none, 4% single, 2% harmonic
            # Set weights inversely proportional to frequency
            self.class_weights = [
                1.0,
                1.0,
                1.0,
                3.0,
            ]  # Give more weight to harmonic class
