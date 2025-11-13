import logging
from pathlib import Path

try:
    import torch  # pyright: ignore[reportMissingImports]
    from transformers import Wav2Vec2Processor, Wav2Vec2Model  # pyright: ignore[reportMissingImports]
except ImportError as e:
    logging.error(e)
    raise

import numpy as np
import librosa

from skactiveml_annotation.core.shared_types import DashProgressFunc
from .base import (
    relative_to_root,
    EmbeddingBaseAdapter
)

class Wav2Vec2EmbeddingAdapter(EmbeddingBaseAdapter):

    def __init__(
        self, 
        model_name: str = "facebook/wav2vec2-base", 
        batch_size: int = 8,
        sample_rate: float | int = 16000 # Default for Speech recognition
    ):
        """
        Initialize the Wav2Vec2 embedding adapter.

        Args:
            model_name (str): Hugging Face model name
            device (str): 'cuda' or 'cpu'. Defaults to CUDA if available.
            batch_size (int): Number of audio samples per batch
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.sample_rate = sample_rate

    def compute_embeddings(
        self, 
        data_path: Path, 
        progress_func: DashProgressFunc
    ) -> tuple[np.ndarray, list[Path]]:
        file_paths = sorted([p for p in data_path.iterdir() if p.suffix.lower() == ".wav"])
        embeddings = []

        # Load all audio first
        logging.info("load all audio waveforms ...")

        audio_list = []
        sampling_rate = self.sample_rate
        for path in file_paths:
            # To preserve sampling rate None can be passed but that assumes all
            # sampels have the same sampling rate
            waveform, sampling_rate = librosa.load(path, sr=sampling_rate, mono=True)
            audio_list.append(waveform)


        logging.info("start preprocessing and embedding")

        # Progress tracking
        update_step = 100
        total_files = len(audio_list)
        processed_count = 0
        next_report_value = update_step

        for i in range(0, len(audio_list), self.batch_size):
            batch_waveforms = audio_list[i:i+self.batch_size]

            # Prepare batch inputs with padding
            # Adding padding for variable lenght samples and convert to tensor
            inputs = self.processor(
                batch_waveforms, 
                sampling_rate=sampling_rate, 
                return_tensors="pt", 
                padding=True
            )

            # PyTorch models require that all inputs are on the same device as the model.
            # If your model is on the GPU, but your input tensors are on the CPU, PyTorch will throw an error.
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad(): # for inference disable gradient tracking. It is not needed
                outputs = self.model(**inputs)
                last_hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

                # Pool over time dimension for each sample
                batch_embeddings = last_hidden_states.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)

            # update progress count
            processed_count += len(batch_embeddings)

            if processed_count >= next_report_value:
                percent = (processed_count / total_files) * 100
                logging.info(f"{processed_count}/{total_files} samples embedded ({percent:.2f}%)")
                progress_func(percent)
                next_report_value += update_step

        embeddings = np.vstack(embeddings)

        logging.info("Final embedding matrix shape:", embeddings.shape)

        relative_paths = [relative_to_root(p) for p in file_paths]

        return embeddings, relative_paths
