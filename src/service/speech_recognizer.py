from typing import List

import torch
from transformers import pipeline

from src.models.audio_clip import AC
from src.models.timestamped_content.timestamp_content import TimestampedContent, from_dict


class SpeechRecognizer:
    """
    Model used to analyze and decode speech in audio
    """

    def __init__(self, model="openai/whisper-base"):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=30,
            device=device,
        )

    def decode(self, audio_file: AC) -> List[TimestampedContent]:
        """
        Extracts the transcription of an audio file
        """
        return from_dict(self.pipe(audio_file.array, batch_size=8, return_timestamps=True)["chunks"], "text")
