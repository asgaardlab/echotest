import functools
import hashlib
import json
import os

import scipy
from transformers import AutoProcessor, AutoModel

from phrase_generator import generate_phrases

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")
clip_path = "/Users/igauk/Documents/Unity Projects/subtitle-test-environment/Assets/AudioClips"
expected_path = "/Users/igauk/PycharmProjects/accessibility-toolkit/src/benchmarks/unity"


def speak(text: str, output_path: str):
    inputs = processor(
        text=[text],
        return_tensors="pt",
    )
    speech_values = model.generate(**inputs, do_sample=True)
    sampling_rate = model.generation_config.sample_rate
    if not os.path.isdir(clip_path):
        os.mkdir(clip_path)
    output_wav_path = f"{clip_path}/{output_path}.wav"
    scipy.io.wavfile.write(output_wav_path, rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())


if __name__ == "__main__":
    phrases = generate_phrases(num_phrases=1)
    for phrase_statistics_groups in phrases:
        phrase_statistic_flattened = functools.reduce(lambda x, y: x + y, phrase_statistics_groups, [])
        original_phrase = " ".join([statistics["word"] for statistics in phrase_statistic_flattened])
        phrase_hash = hashlib.md5(original_phrase.encode()).hexdigest()[:10]
        with open(f"{expected_path}/{phrase_hash}-expected.json", "w") as statistics_file:
            statistics_file.write(json.dumps(phrase_statistic_flattened))

        for i, phrase_statistics in enumerate(phrase_statistics_groups):
            modified_phrase = " ".join([statistics["final_word"] for statistics in phrase_statistics if statistics["final_word"]])
            audio_phrase = " ".join([statistics["word"] for statistics in phrase_statistics])
            section_hash = phrase_hash + f"-{i + 1}"
            with open(f"{clip_path}/{section_hash}.txt", "w") as subtitle_file:
                subtitle_file.write(modified_phrase)
            speak(text=audio_phrase, output_path=section_hash)
