"""
Main module
"""
import argparse
import os
from typing import List

import cv2
import librosa
from moviepy.editor import VideoFileClip

from src.models.audio_clip import AC
from src.models.audio_subtitle_match.audio_subtitle_match import AudioSubtitleMatch
from src.models.processing_pipeline import ProcessingPipeline
from src.models.text_comparison_token import TextComparisonToken
from src.models.timestamped_content.timestamp_content import TimestampedContent, merge_sequential_timestamps
from src.service.optical_character_recognizer import OpticalCharacterRecognizer
from src.service.speech_recognizer import SpeechRecognizer
from src.service.tokenization import tokenize

ocr = OpticalCharacterRecognizer()
speech_recognizer = SpeechRecognizer("openai/whisper-medium")


def extract_subtitles(video_filepath: str, processing_pipeline: ProcessingPipeline) -> str:
    """
    Extracts all subtitle text from the video at the given filepath
    :param processing_pipeline: group transformations on pipeline are applied prior to ocr extraction
    :param video_filepath: Video filepath to extract subtitles from
    :return: Joined subtitle string
    """
    frames: List[TimestampedContent] = _extract_frames(video_filepath)
    captured_texts = []
    images = [frame.content for frame in frames]

    transformed = processing_pipeline.apply_on_images(images)

    for image in transformed:
        text = ocr.extract_text(processing_pipeline.apply_on_image(image)).strip()
        if text not in captured_texts:
            captured_texts.append(text)
    return ' '.join(captured_texts)


def write_audio_to_file(video_filepath: str, output_path: str):
    """
    Extracts audio from a video file and writes it to an audio file.

    :param video_filepath: Path to the video file.
    :param output_path: Path to the output audio file.
    """
    audio = VideoFileClip(video_filepath).audio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio.write_audiofile(output_path, codec='mp3')


def extract_audio(video_filepath: str):
    """
    :param video_filepath:
    """
    return AC(*librosa.load(video_filepath))


def audio_subtitle_match(subtitle_text: str,
                         audio_text: str,
                         speaker_indication=":",
                         window_size=5) -> List[AudioSubtitleMatch]:
    """
    From the subtitle and audio text extracted from a video, compares the two using metaphone approximations to provide
    a matching score.

    :param subtitle_text: Text extracted from the subtitles of a video
    :param audio_text: Text extracted from the audio of a video
    :param speaker_indication: In subtitles, this suffix indicates that the previous word indicates a speaker
    as such this word is not considered when matching audio and subtitles
    :param window_size: Represents the number of words further into the subtitles to find the perfect match
    to account for noise
    :return: Scores by audio text word, 1.0 represents a perfect match None is no match
    """
    cleaned_audio = tokenize(audio_text, split_contractions=False, segment=False).split()
    subtitle_text = " ".join([word for word in subtitle_text.split() if not word.endswith(speaker_indication)])
    cleaned_subtitles = tokenize(subtitle_text, split_contractions=False, segment=False).split()
    audio_tokens = [TextComparisonToken(word) for word in cleaned_audio]
    subtitle_tokens = [TextComparisonToken(word) for word in cleaned_subtitles]
    similarity_scores = []
    subtitle_index = 0

    for audio_token in audio_tokens:
        extension = 0
        match = AudioSubtitleMatch(audio_token.word, None, 0.0)
        best_extension = 0
        while extension < window_size and (subtitle_index + extension) < len(subtitle_tokens):
            subtitle_considered = subtitle_tokens[subtitle_index + extension]
            current_score = audio_token.compare(subtitle_considered)
            if match.score or current_score and extension == window_size - 1:
                audio_index = audio_tokens.index(audio_token)
                remaining_audio_words = [token.word for token in
                                         audio_tokens[audio_index + 1:audio_index + window_size]]
                passed_subtitles = [token.word for token in subtitle_tokens[subtitle_index:subtitle_index + extension]]
                full_matches_skipped = [subtitle for subtitle in passed_subtitles if subtitle in remaining_audio_words]
                if passed_subtitles and len(full_matches_skipped) / len(passed_subtitles) >= 0.5:
                    # We assume that any match is invalid here since there is evidence that there are matches we ignored
                    match.score = 0.0
                    match.subtitle_token = None
                    best_extension = 0
                    break
            if current_score > match.score:
                match.score = current_score
                match.subtitle_token = subtitle_considered.word
                best_extension = extension
            if match.score == 1.0 and extension == 0:
                break
            extension += 1
        subtitle_index += best_extension + 1 if match.subtitle_token else 0
        similarity_scores.append(match)
    return similarity_scores


def run_audio_subtitle_match(file_name: str, processing_pipeline: ProcessingPipeline):
    """
    Extracts audio and subtitles then checks for discrepancies
    :param file_name: file to extract audio and subtitles from (mp4)
    :param processing_pipeline: Image pre-processing pipeline
    """
    ocr_text = extract_subtitles(file_name, processing_pipeline)
    decoded_speech = merge_sequential_timestamps(speech_recognizer.decode(extract_audio(file_name))).content
    return audio_subtitle_match(ocr_text, decoded_speech)


def main():
    """
    ECHOTEST Audio-Subtitle Matching Tool.

    Command-line tool for matching audio and subtitles in a video file.

    Usage:
    python your_script_name.py path/to/your/video/file.mp4
    """
    parser = argparse.ArgumentParser(description="ECHOTEST Audio-Subtitle Matching Tool")
    parser.add_argument("file", help="Path to the video file for testing")
    parser.add_argument("--darkText", help="If subtitle text is dark on light", default=True, required=False)

    args = parser.parse_args()
    file_path = args.file
    is_dark = args.darkText

    integration_pipeline = ProcessingPipeline() \
        .set_reduction_step(ocr.mean_fill_reduce, is_dark) \
        .add_step(ocr.crop_image) \
        .add_step(ocr.canny_edge_detection_text_fill, is_dark)

    pipeline = ProcessingPipeline() \
        .add_group_step(ocr.set_thresholds_from, is_dark) \
        .add_group_step(ocr.integrate_frames, integration_pipeline)

    for result in run_audio_subtitle_match(file_path, pipeline):
        print(result)


def _get_video_capture(video_filepath) -> cv2.VideoCapture:
    video_capture = None
    if not os.path.exists(video_filepath):
        print("Video path does not exist")
    try:
        video_capture = cv2.VideoCapture()
        video_capture.setExceptionMode(True)
        video_capture.open(video_filepath)
        video_capture.setExceptionMode(False)
    except cv2.error:
        print("Cannot open video")
    return video_capture


def _extract_frames(video_filepath: str, sampling_rate_ms=500):
    video_capture = cv2.VideoCapture(video_filepath)
    keyframes = []
    key_frame_index = 1
    video_capture.set(cv2.CAP_PROP_POS_MSEC, sampling_rate_ms * key_frame_index)
    read, frame = video_capture.read()
    while read:
        keyframe = TimestampedContent(
            timestamp=(sampling_rate_ms * (key_frame_index - 1) / 1000, sampling_rate_ms * key_frame_index / 1000),
            content=frame
        )
        keyframes.append(keyframe)
        key_frame_index += 1
        video_capture.set(cv2.CAP_PROP_POS_MSEC, sampling_rate_ms * key_frame_index)
        read, frame = video_capture.read()
    return keyframes


if __name__ == '__main__':
    main()
