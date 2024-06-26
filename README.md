# *ECHOTEST* Audio-Subtitle Matching Tool

The *ECHOTEST* Audio-Subtitle Matching Tool is a command-line utility for identifying discrepancies between audio and subtitles in video. In particular, *ECHOTEST* has been designed to identifying discrepancies beetween audio and video in video game gameplay footage.

## Usage
### Prerequisites

- Python (>=3.6)
- Other dependencies:

  ```bash
  pip install -r requirements.txt

## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/asgaardlab/echotest
cd echotest
pip install -r requirements.txt
```

## Running the tool

```python echotest.py --file=path/to/your/video/file.mp4 --darkText=False```

- **file**: Path to the video file for testing. It should have a video extension.
- **darkText**: True if the text is dark on light, can be omitted otherwise

## Benchmarking
All benchmark files needed to automatically generate benchmark videos for *ECHOTEST* are available at [*ECHOTEST* Benchmark GitHub](https://github.com/asgaardlab/echotest-benchmark)