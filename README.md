# blind-video-watermark
Blind video watermarking in DT CWT domain.

[![PyPI](https://img.shields.io/pypi/v/blind-video-watermark)](https://pypi.org/project/blind-video-watermark/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/eluv-io/blind-video-watermark/LICENSE)

# Installation
```bash
pip install blind-video-watermark
```
For the latest in-development version:
```bach
git clone https://github.com/eluv-io/blind-video-watermark
cd blind-video-watermark
pip install .
```

# Getting Started
## Embed images
Check the [example](https://github.com/eluv-io/blind-video-watermark/blob/main/examples/example_dtcwt_img.py)
```python
import blind_video_watermark as bvw

wm_path = "wms/wm.jpg"
video_path = "videos/bbb-short.mp4"
output_path = "output/output.mp4"
extracted_folder = "output/extracted"
# Embed watermark
bvw.DtcwtImgEncoder().embed_video(wm_path, video_path, output_path)
# Extract watermark
bvw.DtcwtImgDecoder().extract_video(output_path, extracted_folder)
```
## Concurrency
```python
# Ensure "__name__" == "__main__"
bvw.DtcwtEncoder().embed_video_async(wm_path, video_path, output_path)
```
