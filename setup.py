from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]

setup(name='blind-video-watermark',
      python_requires='>=3.6',
      version='0.1.3',
      description='Blind Video Watermarking in Python',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/eluv-io/blind-video-watermark',
      author='Qingyuan Liu',
      author_email='pixelledliu@gmail.com',
      license='MIT',
      packages=find_packages(),
      platforms=['linux', 'windows', 'macos'],
      install_requires=['numpy', 'opencv-python', 'dtcwt', 'tqdm'],
      entry_points= {
          'console_scripts': ['bvw = blind_video_watermark.client:main']
      }
)
