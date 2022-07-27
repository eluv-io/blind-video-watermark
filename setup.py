from setuptools import setup, find_packages
from os import path as os_path

this_directory = os_path.abspath(os_path.dirname(__file__))

def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]

setup(name='blind-video-watermark',
      python_requires='>=3.6',
      version='0.1.0',
      description='Blind Video Watermarking in Python',
      # long_description=read_file('docs/en/README.md'),
      # long_description_content_type="text/markdown",
      url='https://github.com/eluv-io/blind-video-watermark',
      author='Qingyuan Liu',
      author_email='pixelledliu@gmail.com',
      license='MIT',
      packages=find_packages(),
      platforms=['linux', 'windows', 'macos'],
      install_requires=['numpy', 'opencv-python', 'dtcwt', 'tqdm'],
)
