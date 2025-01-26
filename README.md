# Flir Image Extractor CLI

The email address attached to this on PyPi may not be monitored, open issues on the [GitHub repo](https://github.com/nationaldronesau/FlirImageExtractor) to ensure a response

Feel free to submit any pull requests or issues, this is in active development. Also let me know if you are successful in using this on cameras not listed below.

FLIR® thermal cameras like the FLIR ONE® include both a thermal and a visual light camera.
The latter is used to enhance the thermal image using an edge detector. The resulting image is saved as a
jpg image but both the original visual image and the raw thermal sensor data are embedded in the jpg metadata.

This Python CLI that allows you to extract the original photo and thermal sensor values converted to temperatures, normalize the temperature range and output the photos to different color maps.

## Requirements and Install

To install and Run on Linux and windows if repo is cloned:

```bash
poetry install
poetry run flir-image-extractor
```

You can install the CLI using pip:

```bash
pip install flirimageextractor or pip install --upgrade flirimageextractor (windows)
```

## Usage: Sample Basic code Snppet

```
import numpy as np
from flirimageextractor import FlirImageExtractor

flir = FlirImageExtractor()
flir.process_image(flir_img_file='images/DJI_H20T.jpg')
# Save processed images and extract thermal data
images = flir.save_images(bytesIO=True)
temperature = flir.get_thermal_np()
# Calculate and upload thermal data
max_temp, min_temp = np.amax(temperature), np.amin(temperature)
print(max_temp)
print(min_temp)
```


#### Resulting Plot and Saved Images
The CLI is able to output 3 folders of images with the `bwr`, `gnuplot`, and `gist_ncar` colormaps from matplotlib. You can define the pallete(s) that you would rather use.

## Supported/Tested Cameras

- FLIR R-JPEG Camera Model
    - FLIR
    - FLIR AX8
    - FLIR B60
    - FLIR E40
    - FLIR T640

- DJI R-JPEG Camera Model
    - DJI H20T
    - DJI XT2
    - DJI XTR
    - DJI XTS
    - DJI R-JPEG Camera Model DTAT3.0

- DJI M2EA / DJI MAVIC2-ENTERPRISE-ADVANCED
    - DJI H20N
    - DJI M3T / DJI MAVIC3
    - DJI M30T

Other cameras might need some small tweaks (the embedded raw data can be in multiple image formats). Let me know if you succesfully use other cameras so they can be added to this list.

## Development
Install the required packages using [Poetry](https://python-poetry.org//). 
Note that this tool is intended to work on Windows as well as Unix operating systems so use os.path functions to manipulate file paths instead of string manipulation.

## Build Command for Dev (uses poetry or twine)
- python -m build --sdist --wheel
- poetry build
- poetry add "packagename" --group=extras --optional
- sphinx-build -b html docs/source docs/build (To Generate Docs)

## Credits

This CLi was developed using this repos:
- https://github.com/Nervengift/read_thermal.py
- https://github.com/detecttechnologies/thermal_base/
- https://github.com/SanNianYiSi/thermal_parser/
