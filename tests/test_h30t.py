from ctypes import sizeof, c_float

import numpy as np

from flirimageextractor import thermal
from flirimageextractor.thermal import Thermal


def test_linux_uses_dji_sdk_17(monkeypatch):
    monkeypatch.setattr(thermal.platform, "system", lambda: "Linux")
    monkeypatch.setattr(thermal.platform, "architecture", lambda: ("64bit", ""))
    monkeypatch.setattr(thermal.Path, "exists", lambda self: True)

    libdirp, libv_dirp, libv_iirp, exiftool = thermal.get_default_filepaths()

    assert "dji_thermal_sdk_v1.7" in libdirp
    assert libdirp.endswith("linux/release_x64/libdirp.so")
    assert libv_dirp.endswith("linux/release_x64/libv_dirp.so")
    assert libv_iirp.endswith("linux/release_x64/libv_iirp.so")
    assert exiftool == "exiftool"


def test_h30t_is_a_supported_dirp2_camera():
    assert Thermal.DJI_H30T == "H30T"
    assert Thermal.DJI_H30T in Thermal.SUPPORTED_CAMERA_MODELS
    assert Thermal.DJI_H30T in Thermal.DJI_DIRP2_CAMERA_MODELS


def test_linux_preloads_h30t_sdk_plugin(monkeypatch, tmp_path):
    library_directory = tmp_path / "dji_thermal_sdk_v1.7" / "linux" / "release_x64"
    library_directory.mkdir(parents=True)
    library_paths = {
        name: library_directory / name
        for name in [
            "libdirp.so",
            "libv_dirp.so",
            "libv_girp.so",
            "libv_hirp.so",
            "libv_iirp.so",
            "libv_cirp.so",
        ]
    }
    for library_path in library_paths.values():
        library_path.touch()

    monkeypatch.setattr(thermal.platform, "system", lambda: "Linux")
    monkeypatch.setattr(thermal.platform, "architecture", lambda: ("64bit", ""))
    monkeypatch.setattr(
        thermal,
        "get_default_filepaths",
        lambda: (
            str(library_paths["libdirp.so"]),
            str(library_paths["libv_dirp.so"]),
            str(library_paths["libv_iirp.so"]),
            "exiftool",
        ),
    )
    loaded_paths = []

    class FakeFunction:
        def __call__(self, *args):
            return Thermal.DIRP_SUCCESS

    class FakeLibrary:
        def __getattr__(self, name):
            return FakeFunction()

    def fake_cdll(path, **kwargs):
        loaded_paths.append(path)
        return FakeLibrary()

    monkeypatch.setattr(thermal, "CDLL", fake_cdll)

    Thermal()

    assert str(library_paths["libv_hirp.so"]) in loaded_paths
    assert loaded_paths.index(str(library_paths["libv_hirp.so"])) < loaded_paths.index(
        str(library_paths["libdirp.so"])
    )


def test_h30t_metadata_routes_to_dirp2(monkeypatch, tmp_path):
    image_path = tmp_path / "DJI_0001_R.JPG"
    image_path.write_bytes(b"rjpeg")
    metadata = b"\n".join([
        b"Camera Model Name : H30T",
        b"Image Height : 1024",
        b"Image Width : 1280",
        b"Emissivity : 95",
    ])

    class FakePopen:
        def __init__(self, *args, **kwargs):
            pass

        def communicate(self):
            return metadata, b""

    monkeypatch.setattr(thermal.subprocess, "Popen", FakePopen)
    parser = Thermal.__new__(Thermal)
    parser._filepath_exiftool = "exiftool"
    parser._support_camera_model = set(Thermal.SUPPORTED_CAMERA_MODELS)
    captured = {}

    def fake_parse_dirp2(**kwargs):
        captured.update(kwargs)
        return np.zeros((1024, 1280), dtype=np.float32)

    parser.parse_dirp2 = fake_parse_dirp2

    result = parser.parse(str(image_path))

    assert result.shape == (1024, 1280)
    assert captured["m2ea_mode"] is True
    assert captured["emissivity"] == 0.95
    assert "image_width" not in captured
    assert "image_height" not in captured


def test_dirp2_uses_resolution_reported_by_sdk(tmp_path):
    image_path = tmp_path / "DJI_0001_R.JPG"
    image_path.write_bytes(b"rjpeg")
    parser = Thermal.__new__(Thermal)
    parser._dtype = np.float32
    measured_sizes = []

    parser._dirp_create_from_rjpeg = lambda raw, raw_size, handle: Thermal.DIRP_SUCCESS
    parser._dirp_get_rjpeg_version = lambda handle, version: Thermal.DIRP_SUCCESS

    def get_resolution(handle, resolution):
        resolution.width = 1280
        resolution.height = 1024
        return Thermal.DIRP_SUCCESS

    def measure(handle, data, data_size):
        measured_sizes.append(data_size.value)
        return Thermal.DIRP_SUCCESS

    parser._dirp_get_rjpeg_resolution = get_resolution
    parser._dirp_measure_ex = measure
    parser._dirp_destroy = lambda handle: Thermal.DIRP_SUCCESS

    result = parser.parse_dirp2(str(image_path), m2ea_mode=True)

    assert result.shape == (1024, 1280)
    assert measured_sizes == [1280 * 1024 * sizeof(c_float)]
