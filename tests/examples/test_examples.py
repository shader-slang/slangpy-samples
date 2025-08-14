# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy
import sys
import os
import subprocess
import re
from urllib.parse import urlparse
import http.client
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional

DIR = Path(__file__).parent.absolute()
EXAMPLES_DIR = DIR.parent.parent / "examples"

if sys.platform == "win32":
    DEVICE_TYPES = ["d3d12", "vulkan", "cuda"]
elif sys.platform == "linux" or sys.platform == "linux2":
    DEVICE_TYPES = ["vulkan", "cuda"]
elif sys.platform == "darwin":
    DEVICE_TYPES = ["metal"]
else:
    raise RuntimeError("Unsupported platform")


def check_url_exists(url: str) -> bool:
    parsed_url = urlparse(url)
    connection = None
    try:
        if parsed_url.scheme == "http":
            connection = http.client.HTTPConnection(parsed_url.netloc)
        elif parsed_url.scheme == "https":
            connection = http.client.HTTPSConnection(parsed_url.netloc)
        else:
            return False

        connection.request("HEAD", parsed_url.path or "/")
        response = connection.getresponse()
        return response.status == 200
    except (http.client.HTTPException, OSError):
        return False
    except Exception as e:
        raise ConnectionError(f"Error checking URL: {e}")
    finally:
        if connection:
            connection.close()


def find_urls_in_file(path: Path) -> List[str]:
    # Regular expression to detect URLs
    url_pattern = re.compile(
        r"http[s]?://"  # http:// or https://
        r"(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|"  # Domain name or IP address
        r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"  # Percent-encoded characters
    )

    # Find all matches in the text
    urls = url_pattern.findall(path.read_text())

    # Handle trailing characters
    cleaned_urls = [url.rstrip(").,") for url in urls]

    return cleaned_urls


def test_check_urls():
    # Some URLs are known to be unstable or temporary, so we ignore them
    ignored_urls = ["https://intro-to-restir.cwyman.org/"]
    invalid_urls: dict[Path, List[str]] = defaultdict(list)
    for path in EXAMPLES_DIR.glob("**/*.py"):
        urls = find_urls_in_file(path)
        for url in urls:
            if url in ignored_urls:
                continue
            if not check_url_exists(url):
                invalid_urls[path].append(url)
    assert invalid_urls == {}


def normalize_string(text: str) -> str:
    return "\n".join([line.strip() for line in text.strip().splitlines()])


def is_image_like(array: numpy.ndarray) -> bool:
    # Check if the shape is compatible with image data
    return array.ndim == 2 or (array.ndim == 3 and array.shape[2] <= 4)


def downsample_image_like(array: numpy.ndarray, factor: int = 2) -> numpy.ndarray:
    if not is_image_like(array):
        raise ValueError("Input array is not image-like")

    # Pad image to multiple of factor
    pad_height = (factor - array.shape[0] % factor) % factor
    pad_width = (factor - array.shape[1] % factor) % factor
    if pad_height > 0 or pad_width > 0:
        array = numpy.pad(array, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant")

    # Downsample using box-filter
    array = array.reshape(
        array.shape[0] // factor, factor, array.shape[1] // factor, factor, -1
    ).mean(axis=(1, 3))

    return array


class ExampleRunner:
    def __init__(self, tmp_path_factory: pytest.TempPathFactory):
        super().__init__()
        self.tmp_path_factory = tmp_path_factory

    def run_script(
        self, script_path: Path, device_type: str, capture_frames: list[int]
    ) -> Tuple[str, str, int, dict[str, numpy.ndarray]]:
        # Run the script using it's parent directory as the working directory
        cwd_dir = script_path.parent

        data_path = self.tmp_path_factory.mktemp("data") / "data.npz"
        if data_path.exists():
            data_path.unlink()

        # Prepare the command to execute the script
        command = [
            "python",
            DIR / "wrapper.py",
            script_path,
            data_path,
            ",".join(map(str, capture_frames)),
        ]

        # Run the script
        env = dict(os.environ)
        env["SLANGPY_DEVICE_TYPE_OVERRIDE"] = device_type
        result = subprocess.run(command, cwd=cwd_dir, capture_output=True, text=True, env=env)

        # Capture stdout and return code
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode

        # Load data if available
        if data_path.exists():
            data = numpy.load(data_path)
        else:
            data = {}

        return stdout, stderr, return_code, data

    def run(
        self,
        example: str,
        device_type: str,
        ignore_stdout: bool = False,
        capture_frames: list[int] = [0],
        downsample_factor: int = 1,
        include_data: Optional[List[str]] = None,
        rtol: float = 0.0,
        atol: float = 0.0,
    ):

        script_path = EXAMPLES_DIR / example
        base_name = example.replace(".py", "").replace("/", ".")
        expected_path = DIR / f"{base_name}.expected.txt"
        actual_path = DIR / f"{base_name}.actual.txt"
        expected_data_path = DIR / f"{base_name}.expected.npz"
        actual_data_path = DIR / f"{base_name}.actual.npz"
        diff_data_path = DIR / f"{base_name}.diff.npz"

        (stdout, stderr, return_code, data) = self.run_script(
            script_path, device_type, capture_frames
        )

        assert return_code == 0, f"Script failed with return code {return_code}. stderr: {stderr}"

        # Turn data into proper dictionary and filter out data if include_data is specified
        if include_data is not None:
            data = {key: data[key] for key in include_data if key in data}
        else:
            data = {key: data[key] for key in data.keys()}

        # Downsample image-like tensors if requested
        for key in data.keys():
            if is_image_like(data[key]):
                data[key] = downsample_image_like(data[key], downsample_factor)

        # Compare stdout
        if not ignore_stdout:
            try:
                expected = normalize_string(expected_path.read_text())
            except FileNotFoundError:
                expected = ""
            result = normalize_string(stdout)
            if result != expected:
                actual_path.write_text(result)
            assert result == expected

        # Compare numpy data
        try:
            expected_data = numpy.load(expected_data_path)
        except FileNotFoundError:
            expected_data = {}
        data_equal = False
        if list(data.keys()) == list(expected_data.keys()):
            data_equal = True
            for key in data.keys():
                if data[key].shape != expected_data[key].shape:
                    data_equal = False
                    break
                if data[key].dtype != expected_data[key].dtype:
                    data_equal = False
                    break
                if not numpy.allclose(data[key], expected_data[key], rtol=rtol, atol=atol):
                    data_equal = False
                    break
        if not data_equal:
            with open(actual_data_path, "wb") as actual_data_file:
                numpy.savez(actual_data_file, **data, allow_pickle=False)
                # ensure file is written to disk before assert
                actual_data_file.flush()
            with open(diff_data_path, "wb") as diff_data_file:
                diff_data = {
                    key: data[key] - expected_data[key]
                    for key in data.keys()
                    if key in expected_data
                }
                numpy.savez(diff_data_file, **diff_data)
                # ensure file is written to disk before assert
                diff_data_file.flush()

            # Dump image-like tensors as EXRs
            for key in data.keys():
                if is_image_like(data[key]) and data[key].dtype in [numpy.float32]:
                    from slangpy import Bitmap

                    # Get maximum absolute difference from all diff_data
                    max_diff = numpy.max(numpy.abs(diff_data[key]))
                    print(f"Max absolute difference for {key}: {max_diff}")

                    exr_path = DIR / f"{base_name}.{key}.actual.exr"
                    Bitmap(data[key]).write_async(exr_path)
                    if key in expected_data:
                        exr_path = DIR / f"{base_name}.{key}.expected.exr"
                        Bitmap(expected_data[key]).write_async(exr_path)

        assert list(data.keys()) == list(expected_data.keys())
        for key in data.keys():
            assert numpy.allclose(
                data[key], expected_data[key], rtol=rtol, atol=atol
            ), f'Data mismatch for "{key}"'


@pytest.fixture(scope="module")
def example_runner(tmp_path_factory: pytest.TempPathFactory) -> ExampleRunner:
    return ExampleRunner(tmp_path_factory)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_autodiff(example_runner: ExampleRunner, device_type: str):
    example_runner.run("autodiff/main.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_broadcasting(example_runner: ExampleRunner, device_type: str):
    example_runner.run("broadcasting/main.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_buffers(example_runner: ExampleRunner, device_type: str):
    example_runner.run("buffers/main.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_first_function_numpy(example_runner: ExampleRunner, device_type: str):
    example_runner.run("first_function/main_numpy.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_first_function_scalar(example_runner: ExampleRunner, device_type: str):
    example_runner.run("first_function/main_scalar.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_fwd_rasterizer(example_runner: ExampleRunner, device_type: str):
    example_runner.run(
        "fwd-rasterizer/main.py",
        device_type,
        downsample_factor=16,
        rtol=0.01,
        atol=0.05,
    )


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_generators_grid(example_runner: ExampleRunner, device_type: str):
    example_runner.run("generators/main_grid.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_generators_ids(example_runner: ExampleRunner, device_type: str):
    example_runner.run("generators/main_ids.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_generators_random(example_runner: ExampleRunner, device_type: str):
    example_runner.run("generators/main_random.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_mapping(example_runner: ExampleRunner, device_type: str):
    example_runner.run("mapping/main.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_nested(example_runner: ExampleRunner, device_type: str):
    example_runner.run("nested/main.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_pytorch(example_runner: ExampleRunner, device_type: str):
    # TODO implement
    pytest.skip("Not implemented")


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_ray_casting(example_runner: ExampleRunner, device_type: str):
    example_runner.run(
        "ray-casting/main.py",
        device_type,
        ignore_stdout=True,
        downsample_factor=16,
        rtol=0.01,
        atol=0.05,
    )


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_return_type(example_runner: ExampleRunner, device_type: str):
    example_runner.run("return_type/main.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_sdf_match(example_runner: ExampleRunner, device_type: str):
    # TODO implement
    pytest.skip("Not implemented")


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_signed_distance_field(example_runner: ExampleRunner, device_type: str):
    example_runner.run(
        "signed_distance_field/main.py",
        device_type,
        include_data=[
            "initial_distances_aliased",
            "final_distances_aliased",
            "initial_distances_antialiased",
            "final_distances_antialiased",
        ],
        rtol=1e-3,
        atol=1e-5,
    )


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_simplified_splatting(example_runner: ExampleRunner, device_type: str):
    # TODO implement
    pytest.skip("Example is currently broken and needs fixing (slow performance)")


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_soft_rasterizer(example_runner: ExampleRunner, device_type: str):
    example_runner.run(
        "soft-rasterizer/main.py",
        device_type,
        ignore_stdout=True,
        capture_frames=[0, 150, 300],
        downsample_factor=16,
        rtol=0.01,
        atol=0.05,
    )


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_textures(example_runner: ExampleRunner, device_type: str):
    example_runner.run("textures/main.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_toy_restir(example_runner: ExampleRunner, device_type: str):
    example_runner.run(
        "toy-restir/main.py",
        device_type,
        ignore_stdout=True,
        capture_frames=[0, 200, 400],
        downsample_factor=16,
        rtol=0.01,
        atol=0.05,
    )


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_type_methods(example_runner: ExampleRunner, device_type: str):
    if sys.platform == "linux" or sys.platform == "linux2":
        pytest.skip("Example currently crashes on Linux")
    example_runner.run("type_methods/main.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_type_methods_main_instancelists(example_runner: ExampleRunner, device_type: str):
    if sys.platform == "linux" or sys.platform == "linux2":
        pytest.skip("Example currently crashes on Linux")
    example_runner.run("type_methods/main_instancelists.py", device_type)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_type_methods_extend_instancelists(example_runner: ExampleRunner, device_type: str):
    if sys.platform == "linux" or sys.platform == "linux2":
        pytest.skip("Example currently crashes on Linux")
    example_runner.run("type_methods/extend_instancelists.py", device_type)


if __name__ == "__main__":
    pytest.main([__file__, "-vvvs"])
    # pytest.main([__file__, "-vvvs", "-k", "test_autodiff"])
