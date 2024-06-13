import re
import setuptools
from datetime import datetime
from distutils.cmd import Command
from pathlib import Path

init_file_path = "forecaster/__init__.py"


def read(rel_path):
    with open(Path(__file__).parent / rel_path, "r") as fp:
        return fp.read()


def write(rel_path, content):
    with open(Path(__file__).parent / rel_path, "w") as fp:
        fp.write(content)


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


class BumpDevVersion(Command):
    description = "Bump to dev version"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        raw_content = read(init_file_path)
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        pattern = r"__version__\s*=\s*(\'|\")([\d\.]+?)(\.(dev|a|b|rc).*?)?(\'|\")"
        repl = re.sub(pattern, r"__version__ = '\2.dev" + current_time + "'", raw_content)
        write(init_file_path, repl)
        print(f"Version bumped to {get_version(init_file_path)}")


def setup():
    setuptools.setup(
        name="forecaster",
        version=get_version(init_file_path),
        author="Forecaster team",
        author_email="forecaster@microsoft.com",
        description="A toolkit for time-series forecasting tasks",
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        url="https://dev.azure.com/MSForecast/_git/Forecaster",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.8",
        install_requires=[
            "torch",
            "protobuf==4.23",
            "utilsd",
            "click",
            "sktime",
            "tqdm",
            "tensorboard",
            "numba",
            "numpy",
            "scikit_learn==1.3",
            "pandas==2.1",
        ],
        extras_require={"dev": ["flake8", "pytest", "pytest-azurepipelines", "pytest-cov", "pylint"]},
        cmdclass={"bumpver": BumpDevVersion},
    )


if __name__ == "__main__":
    setup()
