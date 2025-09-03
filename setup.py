

import os
from typing import List

from setuptools import find_packages, setup
from setuptools_scm import get_version

ROOT_DIR = os.path.dirname(__file__)
try: 
    VERSION = get_version(write_to="vllm_hymeta/_version.py")
except LookupError:

    VERSION = "0.0.1"

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        with open(get_path("README.md"), encoding="utf-8") as f:
            return f.read()
    else:
        return ""

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    try:
        requirements = _read_requirements("requirements.txt")
    except ValueError:
        print("Failed to read requirements.txt in vllm_hymeta.")
    return requirements

setup(
    name="vllm_hymeta",
    version=VERSION,
    author="FAndromedA",
    license="Apache 2.0",
    description="vllm hymeta backend plugin",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(exclude=["tests*", "examples"]),
    python_requires=">=3.10",
    install_requires=get_requirements(),
    extras_require={},
    entry_points={
        "vllm.platform_plugins": [
            "hymeta = vllm_hymeta:register"
        ],
        "vllm.general_plugins": [
            "hymeta_model = vllm_hymeta:register_model"
        ],
    }
)