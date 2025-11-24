from pathlib import Path

from setuptools import setup


def read_version() -> str:
    for line in Path("app.py").read_text().splitlines():
        if line.startswith("__version__"):
            return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to determine version from app.py")


setup(
    name="fedmed-pl-superlink",
    version=read_version(),
    description="Flower-based coordinator ChRIS plugin for the FedMed demo",
    author="FedMed BU Team",
    author_email="rpsmith@bu.edu",
    url="https://github.com/EC528-Fall-2025/FedMed-ChRIS",
    py_modules=["app"],
    packages=["fedmed_flower_app"],
    package_dir={"fedmed_flower_app": "fedmed_flower_app/fedmed_flower_app"},
    include_package_data=True,
    package_data={"fedmed_flower_app": ["pyproject.toml"]},
    install_requires=[
        "chris_plugin==0.4.0",
        "flwr>=1.23.0,<2",
        "numpy>=1.26,<3",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "medmnist>=3.0.1",
    ],
    license="MIT",
    entry_points={
        "console_scripts": [
            "fedmed-pl-superlink = app:main",
        ]
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    extras_require={"none": [], "dev": []},
)
