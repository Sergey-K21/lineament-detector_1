from setuptools import setup, find_packages

setup(
    name="lineament_detector",
    version="1.0.0",
    description="A Python library for geological lineament detection in images",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "rasterio>=1.3.0",
    ],
)
