from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = (
    "Package containing a set of tools to "
    "process and analyze geological microscopic images."
)
LONG_DESCRIPTION = """Python package to work with geological microscopic
images (mainly images of polished sections). Includes modules for panorama
stitching, segmentation, interactive annotation, color adaptation, etc."""

setup(
    name="petroscope",
    version=VERSION,
    author="Alexander Khvostikov",
    author_email="<khvostikov@cs.msu.ru>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    setup_requires=[
        "numpy>=1.16, <2.0.0",
        "pillow",
        "matplotlib",
        "tqdm",
        "scipy",
    ],
    install_requires=[
        "numpy>=1.16, <2.0.0",
        "pillow",
        "matplotlib",
        "tqdm",
        "scipy",
    ],
    python_requires=">=3.10",
    keywords=["python", "first package"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
)
