from setuptools import setup

PACKAGE_NAME = "elemental"

setup(
    name=PACKAGE_NAME,
    version="0.9.0",
    description=("A tool for producing sound based off on atomic spectra.",),
    author="Mathias Vege",
    license="MIT",
    packages=[
        PACKAGE_NAME,
    ],
    install_requires=[
        "numpy>=1.20.1",
        "scipy>=1.6.1",
        "numba>=0.52.0",
        "tqdm>=4.59.0",
        "click>=7.1.2",
        "pytest>=6.2.3",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.3.4",
            "pydub>=0.25.1",
        ],
    },
    scripts=[
        "scripts/create_elements_video.sh",
        "scripts/generate_dscr.py",
        "scripts/generate_emission_spectra.py",
        "scripts/rename_title.py",
        "scripts/create_videos.sh",
        "scripts/generate_elements_audio.py",
        "scripts/get_viable_elements.py",
    ],
    python_requires=">=3.8",
    setup_requires=["setuptools_scm"],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "elemental=elemental.cli:elemental",
            "rydeberg=elemental.cli:rydeberg",
            "download_spectra=elemental.cli:download_spectra",
        ]
    },
)
