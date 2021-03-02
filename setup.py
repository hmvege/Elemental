from setuptools import setup

PACKAGE_NAME = "elemental"

setup(
    name=PACKAGE_NAME,
    version=0.1,
    description=(
        "A tool for producing sound based off on atomic spectra.",
    ),
    author="Mathias Vege",
    license="MIT",
    packages=[
        PACKAGE_NAME,
    ],
    install_requires=[
        "numpy>=1.20.1",
        "scipy>=1.6.1",
        "matplotlib>=3.3.4",
        "numba>=0.52.0",
        "tqdm>=4.59.0",
    ],
    python_requires=">=3.8",
    setup_requires=["setuptools_scm"],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "elemental=elemental.cli:elemental",
            "rydeberg=elemental.cli:rydeberg",
        ]
    },
)
