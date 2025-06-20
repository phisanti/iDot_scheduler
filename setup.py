from setuptools import setup, find_packages
import os


# Read the README file for long description
def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        return f.read()


setup(
    name="iDot_scheduler",
    version="0.8.2",
    description="A Python tool for automating pipetting worksheets generation for the iDot liquid dispenser",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/phisanti/iDot_scheduler",
    project_urls={
        "Homepage": "https://github.com/phisanti/iDot_scheduler",
        "Repository": "https://github.com/phisanti/iDot_scheduler",
        "Issues": "https://github.com/phisanti/iDot_scheduler/issues",
    },
    license="BSD-3-Clause",
    keywords=["pipetting", "liquid-dispensing", "automation", "worklist"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "iDot_tools": ["resources/instructions.md"],
    },
    install_requires=["gradio", "pandas", "numpy", "openpyxl"],
    entry_points={
        "console_scripts": [
            "idot-scheduler=iDot_tools.cli:main",
        ],
    },
    python_requires=">=3.12",
)
