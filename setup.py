from setuptools import setup, find_packages

setup(
    name="iDot_scheduler",
    version="0.6",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "iDot_tools": ["resources/instructions.md"],
    },
    install_requires=[
        "gradio",
        "pandas",
        "numpy",
        "openpyxl"
    ],
    entry_points={
        "console_scripts": [
            "idot-scheduler=iDot_tools.cli:main",
        ],
    },
    python_requires=">=3.9",
)