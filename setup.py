from setuptools import setup, find_packages

setup(
    name="satquery",
    version="0.1.0",
    description="Agentic vision-language assistant for remote sensing image analysis",
    packages=find_packages(include=["satquery", "satquery.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pydantic>=2.0.0",
        "Pillow>=8.0.0",
    ],
)
