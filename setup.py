from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="davil",
    version="0.1.0",
    author="Mehrish",
    description="DA-VIL: Adaptive Dual-Arm Manipulation with RL and Variable Impedance Control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mehr1sh/dual-arm-variable-impedance-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
