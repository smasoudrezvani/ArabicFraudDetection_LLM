from setuptools import setup, find_packages

setup(
    name="arabic_fraud_detection",
    version="1.0.0",
    description="A package for Arabic text fraud detection using pre-trained language models.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/ArabicFraudDetection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "datasets"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
