from setuptools import setup, find_packages

setup(
    name="my_steering_model",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "inspect_ai.model": [
            "steering = src.steering_model:create_steering_model",
        ],
    },
    install_requires=[
        "inspect_ai",
        "torch",
        "transformers",
    ],
)
