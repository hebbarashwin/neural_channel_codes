from setuptools import setup, find_packages

setup(
    name='deepcommpy',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.8.4',
        'numpy==1.21.5',
        'six==1.16.0',
        'torch==2.3.0',
        'tqdm==4.66.4'
    ],
    entry_points={
        'console_scripts': [
            'deeppolar-train=neural_channel_codes.deeppolar:main',
            # Add other entry points if necessary
        ],
    },
)
