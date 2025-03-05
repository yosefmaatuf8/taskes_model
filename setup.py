from setuptools import setup, find_packages

setup(
    name="taskes_model",
    version="1.0.0",
    author="yosef_maatuf",
    author_email="aiagentdaily8@gmail.com",
    description="A project management and task tracking system using AI and Trello API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yosefmaatuf8/taskes_model.git",  # Update with your actual repo URL
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "boto3>=1.14.0",
        "python-dotenv>=0.15.0",
        "pydub>=0.24.0",
        "openai>=0.27.0",
        "tiktoken>=0.2.0",
        "requests>=2.25.1",
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "tqdm>=4.50.0",
        "pyannote.audio>=2.1.1",
        "pandas>=1.2.0",
        "marshmallow>=3.0.0",
        "google-cloud-storage>=2.1.0",
        "ffmpeg-python>=0.2.0",
        "py-trello>=0.19.0",
        "pyyaml>=6.0",
        "setuptools>=58.0.0"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "init_project=init_project:InitProject.setup_environment",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="task management, trello integration, AI automation, project tracking",
)
