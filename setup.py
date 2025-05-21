from setuptools import setup, find_packages

setup(
    name="survey_gen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    author="Maximilian Kreutner",
    author_email="maximilian.kreutner@uni-mannheim.de",
    description="The SurveyGen Package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)