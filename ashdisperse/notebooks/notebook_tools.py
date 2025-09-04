import subprocess

import importlib_resources


def run_notebook(file):
    subprocess.Popen(["jupyter notebook " + file], shell=True)


def launch_jupyter_example():
    ref = importlib_resources.files('ashdisperse.notebooks') / 'ashdisperse.ipynb'
    with importlib_resources.as_file(ref) as path:
        print(f"Running {path}")
        run_notebook(path)
        
