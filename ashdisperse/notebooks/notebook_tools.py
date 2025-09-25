import subprocess
from importlib import resources


def run_notebook(file):
    subprocess.Popen(["jupyter notebook " + file], shell=True)


def launch_jupyter_example():
    ref = resources.files('ashdisperse.notebooks') / 'ashdisperse.ipynb'
    with resources.as_file(ref) as path:
        print(f"Running {path}")
        run_notebook(path)
        
