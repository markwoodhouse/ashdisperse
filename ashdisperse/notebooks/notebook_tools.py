import subprocess
from importlib import resources


def run_notebook(file):
    subprocess.run(["jupyter", "notebook", file.as_posix()])


def launch_jupyter_example():
    ref = resources.files('ashdisperse.notebooks') / 'ashdisperse.ipynb'
    with resources.as_file(ref) as path:
        print(f"Running {path}")
        run_notebook(path)
        
