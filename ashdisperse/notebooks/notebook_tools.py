import subprocess

from pkg_resources import resource_filename


def run_notebook(file):
    subprocess.Popen(["jupyter notebook " + file], shell=True)


def launch_jupyter_example():
    file = resource_filename(__name__, "/ashdisperse.ipynb")
    print(f"Running {file}")
    run_notebook(file)
