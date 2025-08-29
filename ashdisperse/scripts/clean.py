import os

from ashdisperse import __file__ as _ad_file


def clean():
    _ad_dirc = os.path.dirname(_ad_file)

    for root, _, files in os.walk(_ad_dirc):
        for file in files:
            if file.endswith(".pyc") or file.endswith(".pyo") or file.endswith(".nbc") or file.endswith(".nbi"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")