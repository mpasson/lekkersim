import subprocess
import os

# To modify this file you have to (also) change the make-package script.
mdir = os.path.dirname(os.path.abspath(__file__))

try:
    __version__ = (
        subprocess.check_output(["git", "describe"], cwd=mdir).strip().decode("utf-8")
        + "-dev"
    )
    git_clean = (
        subprocess.check_output(["git", "diff-index", "HEAD"], cwd=mdir)
        .strip()
        .decode("utf-8")
        == ""
    )
except subprocess.CalledProcessError:
    __version__ = "No version available"
    git_clean = True
