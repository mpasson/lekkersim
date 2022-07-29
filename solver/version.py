import subprocess
import os

# To modify this file you have to (also) change the make-package script.
mdir = os.path.dirname(os.path.abspath(__file__))

try:
    subprocess.check_output(["git", "--version"], cwd=mdir).strip().decode("utf-8")
    git = True
except subprocess.CalledProcessError:
    git = False


def get_version():
    if not git:
        return 'no version available', True
    version = (
        subprocess.check_output(["git", "describe"], cwd=mdir).strip().decode("utf-8")
        + "-dev"
        )
    git_clean = (
        subprocess.check_output(["git", "diff-index", "HEAD"], cwd=mdir)
        .strip()
        .decode("utf-8")
        == ""
    )
    return version, git_clean


__version__, git_clean = get_version()

