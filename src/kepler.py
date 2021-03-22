
import subprocess

remote_path = "/work/Tit3/bhusan/Bhusan/2020.05/Run_dat.10.08/sev.83/"
login = "larskue@kepler2.zah.uni-heidelberg.de"


def fetch_single(filepath):
    subprocess.run(["scp", f"{login}:{remote_path + filepath.name}", str(filepath.resolve())])
