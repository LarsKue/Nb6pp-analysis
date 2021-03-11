
import subprocess

def fetch_single(filepath):
    remote_path = "/work/Tit3/bhusan/Bhusan/2020.05/Run_dat.10.08/sev.83/" + filepath.name

    login = "larskue@kepler2.zah.uni-heidelberg.de"

    subprocess.run(["scp", f"{login}:{remote_path}", str(filepath.resolve())])



