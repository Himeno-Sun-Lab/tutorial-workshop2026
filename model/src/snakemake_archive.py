import os
import sys
import importlib.util
import subprocess

from helpers.snakemake import nested_dict_update, get_git_revision_hash
from default_sim_params import params as sim_params
from default_net_params import params as net_params

conf_name, _ = os.path.splitext(os.path.basename(sys.argv[1]))
conf_path = os.path.join(os.getcwd(), sys.argv[1])
spec = importlib.util.spec_from_file_location(conf_name, conf_path)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

nested_dict_update(net_params, exp.net_params)
nested_dict_update(sim_params, exp.sim_params)

outpath = net_params['outpath']

# Read network hash
with open(sys.argv[2], 'r') as f:
    net_hash = f.read()

# Read simulation hash
with open(sys.argv[3], 'r') as f:
    sim_hash = f.read()

# Read analysis hash
with open(sys.argv[4], 'r') as f:
    ana_hash = f.read()

if net_params["N_scaling"] == 1.0 and net_params["K_scaling"] == 1.0:
    scale = "Full"
else:
    scale = f"n{int(net_params["N_scaling"] * 100)}_k{int(net_params["K_scaling"] * 100)}"


archive_name = "_".join([scale, conf_name])

try:
    # tar压缩
    if not os.path.join(archive_name + ".tar.gz") in os.listdir(outpath):
        subprocess.run([
            "cp",
            "-r",
            os.path.join(outpath, net_hash),
            os.path.join(outpath, archive_name)
        ], check=True)
        subprocess.run([
            "tar", "-czvf", 
            os.path.join(outpath, archive_name + ".tar.gz"),
            "-C",
            outpath,
            archive_name
        ], check=True)
        subprocess.run([
            "rm",
            "-r",
            os.path.join(outpath, archive_name)
        ], check=True)
    # 复制目录
    if not os.path.join(archive_name) in os.listdir(os.path.join(outpath, "plots")):
        subprocess.run([
            "cp", "-r",
            os.path.join(outpath, net_hash, sim_hash, ana_hash, "plots"),
            os.path.join(outpath, "plots", archive_name)
        ], check=True)

    # 写入完成标志
    with open(sys.argv[-1], 'w') as f:
        f.write("Done")

except subprocess.CalledProcessError as e:
    print(f"Command Failed: {e.cmd}, with exit code: {e.returncode}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

# try:
#     compress_exitcode = subprocess.run(["tar", "-czf", os.path.join(outpath, net_hash), os.path.join(outpath, archive_name + ".tar.gz")])
# except:
#     print("")
# try:
#     backup_exitcode = subprocess.run(["cp", "-r", os.path.join(outpath, net_hash, sim_hash, ana_hash, "plots"), os.path.join(outpath, "plots", archive_name)])
# except:
#     print("")
# os.system(f"tar -czf {os.path.join(outpath, net_hash)} {os.path.join(outpath, archive_name + ".tar.gz")}" )
# os.system(f"cp -r {os.path.join(outpath, net_hash, sim_hash, ana_hash, "plots")} {os.path.join(outpath, "plots", archive_name)}")
# with open(sys.argv[-1], 'w') as f:
#     f.write("Done")
