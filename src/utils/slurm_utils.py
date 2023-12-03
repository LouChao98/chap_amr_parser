import subprocess


def cleanup_on_slurm(working_dir):
    p = subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "12.12.12.48",
            f"cd {working_dir} && PATH=$HOME/.local/bin wandb sync",
        ],
        capture_output=True,
    )
    print(p.stdout.decode())
    print("---")
    print(p.stderr.decode())
