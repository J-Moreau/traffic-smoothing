import subprocess
import tempfile

import wandb


def log_run(conf, metrics, name="kalman"):
    with wandb.init(project=name) as run:
        wandb.config.update(conf)
        run.log(metrics)
        try:
            git_diff = subprocess.check_output(["git", "diff",":!notebooks/*"], text=True)
            artifact = wandb.Artifact("git_diff_patch", type="patch")
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".patch") as tmpfile:
                tmpfile.write(git_diff)
                tmpfile.flush()
                artifact.add_file(tmpfile.name, name="git_diff.patch")
            run.log_artifact(artifact)
        except FileNotFoundError as e:
            print(f"Error getting git diff: {e}")
