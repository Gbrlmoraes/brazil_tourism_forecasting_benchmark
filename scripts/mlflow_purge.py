"""Permanently delete every MLflow experiment, run and registered model.

Usage (from the project root, with the tracking server running in Docker):

    task mlflow_purge          # asks for confirmation
    task mlflow_purge --yes    # no confirmation

Steps:
1. Delete every registered model (they point to runs that are about to disappear).
2. Soft-delete every experiment. The `Default` experiment (id 0) cannot be deleted in
   MLflow, so only its runs are deleted.
3. Run `mlflow gc` inside the container, which removes the deleted experiments and runs
   from the database and their run artifacts.
4. Remove the artifact folders that `mlflow gc` leaves behind (MLflow 3 logged models
   are stored per experiment and are not collected).

After a purge the experiment names are free again, so the notebooks recreate them.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.entities import ViewType

ROOT_PATH = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ID = '0'
CONTAINER_SERVICE = 'mlflow'
CONTAINER_BACKEND_STORE = 'sqlite:////mlflow/mlflow.db'
CONTAINER_TRACKING_URI = 'http://localhost:5000'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '--yes', action='store_true', help='skip the confirmation prompt'
    )
    return parser.parse_args()


def summarize(client):
    experiments = client.search_experiments(view_type=ViewType.ALL)
    models = client.search_registered_models()
    print(f'Tracking URI: {mlflow.get_tracking_uri()}')
    print(f'Experiments ({len(experiments)}):')
    for exp in experiments:
        n_runs = len(
            client.search_runs(
                [exp.experiment_id], run_view_type=ViewType.ALL, max_results=50000
            )
        )
        print(
            f'  [{exp.experiment_id}] {exp.name} ({exp.lifecycle_stage}, {n_runs} runs)'
        )
    print(f'Registered models ({len(models)}): {[m.name for m in models]}')
    return experiments, models


def confirm():
    answer = input(
        '\nThis PERMANENTLY deletes everything listed above. '
        'Type "DELETE" to continue: '
    )
    if answer.strip() != 'DELETE':
        print('Aborted, nothing was deleted.')
        sys.exit(1)


def soft_delete(client, experiments, models):
    for model in models:
        client.delete_registered_model(model.name)
        print(f'Deleted registered model {model.name}')
    for exp in experiments:
        if exp.experiment_id == DEFAULT_EXPERIMENT_ID:
            runs = client.search_runs(
                [exp.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=50000,
            )
            for run in runs:
                client.delete_run(run.info.run_id)
            print(f'Deleted {len(runs)} runs of the Default experiment')
        elif exp.lifecycle_stage == 'active':
            client.delete_experiment(exp.experiment_id)
            print(f'Deleted experiment [{exp.experiment_id}] {exp.name}')


def garbage_collect():
    command = [
        'docker', 'compose', 'exec', '-T',
        '-e', f'MLFLOW_TRACKING_URI={CONTAINER_TRACKING_URI}',
        CONTAINER_SERVICE,
        'mlflow', 'gc', '--backend-store-uri', CONTAINER_BACKEND_STORE,
    ]  # fmt: skip
    print('Running mlflow gc in the container...', flush=True)
    subprocess.run(command, cwd=ROOT_PATH, check=True)


def remove_leftover_artifacts(experiment_ids):
    data_dir = Path(os.getenv('MLFLOW_DATA_DIR', ROOT_PATH / 'mlflow_data'))
    if not data_dir.is_absolute():
        data_dir = ROOT_PATH / data_dir
    artifacts_dir = data_dir / 'artifacts'
    for experiment_id in experiment_ids:
        folder = artifacts_dir / experiment_id
        if folder.exists():
            shutil.rmtree(folder)
            print(f'Removed leftover artifacts {folder}')


def main():
    args = parse_args()
    load_dotenv(ROOT_PATH / '.env')
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    client = mlflow.MlflowClient()

    experiments, models = summarize(client)
    if not args.yes:
        confirm()
    soft_delete(client, experiments, models)
    garbage_collect()
    remove_leftover_artifacts([exp.experiment_id for exp in experiments])
    print()
    summarize(client)


if __name__ == '__main__':
    main()
