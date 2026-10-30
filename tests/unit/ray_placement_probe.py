"""Prints the GPU IDs Ray assigns after `placement.init_ray(["2"])`, as JSON.

Run as a script in a fresh interpreter by `test_placement.py`, so this Ray instance
(started with fake GPU IDs) stays isolated from the session-wide one used by other
tests.
"""

import json
import os

import ray

from vrollout import placement


@ray.remote(num_gpus=1)
def _assigned_gpus() -> list[str]:
    return [str(i) for i in ray.get_gpu_ids()]


if __name__ == "__main__":
    placement.init_ray(["2"])
    result = {
        "cluster_gpus": ray.cluster_resources().get("GPU"),
        "assigned": ray.get(_assigned_gpus.remote(), timeout=60),
        "driver_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(result))
    ray.shutdown()
