from setuptools import find_packages, setup

setup(
    name="omni_drones",
    version="0.2.1",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "wandb>=0.22.0",
        "PyYAML>=6.0.2",
        "numpy>=1.26.0",
        "scipy>=1.15.3",
        "tqdm>=4.67.1",
        "einops>=0.8.1",
        "pandas>=2.3.2",
        "imageio>=2.37.0",
        "moviepy>=2.2.1",
        "av>=15.1.0",
        "plotly>=5.3.1",
        "tensordict>=0.10.0",
        "torchrl>=0.10.0",
    ],
    extras_require={
        # Optional JAX backend used by the DreamerV3 / SkyDreamer policy
        # (omni_drones.learning.dreamerv3). Install with:
        #   pip install -e .[dreamer]
        # PPO and the other existing algos do not need any of these.
        "dreamer": [
            "jax[cuda12]==0.4.33",
            "optax",
            "flax",
            "elements>=3.19.1",
            "ninjax>=3.5.1",
            "chex",
            "portal>=3.5.0",
        ],
    },
)
