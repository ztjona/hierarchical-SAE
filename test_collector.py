from torchrl.envs.libs.gym import GymEnv
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector

env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
collector = SyncDataCollector(
    create_env_fn=env_maker,
    policy=policy,
    total_frames=2000,
    max_frames_per_traj=50,
    frames_per_batch=200,
    init_random_frames=-1,
    reset_at_each_iter=False,
    device="cpu",
    storing_device="cpu",
)
for i, data in enumerate(collector):
    if i == 2:
        print(data)
        break
