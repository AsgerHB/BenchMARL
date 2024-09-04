import torch
from torchrl.envs import check_env_specs
from torchrl.envs import EnvBase
from torchrl.data import DiscreteTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict

# Debug stuff
from torchrl.collectors import SyncDataCollector
from tensordict.nn import TensorDictModule
from torch import nn

class HierarcialEnvironment(EnvBase):
    def __init__(self,
            scenario,
            seed,
            device,
            **config):

        super().__init__(device=device)

        self.set_seed(seed)

        self.n_agents = config["n_agents"]
        self.agents = [ {"name": f"Car{i}"} for i in range(1, self.n_agents + 1)]

        action_spec = DiscreteTensorSpec(n=3, shape=[self.n_agents], dtype=torch.float32)

        # If provided, must be a CompositeSpec with one (group_name, "action") entry per group.
        self.full_action_spec = CompositeSpec({
                "agents": CompositeSpec({"action": action_spec}, shape=[self.n_agents])
            })

        observation_spec = UnboundedContinuousTensorSpec(shape=[self.n_agents, 3], dtype=torch.float32)

        # Must be a CompositeSpec with one (group_name, observation_key) entry per group.
        self.full_observation_spec = CompositeSpec({
                "agents": CompositeSpec({"observations": observation_spec}, shape=[self.n_agents])
            })

        
        reward_spec = UnboundedContinuousTensorSpec(shape=[self.n_agents, 1], dtype=torch.float32)

        self.full_reward_spec = CompositeSpec({
                "agents": CompositeSpec({"reward": reward_spec}, shape=[self.n_agents])
            })

        self.done_spec = DiscreteTensorSpec(n=2, shape=[1], dtype=torch.bool)

    def _reset(self, tensordict, **kwargs):
        # There are n_agents + 1 agents, because the first car is uncontrollable. That means n_agents distance and n_agents + 1 velocities.
        
        velocity = torch.zeros(self.n_agents + 1, dtype=torch.float32)

        velocity_ego = torch.narrow(velocity, 0, 1, self.n_agents)
        velocity_front = torch.narrow(velocity, 0, 0, self.n_agents)

        distance = torch.tensor([50 for car in range(1, self.n_agents + 1)])

        obsrvations = torch.stack([velocity_ego, velocity_front, distance], dim=1)

        return TensorDict({
            "agents": TensorDict({
                "observations": obsrvations,
            }, batch_size=self.n_agents),
        })
    
    def _step(self, tensordict):
        #print("👉", tensordict["done"])

        actions = tensordict["agents", "action"]
        observations = tensordict["agents", "observations"]
        new_observations = torch.clone(observations)
        new_observations[:, 2] += actions.flatten()
        new_observations[:, 0] += 1
        reward = new_observations[:, 2].unsqueeze(-1)

        done = new_observations[1, 0] > 100

        return  TensorDict({
            "agents": TensorDict({
                "observations": new_observations,
                "reward" : reward,
            }, batch_size=self.n_agents),
            "done": done
        })

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng


if __name__ == "__main__":
    env = HierarcialEnvironment("CruiseControl", 10, 2020, "cpu", n_agents=3)
    check_env_specs(env)

    create_env_fn = lambda: HierarcialEnvironment(
        scenario="ljkæasdljk",
        num_envs=10,  # Number of vectorized envs (do not use this param if the env is not vectorized)
        seed=2020,
        device="cpu",
        n_agents=10,
    )

    policy = TensorDictModule(nn.Linear(30, 10), in_keys=["agents", "observations"], out_keys=["agents", "action"])

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=None,
        total_frames=2000,
        max_frames_per_traj=50,
        frames_per_batch=200,
        init_random_frames=-1,
        reset_at_each_iter=False,
        device="cpu",
        storing_device="cpu",
    )

    for i, data in enumerate(collector):
        print("👉", data["next"][env.reward_key])
        if i > 3:
            break
    
    print("🦑🦑")