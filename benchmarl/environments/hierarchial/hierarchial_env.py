import torch
from torchrl.envs import check_env_specs
from torchrl.envs import EnvBase
from torchrl.data import DiscreteTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict
from torchrl.envs import TransformedEnv, StepCounter

# Debug stuff
import random
from torchrl.collectors import SyncDataCollector
from tensordict.nn import TensorDictModule
from torch import nn

import matplotlib.pyplot as plt

class HierarcialEnvironment(EnvBase):
    def __init__(self,
            scenario,
            seed,
            device,
            **config):

        super().__init__(device=device)

        self.set_seed(seed)

        self.n_agents = config["n_agents"]
        self.t_act = config["t_act"]
        self.distance_min = config["distance_min"]
        self.distance_max = config["distance_max"]
        self.v_min = config["v_min"]
        self.v_max = config["v_max"]

        self.safety_violation_penalty = config["safety_violation_penalty"]
        
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
            }, batch_size=self.n_agents)
        })
    
    def _step(self, tensordict):
        #print("👉", tensordict["done"])

        actions = tensordict["agents", "action"]
        observations = tensordict["agents", "observations"]

        velocity_ego = observations[:, 0]
        velocity_front = observations[:, 1]
        distance = observations[:, 2]

        # Combine the velocities back into a single tensor of shape (n_agents + 1)
        velocity = torch.zeros(self.n_agents + 1, dtype=torch.float32)
        velocity[1:] = velocity_ego
        velocity[0:self.n_agents] = velocity_front

        new_velocity, new_distance = self.simulate_point(velocity, distance, actions)

        new_velocity_ego = torch.narrow(new_velocity, 0, 1, self.n_agents)
        new_velocity_front = torch.narrow(new_velocity, 0, 0, self.n_agents)
        new_observations = torch.stack([new_velocity_ego, new_velocity_front, new_distance], dim=1)

        crash = new_observations[:, 2] < self.distance_min
        fall_behind = new_observations[:, 2] > self.distance_max
        safety_violation = crash.logical_or(fall_behind)

        reward = new_observations[:, 2]

        # saefty_violation bool vector used in the C-style so it is 0 if false and 1 if true
        reward += safety_violation*torch.full([self.n_agents], self.safety_violation_penalty)
        
        return  TensorDict({
            "agents": TensorDict({
                "observations": new_observations,
                "reward" : reward,
            }, batch_size=self.n_agents),
            "done": False
        })

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def random_front_behaviour(self):
        random_variable = torch.rand(1).item()
        if random_variable < 1/3:
            return 0 # backwards
        elif random_variable < 2/3:
            return 1 # neutral
        else:
            return 2 # forwards

    def apply_action(self, velocity, action):
        if action == 0: # backwards
            velocity -= 2
        elif action == 1: # neutral
            pass  # No change
        elif action == 2: # forwards
            velocity += 2
        else:
            raise RuntimeError(f"Unexpected action {action}")
        return torch.clamp(velocity, self.v_min, self.v_max)

    def simulate_point(self, velocity, distance, actions):
        velocity_difference = velocity[:-1] - velocity[1:]
        new_velocity = velocity.clone()
        front_action = self.random_front_behaviour()
        new_velocity[0] = self.apply_action(velocity[0], front_action)

        for i, action in enumerate(actions):
            new_velocity[i + 1] = self.apply_action(velocity[i + 1], action)

        new_velocity_difference = new_velocity[:-1] - new_velocity[1:]
        new_distance = distance + ((velocity_difference + new_velocity_difference) / 2) * self.t_act
        return new_velocity, new_distance

if __name__ == "__main__":

    n_agents = 4
    create_env_fn = lambda: TransformedEnv(
        HierarcialEnvironment(
            scenario="ljkæasdljk",
            seed=random.randint(0, 9999999999),
            device="cpu",

            # config 👇
            n_agents=n_agents,
            t_act=1.0,
            distance_min=0.0,
            distance_max=200.0,
            v_min=-10.0,
            v_max=20.0,
            safety_violation_penalty=100.0,
        ),
        StepCounter(max_steps=100)
    )

    env = create_env_fn()
    #check_env_specs(env)

    policy = TensorDictModule(nn.Linear(30, 10), in_keys=["agents", "observations"], out_keys=["agents", "action"])

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=None,
        frames_per_batch=500,
        total_frames=500*10,
        init_random_frames=-1,
        reset_at_each_iter=False,
        device="cpu",
        storing_device="cpu",
    )

    for i, data in enumerate(collector):
        data_to_plot = data["agents", "observations"][:, :, 2]
        labels = [a["name"] for a in create_env_fn().agents]

        for i in range(data_to_plot.shape[1]):
            plt.plot(data_to_plot[:, i], label=labels[i])

        plt.legend()
        plt.show()

    
    print("🦑🦑")