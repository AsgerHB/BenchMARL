import torch
from torchrl.envs import check_env_specs
from torchrl.envs import EnvBase
from torchrl.data import DiscreteTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict
from torchrl.envs import TransformedEnv, StepCounter
import matplotlib.pyplot as plt
import io
import numpy as np

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

        observation_spec = UnboundedContinuousTensorSpec(shape=[self.n_agents, 4], dtype=torch.float32)

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

        damaged = torch.zeros(self.n_agents, dtype=torch.float32)

        obsrvations = torch.stack([velocity_ego, velocity_front, distance, damaged], dim=1)

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
        damaged = observations[:, 3]

        # Combine the velocities back into a single tensor of shape (n_agents + 1)
        velocity = torch.zeros(self.n_agents + 1, dtype=torch.float32)
        velocity[1:] = velocity_ego
        velocity[0:self.n_agents] = velocity_front

        # The actual update
        new_velocity, new_distance = self.simulate_point(velocity, distance, damaged, actions)

        # Split updated velocities into observations
        new_velocity_ego = torch.narrow(new_velocity, 0, 1, self.n_agents)
        new_velocity_front = torch.narrow(new_velocity, 0, 0, self.n_agents)

        # Compute safety violations
        crash = new_distance < self.distance_min
        fall_behind = new_distance > self.distance_max
        safety_violation = crash.logical_or(fall_behind)

        # Apply damage to cars
        new_damaged = damaged.clone()
        for i, c in enumerate(crash):
            if c:
                new_damaged[i] = 1
                if i == 0:
                    continue
                new_damaged[i - 1] = 1

        new_observations = torch.stack([new_velocity_ego, new_velocity_front, new_distance, new_damaged], dim=1)

        # Reward
        reward = new_observations[:, 2].clone()*0.1
        reward += safety_violation*torch.full([self.n_agents], self.safety_violation_penalty)
        reward = reward.unsqueeze(-1)
        
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

    def render(self, tensordict, mode="rgb_array"):
        car_width = 20
        reward = tensordict.get(("agents", "reward"), default=None)
        observations = tensordict["agents", "observations"]
        uncontrollable_car_velocity = observations[0][1]
        distances = observations[:, 2]
        damages = observations[:, 3]
        positions = [0]
        for distance in distances:
            positions.append(positions[-1] + distance + car_width)
        ys = [0.5 for _ in positions]
        colors = ["r" if damaged else "b" for damaged in damages]
        colors.insert(0, "b") # First car is blue

        info_text = f"Uncontrollable car velocity:\n{uncontrollable_car_velocity}\n\n"
        info_text += f"Velocities:\n{observations[:, 0].tolist()}\n\n"
        if reward != None:
            pretty_reward = [round(r) for r in reward.squeeze().tolist()]
            info_text += f"Immediate reward: \n{pretty_reward}\n\n"

        plt.ioff()
        fig, ax = plt.subplots()
        ax.set_xlim(-50, self.distance_max*self.n_agents*1.5)
        ax.set_ylim(0, 1)
        plt.title("Agent Position in Environment")
        plt.text(0, 0.95, info_text, verticalalignment="top")
        ax.axes.get_yaxis().set_visible(False)

        # Mainmatter
        plt.scatter(positions, ys, marker="<", color=colors)

        if mode == "rgb_array":
            with io.BytesIO() as buff:
                fig.savefig(buff, format='rgba')
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8) # (w*h,)

            w, h = fig.canvas.get_width_height()
            plt.close(fig)

            im = data.reshape((int(h), int(w), -1)) # (w, h, 4)
            im = im[:, :, :3]                       # (w, h, 3); discard alpha
            im = torch.tensor(im)

            return im
        elif mode == "window":
            plt.show()
        elif mode == "plt":
            return plt
        else:
            raise NotImplemented()


    def random_front_behaviour(self):
        random_variable = torch.rand(1).item()
        if random_variable < 1/3:
            return 0 # backwards
        elif random_variable < 2/3:
            return 1 # neutral
        else:
            return 2 # forwards

    def apply_action(self, velocity, damaged, action):
        if not damaged:
            if action == 0: # backwards
                velocity -= 2
            elif action == 1: # neutral
                pass
            elif action == 2: # forwards
                velocity += 2
            else:
                raise RuntimeError(f"Unexpected action {action}")
        else: # Emergency stop
            if velocity < 0:
                velocity += 2
            elif velocity > 0:
                velocity -= 2
        return torch.clamp(velocity, self.v_min, self.v_max)

    def simulate_point(self, velocity, distance, damaged, actions):
        velocity_difference = velocity[:-1] - velocity[1:]
        new_velocity = velocity.clone()
        front_action = self.random_front_behaviour()
        new_velocity[0] = self.apply_action(velocity[0].clone(), 0, front_action) # Front car doesn't get damaged. Because it would be bothersome to implement.

        for i, action in enumerate(actions):
            new_velocity[i + 1] = self.apply_action(velocity[i + 1].clone(), damaged[i], action)

        new_velocity_difference = new_velocity[:-1] - new_velocity[1:]
        new_distance = distance.clone() + ((velocity_difference + new_velocity_difference) / 2) * self.t_act
        new_distance = torch.maximum(new_distance, torch.full(size=[self.n_agents], fill_value=self.distance_min - 1))
        return new_velocity, new_distance

if __name__ == "__main__":
    # Creating the env
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
            safety_violation_penalty=10.0,
        ),
        StepCounter(max_steps=100)
    )

    env = create_env_fn()
    check_env_specs(env)

    # Take some steps manually
    s = env.reset()
    for i in range(0, 100):
        s = env.rand_action(s)
        s = env.step(s)
        s = s["next"]
        if i%2 == 0:
            env.render(s, mode="window")

    # Can't recall what this is.
    policy = TensorDictModule(nn.Linear(30, 10), in_keys=["agents", "observations"], out_keys=["agents", "action"]) 

    # Do some rollouts
    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=None,
        frames_per_batch=100,
        total_frames=100*4,
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