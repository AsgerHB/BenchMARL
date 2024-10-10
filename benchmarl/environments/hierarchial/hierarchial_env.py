import torch
from torchrl.envs import check_env_specs
from torchrl.envs import EnvBase
from torchrl.data import DiscreteTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict
from torchrl.envs import TransformedEnv, StepCounter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
import numpy as np
from enum import Enum
import time

# Debug stuff
import random
from torchrl.collectors import SyncDataCollector
from tensordict.nn import TensorDictModule
from torch import nn
from benchmarl.environments.common import Task

import matplotlib.pyplot as plt

class HierarcialEnvironment(EnvBase):
    def __init__(self,
            scenario,
            seed,
            device,
            **config):

        super().__init__(device=device)

        self.set_seed(seed)
        if scenario == "cruise_control":
            self._cc_set_spec(config)
            self._reset = self._cc_reset
            self._step = self._cc_step
            self.render = self._cc_render
        elif scenario == "chemical_production":
            self._cp_set_spec(config)
            self._reset = self._cp_reset
            self._step = self._cp_step
            self.render = self._cp_render

        else:
            raise NotImplemented
            
    def _reset(self):
        raise Exception("This function should have been replaced in constructor.")

    def _step(self):
        raise Exception("This function should have been replaced in constructor.")

    def _set_seed(self, seed):
        self.rng = torch.manual_seed(seed)

    #<editor-fold desc="Cruise Control">

    def _cc_set_spec(self, config):
        self.n_agents = config["n_agents"]
        self.t_act = config["t_act"]
        self.distance_min = config["distance_min"]
        self.distance_max = config["distance_max"]
        self.v_min = config["v_min"]
        self.v_max = config["v_max"]
        self.initial_distance = config["initial_distance"]
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

    def _cc_reset(self, tensordict, **kwargs):
        # There are n_agents + 1 agents, because the first car is uncontrollable. That means n_agents distance and n_agents + 1 velocities.
        
        velocity = torch.zeros(self.n_agents + 1, dtype=torch.float32)

        velocity_ego = torch.narrow(velocity, 0, 1, self.n_agents)
        velocity_front = torch.narrow(velocity, 0, 0, self.n_agents)

        distance = torch.tensor([self.initial_distance for car in range(1, self.n_agents + 1)])

        damaged = torch.zeros(self.n_agents, dtype=torch.float32)

        obsrvations = torch.stack([velocity_ego, velocity_front, distance, damaged], dim=1)

        return TensorDict({
            "agents": TensorDict({
                "observations": obsrvations,
            }, batch_size=self.n_agents)
        })
    
    def _cc_step(self, tensordict):
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
        new_velocity, new_distance = self._cc_simulate_point(velocity, distance, damaged, actions)

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
        reward = new_observations[:, 2].clone()
        reward = -reward
        reward -= new_damaged*torch.full([self.n_agents], self.safety_violation_penalty)
        reward = reward.unsqueeze(-1)
        
        return  TensorDict({
            "agents": TensorDict({
                "observations": new_observations,
                "reward" : reward,
            }, batch_size=self.n_agents),
            "done": False
        })


    def _cc_render(self, tensordict, mode="rgb_array"):
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
        ax.set_xlim(-50, self.initial_distance*self.n_agents*2.5)
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

    # Probability weights of front action, which depend on front's speed.
    weights_111 = torch.tensor([1, 1, 1], dtype=torch.float32) # Equal weight
    weights_211 = torch.tensor([2, 1, 1], dtype=torch.float32) # Backwards more likely
    weights_112 = torch.tensor([1, 1, 2], dtype=torch.float32) # Forwards more likely

    def random_front_behaviour(self, front_velocity):
        if front_velocity < 0:
            weights = HierarcialEnvironment.weights_112
        elif front_velocity > 10:
            weights = HierarcialEnvironment.weights_211
        else:
            weights = HierarcialEnvironment.weights_111

        return torch.multinomial(weights, 1, generator=self.rng).item()


    def _cc_apply_action(self, velocity, damaged, action):
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

    def _cc_simulate_point(self, velocity, distance, damaged, actions):
        velocity_difference = velocity[:-1] - velocity[1:]
        new_velocity = velocity.clone()
        front_action = self.random_front_behaviour(velocity[0])
        new_velocity[0] = self._cc_apply_action(velocity[0].clone(), 0, front_action) # Front car doesn't get damaged. Because it would be bothersome to implement.

        for i, action in enumerate(actions):
            new_velocity[i + 1] = self._cc_apply_action(velocity[i + 1].clone(), damaged[i], action)

        new_velocity_difference = new_velocity[:-1] - new_velocity[1:]
        new_distance = distance.clone() + ((velocity_difference + new_velocity_difference) / 2) * self.t_act
        new_distance = torch.maximum(new_distance, torch.full(size=[self.n_agents], fill_value=self.distance_min - 1))
        return new_velocity, new_distance

    # </editor-fold>

    #<editor-fold desc="Chemical Production">

    def _cp_layout(self):
        p1 = self.Provider(self, 1, [1.0, 0.5, 0.5, 0.0, 3.0])
        p2 = self.Provider(self, 2, [1.0, 0.5, 0.5, 0.0, 3.0])
        p3 = self.Provider(self, 3, [1.0, 0.5, 0.5, 0.0, 3.0])

        # Units need a parent ref, that's why "self" is in there.
        u1 = self.Unit(self, 1, p1, p1, p1)
        u2 = self.Unit(self, 2, p2, p2, p2)
        u3 = self.Unit(self, 3, p3, p3, p3)

        p4 = self.Provider(self, 4, [2.0, 1.5, 1.5, 0.5, 4.0])
        p5 = self.Provider(self, 5, [2.0, 1.5, 1.5, 0.5, 4.0])

        u4 = self.Unit(self, 4, p4, u1, u2)
        u5 = self.Unit(self, 5, u2, u3, p5)

        p6 = self.Provider(self, 6, [3.0, 3.5, 3.5, 2.5, 2.0])
        p7 = self.Provider(self, 7, [3.0, 3.5, 4.5, 2.5, 2.0])
        p8 = self.Provider(self, 8, [3.0, 4.5, 2.5, 2.5, 2.0])

        u6 = self.Unit(self, 6, p6, p6, u4)
        u7 = self.Unit(self, 7, u4, u5, p7)
        u8 = self.Unit(self, 8, u5, p8, p8)

        p9 = self.Provider(self, 9, [6.0, 6.5, 8.5, 2.5, 2.0])
        p10 = self.Provider(self, 10, [6.0, 6.5, 8.5, 0.5, 6.0])

        u9 = self.Unit(self, 9, p9, u6, u7)
        u10 = self.Unit(self, 10, u7, u8, p10)

        d1 = self.Consumer(self, 1, u9, [0, 2, 2, 1, 0])
        d2 = self.Consumer(self, 2, u10, [1, 1, 0, 2, 1])

        units =     [u1, u2, u3, u4, u5, u6, u7, u8, u9, u10]
        providers = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
        consumers = [d1, d2]

        return units, providers, consumers

    def _cp_set_spec(self, config):
        self.units, self.providers, self.consumers = self._cp_layout()
        self.n_agents = 10 # The layout is hard-coded.
        self.period = 2.5  # So is the period
        self.t_act = config["t_act"]
        self.flow_rate = config["flow_rate"]
        self.flow_variance = config["flow_variance"]
        self.stored_initially = config["stored_initially"]
        self.safety_violation_penalty = config["safety_violation_penalty"]
        
        self.agents = [ {"name": f"Unit{i}"} for i in range(1, self.n_agents + 1)]

        # Actions: CCO=0, COC=1, OCC=2, CCC=3,OOC=4, COO=5, OCO=6, OOO=7, 
        # C: Closed, O: Open.
        # I have no idea what the order is, but this is what the shield uses.
        action_spec = DiscreteTensorSpec(n=8, shape=[self.n_agents], dtype=torch.float32)

        # If provided, must be a CompositeSpec with one (group_name, "action") entry per group.
        self.full_action_spec = CompositeSpec({
                "agents": CompositeSpec({"action": action_spec}, shape=[self.n_agents])
            })

        # Observations are (v, t) where v is volume and t is periodic time (That is time between 0 and self.period; consumers and providers use this.)
        observation_spec = UnboundedContinuousTensorSpec(shape=[self.n_agents, 2], dtype=torch.float32)

        # Must be a CompositeSpec with one (group_name, observation_key) entry per group.
        self.full_observation_spec = CompositeSpec({
                "agents": CompositeSpec({"observations": observation_spec}, shape=[self.n_agents])
            })

        # Cost paid to provider this time-step.
        reward_spec = UnboundedContinuousTensorSpec(shape=[self.n_agents, 1], dtype=torch.float32)

        self.full_reward_spec = CompositeSpec({
                "agents": CompositeSpec({"reward": reward_spec}, shape=[self.n_agents])
            })

        self.done_spec = DiscreteTensorSpec(n=2, shape=[1], dtype=torch.bool)

    def _cp_reset(self, tensordict, **kwargs):
        v = torch.full(fill_value=self.stored_initially, size=[self.n_agents], dtype=torch.float32)
        t = torch.zeros(self.n_agents, dtype=torch.float32)

        observations = torch.stack([v, t], dim=1)

        return TensorDict({
            "agents": TensorDict({
                "observations": observations,
            }, batch_size=self.n_agents)
        })            

    class Provider: 
        def __init__(self, parent, id, cost):
            self.parent = parent
            self.id = id
            self.cost = cost # Array of costs.
        
        def get_cost(self, time):
            return self.cost[int(time/self.parent.t_act)]
        
    class Unit:
        # Note that units etc don't contain internal state, only structure.
        # State is passed as arguments to the class functions.
        def __init__(self, parent, id, left, middle, right):
            self.parent = parent
            self.id = id
            self.left = left
            self.middle = middle
            self.right = right

        @staticmethod
        def is_safe(id, volumes):
            min_stored=2
            max_stored=50
            if volumes[id - 1] > max_stored:
                return False
            
            if id == 9 or id == 10:
                if volumes[id - 1] < min_stored:
                    return False
            
            return True

        def take_from(self, storage, volumes, time):
            flow = self.parent.flow_rate 
            r = torch.rand([], generator=self.parent.rng)
            flow += r*self.parent.flow_variance
            flow = min(flow, volumes[storage.id - 1])
            flow *= self.parent.t_act

            cost = 0

            if isinstance(storage, self.parent.Unit):
                # ID's are 1-10, volume indices are 0-9. Wish I was writing Julia.
                volumes[storage.id - 1] -= flow
                volumes[self.id - 1] += flow
            elif isinstance(storage, self.parent.Provider):
                volumes[self.id - 1] += flow
                cost = flow*storage.get_cost(time)
            else:
                raise NotImplemented("Unexpected storage type.", storage)
            
            volumes[self.id - 1] = min(volumes[self.id - 1], 50 + 1)
            return cost

        
        def take_action(self, action, volumes, time):
            CCO = 0
            COC = 1
            OCC = 2
            CCC = 3
            OOC = 4
            COO = 5
            OCO = 6
            OOO = 7
            
            cost = 0

            if action in [OCC, OOC, OCO, OOO]:
                cost += self.take_from(self.left, volumes, time)

            if action in [COC, OOC, COO, OOO]:
                cost += self.take_from(self.middle, volumes, time)

            if action in [CCO, COO, OCO, OOO]:
                cost += self.take_from(self.right, volumes, time)

            return cost


    class Consumer:
        def __init__(self, parent, id, connected_to, consumption):
            self.parent = parent
            self.id = id
            self.consumption = consumption # Consumption pattern as array
            self.connected_to = connected_to

        def consume(self, volumes, time):
            multiplier = self.consumption[int(time/self.parent.t_act)]
            if multiplier == 0:
                return
            flow = multiplier*self.parent.flow_rate
            r = torch.rand([], generator=self.parent.rng)
            flow += r*self.parent.flow_variance
            flow = min(flow, volumes[self.connected_to.id - 1])
            flow *= self.parent.t_act
            
            volumes[self.connected_to.id - 1] -= flow
    
    def _cp_simulate_point(self, volume, time, actions):
        
        new_volume = torch.clone(volume)
        new_cost = torch.empty([self.n_agents], dtype=torch.float32)

        for action, unit in zip(actions, self.units):
            cost = unit.take_action(action, new_volume, time)
            if not unit.is_safe(unit.id, new_volume):
                cost += self.safety_violation_penalty
            new_cost[unit.id - 1] = cost

        for consumer in self.consumers:
            consumer.consume(new_volume, time)
        
        new_time = (time + self.t_act) % self.period

        return new_volume, new_time, new_cost


    def _cp_step(self, tensordict):
        actions = tensordict["agents", "action"]
        observations = tensordict["agents", "observations"]
        volume = observations[:, 0]
        time = observations[:, 1][1]
        
        new_volume, new_time, new_cost = self._cp_simulate_point(volume, time, actions)

        reward = -new_cost
        reward = reward.unsqueeze(-1)

        new_observations = torch.stack([new_volume, torch.full([self.n_agents], new_time)], dim=1)

        return TensorDict({
            "agents": TensorDict({
                "observations": new_observations,
                "reward": reward
            }, batch_size=self.n_agents),
            "done": False
        })

    def _cp_render(self, tensordict, mode="rgb_array"):
        actions = tensordict["agents", "action"]
        observations = tensordict["agents", "observations"]
        volume = observations[:, 0]
        t = observations[:, 1]

        info_text = f"t={t[1].item()}"

        fig, ax = plt.subplots()
        plt.gca().set_aspect('equal')
        ax.set_axis_off()
        fig.set_size_inches(3, 6)
        plt.title("Chemical Production")
        plt.text(30, 30, info_text, verticalalignment="top")

        # Mainmatter #

        unit_width, unit_height = 10, 12

        # layout[3] is positioned as Unit 3 etc.
        # and then layout[11] to layout[15] are the consumers
        # which are drawn as 2 consumers, but actually have 6 pipes into them.
        layout = [
            (1, 5), (2, 5), (3, 5),
              (1.5, 4), (2.5, 4),
            (1, 3), (2, 3), (3, 3),
              (1.5, 2), (2.5, 2),
            # And then the positions of the consumers 
            (1.5, 1), (2.5, 1),
        ]
        
        plt.plot()
        for i, v in enumerate(volume):
            frame_color = "blue" if self.units[i].is_safe(i + 1, volume) else "red"
            tank_level = unit_height*(v/(50))
            offset = [o*30 for o in layout[i]]
            frame_rect = Rectangle(offset, unit_width, unit_height, linewidth=8, color=frame_color)
            unit_rect = Rectangle(offset, unit_width, unit_height, color="white")
            liquid_rect = Rectangle(offset, unit_width, tank_level, color="gray")
            ax.add_patch(frame_rect)
            ax.add_patch(unit_rect)
            ax.add_patch(liquid_rect)

        if mode == "rgb_array":
            # This is from some stackoverflow question.
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
            plt.ion()
            plt.show()
        elif mode == "save to outputs/frame.png":
            plt.ioff()
            plt.savefig('outputs/frame.png', dpi=100)
            plt.close()
            time.sleep(0.5)
        elif mode == "plt":
            return plt
        else:
            raise NotImplemented()

    #</editor-fold>

if __name__ == "__main__":
    scenario = "chemical_production"
    # Creating the env
    cruise_control_config = dict(n_agents=4,
            t_act=1.0,
            distance_min=0.0,
            distance_max=200.0,
            v_min=-10.0,
            v_max=20.0,
            initial_distance=50.0,
            safety_violation_penalty=10.0,)
    
    chemical_production_config = dict(t_act=0.5,
        flow_rate=2.65,
        flow_variance=0.5,
        stored_initially=11,
        safety_violation_penalty=100,)

    create_env_fn = lambda: TransformedEnv(
        HierarcialEnvironment(
            scenario=scenario,
            seed=random.randint(0, 9999999999),
            device="cpu",

            # config 👇 TODO: I think there's a get_from_yaml() function I could use.
            **(chemical_production_config if scenario == "chemical_production" else cruise_control_config)
        ),
        StepCounter(max_steps=100)
    )

    env = create_env_fn()
    # check_env_specs(env)

    # Take some steps manually
    s = env.reset()
    for i in range(0, 100):
        s = env.rand_action(s)
        s = env.step(s)
        if i%1 == 0:
            env.render(s, mode="save to outputs/frame.png")
        s = s["next"]

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

    # for i, data in enumerate(collector):
    #     data_to_plot = data["agents", "observations"][:, :, 2]
    #     labels = [a["name"] for a in create_env_fn().agents]

    #     for i in range(data_to_plot.shape[1]):
    #         plt.plot(data_to_plot[:, i], label=labels[i])

    #     plt.legend()
    #     plt.show()

    
    print("🦑🦑")