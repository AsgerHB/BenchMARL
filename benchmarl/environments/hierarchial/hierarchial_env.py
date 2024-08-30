import torch
from torchrl.envs import EnvBase
from torchrl.data import DiscreteTensorSpec, CompositeSpec, UnboundedDiscreteTensorSpec

class HierarcialEnvironment(EnvBase):
    def __init__(self,
            scenario,
            num_envs,
            continuous_actions,
            seed,
            device,
            categorical_actions,
            **config):

        super().__init__(device=device)

        n_cars = config["n_cars"]
        self.cars = [ {"name": f"Car{i}"} for i in range(1, n_cars + 1)]

        # If provided, must be a CompositeSpec with one (group_name, "action") entry per group.
        self.full_action_spec = CompositeSpec({("cars", "action"): DiscreteTensorSpec(n=3, shape=[n_cars])})

        # Must be a CompositeSpec with one (group_name, observation_key) entry per group.
        observation_spec = UnboundedDiscreteTensorSpec(shape=[10, 3])
        #agents_observation_spec = CompositeSpec({f"Car{i}": observation_spec for i in range(1, n_cars + 1)}, shape=[n_cars])
        self.full_observation_spec = CompositeSpec({("cars", "observation"): observation_spec}, shape=[n_cars])
        # Set the batch size
        self.batch_size = torch.Size([num_envs])

        # Update the specs with the correct batch size
        # This ensures that the action and observation specs are correctly expanded
        # to match the number of environments being run in parallel.
        self.full_action_spec = self.full_action_spec.expand(self.batch_size)
        self.full_observation_spec = self.full_observation_spec.expand(self.batch_size)



        print("👯👯")

    def _reset(self, tensordict, **kwargs):
        print("Starting over :V")
    
    def _step(self, tensordict):
        print("👉", tensordict)

    def _set_seed(self, seed):
        print("uwu")
