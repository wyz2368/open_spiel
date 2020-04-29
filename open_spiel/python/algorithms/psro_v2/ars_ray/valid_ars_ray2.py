import ray


@ray.remote
class Worker(object):
    def __init__(self,
                 env_name,
                 deltas=None,
                 slow_oracle_kargs=None,
                 fast_oracle_kargs=None
                 ):
        # initialize rl environment.

        self._env_name = env_name
        print(env_name)

ray.init(temp_dir='./ars_temp_dir/')
workers = [Worker.remote("kuhn") for _ in range(3)]