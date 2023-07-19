from mmcv.utils.env import collect_env as collect_base_env

def collect_env():
    env_info = collect_base_env()
    return env_info

if __name__ == "__main__":
    for name, val in collect_env().items():
        print("{}, {}".format(name, val))
