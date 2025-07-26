import sys

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Core.Policies.Agents import RandomAgent
from pyRDDLGym.Core.Policies.Agents import MENTSAgent

def main(env, inst, method_name=None, episodes=1):
    print(f'preparing to launch instance {inst} of domain {env}...')
    env_info = ExampleManager.GetEnvInfo(env)

    log = False if method_name is None else True
    my_env = RDDLEnv.RDDLEnv(domain=env_info.get_domain(),
                             instance=env_info.get_instance(inst),
                             enforce_action_constraints=True,
                             debug=False,
                             log=log,
                             simlogname=method_name)
    
    my_env.set_visualizer(env_info.get_visualizer())
    agent = MENTSAgent(action_space=my_env.action_space, num_actions=my_env.numConcurrentActions)

    for episode in range(episodes):
        total_reward = 0
        state = my_env.reset()
        for step in range(my_env.horizon):
            my_env.render()
            action = agent.sample_action(my_env)
            next_state, reward, done, info = my_env.step(action)
            total_reward += reward
            print()
            print(f'step       = {step}\n'
                  f'state      = {state}\n'
                  f'action     = {action}\n'
                  f'next state = {next_state}\n'
                  f'reward     = {reward}\n')
            state = next_state
            if done:
                break
        print(f'episode {episode} ended with return {total_reward}')
        my_env.close()


if __name__ == "__main__":
    args = sys.argv
    method_name = None 
    episodes = 1
    if len(args) < 3 :
        env, inst = 'Traffic', '0'
    elif len(args) < 4:
        env, inst = args[1:3]
    elif len(args) < 5:
        env, inst, method_name = args[1:4]
    else :
        env, inst, method_name, episodes = args[1:5]
        try:
            episodes = int(episodes)
        except:
            raise ValueError('episodes argument must be an integer, reveived:' + episodes)
    main(env, inst, method_name, episodes)
