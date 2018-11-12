from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

from collections import deque

from tensorboardX import SummaryWriter

import numpy as np


def main():
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        env = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_load_model = False
    is_render = False
    model_path = 'models/{}.model'.format(env_id)
    icm_path = 'models/{}.icm'.format(env_id)

    writer = SummaryWriter()

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    stable_eps = float(default_config['StableEps'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])

    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    reward_rms = RunningMeanStd()
    discounted_reward = RewardForwardFilter(gamma)

    agent = RNDAgent

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    else:
        raise NotImplementedError

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
    )

    if is_load_model:
        agent.model.load_state_dict(torch.load(model_path))

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob, life_done=life_done)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_step = 0
    recent_prob = deque(maxlen=10)

    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward = [], [], [], [], [], []
        global_step += (num_worker * num_step)

        for _ in range(num_step):
            actions = agent.get_action(states)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards = [], [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd, lr = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)

            # total reward = int reward + ext Reward
            intrinsic_reward = agent.compute_intrinsic_reward(next_states)
            intrinsic_reward = np.hstack(intrinsic_reward)
            sample_i_rall += intrinsic_reward[sample_env_idx]

            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_next_state.append(next_states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward', sample_rall, sample_episode)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().reshape([-1])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose().reshape([-1])

        value_ext, value_int, next_value_ext, next_value_int, policy = agent.forward_transition(
            total_state, total_next_state)

        # running mean int reward
        total_int_reward = np.stack(total_int_reward).transpose().reshape([-1])
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.reshape([num_worker, -1]).T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # devided reward by running std
        total_int_reward /= np.sqrt(reward_rms.var)

        writer.add_scalar('data/int_reward', sum(total_int_reward) / num_worker, sample_episode)

        policy = policy.detach()
        m = F.softmax(policy, dim=-1)
        recent_prob.append(m.max(1)[0].mean().cpu().numpy())
        writer.add_scalar('data/max_prob', np.mean(recent_prob), sample_episode)

        total_ext_target = []
        total_int_target = []
        total_ext_adv = []
        total_int_adv = []
        # -----------------------------------------------
        # extrinsic reward calculate
        for idx in range(num_worker):
            target, adv = make_train_data(total_reward[idx * num_step:(idx + 1) * num_step],
                                          total_done[idx * num_step:(idx + 1) * num_step],
                                          value_ext[idx * num_step:(idx + 1) * num_step],
                                          next_value_ext[idx * num_step:(idx + 1) * num_step],
                                          gamma,
                                          num_step)

            total_ext_target.append(target)
            # multiply coef
            total_ext_adv.append(adv * ext_coef)

        # intrinsic reward calculate
        # None Episodic
        for idx in range(num_worker):
            target, adv = make_train_data(total_int_reward[idx * num_step:(idx + 1) * num_step],
                                          np.zeros([num_step]),
                                          value_int[idx * num_step:(idx + 1) * num_step],
                                          next_value_int[idx * num_step:(idx + 1) * num_step],
                                          int_gamma,
                                          num_step)

            total_int_target.append(target)
            # multiply coef
            total_int_adv.append(adv * int_coef)
        # -----------------------------------------------
        # add ext adv and int adv
        total_adv = total_int_adv + total_ext_adv

        agent.train_model(total_state, total_next_state, np.hstack(total_ext_target), np.hstack(total_int_target),
                          total_action, np.hstack(total_adv))

        if global_step % (num_worker * num_step * 100) == 0:
            torch.save(agent.model.state_dict(), model_path)
            if train_method == 'ICM':
                torch.save(agent.icm.state_dict(), icm_path)


if __name__ == '__main__':
    main()
