import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from qlearning_agent import QLearningAgent
from grid_world import GridWorld


if __name__ == '__main__':
    grid_env = GridWorld() # grid worldの環境の初期化
    ini_state = grid_env.start_pos  # 初期状態（エージェントのスタート地点の位置）
    agent = QLearningAgent(epsilon=.1, actions=np.arange(4), observation=ini_state)  # Q学習エージェント
    nb_episode = 1000   #エピソード数
    rewards = []    # 評価用報酬の保存
    is_end_episode = False # エージェントがゴールしてるかどうか？
    for episode in range(nb_episode):
        episode_reward = [] # 1エピソードの累積報酬
        while(is_end_episode == False):    # ゴールするまで続ける
            action = agent.act()  # 行動選択
            state, reward, is_end_episode = grid_env.step(action)
            agent.observe(state, reward)   # 状態と報酬の観測
            episode_reward.append(reward)
        rewards.append(np.sum(episode_reward)) # このエピソードの平均報酬を与える
        state = grid_env.reset()    #  初期化
        agent.observe(state)    # エージェントを初期位置に
        is_end_episode = False

    # 結果のプロット
    plt.plot(np.arange(nb_episode), rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("result.jpg")
    plt.show()
