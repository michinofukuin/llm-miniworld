import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='Half-Cheetah-RM-v0',
         entry_point='env.gym_control_rm:HalfCheetahRMEnv',
         max_episode_steps=1000)

register(
    id='Sign-RM-v0',
         entry_point='env.gym_control_rm:SignRMEnv',
         max_episode_steps=1000)

register(
    id='Sign-RM-v1',
         entry_point='env.gym_control_rm:SignRMEnv_r',
         max_episode_steps=1000)
register(
    id='Sign-v1',
         entry_point='env.gym_control_rm:SignRMEnvEmpty_r',
         max_episode_steps=1000)

register(
    id='Sign-v0',
         entry_point='env.gym_control_rm:SignRMEnvEmpty',
         max_episode_steps=1000)

register(
    id='Sign-RM-Evaluate-v0',
         entry_point='env.gym_control_rm:SignRMEnvEvaluate',
         max_episode_steps=1000)

register(
    id='Sign-Evaluate-v0',
         entry_point='env.gym_control_rm:SignEnvEvaluate',
         max_episode_steps=1000)

register(
    id='Half-Cheetah-RM-v1',
         entry_point='env.gym_control_rm:HalfCheetahRMEnvEmptyRM',
         max_episode_steps=1000)

register(
    id='Cartpole-RM-v0',
         entry_point='env.gym_control_rm:CartPoleRMEnv',
         max_episode_steps=200,)

register(
    id='Cartpole-RM-v1',
         entry_point='env.gym_control_rm:CartPoleRMEnvSparse',
         max_episode_steps=200,)

register(
    id='Cartpole-RM-v2',
         entry_point='env.gym_control_rm:CartPoleRMEnvTransfer',
         max_episode_steps=200,)

register(
    id='Cartpole-NoRM-v1',
         entry_point='env.gym_control_rm:CartPoleNoRMEnvSparse',
         max_episode_steps=200,)

register(
    id='Cartpole-NoRM-v2',
         entry_point='env.gym_control_rm:CartPoleNoRMEnvTransfer',
         max_episode_steps=200,)

register(
    id='Freeway-RM-v1',
    entry_point='env.atari_rm:FreewayRMEnvEmptyRM',
)

register(
    id='Freeway-RM-Evaluate-v0',
    entry_point='env.atari_rm:FreewayRMEnvEvaluate',
)

register(
    id='Freeway-RM-v0',
         entry_point='env.atari_rm:FreewayRMEnvRM',)

register(
    id='Half-Cheetah-RM-Evaluate-v0',
         entry_point='env.gym_control_rm:HalfCheetahRMEnvEvaluate',
         max_episode_steps=1000)

register(
    id='Half-Cheetah-v0',
         entry_point='env.gym_control_rm:HalfCheetahEnvReward',
         max_episode_steps=1000)

register(
    id='Half-Cheetah-v1',
         entry_point='env.gym_control_rm:HalfCheetahEnvOriginal',
         max_episode_steps=1000)

register(
    id='Half-Cheetah-Evaluate-v0',
         entry_point='env.gym_control_rm:HalfCheetahEnvEvaluate',
         max_episode_steps=1000)