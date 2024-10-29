export MUJOCO_GL=egl
export DATA_DIR=data
JOB_NAME=tdmpc_push_cube
# hydra.run.dir=outputs/train/$(date +'%Y-%m-%d/%H-%M-%S')_${JOB_NAME} \

python lerobot/scripts/train.py \
  hydra.job.name=$JOB_NAME \
  hydra.run.dir=outputs/train/2024-10-27/14-21-02_tdmpc_push_cube \
  env=koch_real \
  dataset_repo_id=push_cube \
  policy=tdmpc_real \
  training.log_freq=100 \
  training.offline_steps=10000 \
  training.save_freq=1000 \
  training.save_checkpoint=true \
  training.num_workers=4 \
  training.lr=2e-4 \
  training.batch_size=512 \
  training.online_steps=2000000 \
  training.online_sampling_ratio=0.9 \
  training.do_online_rollout_async=false \
  training.online_rollout_n_episodes=20 \
  training.grad_clip_norm=10.0 \
  +training.do_seed_online_buffer_with_offline_data=false \
  +lerobot_dataset_v2_image_mode=video \
  training.online_update_to_data_ratio=1 \
  training.online_buffer_capacity=30000 \
  training.online_buffer_seed_size=100 \
  eval.n_action_buffer=2 \
  device=cuda \
  use_amp=true \
  wandb.enable=false \
  wandb.disable_artifact=true \
  resume=true

