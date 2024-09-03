import argparse
import logging
import time
from collections import defaultdict
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from termcolor import colored
from torch import nn

from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.policies.rollout_wrapper import PolicyRolloutWrapper
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.koch import KochRobot
from lerobot.common.utils.digital_twin import DigitalTwin
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from lerobot.common.vision import GoalSetter, HSVSegmenter
from lerobot.scripts.eval import get_pretrained_policy_path


def busy_wait(seconds: float):
    # Significantly more accurate than `time.sleep`, and mandatory for our use case,
    # but it consumes CPU cycles.
    # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        time.sleep(0.0001)


def rollout(
    robot: KochRobot,
    policy: Policy,
    fps: float,
    n_action_buffer: int = 0,
    warmup_s: float = 5.0,
    relative_actions_max: float | None = None,
    max_steps: int | None = None,
    visualize: bool = False,
):
    segmenter = HSVSegmenter()
    goal_setter = GoalSetter.from_mask_file("outputs/goal_mask.npy")
    goal_mask = goal_setter.get_goal_mask()
    where_goal = np.where(goal_mask > 0)

    digital_twin = DigitalTwin()
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)
    policy_rollout_wrapper = PolicyRolloutWrapper(policy, fps=fps, n_action_buffer=n_action_buffer)

    policy_rollout_wrapper.reset()

    step = 0
    start_time = time.perf_counter()

    def to_relative_time(t):
        return t - start_time

    episodes_data = defaultdict(list)

    period = 1 / fps
    to_visualize = {}
    reward = 0
    success = False
    first_follower_pos = None  # Will be held during the warmup
    while True:
        is_dropped_cycle = False
        over_time = False
        start_step_time = to_relative_time(time.perf_counter())
        is_warmup = start_step_time <= warmup_s
        observation: dict[str, torch.Tensor] = robot.capture_observation()
        annotated_img = None
        if not is_warmup:
            episodes_data["index"].append(step)
            episodes_data["episode_index"].append(0)
            episodes_data["timestamp"].append(start_step_time)
            episodes_data["frame_index"].append(step)
            for k in observation:
                if k.startswith("observation.image"):
                    episodes_data[k].append(observation[k].permute(2, 0, 1).numpy().astype(float) / 255.0)
                    # img = observation[k].numpy()
                    # if step > 0:
                    #     try:
                    #         obj_mask, annotated_img = segmenter.segment(img)
                    #         reward, success = calc_reward(
                    #             obj_mask,
                    #             goal_mask,
                    #             episodes_data["action"][-1],
                    #             digital_twin.fk_gripper_tip(observation["observation.state"].numpy())[:3, 3]
                    #         )
                    #     except:
                    #         logging.warning(colored("Failed to compute reward", "yellow"))
                    #         reward = -2
                    #         success = False
                else:
                    episodes_data[k].append(observation[k].numpy())

            if step > 0:
                goal = np.array([87, 82, 91, 65, 3, 0])
                curr = observation["observation.state"].numpy()
                reward = -np.abs(goal - curr).mean() / 10
                success = np.abs(goal - curr).max() <= 3
                if digital_twin.fk_gripper_tip(observation["observation.state"].numpy())[2, -1] < 0.004:
                    reward -= 1
                print("REWARD:", reward, ", SUCCESS:", success)
                episodes_data["next.reward"].append(reward)
                episodes_data["next.success"].append(success)
                episodes_data["next.done"].append(success)

        follower_pos = observation["observation.state"].numpy()
        if first_follower_pos is None:
            first_follower_pos = follower_pos.copy()

        elapsed = to_relative_time(time.perf_counter()) - start_step_time
        if elapsed > period:
            over_time = True
            logging.warning(f"Over time after capturing observation! {elapsed=}")

        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if name.startswith("observation.image"):
                if visualize:
                    to_visualize[name] = observation[name].numpy() if annotated_img is None else annotated_img
                    if start_step_time > warmup_s:
                        to_visualize[name][where_goal] = (
                            to_visualize[name][where_goal]
                            - (to_visualize[name][where_goal] - np.array([255, 255, 255])) // 2
                        )
                        to_visualize[name] = cv2.resize(to_visualize[name], (640, 480))
                        cv2.putText(
                            to_visualize[name],
                            org=(10, 25),
                            color=(255, 255, 255),
                            text=f"{reward=:.3f}",
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            thickness=1,
                        )
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        with torch.inference_mode():
            timeout = (
                period - (to_relative_time(time.perf_counter()) - start_step_time) - 0.025
                if step > 0
                else None
            )
            action_sequence = policy_rollout_wrapper.provide_observation_get_actions(
                observation,
                observation_timestamp=start_step_time,
                first_action_timestamp=start_step_time,
                strict_observation_timestamps=step > 0,
                timeout=timeout,
            )

            if action_sequence is not None:
                action_sequence = action_sequence.squeeze(1)  # remove batch dim

        if action_sequence is not None:
            digital_twin.set_twin_pose(follower_pos, follower_pos + action_sequence.numpy())

        if step == 0:
            # On the first step we should just use the first action. We are guaranteed that action_sequence is
            # not None.
            action = action_sequence[0]
            # We also need to store the next action. If the next action is not available, we adopt the
            # strategy of repeating the current action.
            if len(action_sequence) > 1:
                next_action = action_sequence[1].clone()
            else:
                next_action = action.clone()
                is_dropped_cycle = True
        else:
            # All steps after  the first must use the `next_action` from the previous step.
            action = next_action.clone()
            if action_sequence is not None and len(action_sequence) > 1:
                next_action = action_sequence[1].clone()
            else:
                next_action = action.clone()
                is_dropped_cycle = True

        if visualize:
            for name in to_visualize:
                if is_dropped_cycle:
                    red = np.array([255, 0, 0], dtype=np.uint8)
                    to_visualize[name][:10] = red
                    to_visualize[name][-10:] = red
                    to_visualize[name][:, :10] = red
                    to_visualize[name][:, -10:] = red
                if over_time:
                    purple = np.array([255, 0, 255], dtype=np.uint8)
                    to_visualize[name][:20] = purple
                    to_visualize[name][-20:] = purple
                    to_visualize[name][:, :20] = purple
                    to_visualize[name][:, -20:] = purple
                cv2.imshow(name, cv2.cvtColor(to_visualize[name], cv2.COLOR_RGB2BGR))
                k = cv2.waitKey(1)
                if k == ord("q"):
                    return

        # Order the robot to move
        if is_warmup:
            policy_rollout_wrapper.reset()
            robot.send_action(torch.from_numpy(first_follower_pos))
            logging.info("Warming up.")
        else:
            if relative_actions_max is not None:
                relative_action = action
                # Policy provided relative actions.
                relative_action = torch.clamp(relative_action, -relative_actions_max, relative_actions_max)
                action = torch.from_numpy(follower_pos) + relative_action
            action = robot.send_action(action)
            episodes_data["action"].append(relative_action.numpy())

        elapsed = to_relative_time(time.perf_counter()) - start_step_time
        if elapsed > period:
            logging.warning(colored(f"Step took too long! {elapsed=}", "yellow"))
        else:
            busy_wait(period - elapsed - 0.001)

        if digital_twin.quit_signal_is_set():
            break

        if not is_warmup:
            step += 1

        if step >= max_steps:
            episodes_data["next.done"][-1] = True

        if len(episodes_data["next.done"]) > 0 and episodes_data["next.done"][-1]:
            break

    for k in episodes_data:
        if k.startswith("next."):
            episodes_data[k].append(episodes_data[k][-1])

    for k in episodes_data:
        episodes_data[k] = np.stack(episodes_data[k])

    episodes_data["next.done"][-1] = True

    # Hack: drop the first frame because of first inference being slow.
    for k in episodes_data:
        episodes_data[k] = episodes_data[k][1:]
    episodes_data["frame_index"] -= 1
    episodes_data["index"] -= 1

    policy_rollout_wrapper.close_thread()

    digital_twin.close()

    return {
        "episodes": episodes_data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fps", type=float)
    parser.add_argument("--n-action-buffer", type=int, default=0)
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch_.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--warmup-time-s",
        type=int,
        default=5,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    parser.add_argument("-v", "--visualize", action="store_true")

    args = parser.parse_args()

    init_logging()

    pretrained_policy_path = get_pretrained_policy_path(args.pretrained_policy_name_or_path)

    robot_cfg = init_hydra_config(args.robot_path)
    robot = make_robot(robot_cfg)

    try:
        if not robot.is_connected:
            robot.connect()
        hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), args.policy_overrides)

        # Check device is available
        device = get_safe_torch_device(hydra_cfg.device, log=True)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        set_global_seed(hydra_cfg.seed)

        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))

        assert isinstance(policy, nn.Module)
        policy.eval()

        with torch.no_grad(), torch.autocast(device_type=device.type) if hydra_cfg.use_amp else nullcontext():
            rollout(
                robot,
                policy,
                args.fps,
                n_action_buffer=args.n_action_buffer,
                warmup_s=args.warmup_time_s,
                visualize=args.visualize,
            )

        logging.info("End of eval")
    finally:
        if robot.is_connected:
            # Disconnect manually to avoid a "Core dump" during process
            # termination due to camera threads not properly exiting.
            robot.disconnect()
