# apple_pie

## Teleop runbook

```

ls /dev/ | grep ACM 
to check ports, apply ports sequentially:
jason leader: /dev/ttyACM0
jason follower: /dev/ttyACM1
cv leader: /dev/ttyACM2
cv follower: /dev/ttyACM3
```

```
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
sudo chmod 666 /dev/ttyACM2
sudo chmod 666 /dev/ttyACM3


conda activate tinyenv

python lerobot/scripts/control_robot.py   --robot.type=so100   --robot.cameras='{}'   --control.type=calibrate   --control.arms='["main_follower"]'

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate

python lerobot/scripts/control_robot.py --robot.type=so100 --control.type=teleoperate
```

## Cameras

```
Order of connection:
webcam: 0
main: 2
cv: 4

python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras
```

## Collect data

huggingface-cli login

```
REPO_ID="jchun/so100_pickplace_small_$(date +%Y%m%d_%H%M%S)"
echo "Save to: ${REPO_ID}"
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.single_task="Grasp items from white bowl and place in black tray" \
  --control.fps=30 \
  --control.repo_id=${REPO_ID} \
  --control.tags='["pickplace"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=120 \
  --control.reset_time_s=30 \
  --control.num_episodes=100 \
  --control.push_to_hub=true

sh record.sh

python lerobot/scripts/visualize_dataset.py \
  --repo-id jchun/so100_pickplace

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=jchun/so100_pickplace_small_20250322_163234 \
  --control.episode=0
  
```

## inference

```
# gr00t
rm -r  ~/.cache/huggingface/hub/models--nahidalam--nv
idia-gr00t/
conda activate apple_pie

python scripts/inference_service.py --server \
    --model_path nahidalam/nvidia-gr00t \
    --embodiment_tag new_embodiment \
    --data_config bimanual_so100 \
    --denoising_steps 4

python getting_started/examples/eval_gr00t_so100.py \
 --host 0.0.0.0 \
 --port 5555 \
 --action_horizon 16

 # ACT
 
```
