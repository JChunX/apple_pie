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

conda activate tinyenv && python lerobot/scripts/control_robot.py --robot.type=so100 --control.type=teleoperate
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

```
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.fps=30 \
  --control.repo_id=jchun/so100_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=true
```
