#!/usr/bin/env bash
# Resume conservativo da xarm_pose_calibrated_v1/best/ckpt-600 (picco reward 134).
# Fix principale vs v1: --best-metric reward_mean cosi auto-recovery ha un baseline
# e SCATTA davvero sul collasso (v1 aveva success_rate sempre 0 -> recovery inerte).
cd /home/fedyfausto/Lavoro/GodotAgents/godot-gymnasium-keras-marl
exec python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --num-envs 8 --base-port 7200 \
  --num-episodes 8000 --max-steps-per-episode 300 --physics-frames-per-step 3 \
  --batch-size 256 --replay-warmup 20000 --random-exploration-episodes 40 \
  --resume --resume-checkpoint checkpoints/xarm_pose_calibrated_v1/best/ckpt-600 \
  --critic-learning-rate 1e-4 --policy-update-every 4 \
  --critic-warmup-updates 1000 \
  --min-alpha 0.05 \
  --no-caps --grad-clip-adaptive --grad-clip-norm 10 \
  --auto-recovery --best-checkpoint --best-metric reward_mean \
  --best-evaluation-every 100 --best-evaluation-episodes 20 --best-evaluation-training-episode 2200 \
  --collector-mode async --checkpoint-every 50 \
  --checkpoint-dir checkpoints/xarm_pose_calibrated_v2 \
  --headless --log-format compact --dashboard --dashboard-port 8770
