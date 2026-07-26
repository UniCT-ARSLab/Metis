#!/usr/bin/env bash
# Resume da xarm_pose_calibrated_v3/best/ckpt-2100 (frozen reward 165.2, miglior policy finora)
# CON REWARD CORRETTO: get_pose_tracking_reward ora ha un termine di PRECISIONE (precision_distance
# 0.04m) oltre a quello coarse (near_target_distance 0.20m). Il coarse era quasi piatto negli ultimi
# cm (3.2cm->2cm guadagnava solo ~0.06) -> la policy plateava sopra il gate da 2cm e frozen success
# restava 0 pur con reward che saliva (156->165). Il termine fine rende l'ultimo tratto il piu ripido
# -> spinge il tool sotto i 2cm.
#
# best/ dir NON salva replay -> warmup a freddo 20k: VOLUTO, cosi il replay si riempie con transizioni
# sotto il NUOVO reward (non stale). Poi 1000 step critic-only ricalibrano il critic, poi l'attore
# fa fine-tuning con il gradiente piu ripido.
cd /home/fedyfausto/Lavoro/GodotAgents/godot-gymnasium-keras-marl
exec python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --num-envs 8 --base-port 7200 \
  --num-episodes 8000 --max-steps-per-episode 300 --physics-frames-per-step 3 \
  --batch-size 256 --replay-warmup 20000 --random-exploration-episodes 40 \
  --resume --resume-checkpoint checkpoints/xarm_pose_calibrated_v3/best/ckpt-2100 \
  --critic-learning-rate 1e-4 --policy-update-every 4 \
  --critic-warmup-updates 1000 \
  --min-alpha 0.02 \
  --no-caps --grad-clip-adaptive --grad-clip-norm 10 \
  --auto-recovery --best-checkpoint --best-metric reward_mean \
  --best-evaluation-every 100 --best-evaluation-episodes 20 --best-evaluation-training-episode 2200 \
  --collector-mode async --checkpoint-every 50 \
  --checkpoint-dir checkpoints/xarm_pose_calibrated_v4 \
  --headless --log-format compact --dashboard --dashboard-port 8770
