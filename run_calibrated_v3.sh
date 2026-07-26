#!/usr/bin/env bash
# Resume da xarm_pose_calibrated_v1/ckpt-1800 (miglior checkpoint VALIDATO:
# frozen reward 150.3 > 134.5 di ckpt-600; finish 50%, pos 3.4cm, orient 12.7).
# ckpt-1800 main-dir ha replay-1800.npz => resume restaura anche il replay buffer
# (nessun warmup a freddo, update partono subito).
#
# Fix chiave vs v1:
#  --best-metric reward_mean  -> auto-recovery HA un baseline e SCATTA sul collasso
#                                (v1 con success_rate=0 aveva recovery inerte).
# Fine-tuning conservativo per l'ultimo miglio (frozen success 0 -> >0):
#  --critic-learning-rate 1e-4 --policy-update-every 4  -> critic/attore lenti, stabili
#  --resume-actor-learning-rate 1e-5 (default)          -> non fa driftare la policy buona
#  --min-alpha 0.02  -> entropia bassa = policy piu deterministica/precisa
#                       (rete di sicurezza contro collasso = auto-recovery, non entropia alta)
cd /home/fedyfausto/Lavoro/GodotAgents/godot-gymnasium-keras-marl
exec python/.venv/bin/python python/train.py \
  --algorithm sac \
  --godot-bin /home/fedyfausto/Godot/Godot_v4.6.2-stable_linux.x86_64 \
  --godot-project godot \
  --godot-scene res://scenarios/robotarms/XarmScenario.tscn \
  --num-envs 8 --base-port 7200 \
  --num-episodes 8000 --max-steps-per-episode 300 --physics-frames-per-step 3 \
  --batch-size 256 --replay-warmup 20000 --random-exploration-episodes 40 \
  --resume --resume-checkpoint checkpoints/xarm_pose_calibrated_v1/ckpt-1800 \
  --critic-learning-rate 1e-4 --policy-update-every 4 \
  --critic-warmup-updates 1000 \
  --min-alpha 0.02 \
  --no-caps --grad-clip-adaptive --grad-clip-norm 10 \
  --auto-recovery --best-checkpoint --best-metric reward_mean \
  --best-evaluation-every 100 --best-evaluation-episodes 20 --best-evaluation-training-episode 2200 \
  --collector-mode async --checkpoint-every 50 \
  --checkpoint-dir checkpoints/xarm_pose_calibrated_v3 \
  --headless --log-format compact --dashboard --dashboard-port 8770
