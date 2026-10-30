extends SceneTree


func _initialize() -> void:
	if not Engine.is_editor_hint():
		push_error("Training control option test requires --editor")
		quit(1)
		return
	var dialog := MetisTrainDialog.new()
	root.add_child(dialog)
	await process_frame
	dialog.configure(MetisRuntimeManager.new())
	dialog._ensure_argspec()
	for _attempt in 100:
		if not dialog._argspec_is_stale():
			break
		await create_timer(0.2).timeout
	var failures: Array[String] = []
	for algorithm in ["sac", "ppo", "dqn", "ddpg", "td3"]:
		for index in dialog._algorithm.item_count:
			if dialog._algorithm.get_item_text(index) == algorithm:
				dialog._algorithm.select(index)
		dialog._options_algo = ""
		dialog._on_algorithm_changed()
		for destination in ["stop_when_solved", "solved_patience", "lr_decay_on_plateau"]:
			if not dialog._algo_controls.has(destination):
				failures.append("%s_missing_%s" % [algorithm, destination])
		for destination in ["solved_certification_episodes", "solved_certification_confidence", "recovery_lr_cumulative"]:
			if not dialog._all_controls.has(destination):
				failures.append("%s_missing_advanced_%s" % [algorithm, destination])
		if dialog._algo_controls.has("stop_when_solved"):
			dialog._algo_controls["stop_when_solved"]["control"].text = "0.95"
			var arguments: PackedStringArray = dialog._build_args()
			var flag := arguments.find("--stop-when-solved")
			if flag < 0 or arguments[flag + 1] != "0.95":
				failures.append("%s_solved_argv" % algorithm)
		if algorithm in ["sac", "ddpg", "td3"] and not dialog._algo_controls.has("actor_ema_decay"):
			failures.append("%s_missing_ema" % algorithm)
		if algorithm == "ppo":
			for destination in ["demo_path", "demo_bc_epochs", "ppo_demo_coef"]:
				if not dialog._algo_controls.has(destination):
					failures.append("ppo_missing_%s" % destination)
			if dialog._algo_controls.has("actor_ema_decay"):
				failures.append("ppo_unsupported_ema")
		await process_frame
	dialog.queue_free()
	await process_frame
	await process_frame
	if not failures.is_empty():
		push_error("Training control option test failed: %s" % [failures])
		quit(1)
		return
	print("Training control option test passed")
	quit(0)
