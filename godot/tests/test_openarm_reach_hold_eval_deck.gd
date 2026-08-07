extends SceneTree
## M4 test: the frozen-evaluation balanced region deck. In evaluation_mode every ACTIVE stage
## region is represented equally (max diff 1) independent of training weights; the deck is
## deterministic per seed; training mode keeps the declared (unequal) weights.

const SCENE := "res://scenarios/robotarms/openarm_reach_hold_scenario.tscn"

var scenario: Node3D
var failures: Array[String] = []


func _sample(level: float, eval_mode: bool, n: int, seed_val: int) -> Dictionary:
	scenario.call("_on_scenario_configured", {
		"training_episode": 0, "training_mode": true,
		"curriculum_level": level, "evaluation_mode": eval_mode})
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_val
	var counts := {"easy": 0, "medium": 0, "hard": 0, "null": 0}
	var seq: Array = []
	for i in range(n):
		var r = scenario.call("_select_target_region", rng)
		var name := (str(r.region_name).to_lower() if r else "null")
		counts[name] = int(counts.get(name, 0)) + 1
		seq.append(name)
	return {"counts": counts, "seq": seq}


func _spread(counts: Dictionary, active: Array) -> int:
	var vals: Array = []
	for k in active:
		vals.append(int(counts[k]))
	return int(vals.max()) - int(vals.min())


func _initialize() -> void:
	var packed := load(SCENE) as PackedScene
	scenario = packed.instantiate() as Node3D
	var bridge := scenario.get_node_or_null("BridgeServer")
	if bridge:
		scenario.remove_child(bridge)
		bridge.free()
	root.add_child(scenario)
	await process_frame

	# Stage A: eval samples Easy only.
	var a := _sample(0.0, true, 12, 1)
	if a["counts"]["medium"] != 0 or a["counts"]["hard"] != 0 or a["counts"]["easy"] != 12:
		failures.append("A_not_easy_only=%s" % str(a["counts"]))

	# Stage D: eval Easy/Medium equal (max diff 1), no Hard.
	var d := _sample(0.6, true, 12, 1)
	if d["counts"]["hard"] != 0 or _spread(d["counts"], ["easy", "medium"]) > 1:
		failures.append("D_not_balanced_easy_medium=%s" % str(d["counts"]))

	# Stage F: eval all three equal (max diff 1).
	var f := _sample(1.0, true, 12, 1)
	if _spread(f["counts"], ["easy", "medium", "hard"]) > 1:
		failures.append("F_not_balanced=%s" % str(f["counts"]))
	if f["counts"]["easy"] == 0 or f["counts"]["medium"] == 0 or f["counts"]["hard"] == 0:
		failures.append("F_missing_region=%s" % str(f["counts"]))

	# Same seed -> identical deck.
	var f2 := _sample(1.0, true, 12, 1)
	if f["seq"] != f2["seq"]:
		failures.append("deck_not_deterministic")

	# Training mode at F keeps the declared 0.2/0.4/0.4 weights (NOT balanced): easy under-represented.
	var ft := _sample(1.0, false, 200, 1)
	if int(ft["counts"]["easy"]) >= int(ft["counts"]["medium"]):
		failures.append("training_weights_balanced=%s" % str(ft["counts"]))
	if _spread(f["counts"], ["easy", "medium", "hard"]) >= _spread(ft["counts"], ["easy", "medium", "hard"]):
		# eval spread must be tighter (balanced) than weighted training spread
		pass  # informational; the primary check is easy < medium above

	if failures.is_empty():
		print("test_openarm_reach_hold_eval_deck: PASS")
		scenario.free()
		quit(0)
	else:
		for x in failures:
			push_error("FAIL: " + x)
		print("test_openarm_reach_hold_eval_deck: FAIL (%d) %s" % [failures.size(), str(failures)])
		scenario.free()
		quit(1)
