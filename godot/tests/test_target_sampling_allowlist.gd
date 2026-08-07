extends SceneTree
## Unit tests for TargetSamplingRegion3D.allowed_cells (M1.1 hardening).
## Verifies: empty allowlist = all cells; a set allowlist restricts sampling to exactly those
## cells; fail-closed on a broken allowlist (out-of-range indices -> no cells, cell -1, never the
## whole region); active_cells() reflects the effective set.

func _make_region(grid: Vector3i, allowed: PackedInt32Array):
	var region = load("res://addons/metis/runtime/agent/sampling/TargetSamplingRegion3D.gd").new()
	region.grid_size = grid
	region.allowed_cells = allowed
	var cs := CollisionShape3D.new()
	cs.name = "CollisionShape3D"
	var box := BoxShape3D.new()
	box.size = Vector3(0.4, 0.3, 0.4)
	cs.shape = box
	region.add_child(cs)
	# Not added to the SceneTree root on purpose: an Area3D needs a 3D physics world that a bare
	# `--script` SceneTree does not set up. Cell sampling is pure geometry (get_node child +
	# CollisionShape.to_global), so it works without being in the tree.
	return region


func _sample_cells(region, draws: int) -> Dictionary:
	var rng := RandomNumberGenerator.new()
	rng.seed = 4242
	var counts := {}
	for i in range(draws):
		var s: Dictionary = region.sample_transform(rng)
		var c := int(s.get("cell", -99))
		counts[c] = int(counts.get(c, 0)) + 1
	return counts


func _initialize() -> void:
	var failures: Array[String] = []

	# 1) empty allowlist -> every one of the 48 cells appears.
	var r_all = _make_region(Vector3i(4, 3, 4), PackedInt32Array())
	var all_counts := _sample_cells(r_all, 48 * 5)
	for c in range(48):
		if not all_counts.has(c):
			failures.append("empty-allowlist: cell %d never sampled" % c)
	if all_counts.has(-1):
		failures.append("empty-allowlist: produced a degenerate cell -1")

	# 2) allowlist [0,5,10] -> only those cells, each sampled, nothing else.
	var r_sub = _make_region(Vector3i(4, 3, 4), PackedInt32Array([0, 5, 10]))
	var sub_counts := _sample_cells(r_sub, 60)
	for c in sub_counts.keys():
		if c not in [0, 5, 10]:
			failures.append("allowlist[0,5,10]: sampled forbidden cell %d" % c)
	for c in [0, 5, 10]:
		if not sub_counts.has(c):
			failures.append("allowlist[0,5,10]: cell %d never sampled" % c)
	if r_sub.active_cells().size() != 3:
		failures.append("active_cells() size %d != 3" % r_sub.active_cells().size())

	# 3) fail-closed: out-of-range allowlist -> no real cells (only cell -1), never the region.
	var r_bad = _make_region(Vector3i(4, 3, 4), PackedInt32Array([999, 1000]))
	var bad_counts := _sample_cells(r_bad, 30)
	for c in bad_counts.keys():
		if c != -1:
			failures.append("fail-closed: broken allowlist sampled real cell %d" % c)
	if not bad_counts.has(-1):
		failures.append("fail-closed: expected degenerate cell -1, got %s" % str(bad_counts))
	if r_bad.active_cells().size() != 0:
		failures.append("fail-closed: active_cells() should be empty, got %d" % r_bad.active_cells().size())

	# 4) partially-valid allowlist -> keep valid, drop invalid.
	var r_mix = _make_region(Vector3i(4, 3, 4), PackedInt32Array([3, 999, 7]))
	var mix_counts := _sample_cells(r_mix, 40)
	for c in mix_counts.keys():
		if c not in [3, 7]:
			failures.append("partial allowlist: sampled unexpected cell %d" % c)

	# 5) 'valid' field: true for a real sample, false for a broken/degenerate one.
	var rng2 := RandomNumberGenerator.new()
	rng2.seed = 1
	var good_sample: Dictionary = r_sub.sample_transform(rng2)
	if good_sample.get("valid", false) != true:
		failures.append("valid-field: real sample not marked valid")
	var bad_sample: Dictionary = r_bad.sample_transform(rng2)
	if bad_sample.get("valid", true) != false:
		failures.append("valid-field: broken-allowlist sample not marked invalid")

	# 6) forced_cell IN the allowlist -> every draw is exactly that cell, marked valid (M5 demos).
	var rng3 := RandomNumberGenerator.new()
	rng3.seed = 7
	var forced_counts := {}
	for i in range(20):
		var s: Dictionary = r_sub.sample_transform(rng3, Basis.IDENTITY, 5)
		if s.get("valid", false) != true:
			failures.append("forced_cell(5): sample not valid")
		forced_counts[int(s.get("cell", -99))] = true
	if forced_counts.size() != 1 or not forced_counts.has(5):
		failures.append("forced_cell(5): expected only cell 5, got %s" % str(forced_counts.keys()))

	# 7) forced_cell NOT in the allowlist -> fail-closed (cell -1, valid false), never substituted.
	var forbidden: Dictionary = r_sub.sample_transform(rng3, Basis.IDENTITY, 3)
	if forbidden.get("valid", true) != false:
		failures.append("forced_cell(3 not allowed): should be invalid (fail-closed)")
	if int(forbidden.get("cell", -99)) != -1:
		failures.append("forced_cell(3 not allowed): cell should be -1, got %d" % int(forbidden.get("cell", -99)))

	# 8) forced_cell == -1 -> normal stratified sampling (a real, valid cell from the allowlist).
	var normal: Dictionary = r_sub.sample_transform(rng3, Basis.IDENTITY, -1)
	if normal.get("valid", false) != true or int(normal.get("cell", -1)) not in [0, 5, 10]:
		failures.append("forced_cell(-1): expected a normal allowlisted sample, got %s" % str(normal))

	if failures.is_empty():
		print("test_target_sampling_allowlist: PASS (8 checks)")
		quit(0)
	else:
		for f in failures:
			push_error("FAIL: " + f)
		print("test_target_sampling_allowlist: FAIL (%d)" % failures.size())
		quit(1)
