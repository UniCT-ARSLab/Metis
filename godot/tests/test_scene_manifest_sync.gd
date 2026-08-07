extends SceneTree
## Guards against scene <-> manifest divergence (M1.1 hardening): the reach-hold scene's baked
## region allowlists and geometry must match the canonical region manifest, so editing one
## without the other fails CI instead of silently shipping a mismatched curriculum.

const SCENE := "res://scenarios/robotarms/openarm_reach_hold_scenario.tscn"
const MANIFEST := "res://scenarios/robotarms/openarm_reach_hold_region_manifest.json"


func _sorted(arr) -> Array:
	var out: Array = []
	for v in arr:
		out.append(int(v))
	out.sort()
	return out


func _initialize() -> void:
	var failures: Array[String] = []

	var f := FileAccess.open(MANIFEST, FileAccess.READ)
	if f == null:
		push_error("manifest not found: " + MANIFEST)
		quit(1)
		return
	var manifest = JSON.parse_string(f.get_as_text())
	f.close()

	var packed := load(SCENE) as PackedScene
	if packed == null:
		push_error("scene not found: " + SCENE)
		quit(1)
		return
	var scenario := packed.instantiate() as Node3D
	var bridge := scenario.get_node_or_null("BridgeServer")
	if bridge:
		scenario.remove_child(bridge)
		bridge.free()
	root.add_child(scenario)
	await process_frame

	var active: Dictionary = manifest["active_shells_path_confirmed"]
	var region_nodes := {
		"easy": scenario.get_node_or_null("EasyArea"),
		"medium": scenario.get_node_or_null("MediumArea"),
		"hard": scenario.get_node_or_null("HardArea"),
	}
	var expected_grid: Array = manifest["geometry"]["grid"]

	for key in ["easy", "medium", "hard"]:
		var region = region_nodes[key]
		if region == null:
			failures.append("scene missing region node for '%s'" % key)
			continue
		var scene_cells := _sorted(region.allowed_cells)
		var manifest_cells := _sorted(active[key])
		if scene_cells != manifest_cells:
			failures.append("region '%s' allowed_cells diverge:\n  scene=%s\n  manifest=%s"
				% [key, str(scene_cells), str(manifest_cells)])
		var g = region.grid_size
		if [g.x, g.y, g.z] != [int(expected_grid[0]), int(expected_grid[1]), int(expected_grid[2])]:
			failures.append("region '%s' grid_size %s != manifest %s" % [key, str(g), str(expected_grid)])

	if failures.is_empty():
		print("test_scene_manifest_sync: PASS (regions match manifest)")
		quit(0)
	else:
		for x in failures:
			push_error("FAIL: " + x)
		print("test_scene_manifest_sync: FAIL (%d)" % failures.size())
		quit(1)
