extends SceneTree


func _initialize() -> void:
	var manager := MetisRuntimeManager.new()
	var source_root := manager.development_source_root()
	var passed := not source_root.is_empty()
	passed = passed and FileAccess.file_exists(
		source_root.path_join("pyproject.toml"))
	passed = passed and not manager.suggested_python().is_empty()

	var dialog := MetisRuntimeSetupDialog.new()
	root.add_child(dialog)
	await process_frame
	dialog.configure(manager)
	await process_frame
	passed = passed and dialog.title == "Metis Runtime Setup"
	var expected_button := (
		"Set Up Runtime"
		if manager.has_packaged_payload()
		else "Validate Runtime")
	passed = passed and dialog.get_ok_button().text == expected_button
	passed = passed and dialog._profile.item_count == 3
	passed = passed and dialog._include_sb3 != null
	passed = passed and dialog._include_dashboard.button_pressed
	passed = passed and dialog._include_export.button_pressed
	passed = passed and dialog._include_export.text == "Policy export"
	dialog.free()

	var train_dialog := MetisTrainDialog.new()
	root.add_child(train_dialog)
	await process_frame
	train_dialog.configure(manager)
	await process_frame
	var command_prefix: PackedStringArray = train_dialog._command_prefix()
	var train_args: PackedStringArray = train_dialog._build_args()
	passed = passed and command_prefix.size() >= 2
	passed = passed and command_prefix[0] == manager.suggested_python()
	passed = passed and command_prefix[1].ends_with("python/train.py")
	var project_index := train_args.find("--godot-project")
	passed = passed and project_index >= 0
	passed = passed and train_args[project_index + 1] == (
		ProjectSettings.globalize_path("res://").trim_suffix("/"))
	train_dialog.free()

	if not passed:
		push_error(
			"Metis runtime setup test failed: source=%s python=%s" % [
				source_root,
				manager.suggested_python(),
			])
		quit(1)
		return
	print("Metis runtime setup test passed")
	quit(0)
