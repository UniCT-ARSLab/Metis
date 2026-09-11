@tool
extends RefCounted
## Builds editable forms out of a Metis entry point's argparse specification.
##
## The form is generated from `.metis/argspec.json`, keeping editor defaults in sync with the CLI.
## Only changed values are emitted as command-line arguments.

const LABEL_WIDTH := 160

# Parser changes invalidate the cached specification.
const SPEC_SOURCES := [
	"core/argspec.py", "core/training.py", "algorithms/sac.py", "algorithms/ppo.py",
	"algorithms/dqn.py", "algorithms/common.py", "run.py", "recorder.py",
]

var controls := {}  # dest -> {"control": Control, "arg": Dictionary}

static func spec_path() -> String:
	return ProjectSettings.globalize_path("res://.metis").path_join("argspec.json")


static func is_spec_stale(source_root: String) -> bool:
	var path := spec_path()
	if not FileAccess.file_exists(path):
		return true
	if source_root.is_empty():
		# Packaged runtimes have no source tree to compare against.
		return false
	var spec_time := FileAccess.get_modified_time(path)
	for relative in SPEC_SOURCES:
		var source := source_root.path_join(relative)
		if FileAccess.file_exists(source) and FileAccess.get_modified_time(source) > spec_time:
			return true
	return false


static func ensure_spec(python: String, source_root: String) -> void:
	## Regenerates a stale specification without blocking the editor on Python imports.
	if not is_spec_stale(source_root):
		return
	if not FileAccess.file_exists(python):
		return
	var path := spec_path()
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	if not source_root.is_empty():
		var script := source_root.path_join("core/argspec.py")
		if FileAccess.file_exists(script):
			OS.create_process(python, PackedStringArray([script, "--all", "--output", path]))
	else:
		OS.create_process(python, PackedStringArray(["-m", "core.argspec", "--all", "--output", path]))


static func load_spec() -> Dictionary:
	var path := spec_path()
	if not FileAccess.file_exists(path):
		return {}
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	return parsed if parsed is Dictionary else {}


static func spec_for(entry_point: String) -> Array:
	## Returns the options for an entry point in declaration order.
	var spec: Variant = load_spec().get(entry_point, [])
	return spec if spec is Array else []


static func spec_by_dest(entry_point: String) -> Dictionary:
	var out := {}
	for arg in spec_for(entry_point):
		if arg is Dictionary:
			out[str(arg.get("dest", ""))] = arg
	return out


static func make_grid() -> GridContainer:
	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 12)
	grid.add_theme_constant_override("v_separation", 6)
	grid.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	return grid


static func row(grid: GridContainer, label_text: String, control: Control) -> void:
	var label := Label.new()
	label.text = label_text
	label.custom_minimum_size = Vector2(LABEL_WIDTH, 0)
	grid.add_child(label)
	control.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	grid.add_child(control)


static func clear(box: Control) -> void:
	## Detaches children immediately so the form can be rebuilt in the same frame.
	for child in box.get_children():
		box.remove_child(child)
		child.queue_free()


static func format_default(arg: Dictionary, value: Variant) -> String:
	## Formats JSON defaults as the CLI expects to receive them.
	var kind := str(arg.get("kind", "str"))
	if value is Array:
		var parts := PackedStringArray()
		for item in value:
			parts.append(format_default(arg, item))
		return " ".join(parts)
	if kind == "int" and (value is float or value is int):
		return str(int(round(float(value))))
	if kind == "float" and value is float and value == floor(value) and absf(value) < 1e15:
		return str(int(value))
	return str(value)


static func mark_validity(edit: LineEdit, arg: Dictionary) -> void:
	## Marks invalid input without rewriting what the user typed.
	var apply := func(text: String):
		if value_error(arg, text).is_empty():
			edit.remove_theme_color_override("font_color")
			edit.tooltip_text = str(arg.get("help", ""))
		else:
			edit.add_theme_color_override("font_color", Color(1.0, 0.45, 0.45))
			edit.tooltip_text = value_error(arg, text)
	edit.text_changed.connect(apply)
	apply.call(edit.text)


static func value_error(arg: Dictionary, text: String) -> String:
	## Returns the CLI validation error for a value, or an empty string when valid.
	var trimmed := text.strip_edges()
	if trimmed.is_empty():
		return ""
	var kind := str(arg.get("kind", "str"))
	if kind not in ["int", "float"]:
		return ""
	var flag_list: Array = arg.get("flags", [])
	var flag := str(flag_list[0]) if not flag_list.is_empty() else str(arg.get("dest", "?"))
	var tokens := trimmed.split(" ", false) if takes_many_values(arg) else PackedStringArray([trimmed])
	for token in tokens:
		var value := token.strip_edges()
		if value.is_empty():
			continue
		if kind == "int" and not value.is_valid_int():
			return "%s expects a whole number, got \"%s\"" % [flag, value]
		if kind == "float" and not value.is_valid_float():
			return "%s expects a number, got \"%s\"" % [flag, value]
	return ""


static func takes_many_values(arg: Dictionary) -> bool:
	var nargs: Variant = arg.get("nargs", null)
	if nargs == null:
		return false
	var text := str(nargs)
	return text in ["+", "*"] or (text.is_valid_int() and text.to_int() > 0)


func add(grid: GridContainer, arg: Dictionary, label_override: String = "") -> Control:
	## Adds and registers one option row.
	var kind := str(arg.get("kind", "str"))
	var control: Control
	if kind in ["bool", "flag_true", "flag_false"]:
		var check := CheckBox.new()
		check.button_pressed = bool(arg.get("default", false))
		control = check
	elif kind == "choice":
		var option := OptionButton.new()
		var choices: Array = arg.get("choices", [])
		for i in choices.size():
			option.add_item(str(choices[i]))
			if str(choices[i]) == str(arg.get("default", "")):
				option.select(i)
		control = option
	else:
		var edit := LineEdit.new()
		var default_value: Variant = arg.get("default", null)
		if default_value != null:
			edit.text = format_default(arg, default_value)
		elif takes_many_values(arg):
			edit.placeholder_text = "space-separated, e.g. 256 256"
		else:
			edit.placeholder_text = "(unset)"
		mark_validity(edit, arg)
		control = edit
	control.tooltip_text = str(arg.get("help", ""))
	var label := label_override
	if label.is_empty():
		var flag_list: Array = arg.get("flags", [])
		label = (str(flag_list[0]).trim_prefix("--") if not flag_list.is_empty()
			else str(arg.get("dest", "")))
	row(grid, label, control)
	controls[str(arg.get("dest", ""))] = {"control": control, "arg": arg}
	return control


func add_all(box: VBoxContainer, entry_point: String, skip: PackedStringArray,
		collapsed := true) -> int:
	## Adds the remaining options in collapsible argparse groups.
	var by_group := {}
	var order: Array[String] = []
	for arg in spec_for(entry_point):
		if not (arg is Dictionary) or str(arg.get("dest", "")) in skip:
			continue
		var group_name := str(arg.get("group", ""))
		if group_name.is_empty():
			group_name = "General"
		if not by_group.has(group_name):
			by_group[group_name] = []
			order.append(group_name)
		by_group[group_name].append(arg)
	var total := 0
	for group_name in order:
		var grid := make_grid()
		grid.visible = not collapsed
		var count: int = by_group[group_name].size()
		var section := Button.new()
		section.toggle_mode = true
		section.button_pressed = not collapsed
		section.alignment = HORIZONTAL_ALIGNMENT_LEFT
		section.text = "%s %s (%d)" % [("▼" if not collapsed else "▸"), group_name, count]
		section.add_theme_color_override("font_color", Color(0.6, 0.72, 1.0))
		section.toggled.connect(func(pressed):
			grid.visible = pressed
			section.text = "%s %s (%d)" % [("▼" if pressed else "▸"), group_name, count])
		box.add_child(section)
		box.add_child(grid)
		for arg in by_group[group_name]:
			add(grid, arg)
			total += 1
	return total


func value_of(dest: String) -> Variant:
	## Returns the current value of a registered control.
	if not controls.has(dest):
		return null
	var entry: Dictionary = controls[dest]
	var control = entry["control"]
	var kind := str(entry["arg"].get("kind", "str"))
	if kind in ["bool", "flag_true", "flag_false"]:
		return control.button_pressed
	if kind == "choice":
		return control.get_item_text(control.selected)
	return str(control.text).strip_edges()


func set_value(dest: String, value: Variant) -> void:
	if not controls.has(dest):
		return
	var entry: Dictionary = controls[dest]
	var control = entry["control"]
	var kind := str(entry["arg"].get("kind", "str"))
	if kind in ["bool", "flag_true", "flag_false"]:
		control.button_pressed = bool(value)
	elif kind == "choice":
		for i in control.item_count:
			if control.get_item_text(i) == str(value):
				control.select(i)
	else:
		control.text = str(value)


func flags() -> PackedStringArray:
	## Returns changed values as command-line tokens.
	var out := PackedStringArray()
	for dest in controls:
		var entry: Dictionary = controls[dest]
		var arg: Dictionary = entry["arg"]
		var control = entry["control"]
		var kind := str(arg.get("kind", "str"))
		var flag_list: Array = arg.get("flags", [])
		if flag_list.is_empty():
			continue
		if kind == "bool":
			var value: bool = control.button_pressed
			if value != bool(arg.get("default", false)):
				var positive := ""
				var negative := ""
				for flag in flag_list:
					if str(flag).begins_with("--no-"):
						negative = str(flag)
					else:
						positive = str(flag)
				out.append(positive if value else negative)
		elif kind == "flag_true":
			if control.button_pressed and not bool(arg.get("default", false)):
				out.append(str(flag_list[0]))
		elif kind == "flag_false":
			if not control.button_pressed and bool(arg.get("default", true)):
				out.append(str(flag_list[0]))
		elif kind == "choice":
			var selected: String = control.get_item_text(control.selected)
			if selected != str(arg.get("default", "")):
				out.append(str(flag_list[0]))
				out.append(selected)
		else:
			var text := str(control.text).strip_edges()
			if not text.is_empty() and text != format_default(arg, arg.get("default", "")):
				out.append(str(flag_list[0]))
				if takes_many_values(arg):
					# argparse expects one token per item for nargs options.
					for token in text.split(" ", false):
						if not token.strip_edges().is_empty():
							out.append(token.strip_edges())
				else:
					out.append(text)
	return out


func validation_errors() -> PackedStringArray:
	## Returns one message for each value the CLI would reject.
	var problems := PackedStringArray()
	for dest in controls:
		var entry: Dictionary = controls[dest]
		var control = entry["control"]
		if not (control is LineEdit):
			continue
		var problem := value_error(entry["arg"], str(control.text))
		if not problem.is_empty():
			problems.append(problem)
	return problems
