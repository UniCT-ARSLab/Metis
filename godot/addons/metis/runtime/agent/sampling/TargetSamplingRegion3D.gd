@tool
extends Area3D
class_name TargetSamplingRegion3D

## A reusable 3D target volume with stratified sampling.
##
## The box is divided into cells. Every cell is visited once in a shuffled order
## before a new coverage cycle begins, avoiding the clustering of uniform random
## sampling while retaining randomness inside each cell.

@export var region_name: StringName = &"region"
@export var grid_size := Vector3i(4, 3, 4)
@export var sampling_inset := Vector3(0.005, 0.005, 0.005)
@export_node_path("CollisionShape3D") var collision_shape_path := NodePath("CollisionShape3D")
## Optional allowlist of cell indices this region may sample. Empty = every cell (default,
## backward-compatible). When set, stratified sampling visits ONLY these cells, so a workspace
## probe can mask cells that are physically invalid (unreachable / endpoint-collision / joint
## limit) while keeping the rest. Indices outside [0, total_cells) are ignored.
@export var allowed_cells: PackedInt32Array = PackedInt32Array()

var _cell_order: Array[int] = []
var _cell_cursor := 0
var _coverage_cycle := 0
var _cached_grid_size := Vector3i.ZERO
var _cached_allowed_signature := ""
var _warned_invalid_shape := false
var _warned_empty_allowlist := false


func _ready() -> void:
	# Sampling volumes are editor/runtime guides, not physical obstacles.
	collision_layer = 0
	collision_mask = 0
	monitoring = false
	monitorable = false


func reset_sampling_sequence() -> void:
	_cell_order.clear()
	_cell_cursor = 0
	_coverage_cycle = 0
	_cached_grid_size = Vector3i.ZERO
	_cached_allowed_signature = ""


## Cells this region will actually sample (allowlist filtered to range, or all cells).
func active_cells() -> Array[int]:
	return _eligible_cells(total_cells())


## Sample a target inside the region. ``forced_cell`` >= 0 pins the sample to that exact cell
## (deterministic demo generation): it is honoured ONLY when the cell is in the eligible set
## (allowlist-filtered), otherwise the call fails closed (cell -1, valid:false, error) — a forced
## cell that is not allowed must never silently fall back to another cell. ``forced_cell`` == -1
## (default) keeps the shuffled stratified coverage-cycle behaviour.
func sample_transform(
	rng: RandomNumberGenerator,
	target_basis := Basis.IDENTITY,
	forced_cell := -1
) -> Dictionary:
	var collision_shape := _collision_shape()
	var box := collision_shape.shape as BoxShape3D if collision_shape else null
	if box == null:
		if not _warned_invalid_shape:
			_warned_invalid_shape = true
			push_warning(
				"TargetSamplingRegion3D '%s' requires a BoxShape3D." % name)
		return {
			"transform": Transform3D(target_basis, global_position),
			"region": str(region_name),
			"cell": -1,
			"valid": false,
			"coverage_cycle": _coverage_cycle,
			"coverage_fraction": 0.0,
		}

	var resolved_grid := Vector3i(
		maxi(grid_size.x, 1),
		maxi(grid_size.y, 1),
		maxi(grid_size.z, 1))

	var cell_index: int
	var coverage_fraction: float
	if forced_cell >= 0:
		# Explicit deterministic cell (demo generation). Fail-closed if the caller asks for a cell
		# outside the allowlist/range: never substitute a different cell.
		var eligible := _eligible_cells(resolved_grid.x * resolved_grid.y * resolved_grid.z)
		if not eligible.has(forced_cell):
			push_error(
				("TargetSamplingRegion3D '%s': forced_cell %d is not an allowed cell "
				+ "(fail-closed).") % [name, forced_cell])
			return {
				"transform": Transform3D(target_basis, global_position),
				"region": str(region_name),
				"cell": -1,
				"valid": false,
				"coverage_cycle": _coverage_cycle,
				"coverage_fraction": 0.0,
			}
		cell_index = forced_cell
		coverage_fraction = 0.0
	else:
		_ensure_cell_order(rng, resolved_grid)
		# Fail-closed: a broken/empty allowlist leaves no cells to sample -> return a degenerate
		# sample (cell -1) rather than indexing an empty order or opening the whole region.
		if _cell_order.is_empty():
			return {
				"transform": Transform3D(target_basis, global_position),
				"region": str(region_name),
				"cell": -1,
				"valid": false,
				"coverage_cycle": _coverage_cycle,
				"coverage_fraction": 0.0,
			}
		cell_index = _cell_order[_cell_cursor]
		coverage_fraction = float(_cell_cursor + 1) / float(_cell_order.size())
		_cell_cursor += 1

	var half_size := box.size * 0.5
	var inset := Vector3(
		clampf(sampling_inset.x, 0.0, maxf(half_size.x - 0.0001, 0.0)),
		clampf(sampling_inset.y, 0.0, maxf(half_size.y - 0.0001, 0.0)),
		clampf(sampling_inset.z, 0.0, maxf(half_size.z - 0.0001, 0.0)))
	var local_min := -half_size + inset
	var local_max := half_size - inset
	var usable_size := local_max - local_min
	var cell_size := Vector3(
		usable_size.x / float(resolved_grid.x),
		usable_size.y / float(resolved_grid.y),
		usable_size.z / float(resolved_grid.z))

	var x := cell_index % resolved_grid.x
	var yz := floori(float(cell_index) / float(resolved_grid.x))
	var y := yz % resolved_grid.y
	var z := floori(float(yz) / float(resolved_grid.y))
	var cell_min := local_min + Vector3(
		float(x) * cell_size.x,
		float(y) * cell_size.y,
		float(z) * cell_size.z)
	var local_point := cell_min + Vector3(
		rng.randf_range(0.0, cell_size.x),
		rng.randf_range(0.0, cell_size.y),
		rng.randf_range(0.0, cell_size.z))
	var sampled_position := collision_shape.to_global(local_point)

	return {
		"transform": Transform3D(target_basis, sampled_position),
		"region": str(region_name),
		"cell": cell_index,
		"valid": true,
		"coverage_cycle": _coverage_cycle,
		"coverage_fraction": coverage_fraction,
	}


func contains_global_position(point: Vector3) -> bool:
	var collision_shape := _collision_shape()
	var box := collision_shape.shape as BoxShape3D if collision_shape else null
	if box == null:
		return false
	var local_point := collision_shape.to_local(point)
	var half_size := box.size * 0.5 + Vector3.ONE * 0.0001
	return (
		absf(local_point.x) <= half_size.x
		and absf(local_point.y) <= half_size.y
		and absf(local_point.z) <= half_size.z)


func total_cells() -> int:
	return (
		maxi(grid_size.x, 1)
		* maxi(grid_size.y, 1)
		* maxi(grid_size.z, 1))


## Cells eligible for sampling this cycle: the allowlist (filtered to the valid range) when set,
## otherwise every cell. FAIL-CLOSED: an allowlist that is set but filters to nothing (every
## index out of range) yields NO cells and an error — never the whole region — so a broken
## allowlist cannot silently train on masked/invalid targets.
func _eligible_cells(cell_total: int) -> Array[int]:
	var out: Array[int] = []
	if allowed_cells.is_empty():
		for i in range(cell_total):
			out.append(i)
		return out
	var seen := {}
	for c in allowed_cells:
		var ci := int(c)
		if ci >= 0 and ci < cell_total and not seen.has(ci):
			seen[ci] = true
			out.append(ci)
	if out.is_empty() and not _warned_empty_allowlist:
		_warned_empty_allowlist = true
		push_error(
			("TargetSamplingRegion3D '%s': allowed_cells is set but no index falls in [0, %d); "
			+ "sampling nothing (fail-closed).") % [name, cell_total])
	return out


func _ensure_cell_order(
	rng: RandomNumberGenerator,
	resolved_grid: Vector3i
) -> void:
	var cell_total := resolved_grid.x * resolved_grid.y * resolved_grid.z
	var eligible := _eligible_cells(cell_total)
	var signature := "%d:%s" % [cell_total, str(allowed_cells)]
	var begin_new_cycle := (
		_cell_order.size() != eligible.size()
		or _cached_grid_size != resolved_grid
		or _cached_allowed_signature != signature
		or _cell_cursor >= _cell_order.size())
	if not begin_new_cycle:
		return
	if not _cell_order.is_empty() and _cell_cursor >= _cell_order.size():
		_coverage_cycle += 1
	_cell_order = eligible.duplicate()
	for index in range(_cell_order.size() - 1, 0, -1):
		var swap_index := rng.randi_range(0, index)
		var temporary := _cell_order[index]
		_cell_order[index] = _cell_order[swap_index]
		_cell_order[swap_index] = temporary
	_cell_cursor = 0
	_cached_grid_size = resolved_grid
	_cached_allowed_signature = signature


func _collision_shape() -> CollisionShape3D:
	return get_node_or_null(collision_shape_path) as CollisionShape3D
