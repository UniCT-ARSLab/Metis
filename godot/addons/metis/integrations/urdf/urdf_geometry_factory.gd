class_name URDFGeometryFactory

static var resource_cache: Dictionary = {}

static func _clean_path(
		package_path: String,
		options: Dictionary,
		source_path: String) -> String:
	return URDFUtils.resolve_reference_path(
		package_path,
		source_path,
		str(options.get("package_folder", "")))

static func load_resource(
		path: String, opts: Dictionary, source_path: String) -> Resource:
	path = _clean_path(path, opts, source_path)
	if resource_cache.has(path): return resource_cache[path]
	var res = load(path)
	resource_cache[path] = res
	return res

static func get_visual_callable(type: int) -> Callable:
	match type:
		URDFVisual.Type.BOX:
			return create_box_visual
		URDFVisual.Type.CYLINDER:
			return create_cylinder_visual
		URDFVisual.Type.SPHERE:
			return create_sphere_visual
		URDFVisual.Type.MESH:
			return create_mesh_resource_visual
	return Callable()

static func get_collision_callable(type: int) -> Callable:
	match type:
		URDFCollider.Type.BOX:
			return create_box_collision
		URDFCollider.Type.CYLINDER:
			return create_cylinder_collision
		URDFCollider.Type.SPHERE:
			return create_sphere_collision
		URDFCollider.Type.MESH:
			return create_mesh_resource_collision
	return Callable()


# Visual
static func create_box_visual(
		parent: Node3D, owner: Node, data: URDFVisual,
		_opts: Dictionary, _path: String, material: BaseMaterial3D):
	var mesh_inst = MeshInstance3D.new()
	mesh_inst.mesh = BoxMesh.new()
	mesh_inst.mesh.size = data.size
	_finalize(
		mesh_inst, parent, owner, material,
		data.origin_xyz, data.origin_rpy)

static func create_cylinder_visual(
		parent: Node3D, owner: Node, data: URDFVisual,
		_opts: Dictionary, _path: String, material: BaseMaterial3D):
	var mesh_inst = MeshInstance3D.new()
	var cm = CylinderMesh.new()
	cm.height = data.length
	cm.top_radius = data.radius
	cm.bottom_radius = data.radius
	mesh_inst.mesh = cm
	_finalize(
		mesh_inst, parent, owner, material,
		data.origin_xyz, data.origin_rpy)

static func create_sphere_visual(
		parent: Node3D, owner: Node, data: URDFVisual,
		_opts: Dictionary, _path: String, material: BaseMaterial3D):
	var mesh_inst = MeshInstance3D.new()
	var sphere = SphereMesh.new()
	sphere.radius = data.radius
	sphere.height = data.radius * 2
	mesh_inst.mesh = sphere
	_finalize(
		mesh_inst, parent, owner, material,
		data.origin_xyz, data.origin_rpy)

static func create_mesh_resource_visual(
		parent: Node3D, owner: Node, data: URDFVisual,
		opts: Dictionary, source_path: String, material: BaseMaterial3D):
	var resource = load_resource(data.mesh_path, opts, source_path)
	var instance
	if resource is PackedScene:
		instance = resource.instantiate()
		_finalize(
			instance, parent, owner, material,
			data.origin_xyz, data.origin_rpy)
	elif resource is Mesh:
		instance = MeshInstance3D.new()
		instance.mesh = resource
		_finalize(
			instance, parent, owner, material,
			data.origin_xyz, data.origin_rpy)
	else:
		push_error(
			"Error loading " + data.mesh_path +
			" - Unknown Resource type:" + type_string(typeof(resource)))
		return

	var _scale = 1
	if opts.has("scale"):
		_scale = opts.get("scale")
	# Prefer the URDF's per-mesh <mesh scale="x y z"> (supports non-uniform + mirror) when present;
	# otherwise fall back to the uniform importer scale option.
	if data.mesh_scale != Vector3.ZERO:
		instance.scale = data.mesh_scale
	else:
		instance.scale = Vector3(_scale, _scale, _scale)

	if opts.get('rotate_x', null) != null:
		# overwrite rotation
		instance.rotate_x(opts['rotate_x'])
	else:
		var ext = data.mesh_path.get_extension().to_lower()
		if ext == "stl":
			instance.rotate_x(-PI / 2)

# Collision
static func create_box_collision(
		parent: Node3D, owner: Node, data: URDFCollider,
		_opts: Dictionary, _path: String):
	var coll = CollisionShape3D.new()
	coll.shape = BoxShape3D.new()
	coll.shape.size = data.size
	_finalize(
		coll, parent, owner, null, data.origin_xyz, data.origin_rpy)

static func create_cylinder_collision(
		parent: Node3D, owner: Node, data: URDFCollider,
		_opts: Dictionary, _path: String):
	var coll = CollisionShape3D.new()
	var shape = CylinderShape3D.new()
	shape.height = data.length
	shape.radius = data.radius
	coll.shape = shape
	_finalize(
		coll, parent, owner, null, data.origin_xyz, data.origin_rpy)

static func create_sphere_collision(
		parent: Node3D, owner: Node, data: URDFCollider,
		_opts: Dictionary, _path: String):
	var coll = CollisionShape3D.new()
	var shape = SphereShape3D.new()
	shape.radius = data.radius
	coll.shape = shape
	_finalize(
		coll, parent, owner, null, data.origin_xyz, data.origin_rpy)


static func create_mesh_resource_collision(
		parent: Node3D, owner: Node, data: URDFCollider,
		opts: Dictionary, source_path: String):
	var resource = load_resource(data.mesh_path, opts, source_path)
	var urdf_transform = URDFUtils.xyz_rpy_to_transform3d(
		data.origin_xyz, data.origin_rpy)
	# Same scale the visual uses: the URDF per-mesh scale (non-uniform + mirror) when present, else
	# the uniform importer scale option.
	var uniform := float(opts.get("scale", 1.0))
	var scale_vec := data.mesh_scale if data.mesh_scale != Vector3.ZERO \
		else Vector3(uniform, uniform, uniform)

	if resource is Mesh:
		_create_col_shape_from_mesh(
			resource, urdf_transform, parent, owner, opts, scale_vec)
	elif resource is PackedScene:
		var temp_scene = resource.instantiate()
		_recursive_collision_gen(
			temp_scene, urdf_transform, parent, owner, opts, scale_vec)
		temp_scene.queue_free()


static func _recursive_collision_gen(
		node: Node, base_transform: Transform3D,
		parent: Node3D, owner: Node,
		opts: Dictionary, scale_vec: Vector3):
	if node is MeshInstance3D:
		# Combine the URDF offset with the mesh's internal local transform
		var final_transform = base_transform * node.transform
		_create_col_shape_from_mesh(
			node.mesh, final_transform, parent, owner, opts, scale_vec)

	for child in node.get_children():
		_recursive_collision_gen(
			child, base_transform, parent, owner, opts, scale_vec)


static func _create_col_shape_from_mesh(
		mesh: Mesh, tr: Transform3D,
		parent: Node3D, owner: Node,
		opts: Dictionary, scale_vec: Vector3):
	var shape = mesh.create_convex_shape(true, true)
	if shape:
		# The mesh vertices are in the URDF mesh units (e.g. millimetres); the visual scales the
		# MeshInstance3D, but the collision is built straight from the raw vertices. Bake the SAME
		# per-axis scale (including a mirror from a negative component) into the shape points so the
		# collider matches the model instead of being ~1000x too large or unmirrored. Baking (rather
		# than scaling the CollisionShape3D node) keeps the physics shape correct for RigidBody3D.
		if scale_vec != Vector3.ONE and shape is ConvexPolygonShape3D:
			var points: PackedVector3Array = shape.points
			for i in points.size():
				points[i] = points[i] * scale_vec
			shape.points = points
		var coll = CollisionShape3D.new()
		coll.shape = shape
		coll.name = parent.name + "_collision"
		parent.add_child(coll)
		coll.owner = owner
		coll.transform = tr
		# TODO: check, if we always need to rotate by -90°
		if opts.get('rotate_x', null) != null:
			# overwrite rotation
			coll.rotate_x(opts['rotate_x'])
		else:
			coll.rotate_x(-PI / 2)

static func _finalize(
		node: Node3D, parent: Node, owner: Node, 
		material: BaseMaterial3D,
		xyz: Vector3, rpy: Vector3, num: int = 0):
	parent.add_child(node)
	if node is CollisionShape3D:
		node.name = parent.name + "_collision"
	else:
		node.name = parent.name + "_mesh"
		if material:
			node.material_override = material
	if num > 0:
		node.name += "_" + str(num)
	node.owner = owner
	node.transform = URDFUtils.xyz_rpy_to_transform3d(xyz, rpy)
