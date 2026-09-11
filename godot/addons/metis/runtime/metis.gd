extends Node
class_name Metis

## Common editor-facing base for Metis runtime nodes.
##
## Godot builds the Create New Node tree from script-class inheritance. Keeping the framework's Node-based types under this otherwise behaviorless base gives them one discoverable Metis branch without changing their runtime contract.
