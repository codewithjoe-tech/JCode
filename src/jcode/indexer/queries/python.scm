; function_definition covers both def and async def in tree-sitter-python >= 0.23
(function_definition) @function

; Classes
(class_definition) @class

; Inheritance base identifiers
(class_definition
  superclasses: (argument_list
    (identifier) @base))

; Imports
(import_statement) @import
(import_from_statement) @import

; All call expressions
(call) @call

; Module-level variable assignments (globals / constants)
(module (expression_statement (assignment) @variable))
(module (expression_statement (annotated_assignment) @variable))

; Class-body attribute assignments (Pydantic fields, dataclass fields, settings)
(class_definition body: (block (expression_statement (assignment) @variable)))
(class_definition body: (block (expression_statement (annotated_assignment) @variable)))
