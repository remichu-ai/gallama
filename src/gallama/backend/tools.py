from pydantic import BaseModel, Field, ValidationError, create_model
from typing import Literal, Type, Union, Optional, List, Any, Dict
from formatron.schemas.pydantic import ClassSchema, Schema


class Tools:
    def __init__(self, prompt_eng, tools, tool_choice):
        self.prompt_eng = prompt_eng
        self.tools = tools
        self.tools_list = self.create_pydantic_model_from_tools(self.tools, mode="pydantic_v2")
        self.tools_list_formatron = self.create_pydantic_model_from_tools(self.tools, mode="formatron")
        self.tool_dict = {tool.schema()['title']: tool for tool in self.tools_list}
        self.tool_dict_formatron = {tool.schema()['title']: tool for tool in self.tools_list_formatron}
        self.answer_format = None
        self.json_parser = None
        self.tool_choice = tool_choice

    @property
    def tool_name_list(self):
        return ", ".join(self.tool_dict.keys())

    @staticmethod
    def type_from_json_schema(schema: dict):
        """Map JSON Schema types to Python types with appropriate handling for enums and additional types."""
        if 'anyOf' in schema:
            return Union[tuple(Tools.type_from_json_schema(sub_schema) for sub_schema in schema['anyOf'])]

        schema_type = schema['type']

        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'object': Dict[str, Any],  # Represents JSON objects
            'array': List[Any],        # Represents JSON arrays
            'null': type(None)         # Represents JSON null
        }

        if schema_type not in type_mapping:
            raise ValueError(f"Unsupported JSON schema type: {schema_type}")

        if 'enum' in schema:
            # Define the type as a Python Literal with enumerated values.
            return Literal[tuple(schema['enum'])]

        if schema_type == 'object':
            # Recursively create a Pydantic model for nested objects.
            return Tools.create_pydantic_model_v2({
                'name': schema.get('title', 'NestedObject'),
                'description': schema.get('description', ''),
                'parameters': schema
            })

        if schema_type == 'array':
            # Handle array type with items
            items_schema = schema.get('items', {})
            items_type = Tools.type_from_json_schema(items_schema)
            return List[items_type]

        return type_mapping[schema_type]


    @staticmethod
    def create_pydantic_model_v1(function_info: dict) -> Type[BaseModel]:
        parameters = function_info['parameters']
        properties = parameters.get('properties', {})
        required_properties = set(function_info['parameters'].get('required', []))

        # Redefined how fields are set up
        attributes = {}
        for prop_name, prop_info in properties.items():
            field_type = Tools.type_from_json_schema(prop_info)
            field_description = prop_info.get('description', None)

            if prop_name in required_properties:
                # Required properties should not have a default value
                attributes[prop_name] = (field_type, ..., field_description)
            else:
                # Optional properties default to None
                attributes[prop_name] = (Optional[field_type], None, field_description)

        # Dynamically create Pydantic model with explicit annotations
        #print(attributes)
        namespace = {'__annotations__': {}}
        for attr_name, (attr_type, default, field_description) in attributes.items():
            namespace['__annotations__'][attr_name] = attr_type
            if default is not ...:
                namespace[attr_name] = Field(default=default, description=field_description)
            else:
                namespace[attr_name] = Field(..., description=field_description)

        # Dynamically create Pydantic model with explicit annotations
        # The comma is crucial here—it signifies that this is not merely a parenthesized expression but a tuple.
        model = type(function_info['name'], (BaseModel,), namespace)
        model.__doc__ = function_info['description']

        return model

    @staticmethod
    def create_pydantic_model_v2(function_info: dict) -> Type[BaseModel]:
        parameters = function_info['parameters']
        properties = parameters.get('properties', {})
        required_properties = set(parameters.get('required', []))

        # Redefined how fields are set up
        attributes = {}
        for prop_name, prop_info in properties.items():
            field_type = Tools.type_from_json_schema(prop_info)
            field_description = prop_info.get('description', None)

            if prop_name in required_properties:
                # Required properties should not have a default value
                attributes[prop_name] = (field_type, Field(description=field_description))
            else:
                # Optional properties default to None
                attributes[prop_name] = (Optional[field_type], Field(default=None, description=field_description))

        # Dynamically create Pydantic model with explicit annotations
        namespace = {'__annotations__': {}}
        for attr_name, (attr_type, field_info) in attributes.items():
            namespace['__annotations__'][attr_name] = attr_type
            namespace[attr_name] = field_info

        model = type(function_info['name'], (BaseModel,), namespace)
        model.__doc__ = function_info.get('description', '')

        return model

    @staticmethod
    def create_class_schema(function_info: dict) -> Type[ClassSchema]:
        parameters = function_info['parameters']
        properties = parameters.get('properties', {})
        required_properties = set(parameters.get('required', []))

        attributes = {}
        for prop_name, prop_info in properties.items():
            field_type = Tools.type_from_json_schema(prop_info)
            field_description = prop_info.get('description', None)

            if prop_name in required_properties:
                attributes[prop_name] = (field_type, Field(description=field_description))
            else:
                attributes[prop_name] = (Optional[field_type], Field(default=None, description=field_description))

        namespace = {'__annotations__': {}}
        for attr_name, (attr_type, field_info) in attributes.items():
            namespace['__annotations__'][attr_name] = attr_type
            namespace[attr_name] = field_info

        model = type(function_info['name'], (ClassSchema,), namespace)
        model.__doc__ = function_info.get('description', '')

        return model

    @staticmethod
    def create_pydantic_model_from_tools(tools: list, mode: Literal["pydantic_v2", "formatron", "pydantic_v1"]="pydantic_v2"):
        """
        this function create a list of pydantic model based on
        tool_list in the API call
        """
        models = []
        for tool in tools:
            function_info = tool.dict()['function']
            if mode == "pydantic_v2":
                model = Tools.create_pydantic_model_v2(function_info)
            elif mode == "formatron":
                model = Tools.create_class_schema(function_info)
            elif mode == "pydantic_v1":
                model = Tools.create_pydantic_model_v1(function_info)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            models.append(model)

        return models

    @staticmethod
    def replace_refs_with_definitions_v2(schema: Dict[str, Any], defs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Recursively replace all `$ref` in schema with definitions from `$defs`.
        """
        if defs is None:
            defs = schema.get('$defs', {})

        if isinstance(schema, dict):
            if '$ref' in schema:
                # $ref in Pydantic v2 will point to '#/$defs/Child'
                ref_path = schema['$ref']
                assert ref_path.startswith('#/$defs/'), f"Unhandled $ref format: {ref_path}"
                ref_name = ref_path.split('/')[-1]
                # Proceed to replace with the actual schema from $defs
                # Making a copy of the definition to avoid modifications
                return Tools.replace_refs_with_definitions_v2(defs[ref_name], defs)
            else:
                # Recursively replace in all dictionary items
                return {k: Tools.replace_refs_with_definitions_v2(v, defs) for k, v in schema.items()}
        elif isinstance(schema, list):
            # Recursively replace in all list items
            return [Tools.replace_refs_with_definitions_v2(item, defs) for item in schema]
        return schema


    @staticmethod
    def get_schema_without_refs_from_pydantic_v2(model: BaseModel) -> Dict[str, Any]:
        """
        Generate JSON schema from a Pydantic model and replace all `$ref` with actual definitions.
        """
        raw_schema = model.schema()
        return Tools.replace_refs_with_definitions_v2(raw_schema)

    @staticmethod
    def replace_refs_with_definitions_v1(schema: Dict[str, Any], definitions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Recursively replace all `$ref` in schema with definitions.
        """
        if definitions is None:
            definitions = schema.get('definitions', {})

        if isinstance(schema, dict):
            if '$ref' in schema:
                ref_path = schema['$ref']
                assert ref_path.startswith('#/definitions/'), f"Unhandled $ref format: {ref_path}"
                ref_name = ref_path.split('/')[-1]
                # Proceed to replace with the actual schema definition
                # Important: We make a deep copy to avoid unintentional modifications
                return Tools.replace_refs_with_definitions_v1(definitions[ref_name], definitions)
            else:
                # Recursively replace in all dictionary items
                return {k: Tools.replace_refs_with_definitions_v1(v, definitions) for k, v in schema.items()}
        elif isinstance(schema, list):
            # Recursively replace in all list items
            return [Tools.replace_refs_with_definitions_v1(item, definitions) for item in schema]
        return schema

    @staticmethod
    def get_schema_without_refs_from_pydantic_v1(model: BaseModel) -> Dict[str, Any]:
        """
        Generate JSON schema from a Pydantic model and replace all `$ref`.
        """
        raw_schema = model.schema()
        return Tools.replace_refs_with_definitions_v1(raw_schema)



def create_function_models_v1(functions: Dict[str, Type[BaseModel]]) -> List[Type[BaseModel]]:
    """ create a list of pydantic models v1 for the function schemas that passed in via OpenAI request call"""
    function_model_list: List[Type[BaseModel]] = []
    for func_name, arg_model in functions.items():
        # Dynamic Pydantic model creation
        NewModel = create_model(
            func_name.title(),
            name=(Literal[func_name], ...),  # ... mean required
            arguments=(arg_model, ...),
            __config__=type('Config', (BaseModel.Config,), {'arbitrary_types_allowed': True})
            # Nested Config class
        )
        function_model_list.append(NewModel)
    return function_model_list

def create_function_models_v2(functions: Dict[str, Type[BaseModel]]) -> List[Type[BaseModel]]:
    """Create a list of Pydantic models v2 for the function schemas passed in via OpenAI request call."""
    function_model_list: List[Type[BaseModel]] = []
    for func_name, arg_model in functions.items():
        class Config:
            arbitrary_types_allowed = True

        NewModel = create_model(
            func_name.title(),
            name=(Literal[func_name], Field(...)),  # '...' means required
            arguments=(arg_model, Field(...)),
            __config__=Config
        )
        function_model_list.append(NewModel)
    return function_model_list


def create_function_models_formatron(functions: Dict[str, Type[ClassSchema]]) -> List[Type[ClassSchema]]:
    """Create a list of ClassSchema models for the function schemas passed in via OpenAI request call."""
    function_model_list: List[Type[ClassSchema]] = []
    for func_name, arg_model in functions.items():
        class Config:
            arbitrary_types_allowed = True

        # Create a new ClassSchema subclass
        class NewModel(ClassSchema):
            name: Literal[func_name] = Field(...)
            arguments: arg_model = Field(...)

            class Config:
                arbitrary_types_allowed = True

        # Set the name of the class to match the function name
        NewModel.__name__ = func_name.title()

        function_model_list.append(NewModel)
    return function_model_list