from pydantic import BaseModel, Field, ValidationError, create_model
from typing import Literal, Type, Union, Optional, List, Any, Dict
from formatron.schemas.pydantic import ClassSchema, Schema
from pathlib import Path
from tempfile import TemporaryDirectory
from ...data_classes import ToolSpec
import json
from datamodel_code_generator import generate, InputFileType, DataModelType
from ...logger.logger import logger


class Tools:
    def __init__(self, prompt_eng, tools: [List[ToolSpec]], tool_choice):
        self.prompt_eng = prompt_eng
        self.tools = tools
        self.tools_list = self.create_pydantic_model_from_tools(self.tools, mode="pydantic_v2")
        self.tools_list_formatron = self.create_pydantic_model_from_tools(self.tools, mode="formatron")
        self.tool_dict = {tool.schema()['title']: tool for tool in self.tools_list}
        self.tool_dict_formatron = {tool.schema()['title']: tool for tool in self.tools_list_formatron}
        self.answer_format = None
        self.json_parser = None
        self.tool_choice = tool_choice

        # initialize tool as code
        self.tool_def_as_code: str = self.create_tool_def_as_code()
        logger.debug(f"tool_def_as_code: {self.tool_def_as_code}")

    def create_tool_def_as_code(self):
        _temp_tool_list = [
            self.generate_pydantic_model_from_json_schema(tool.model_json_schema())
            for tool in self.tools_list
        ]
        return self.append_code_without_duplicate_imports(_temp_tool_list)

    @property
    def tool_name_list(self):
        return ", ".join(f'"{name}"' for name in self.tool_dict.keys())


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

    @staticmethod
    def generate_pydantic_model_from_json_schema(json_schema: dict) -> str:
        """
        Generates Pydantic model code from a JSON schema using datamodel-code-generator.

        Args:
            json_schema (dict): The JSON schema to generate the Pydantic model from.

        Returns:
            str: The generated Pydantic model code as a string, with comments removed,
                 duplicate imports removed, and normalized newlines.
        """
        with TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)

            # Write the JSON schema to a temporary file
            schema_file = temporary_directory / 'schema.json'
            with open(schema_file, 'w') as f:
                json.dump(json_schema, f)

            # Define the output file for the generated model
            output = temporary_directory / 'model.py'

            # Generate the Pydantic model code
            generate(
                input_=schema_file,  # Pass the file path instead of the raw dictionary
                input_file_type=InputFileType.JsonSchema,
                output=output,
                output_model_type=DataModelType.PydanticV2BaseModel,
            )

            # Read the generated code from the output file
            with open(output, 'r') as file:
                generated_code = file.read()

        # Remove the comment lines at the top
        generated_code = '\n'.join(
            line for line in generated_code.splitlines()
            if not line.strip().startswith('#')
        )

        # Normalize newlines to ensure a maximum of one empty line
        generated_code = '\n'.join(
            line for line in generated_code.splitlines()
            if line.strip()  # Remove empty lines
        )
        generated_code = generated_code.replace('\n\n', '\n')  # Ensure max one empty line

        return generated_code

    @staticmethod
    def append_code_without_duplicate_imports(code_list: list[str]) -> str:
        """
        Appends multiple code strings into one, removing duplicate imports and normalizing newlines.

        Args:
            code_list (list[str]): A list of code strings to combine.

        Returns:
            str: The combined code with duplicate imports removed and normalized newlines.
        """
        if not code_list:
            return ""

        # Initialize the combined code with the first code string
        combined_code = code_list[0]

        # Iterate over the remaining code strings
        for new_code in code_list[1:]:
            # Split the code into lines
            existing_lines = combined_code.splitlines()
            new_lines = new_code.splitlines()

            # Extract imports from existing code
            existing_imports = set(
                line for line in existing_lines
                if line.strip().startswith(('from ', 'import '))
            )

            # Filter out duplicate imports from new code
            new_lines_filtered = [
                line for line in new_lines
                if not (line.strip().startswith(('from ', 'import ')) and line in existing_imports)
            ]

            # Append the filtered new code to the combined code
            combined_code += '\n' + '\n'.join(new_lines_filtered)

        # Normalize newlines to ensure a maximum of one empty line
        combined_code = '\n'.join(
            line for line in combined_code.splitlines()
            if line.strip()  # Remove empty lines
        )
        combined_code = combined_code.replace('\n\n', '\n')  # Ensure max one empty line

        return combined_code



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