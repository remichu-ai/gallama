import uuid
from exllamav2 import ExLlamaV2Tokenizer
from llama_cpp import Llama
import yaml
import base64
import struct
from lxml import etree
from pathlib import Path


def get_response_uid():
    return "cmpl-" + str(uuid.uuid4().hex)


def get_response_tool_uid():
    return "call_" + str(uuid.uuid4().hex)


def get_token_length(tokenizer, text):
    str_text = text
    if not isinstance(text, str):
        str_text = str(text)

    if isinstance(tokenizer, ExLlamaV2Tokenizer):
        return tokenizer.num_tokens(str_text)
    elif isinstance(tokenizer, Llama):
        return len(tokenizer.tokenize(str_text.encode("utf-8"), add_bos=False))
    else:
        return len(tokenizer.encode(str_text))


def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error reading YAML file: {exc}")
        return None


def floats_to_base64(float_array):
    """" convert list of floats to base64 for embedding response """

    # Pack the float array into binary format
    binary_data = struct.pack('f' * len(float_array), *float_array)

    # Encode the binary data to base64
    base64_encoded = base64.b64encode(binary_data)

    # Convert bytes to string for easy handling
    base64_string = base64_encoded.decode('utf-8')

    return base64_string


def parse_xml_to_dict(xml_string):
    # Parse the XML string
    root = etree.fromstring(xml_string)

    # Get the root tag
    root_tag = root.tag

    # Function to convert XML element to a nested dictionary
    def xml_to_dict(element):
        children = list(element)
        if not children:
            return element.text
        result = {}
        for child in children:
            child_dict = xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = child_dict
        return result

    # Convert the XML to a nested dictionary
    xml_dict = {root.tag: xml_to_dict(root)}

    return xml_dict


def get_package_file_path(file_name: str) -> str:
    """
    Get the absolute path of a file within the gallama package.

    Args:
    file_name (str): The name of the file (e.g., 'app.py')

    Returns:
    str: The absolute path to the file
    """
    current_dir = Path(__file__).parent
    file_path = current_dir / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_name} not found in the gallama package.")
    return str(file_path.resolve())
