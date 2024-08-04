import os
import sys
# Get the directory of the current script
# current_dir = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
from lxml import etree
from typing import Dict, Optional, List, Any
from gallama.logger import logger
import re


def get_package_root():
    """
    Get the root directory of the installed package or the current script directory.
    """
    if getattr(sys, 'frozen', False):
        # If the application is frozen (e.g., PyInstaller executable)
        return Path(sys.executable).parent.parent
    elif __package__:
        # If it's an installed package
        return Path(__file__).parent.parent
    else:
        # If it's a standalone script
        return Path(__file__).resolve().parent.parent

# Use the function to get the package root
current_dir = get_package_root()

class Thinking:
    def __init__(self, xml: str, regex: Optional[str] = None, xml_is_file: bool = False, regex_is_file: bool = False):
        self.xml: str = self._process_input(xml, xml_is_file, self.read_xml_file_to_string)
        self.regex: Optional[str] = self._process_input(regex, regex_is_file, self.read_regex_file_to_string) if regex else None
        self.xml_dict: Dict[str, Any] = {}
        self.root_tag: Optional[str] = None
        self.root_key_stop_words: List[str] = []
        self._parse_xml()
        self._generate_stop_words()

    def _process_input(self, input_value: Optional[str], is_file: bool, read_func) -> Optional[str]:
        if input_value is None:
            return None
        if not isinstance(input_value, str):
            raise ValueError("Input must be a string (content or file path)")
        if is_file:
            if not os.path.isfile(input_value):
                raise FileNotFoundError(f"File not found: {input_value}")
            return read_func(input_value)
        return input_value

    def _parse_xml(self):
        root = etree.fromstring(self.xml)
        self.root_tag = root.tag
        self.xml_dict = self.parse_xml_to_dict(self.xml)

    def _generate_stop_words(self):
        # self.root_key_stop_words = [f"</{self.root_tag}>", f"</{self.root_tag}>\n", f"</{self.root_tag}>\n```"]
        self.root_key_stop_words = [f"</{self.root_tag}>"]

    @staticmethod
    def read_xml_file_to_string(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            xml_string = file.read()
        return xml_string

    @staticmethod
    def read_regex_file_to_string(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            regex_string = file.read().strip()
        return regex_string

    @staticmethod
    def parse_xml_to_dict(xml_string: str) -> Dict[str, Any]:
        root = etree.fromstring(xml_string)

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

        return {root.tag: xml_to_dict(root)}

    def validate_with_regex(self, value: str) -> str:
        if self.regex:
            pattern = re.compile(self.regex)
            if not pattern.match(value):
                raise ValueError(f"Value '{value}' does not match the regex pattern.")
        return value

    def get_stop_words(self) -> List[str]:
        return self.root_key_stop_words


def find_xml_regex_pairs(directory:str = f"{current_dir}/data/thinking_template/"):
    # Define a named tuple for our pairs, now including the XML filename without extension
    thinking_dict = dict()


    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.xml'):
                xml_file = os.path.join(root, file)
                regex_file = os.path.join(root, file[:-4] + '.regex')

                # Get the XML filename without extension
                xml_filename, _ = os.path.splitext(file)

                # Check if the corresponding .regex file exists
                if not os.path.exists(regex_file):
                    regex_file = None

                thinking_dict[xml_filename] = Thinking(
                    xml=xml_file,
                    xml_is_file=True,
                    regex=regex_file,
                    regex_is_file=True
                )

    return thinking_dict

THINKING_TEMPLATE = find_xml_regex_pairs(f"{current_dir}/data/thinking_template/")

