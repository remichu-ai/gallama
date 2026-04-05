from gallama.backend.llm.prompt_engine.by_model.minimax import minimax_tool_parser


def test_minimax_tool_parser_allows_shell_operators_in_parameter_text():
    tool_text = """
    <invoke name="Bash">
    <parameter name="command">pip install youtube-transcript-api -q && python3 -c "print('ok')"</parameter>
    <parameter name="description">Install library and scrape transcript</parameter>
    <parameter name="timeout">60000</parameter>
    </invoke>
    """

    parsed = minimax_tool_parser(tool_text)

    assert len(parsed) == 1
    arguments = parsed[0].arguments
    assert arguments["command"] == 'pip install youtube-transcript-api -q && python3 -c "print(\'ok\')"'
    assert arguments["description"] == "Install library and scrape transcript"
    assert arguments["timeout"] == 60000


def test_minimax_tool_parser_preserves_literal_angle_brackets_in_arguments():
    tool_text = """
    <invoke name="Write">
    <parameter name="content"><system-reminder>
    Use the Skill tool first.
    </system-reminder></parameter>
    </invoke>
    """

    parsed = minimax_tool_parser(tool_text)

    assert len(parsed) == 1
    arguments = parsed[0].arguments
    assert "<system-reminder>" in arguments["content"]
    assert "</system-reminder>" in arguments["content"]
