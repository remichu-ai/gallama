import json

from gallama.utils.utils import format_request_body_for_logging, truncate_base64_data_uri


def test_truncate_base64_data_uri_shortens_large_image_payloads():
    image_data = "a" * 256
    data_uri = f"data:image/png;base64,{image_data}"

    truncated = truncate_base64_data_uri(data_uri, preview_chars=24)

    assert truncated.startswith("data:image/png;base64," + ("a" * 24))
    assert "[truncated base64, total=256 chars]" in truncated


def test_format_request_body_for_logging_truncates_nested_image_urls_by_default():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + ("b" * 300),
                        },
                    },
                ],
            }
        ]
    }

    formatted = format_request_body_for_logging(json.dumps(payload).encode("utf-8"))

    assert '"text": "describe this image"' in formatted
    assert "[truncated base64, total=300 chars]" in formatted
    assert ('"url": "data:image/jpeg;base64,' + ("b" * 300) + '"') not in formatted


def test_format_request_body_for_logging_keeps_full_base64_at_max_verbosity():
    payload = {
        "image": "data:image/png;base64," + ("c" * 180),
    }

    formatted = format_request_body_for_logging(payload, include_full_base64=True)

    assert "[truncated base64" not in formatted
    assert ('"image": "data:image/png;base64,' + ("c" * 180) + '"') in formatted
