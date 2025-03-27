from llamore.extractors import OpenaiExtractor
from llamore.prompter import SchemaPrompter


def test_openai_extractor():
    extractor = OpenaiExtractor(api_key="api_key")

    assert isinstance(extractor._prompter, SchemaPrompter)
    assert extractor._model == "None"
