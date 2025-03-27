import json
import logging
import re
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.json_schema import GenerateJsonSchema

from llamore.reference import Reference, References

_LOGGER = logging.getLogger(__name__)


class LineByLinePrompter:
    """Prompts instruction models to output the references as JSON formatted strings line by line.
    So each line contains one JSON formatted `Reference`.
    """

    @property
    def _json_schema(self) -> str:
        return Reference.model_json_schema(
            mode="serialization", schema_generator=_GenerateJsonSchemaNoTitle
        )

    def system_prompt(self) -> str:
        """The system prompt."""
        return (
            "You are an expert in scholarly references and citations. "
            "You help the user to extract citation data from scientific works."
        )

    def user_prompt(self, text: Optional[str] = None) -> str:
        """The user prompt.

        Args:
            text: The input text from which to extract the references.
                If None, we assume we are extracting references from a PDF.

        Returns:
            The prompt for the user role.
        """
        prompt = (
            f"Extract all references from the given {'text' if text else 'PDF'}. "
            "Output the references in JSON format with following schema:"
            f"\n\n{self._json_schema}\n\n"
            "Output the references as JSON formatted strings, each reference in a new line. "
            "Only output the JSON strings, nothing else. Don't use markdown."
        )

        if text:
            prompt += f"\n\nTEXT: <<<{text}>>>"

        return prompt

    def parse(self, response: str) -> References:
        """Parse the LLM response in json format to a list of references.

        We post-process the LLM response line by line and pass it on to the Reference model.

        Args:
            response: The response of the LLM in json format.

        Returns:
            A list of `Reference`s.
        """
        # TODO: Common practice is also to repeatedly prompt the LLM when failing to parse its response.
        references = References()

        response = response.strip()

        # Parse line by line, everything between curly brackets
        prog = re.compile("{.*}")
        for line in response.split("\n"):
            match = prog.search(line)
            if match is not None:
                try:
                    instance = Reference.model_validate_json(match.group())
                except ValidationError as err:
                    _LOGGER.debug(f"{err}, {match.group()}")
                else:
                    if instance != Reference():
                        references.append(instance)

        # Try to parse the response as is (bigger models are sometimes too smart ...)
        if not references:
            # remove markdown
            if response.startswith("```"):
                response = "\n".join([line for line in response.split("\n")][1:-1])
            try:
                response_json = json.loads(response)
            except json.JSONDecodeError as err:
                _LOGGER.debug(err)
            else:
                if isinstance(response_json, dict):
                    response_json = [response_json]
                for instance_json in response_json:
                    try:
                        instance = Reference(**instance_json)
                    except ValidationError as err:
                        _LOGGER.debug(err)
                    else:
                        if instance != Reference():
                            references.append(instance)

        return references


class SchemaPrompter:
    """Prompts instruction models to output all references as one valid JSON formatted string.

    So we basically provide following new JSON schema to the model: {"references": List[Reference]}

    Args:
        print_pretty: Instruct the model to print the response pretty?
        step_by_step: Instruct the model to extract the references step by step?
            In this case the new schema looks like this: {"steps": List[str], "references": List[Reference]}
    """

    def __init__(
        self,
        print_pretty: bool = True,
        step_by_step: bool = False,
    ):
        self._print_pretty = print_pretty
        self._step_by_step = step_by_step

        if self._step_by_step:
            self._schema_model = create_model(
                "References",
                steps=(
                    List[str],
                    Field(
                        description="A list of necessary steps to extract all the references."
                    ),
                ),
                references=(List[Reference], Field(description="A list of references")),
            )
        else:
            self._schema_model = create_model(
                "References",
                references=(List[Reference], Field(description="A list of references")),
            )

    @property
    def schema_model(self) -> type[BaseModel]:
        """The pydantic BaseModel of the JSON schema in the prompt."""
        return self._schema_model

    @property
    def json_schema(self) -> str:
        """The JSON schema for the prompt."""
        return self._schema_model.model_json_schema(
            mode="serialization", schema_generator=_GenerateJsonSchemaNoTitle
        )

    def system_prompt(self) -> str:
        """The system prompt for extracting References in json format."""
        return (
            "You are an expert in scholarly references and citations. "
            "You help the user to extract citation data from scientific works."
        )

    def user_prompt(self, text: Optional[str] = None) -> str:
        """The user prompt.

        Args:
            text: The input text from which to extract the structured information.
                If None, we assume we are extracting references from a PDF.

        Returns:
            The prompt for the user role.
        """
        prompt = (
            f"Extract all references from the given {'text' if text else 'PDF'}. "
            "Output the references in JSON format with following schema:"
            f"\n\n{self.json_schema}\n\n"
            f"{'Extract the references step by step. ' if self._step_by_step else ''}"
            f"Only output the JSON string, nothing else. {'Print it pretty. ' if self._print_pretty else ''}"
            "Don't use markdown."
        )
        if text:
            prompt += f"\n\nTEXT: <<<{text}>>>"

        return prompt

    def parse(self, response: str) -> References:
        # remove markdown
        if response.startswith("```"):
            response = "\n".join([line for line in response.split("\n")][1:-1])

        try:
            references = self.schema_model.model_validate_json(response).references
        except ValidationError as e:
            _LOGGER.debug(f"ValidationError: {e}")
            references = []

        references = [ref for ref in references if ref != Reference()]

        return References(references)


class _GenerateJsonSchemaNoTitle(GenerateJsonSchema):
    def generate(self, schema, mode="validation"):
        json_schema = super().generate(schema, mode=mode)
        if "title" in json_schema:
            del json_schema["title"]
        return json_schema

    def get_schema_from_definitions(self, json_ref):
        json_schema = super().get_schema_from_definitions(json_ref)
        if json_schema and "title" in json_schema:
            del json_schema["title"]
        return json_schema

    def field_title_should_be_set(self, schema) -> bool:
        return False
