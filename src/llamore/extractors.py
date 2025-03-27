import base64
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pymupdf
from google import genai
from openai import OpenAI

from .prompter import (
    LineByLinePrompter,
    SchemaPrompter,
)
from .reference import References

_LOGGER = logging.getLogger(__name__)

Prompter = Union[LineByLinePrompter, SchemaPrompter]


class OpenaiExtractor:
    """Use an OpenAI endpoint to extract the references.

    We also use this extractor with compatible OpenAI endpoints provided by Ollama, vLLM, SGLang, TGI, etc.
    In this case you do not need to provide an `api_key` and/or `model`.

    Args:
        api_key: Your OpenAI API key.
        model: The model to use for the extraction.
        prompter: The prompter to use for the extraction. By default, we use the `llamore.SchemaPrompter`.
        endpoint: The endpoint to use for the extraction.
        **kwargs: Additional keyword arguments are passed on to the `openai.OpenAI` client.
    """

    def __init__(
        self,
        api_key: str = "None",
        model: str = "None",
        prompter: Optional[Prompter] = None,
        endpoint: Literal["create", "parse"] = "create",
        **kwargs,
    ):
        self._prompter = prompter or SchemaPrompter()
        self._endpoint = endpoint

        if endpoint == "parse" and isinstance(prompter, LineByLinePrompter):
            raise ValueError(
                "The 'parse' endpoint does not support the 'LineByLinePrompter'."
            )

        self._client = OpenAI(api_key=api_key, **kwargs)
        self._model = model

    def __call__(
        self,
        pdf: Optional[Union[str, Path]] = None,
        text: Optional[str] = None,
        **kwargs,
    ) -> References:
        """Extract references from a PDF file or raw text input.

        Args:
            pdf: The file path to the PDF document to extract references from. Defaults to None.
            text: A string containing raw text to extract references from. Defaults to None.
            **kwargs: Additional keyword arguments are passed on to the
                `OpenAI.chat.completions.create` or `OpenAI.beta.chat.completions.parse` call.

        Returns:
            References: An object containing the extracted references.
        """
        if pdf is None and text is None:
            raise ValueError("Either a `pdf` or a `text` must be provided.")

        if pdf is not None and text is not None:
            raise ValueError("Only one of `pdf` or `text` must be provided.")

        if pdf:
            messages = self._messages_for_pdf(Path(pdf))
        else:
            messages = self._messages_for_text(text)

        _LOGGER.debug(f"Prompt: {messages}")

        if self._endpoint == "create":
            references = self._call_create_endpoint(messages, **kwargs)
        elif self._endpoint == "parse":
            references = self._call_parse_endpoint(messages, **kwargs)
        else:
            raise ValueError(f"Invalid endpoint: {self._endpoint}")

        return references

    def _messages_for_pdf(self, pdf: Path) -> List[Dict[str, str]]:
        """Create the messages for the PDF input."""
        image_encodings = self._extract_image_encodings(pdf)
        image_messages = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}",
                    "detail": "high",
                },
            }
            for image in image_encodings
        ]

        messages = [
            {"role": "system", "content": self._prompter.system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._prompter.user_prompt()},
                    *image_messages,
                ],
            },
        ]

        return messages

    def _extract_image_encodings(self, pdf: Path, dpi: int = 150) -> List[str]:
        """Save PDF as pngs and encode them as base64."""
        image_encodings = []
        with tempfile.TemporaryDirectory() as tmp_path:
            doc = pymupdf.open(pdf)
            for i, page in enumerate(doc):
                png_path = Path(tmp_path) / f"{pdf.stem}_{i}.png"
                page.get_pixmap(dpi=dpi).save(png_path)
                with png_path.open("rb") as png_file:
                    base64_png = base64.b64encode(png_file.read()).decode("utf-8")
                image_encodings.append(base64_png)

        return image_encodings

    def _messages_for_text(self, text: str) -> List[Dict[str, str]]:
        """Create the messages for the text input."""
        messages = [
            {"role": "system", "content": self._prompter.system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._prompter.user_prompt(text=text)}
                ],
            },
        ]

        return messages

    def _call_create_endpoint(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> References:
        if (
            isinstance(self._prompter, SchemaPrompter)
            and "response_format" not in kwargs
        ):
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "References",
                    "schema": self._prompter.json_schema,
                },
            }

        output = self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs
        )
        _LOGGER.debug(f"Output: {output}")

        references = self._prompter.parse(output.choices[0].message.content)

        return references

    def _call_parse_endpoint(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> References:
        if (
            isinstance(self._prompter, SchemaPrompter)
            and "response_format" not in kwargs
        ):
            kwargs["response_format"] = self._prompter.schema_model

        output = self._client.beta.chat.completions.parse(
            model=self._model, messages=messages, **kwargs
        )
        try:
            references = output.choices[0].message.parsed.references
        except Exception as e:
            _LOGGER.debug(f"Error parsing response: {e}")
            references = []

        if output.choices[0].message.refusal:
            _LOGGER.debug(f"Refusal: {output.choices[0].message.refusal}")

        return References(references)


class GeminiExtractor:
    """Use a Google Gemini endpoint to extract the references.

    Args:
        api_key: Your Google Gemini API key.
        model: The model to use for the extraction.
        prompter: The prompter to use for the extraction. By default, we use the `llamore.LineByLinePrompter`.
        **kwargs: Additional keyword arguments are passed on to the `genai.Client`.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        prompter: Optional[Prompter] = None,
    ):
        self._client = genai.Client(api_key=api_key)

        self._model = model
        self._prompter = prompter or LineByLinePrompter()

    def __call__(
        self,
        pdf: Optional[Union[str, Path]] = None,
        text: Optional[str] = None,
        **kwargs,
    ) -> References:
        """Extract references from a PDF file or raw text input.

        Args:
            pdf: The file path to the PDF document to extract references from. Defaults to None.
            text: A string containing raw text to extract references from. Defaults to None.
            **kwargs: Additional keyword arguments are passed on to the `genai.Client.models.generate_content` call.

        Returns:
            References: An object containing the extracted references.
        """
        if pdf is None and text is None:
            raise ValueError("Either a `pdf` or a `text` must be provided.")

        if pdf is not None and text is not None:
            raise ValueError("Only one of `pdf` or `text` must be provided.")

        contents = [self._prompter.user_prompt(text=text)]
        if pdf:
            file = self._client.files.upload(
                file=pdf, config={"mime_type": "application/pdf"}
            )
            contents.insert(0, file)

        config = {"system_instruction": self._prompter.system_prompt()}
        config.update(**kwargs)

        # config = {
        #     "system_instruction": self._prompter.system_prompt(),
        #     "temperature": temperature,
        #     "top_p": 0.95,
        #     "top_k": 40,
        #     "max_output_tokens": 8192,
        #     # FOR THE STRUCTURED OUTPUT TO WORK WE NEED A SIMPLER JSON SCHEMA, GENAI DOES NOT SUPPORT ANYOF FOR NOW!
        #     # "response_mime_type": "application/json",
        #     # "response_schema": prompter.schema_model
        # }

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        references = self._prompter.parse(response.text)

        return references
