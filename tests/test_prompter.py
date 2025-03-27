import pytest
from llamore import Reference, LineByLinePrompter
from llamore.reference import Person


def test_line_by_line_prompter():
    references = [
        Reference(
            analytic_title="Analytic Title",
            authors=[
                Person(forename="Forename", surname="Surname"),
                Person(forename="Forename", surname="Surname"),
            ],
            volume="1",
        ),
        Reference(
            analytic_title="Analytic Title",
            authors=[Person(forename="J", surname="Doe"), Person(surname="Surname")],
            cited_range="2ff",
        ),
    ]

    prompter = LineByLinePrompter()

    response = f"{references[0].model_dump_json()}\n{references[1].model_dump_json()}"
    parsed_refs = prompter.parse(response)
    assert parsed_refs == references

    response = f"```json\n{references[0].model_dump_json()}\n{references[1].model_dump_json()}```"
    parsed_refs = prompter.parse(response)
    assert parsed_refs == references
