from pathlib import Path

import pytest
from llamore.parser import TeiBiblStruct
from llamore.reference import Organization, Person, Reference
from lxml import etree

ROOT = Path(__file__).parent.parent
SCHEMA = ROOT / "docs" / "schema" / "llamore-default.rng"


@pytest.fixture(scope="module")
def relaxng():
    return etree.RelaxNG(etree.parse(SCHEMA))


def test_example_document(relaxng):
    doc = etree.parse(Path(__file__).parent / "fixtures" / "schema-example.xml")
    assert relaxng.validate(doc), relaxng.error_log


def test_parser_output(relaxng):
    reference = Reference(
        analytic_title="Article",
        journal_title="Journal",
        monographic_title="Book",
        authors=[
            Person(first_name="A.", middle_name="B.", surname="Doe", name_link="von", role_name="Dr."),
            Organization(name="ACME"),
        ],
        editors=[Person(surname="Smith")],
        translator=Person(surname="Jones"),
        publisher="Publisher",
        publication_date="2020",
        publication_place="Berlin",
        volume="1",
        issue="2",
        pages="3-4",
        cited_range="S. 3",
        footnote_number="12",
        refs="ebd.",
    )
    xml = TeiBiblStruct().to_xml([[reference], [reference]])
    doc = etree.fromstring(xml.encode())
    assert relaxng.validate(doc), relaxng.error_log
