import pytest
from llamore.reference import Organization, Person, Reference, References


def test_reference():
    json_str = (
        """{"authors": [{"forename": "forename  check\\t", "surname": "surname"}]}"""
    )
    assert Reference.model_validate_json(json_str).authors == [
        Person(forename="forename check", surname="surname")
    ]

    json_str = """{"date": "", "analytic_title": "title"}"""
    ref = Reference.model_validate_json(json_str)
    assert ref.publication_date is None

    json_str = """{"authors": [{"name": ""}]}"""
    ref = Reference.model_validate_json(json_str)
    assert ref.authors is None


def test_authitor():
    p = Person(forename="", surname="Check")
    assert p.forename is None
    assert p.surname == "Check"

    p = Person(forename="Check\t", surname="")
    assert p.forename == "Check"
    assert p.surname is None

    o = Organization(name=" GmbH\t")
    assert o.name == "GmbH"
    o = Organization(name=123)
    assert o.name == "123"


@pytest.mark.parametrize(
    "inp,out", [(1988, "1988"), (1.8, "1.8"), (None, None), ("None", "None")]
)
def test_date(inp, out):
    c = Reference(analytic_title="title", publication_date=inp)
    assert c.publication_date == out


def test_references(tmpdir_factory):
    pass


def test_references_to_xml(tmpdir_factory):
    references = References(
        [
            Reference(
                analytic_title="Article Title",
                authors=[Person(forename="John", surname="Doe")],
                journal_title="Journal Name",
                publication_date="2023",
            ),
            Reference(
                monographic_title="Book Title",
                authors=[Organization(name="Some Organization")],
                publisher="Publisher Name",
                publication_date="2022",
            ),
        ]
    )
    xml_output = references.to_xml()

    assert isinstance(xml_output, str)
    assert "<listBibl>" in xml_output
    assert "<biblStruct>" in xml_output
    assert "Article Title" in xml_output
    assert "John" in xml_output
    assert "Doe" in xml_output
    assert "Journal Name" in xml_output
    assert "Book Title" in xml_output
    assert "Some Organization" in xml_output
    assert "Publisher Name" in xml_output

    temp_file = tmpdir_factory.mktemp("data").join("output.xml")
    references.to_xml(file_path=str(temp_file))
    assert temp_file.check(file=1)

    references2 = References.from_xml(file_path=str(temp_file))
    assert references == references2


def test_references_from_xml():
    xml_input = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <listBibl>
            <biblStruct>
                <analytic>
                    <title level="a">Article Title</title>
                    <author>
                        <persName>
                            <forename>John</forename>
                            <surname>Doe</surname>
                        </persName>
                    </author>
                </analytic>
                <monogr>
                    <title level="j">Journal Name</title>
                    <imprint>
                        <date>2023</date>
                    </imprint>
                </monogr>
            </biblStruct>
            <biblStruct>
                <monogr>
                    <title level="m">Book Title</title>
                    <author>
                        <orgName>Some Organization</orgName>
                    </author>
                    <imprint>
                        <publisher>Publisher Name</publisher>
                        <date>2022</date>
                    </imprint>
                </monogr>
            </biblStruct>
        </listBibl>
    </TEI>
    """
    references = References.from_xml(xml_str=xml_input)
    assert len(references) == 2

    ref1 = references[0]
    assert ref1.analytic_title == "Article Title"
    assert ref1.authors[0].forename == "John"
    assert ref1.authors[0].surname == "Doe"
    assert ref1.journal_title == "Journal Name"
    assert ref1.publication_date == "2023"

    ref2 = references[1]
    assert ref2.monographic_title == "Book Title"
    assert ref2.authors[0].name == "Some Organization"
    assert ref2.publisher == "Publisher Name"
    assert ref2.publication_date == "2022"
