import pytest
from llamore.parser import TeiBiblStruct
from llamore.reference import Organization, Person, Reference
from lxml.etree import Element, SubElement, _Element
from lxml import etree


@pytest.mark.skip("Wait for the new gold data to be commited")
def test_read_gold_data(pytestconfig):
    from notebooks.utils import read_gold_data

    xml_path = pytestconfig.rootpath / "data" / "gold" / "10.1111_1467-6478.00057.xml"

    input_texts, references, labels_raw = read_gold_data(xml_path)
    assert len(input_texts) == len(references) == len(labels_raw) == 74
    assert (
        input_texts[0]
        == "1 A. Phillips, ‘Citizenship and Feminist Politics’ in Citizenship, ed. G. Andrews (1991) 77."
    )
    assert len(references) == 74
    assert sum([len(c) for c in references]) == 110

    reference = references[0][0]
    assert reference.analytic_title == "Citizenship and Feminist Politics"
    assert reference.authors == [Person(first_name="A.", surname="Phillips")]
    assert reference.monographic_title == "Citizenship"
    assert reference.editors == [Person(first_name="G.", surname="Andrews")]
    assert reference.journal_title is None
    assert reference.publication_date == "1991"

    reference = references[1][0]
    assert (
        reference.analytic_title
        == "“Mere Auxiliaries to the Commonwealth”: Women and the Origins of Liberalism"
    )
    assert reference.authors == [
        Person(first_name="T.", surname="Brennan"),
        Person(first_name="C.", surname="Pateman"),
    ]
    assert reference.monographic_title is None
    assert reference.editors is None
    assert reference.journal_title == "Political Studies"
    assert reference.publication_date == "1979"

    reference = references[3][1]
    assert reference.volume == "20"

    reference = references[20][0]
    assert reference.publication_place == "N.S.W."

    xml_path = pytestconfig.rootpath / "data" / "gold" / "10.1515_zfrs-1980-0103.xml"

    input_texts, references, labels_raw = read_gold_data(xml_path)
    assert references[36][22].publisher == "Wissenschaftszentrum Berlin"


def test_to_from_xml(tmp_path_factory):
    reference = Reference(
        analytic_title="at",
        monographic_title="mt",
        journal_title="jt",
        authors=[
            Person(
                first_name="fn",
                middle_name="md",
                name_link="nl",
                role_name="rl",
                surname="sn",
            )
        ],
        editors=[Person(first_name="a"), Person(first_name="b", surname="b2")],
        publication_date="1978",
        publication_place="Moon",
        publisher="Mann im Mond",
        translator=Person(first_name="t", surname="s"),
        pages="666",
        cited_range="666-9",
        volume="42",
        footnote_number="43",
    )
    reader = TeiBiblStruct()

    xml_str = reader.to_xml(reference)
    print(xml_str)
    references = reader.from_xml(xml_str=xml_str)
    print(references)
    assert reference == references[0][0]

    tmp_file = tmp_path_factory.mktemp("test") / "tei_biblstruct.xml"
    reader.to_xml(reference, file_path=tmp_file)
    assert reference == reader.from_xml(tmp_file)[0][0]


def test_to_from_xml_for_pubPlace():
    parser = TeiBiblStruct(namespaces=None)
    ref = Reference(publication_place="Berlin,  Brandenburg")
    xml_string = parser.to_xml(ref)

    expected = """<TEI>
  <listBibl>
    <biblStruct>
      <monogr>
        <imprint>
          <date></date>
          <pubPlace>Berlin</pubPlace>
          <pubPlace>Brandenburg</pubPlace>
        </imprint>
      </monogr>
    </biblStruct>
  </listBibl>
</TEI>
"""
    assert xml_string == expected

    ref2 = parser.from_xml(xml_str=xml_string)[0][0]
    assert ref2 == ref




class TestFindPersonsAndOrganizations:
    @pytest.fixture
    def parser(self) -> TeiBiblStruct:
        return TeiBiblStruct(namespaces=None)

    @pytest.fixture(params=["author", "editor"])
    def authitor(self, request) -> _Element:
        biblStruct = Element("biblStruct")
        return SubElement(biblStruct, request.param)

    @pytest.mark.parametrize("with_or_without_type", ["with", "without"])
    def test_std(self, parser: TeiBiblStruct, authitor: _Element, with_or_without_type: str):
        persName = SubElement(authitor, "persName")
        if with_or_without_type == "with":
            first_name = SubElement(persName, "forename", attrib={"type": "first"})
        else:
            first_name = SubElement(persName, "forename")
        first_name.text = "John"
        surname = SubElement(persName, "surname")
        surname.text = "Wayne"

        person = parser._find_persons_and_organizations(
            authitor.getparent(), author_or_editor=authitor.tag
        )
        assert person == [Person(first_name="John", surname="Wayne")]

    def test_raw(self, parser: TeiBiblStruct, authitor: _Element):
        authitor.text = "John Wayne"

        assert parser._find_persons_and_organizations(
            authitor.getparent(), author_or_editor=authitor.tag
        ) == [Person(surname="John Wayne")]

    @pytest.mark.parametrize("with_or_without_type", ["with", "without"])
    def test_without_persName(self, parser: TeiBiblStruct, authitor: _Element, with_or_without_type: str):
        if with_or_without_type == "with":
            first_name = SubElement(authitor, "forename", attrib={"type": "first"})
        else:
            first_name = SubElement(authitor, "forename")
        first_name.text = "John"
        surname = SubElement(authitor, "surname")
        surname.text = "Wayne"

        assert parser._find_persons_and_organizations(
            authitor.getparent(), author_or_editor=authitor.tag
        ) == [Person(first_name="John", surname="Wayne")]

    def test_organization(self, parser: TeiBiblStruct, authitor: _Element):
        orgName = SubElement(authitor, "orgName")
        orgName.text = "JohnWayne GmbH"

        assert parser._find_persons_and_organizations(
            authitor.getparent(), author_or_editor=authitor.tag
        ) == [Organization(name="JohnWayne GmbH")]

    def test_translator(self, parser: TeiBiblStruct):
        biblStruct = Element("biblStruct")
        editor = SubElement(biblStruct, "editor", attrib={"role": "translator"})
        persName = SubElement(editor, "persName")
        first_name = SubElement(persName, "forename", attrib={"type": "first"})
        first_name.text = "John"
        surname = SubElement(persName, "surname")
        surname.text = "Wayne"

        editors = parser._find_persons_and_organizations(biblStruct, author_or_editor="editor")
        assert editors == []

        translator = parser._find_translator(biblStruct)

        assert translator == Person(first_name="John", surname="Wayne")


def test_find_ref():
    biblStruct = Element("biblStruct")
    ref = SubElement(biblStruct, "ref")
    persName = SubElement(ref, "persName")
    persName.text = "John Wayne"
    op_cit = SubElement(biblStruct, "ref", attrib={"type": "op-cit"})
    op_cit.text = "op. cit"
    footnote = SubElement(biblStruct, "ref", attrib={"type": "footnote"})
    footnote.text = "n. 18"

    # print(tostring(ElementTree(biblStruct), pretty_print=True).decode())

    refs = TeiBiblStruct(namespaces=None)._find_and_join_all_refs(biblStruct)
    assert refs == "John Wayne op. cit n. 18"
