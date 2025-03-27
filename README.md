# Llamore

***L**arge **LA**nguage **MO**dels for **R**eference **E**xtraction*

A framework to extract and evaluate scientific references and citations from free-form text and PDFs using LLM/VLMs.

## Setup

```bash
pip install llamore
```

## Quick start

A few things you can do with Llamore.

### Extract references

```python
from llamore import GeminiExtractor

extractor = GeminiExtractor(api_key="MY_GEMINI_API_KEY")
references = extractor(pdf="path/to/my.pdf")
```
or

```python
text = """4 I have explored the gendered nature of citizenship at greater length in two complementary
papers: ‘Embodying the Citizen’ in Public and Private: Feminist Legal Debates, ed. M.
Thornton (1995) and ‘Historicising Citizenship: Remembering Broken Promises’ (1996) 20
Melbourne University Law Rev. 1072."""
references = extractor(text=text)
```

### Export as TEI biblStructs

```python
references.to_xml("./my_references.xml")
```

### Evaluate with gold references

```python
from llamore import F1

f1 = F1()
f1.compute_macro_average(references, gold_references)
```


You can also have a look into the jupyter notebook at [notebooks/quick_start.ipynb](notebooks/quick_start.ipynb).

## Reference JSON schema

Llamore internally defines a reference via a pydantic BaseModel in `llamore.reference.Reference`.
It is based on the TEI biblStruct model and its JSON schema is the following:

```json
{
  "$defs": {
    "Organization": {
      "description": "Contains information about an identifiable organization such as a business, a tribe, or any other grouping of people.",
      "properties": {
        "name": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Contains an organizational name.",
          "title": "Name"
        }
      },
      "title": "Organization",
      "type": "object"
    },
    "Person": {
      "description": "Contains a proper noun or proper-noun phrase referring to a person, possibly including one or more of the person's forenames, surnames, honorifics, added names, etc.",
      "properties": {
        "forename": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Contains a forename, given or baptismal name.",
          "title": "Forename"
        },
        "surname": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "description": "Contains a family (inherited) name of a person, as opposed to a given, baptismal, or nick name.",
          "title": "Surname"
        }
      },
      "title": "Person",
      "type": "object"
    }
  },
  "description": "A reference based on the TEI biblstruct format.",
  "properties": {
    "analytic_title": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "This title applies to an analytic item, such as an article, poem, or other work published as part of a larger item.",
      "title": "Analytic Title"
    },
    "monographic_title": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "This title applies to a monograph such as a book or other item considered to be a distinct publication, including single volumes of multi-volume works.",
      "title": "Monographic Title"
    },
    "journal_title": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "This title applies to any serial or periodical publication such as a journal, magazine, or newspaper.",
      "title": "Journal Title"
    },
    "authors": {
      "anyOf": [
        {
          "items": {
            "anyOf": [
              {
                "$ref": "#/$defs/Person"
              },
              {
                "$ref": "#/$defs/Organization"
              }
            ]
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Contains the name or names of the authors, personal or corporate, of a work; for example in the same form as that provided by a recognized bibliographic name authority.",
      "title": "Authors"
    },
    "editors": {
      "anyOf": [
        {
          "items": {
            "anyOf": [
              {
                "$ref": "#/$defs/Person"
              },
              {
                "$ref": "#/$defs/Organization"
              }
            ]
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Contains a secondary statement of responsibility for a bibliographic item, for example the name of an individual, institution or organization, (or of several such) acting as editor, compiler, translator, etc.",
      "title": "Editors"
    },
    "publisher": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Contains the name of the organization responsible for the publication or distribution of a bibliographic item.",
      "title": "Publisher"
    },
    "publication_date": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Contains the date of publication in any format.",
      "title": "Publication Date"
    },
    "publication_place": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Contains the name of the place where a bibliographic item was published.",
      "title": "Publication Place"
    },
    "volume": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Defines the scope of a bibliographic reference in terms of the volume of a larger work.",
      "title": "Volume"
    },
    "issue": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Contains an issue number, or issue numbers.",
      "title": "Issue"
    },
    "pages": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Defines the scope of a bibliographic reference in terms of page numbers.",
      "title": "Pages"
    },
    "cited_range": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Defines the range of cited content, often represented by pages or other units.",
      "title": "Cited Range"
    },
    "refs": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Defines references to another location, possibly modified by additional text or comment. ",
      "title": "Refs"
    }
  },
  "title": "Reference",
  "type": "object"
}
```
