<h1 align="center">
  <a href=""><img src="llamore.png" alt="Llamore logo" width="140"></a>
  <br>
  Llamore
</h1>
<p align="center"><b>L</b>arge <b>LA</b>nguage <b>MO</b>dels for <b>R</b>eference <b>E</b>xtraction</b></p>

<p align="center"><b>A framework to extract and evaluate scientific references and citations from free-form text and PDFs using LLM/VLMs.</b></p>

## Setup

```bash
pip install llamore
```

## Quick start

A few things you can do with Llamore.

### Extract references

Define your extractor. You can use the `OpenaiExtractor` for most of the open model serving frameworks like Ollama, vLLM, etc.

```python
from llamore import GeminiExtractor, OpenaiExtractor

extractor = GeminiExtractor(api_key="MY_GEMINI_API_KEY")
```

Extract references from a PDF or a raw input string.

```python
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

f1 = F1(levenshtein_distance=0.9)

f1.compute_macro_average(references, gold_references)
# or compute metrics per field
f1.compute_micro_average(references, gold_references)
```

You can also have a look at the [quick start notebook](notebooks/quick_start.ipynb).

## Reference JSON and RNG schema

Llamore internally defines a reference via a pydantic BaseModel in `llamore.reference.Reference`.
It is based on the TEI biblStruct model. Schema files are published on this repository's GitHub page:

- JSON schema: [Code](./docs/schema/llamore.schema.json) | [Download](https://mpilhlt.github.io/llamore/schema/llamore.schema.json)
- Relax NG schema: [Code](./docs/schema/llamore.rng) | [Download](https://mpilhlt.github.io/llamore/schema/llamore.rng)
