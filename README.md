# Wiktionary Sense Collector

Collects English Wiktionary senses and aligns them with translations and synsets.

<!-- installation -->

## Installation

### Requirements

- Python 3.14 or later

### From source

Clone the repository:

```sh
git clone git@github.com:evaevangelisti/WSC.git
cd WSC
```

Create a virtual environment, activate it, and install the package into it:

```sh
python -m venv .venv
source .venv/bin/activate
pip install .
```

With [uv](https://docs.astral.sh/uv/), one command does all three:

```sh
uv tool install .
```

<!-- usage -->

## Usage

Collection has three steps. Each caches its output for subsequent commands and reuse.

The options a whole session shares can be set once, through `WSC_DUMP_DATE` and `WSC_CACHE_DIR`.

Run `wsc <command> --help` for the whole of it.

### fetch

Downloads a Wiktionary dump.

```sh
wsc fetch
```

| Option | Default | |
| --- | --- | --- |
| `--dump-date` | `latest` | Dump to use, as `20260801` |
| `--cache-dir` | your platform's cache directory | Where the sources and what is made of them are kept |

### parse

Reads the dump with [wiktextract](https://github.com/tatuylonen/wiktextract), which turns Wiktionary's markup into entries, keeping the fields the collector reads.

```sh
wsc parse
```

| Option | Default | |
| --- | --- | --- |
| `--dump-date` | `latest` | Dump to use, as `20260801` |
| `--archive` | off | Download the parse kaikki.org publishes instead of making one |
| `--processes` | `1` | Processes wiktextract may run, at 4 GB each |
| `--db-path` | a temporary file | Where the pages extracted from the dump are kept |
| `--cache-dir` | your platform's cache directory | Where the sources and what is made of them are kept |

### collect

Collects senses and reports into a directory, with one JSONL entry per headword and part of speech.

Wiktextract's bold ranges and [kwic](https://github.com/evaevangelisti/kwic) independently locate lemmas. Offsets record `bold`, `lemmatizer`, or both, preserving disagreements for review.

```sh
wsc collect
```

| Option | Default | |
| --- | --- | --- |
| `--dump-date` | `latest` | Dump to use, as `20260801` |
| `--pos` | every part of speech | Parts of speech to keep; repeat to name several |
| `--min-year` | no limit | Oldest quotation to keep |
| `--max-year` | no limit | Newest quotation to keep |
| `--engine` | `spacy` | What reads a sentence to locate the lemma in it |
| `--processes` | `1` | Processes the reading may run; Stanza runs one regardless |
| `--batch-size` | `8` | How many sentences the reading takes at a time |
| `--gpu` | off | Read on the graphics card |
| `--cache-dir` | your platform's cache directory | Where the sources and what is made of them are kept |
| `--output-dir` | `collection` | Directory for the collection and its reports |

| Engine | Reads with | Speed |
| --- | --- | --- |
| `spacy` | a transformer pipeline, the most accurate of the three | tens a second |
| `stanza` | Stanza, whose parser finds a phrasal verb written apart | tens a second |
| `lemminflect` | spaCy for the tags and LemmInflect for the lemmas | hundreds a second |

Each collection contains the collected senses, a manifest, and reports in JSON and Markdown:

| File | Contents |
| --- | --- |
| `senses.jsonl` | Collected entries and their senses |
| `report.json` | Numeric statistics and complete frequency distributions |
| `report.md` | Readable tables describing coverage and offset agreement |
| `manifest.json` | Resolved dump, source files, filters, runtime versions, and timestamps |

### align

Aligns collected senses with translations and supplied synsets through offline [vLLM](https://vllm.ai/) inference.

```sh
wsc align collection/senses.jsonl
```

| Option | Default | |
| --- | --- | --- |
| `--task` | both resources | `translations` or `synsets`; repeat to select both |
| `--synsets` | `synsets.jsonl` | Synsets in JSON Lines format |
| `--model` | `openai/gpt-oss-120b` | Local model path or Hugging Face identifier |
| `--gloss-mode` | `last` | Last gloss or full hierarchy joined with ` > ` |
| `--prompts` | bundled `prompts.toml` | One customizable template per task |
| `--temperature` | `0.0` | Sampling temperature |
| `--maximum-tokens` | `4096` | Generated token limit |
| `--batch-size` | `32768` | Prompts prepared for one vLLM inference call |
| `--reasoning-parser` | unset | vLLM reasoning parser |
| `--reasoning-effort` | unset | Reasoning effort supported by the chat template |
| `--chat-template-option` | unset | Chat template `KEY=VALUE` argument; repeat to set multiple |
| `--engine-option`, `-o` | unset | Additional vLLM engine parameter; repeat to set multiple |
| `--reuse` | off | Reuse decisions by source ID and infer only missing senses |
| `--verbose` | off | Show logs for individual alignment requests |
| `--cache-dir` | platform cache | Also configurable through `WSC_CACHE_DIR` |
| `--output-dir` | `alignment` | Directory for aligned senses and its reports |

Each alignment contains the aligned collection, a manifest, and one report per selected task:

| File | Contents |
| --- | --- |
| `senses.jsonl` | Collected entries with aligned translations and synset associations |
| `reports/translations.json` | Evaluated, aligned, and unaligned senses, translation associations, and relation counts |
| `reports/synsets.json` | Evaluated, aligned, and unaligned senses, synset associations, and relation counts |
| `manifest.json` | Input collection, tasks, model settings, runtime versions, timestamps, and output files |
