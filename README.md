# Wiktionary Sense Collector

Collects every English Wiktionary lemma into one file, with its glosses, synonyms, labels, attesting sentences and translations.

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

Collecting senses takes three steps. Each keeps what it made in a cache, so it is only ever run once, and the next step picks it up from there.

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

Parsing runs for hours. `--archive` downloads what [kaikki.org](https://kaikki.org) holds instead, whatever that site currently publishes.

The dump is then walked for the translations wiktextract misses: a table on a subpage, a meaning another headword translates.

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

Reads the senses into a file, one entry per headword and part of speech, the suffix picking the format.

[kwic](https://github.com/evaevangelisti/kwic) reads each sentence to place the lemma; where the reading finds nothing, the listed forms are matched.

`--gpu` needs CuPy, which kwic offers as an extra named after your CUDA release: `pip install "kwic[cuda13x]"`.

```sh
wsc collect senses.jsonl
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

| Engine | Reads with | Speed |
| --- | --- | --- |
| `spacy` | a transformer pipeline, the most accurate of the three | tens a second |
| `stanza` | Stanza, whose parser finds a phrasal verb written apart | tens a second |
| `lemminflect` | spaCy for the tags and LemmInflect for the lemmas | hundreds a second |

### wordnet

Downloads [Open English WordNet](https://en-word.net) and reads its synsets, worth running only if you mean to align senses.

```sh
wsc wordnet
```

| Option | Default | |
| --- | --- | --- |
| `--edition` | `latest` | Wordnet edition to use, as `2025` |
| `--cache-dir` | your platform's cache directory | Where the sources and what is made of them are kept |
