# WSC

Collects the senses of every Wiktionary lemma into one file, each with its glosses, its labels and the sentences that illustrate it.

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

## Usage

Collecting senses takes three steps. Each keeps what it made in a cache, so it is only ever run once, and the next step picks it up from there.

The options a whole session shares can be set once, through `WSC_LANGUAGE`, `WSC_DUMP_DATE` and `WSC_CACHE_DIR`. 

Run `wsc <command> --help` for the whole of it.

### fetch

Downloads a Wiktionary dump.

```sh
wsc fetch
```

| Option | Default | |
| --- | --- | --- |
| `--language` | `en` | Wiktionary edition to read, by language code |
| `--dump-date` | `latest` | Dump to use, as `20260801` |
| `--cache-dir` | your platform's cache directory | Where dumps and what is made of them are kept |

### parse

Reads the dump with [wiktextract](https://github.com/tatuylonen/wiktextract), which turns Wiktionary's markup into entries.

```sh
wsc parse
```

| Option | Default | |
| --- | --- | --- |
| `--language` | `en` | Wiktionary edition to read, by language code |
| `--dump-date` | `latest` | Dump to use, as `20260801` |
| `--processes` | `1` | Processes wiktextract may run, at 4 GB each |
| `--cache-dir` | your platform's cache directory | Where dumps and what is made of them are kept |

### collect

Reads the senses out of those entries and writes them where you asked. The suffix of the file picks the format.

```sh
wsc collect <output>
```

| Option | Default | |
| --- | --- | --- |
| `--language` | `en` | Wiktionary edition to read, by language code |
| `--dump-date` | `latest` | Dump to use, as `20260801` |
| `--pos` | every part of speech | Parts of speech to keep; repeat to name several |
| `--min-year` | no limit | Oldest quotation to keep |
| `--max-year` | no limit | Newest quotation to keep |
| `--cache-dir` | your platform's cache directory | Where dumps and what is made of them are kept |
