# Text cleaning

Collection applies the same text policies to Wiktextract entries and supplementary translation tables.

| Field | Reformatted | Excluded | Preserved |
| --- | --- | --- | --- |
| Sense glosses | Complete links, emphasis, HTML scripts, bounded references, editorial URLs | Bibliographies, unresolved templates, damaged notation | Numeric definitions, chemical brackets, lexical uses of “see” |
| Sentences | Formatting remnants, audited encoding errors, layout characters | Lexical metadata, explicit navigation, unrecoverable fragments | Referenced quotations, literal code, verse and dialogue line breaks |
| Table glosses | The same display formatting and bounded references | Exact placeholder headings and incomplete links | Meaningful headings beginning with “see” |
| Translation words | Complete links and recognized translation templates | Reference instructions, incomplete markup, invisible-only values | Geographic names, script joiners, Skolt Sami orthography |
| Languages | Leading and trailing whitespace | Malformed code syntax | Source codes and dialect subtags absent from the local registry |

The dump reader splits arguments only outside balanced templates and links. It reads templates across line breaks and processes table boundaries in source order. Unknown templates are not expanded by a partial imitation of MediaWiki.

Cleaning tracks exact text edits. Existing offsets shift with their text and retain their provenance. A range whose boundary falls inside a rewritten token is omitted when its new position is ambiguous. During collection, lemma location runs on the cleaned sentence.

An unusable leaf definition excludes its sense instead of silently replacing it with a broader parent. A navigation-only hierarchy level can be removed while retaining a defining level. Bibliographic glosses are not reassigned as quotations without an identifiable source sense.

## Coverage and limits

These policies address every category in the collection audit, but pattern handling is not a guarantee of semantic correctness. Ambiguous prose remains conservative: genuine quotations and ordinary examples containing “see” are preserved. Unsupported TeX and incomplete markup are excluded when the cleaner cannot recover their meaning. Complete mathematical alphabet groups, known symbol commands, and HTML superscripts and subscripts can be rendered without guessing missing command boundaries.

Previously truncated supplementary glosses and translations require recovery from the cached source. The corrected parser prevents the same truncation in future extractions; it cannot reconstruct missing characters from an existing JSONL value. Source retrieval, existing collection migration, and alignment reconciliation are separate from these extraction rules.

## Read-only validation

The 2026-09-20 preview processed every entry in `data/spacy/senses.jsonl`: 739,743 entries and 7,655,867 field occurrences, including language keys. It did not write collection or cache files.

| Field | Reformatted occurrences | Excluded occurrences |
| --- | ---: | ---: |
| Sense glosses | 849 | 85 |
| Sentences | 1,084 | 5,018 |
| Table glosses | 367 | 132 |
| Translation words | 23 | 171 |
| Language keys | 0 | 0 |

These are independent field-level outcomes, before source recovery, sense removal, merging, or alignment reconciliation. They are not final migration totals. Every retained offset stayed within its sentence; three ranges had ambiguous rewritten boundaries and were omitted. SHA-256 checks matched the previous audit for all four collections, both alignment TSVs, and the supplementary translation cache.

## Identifier migration

New entry IDs replace ASCII spaces with underscores; headword text and existing hyphens remain unchanged. Sense, table, and alignment-query IDs inherit the entry prefix. An identifier-only migration preserves existing hash suffixes and WordNet synset IDs.

Existing collection files, alignment TSVs, and supplementary translation-cache keys must migrate together before mixing old artifacts with new extraction output. Text cleanup additionally changes sense and table hashes when their defining glosses change. Preserve an old-to-new identity map, remap retained offsets, and reconcile affected alignment decisions before publishing migrated files. The migration records text changes separately from the identifier-only rule.

## Applied migration

The 2026-09-21 migration updates all four collections, their reports and manifests, both alignment TSVs, and the supplementary translation cache. Each collection retains 739,717 entries and 1,006,066 senses. Source recovery repairs 141 damaged table records. No NLP or model inference is repeated.

Existing offset provenance is preserved; ambiguous rewritten boundaries are omitted. Unsafe alignment decisions are invalidated for 142 translation sources and one WordNet source. Historical cache records remain available. The complete results, identity mappings, exclusions, and checksums are recorded in `data/quality-audit/migration-20260921.md` and its JSON ledger.

## API

See the functions documented under [glosses](wiktionary/parts/glosses.md), [sentences](wiktionary/parts/sentences.md), [translations](translations.md), [markup](markup.md), and [offsets](offsets.md).
