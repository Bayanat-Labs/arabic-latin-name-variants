# arabic-latin-name-variants

By **Bayanat Labs**. The code behind the [Arabic-Latin Name Variants](https://huggingface.co/datasets/Bayanat-Labs/arabic-latin-name-variants) dataset: it streams the Wikidata JSON dump once, keeps the Arabic-script and Latin-script names of people, places and organisations, labels each Arabic–Latin pair as a transliteration, a mixed form or a translation/exonym, aligns names word by word (with Arabic compounds such as عبد الله and نور الدين kept together), groups the Latin spellings of each Arabic word into variant tables, and builds a small Arabic–Latin name-matching benchmark with four baselines.

Code: Apache-2.0. The dataset built with it is CC0 1.0 and derived from Wikidata (CC0).

## Install

Python 3.10 or later.

```bash
pip install "arabic-latin-name-variants @ git+https://github.com/Bayanat-Labs/arabic-latin-name-variants@v0.1.0"
```

This also installs [arabic-normalize](https://github.com/Bayanat-Labs/arabic-normalize) from GitHub at a fixed commit (Arabic text normalisation, policy `bayanat-norm-v1`), PyArrow, DuckDB, zstandard, regex and requests. `pip install "...[fast]"` adds orjson for faster parsing.

## Use the released tables

```python
from datasets import load_dataset

tokens = load_dataset("Bayanat-Labs/arabic-latin-name-variants", "token_variants", split="train")
```

```bash
arabic-latin-name-variants lookup "عبد الله" --release path/to/downloaded/dataset
```

## Rebuild from the dump

The dump is about 157 GB (gzip). It is read once over one HTTP connection, inflated in memory and never stored; a break in the connection resumes with an HTTP Range request, and a stopped run resumes from its last checkpoint (`--resume`). Every request sends the User-Agent `BayanatLabs-OpenData/1.0 (+https://bayanatlabs.com/open; hello@bayanatlabs.com)`, and a 403 or 429 reply stops the run.

```bash
# 1. stream: compact name records and the subclass (P279) table
arabic-latin-name-variants stream \
  --url https://dumps.wikimedia.org/wikidatawiki/entities/20261026/wikidata-20261026-all.json.gz \
  --expect-md5 356990468877421dc061694ad61c487f --expect-sha1 d2112c5243f3842e32fef6fc0b4b42e533954ee4 \
  --keep-classes keep_classes.txt --person-classes person_classes.txt --out stream/
# 2. build: name strings, relation filter, alignment (Arab-World Wikidata slice for the arab_world scope)
arabic-latin-name-variants build --stream-dir stream/ --slice entities.parquet --work work/
# 3. release: entity_variants, token_variants, clusters, name_matching_bench, meta/bench_results.json
arabic-latin-name-variants release --work work/ --out release/
arabic-latin-name-variants stats --out release/
```

`--keep-classes` and `--person-classes` limit the stream to items whose class (P31) reaches the human, place or organisation roots through subclass of (P279). Both lists come from a class table of the same dump: `arabic-latin-name-variants class-filter --stream-dir DIR --out DIR` writes them from the `classes_*.jsonl.zst` files of an earlier stream. Without them, the stream keeps every item with a P31 value and Arabic and Latin names, and the build applies the same class rule afterwards. The slice file is the `entities` config of [Bayanat-Labs/arab-world-wikidata](https://huggingface.co/datasets/Bayanat-Labs/arab-world-wikidata) built from the same dump.

## Method

- **Scope.** Items whose nearest class root (through P279) is human (Q5), a place root (geographic location, administrative territorial entity, geographical feature, human settlement, architectural structure) or organization (Q43229), with at least one Arabic-script name in `ar`, `arz` or `ary` and at least one Latin-script name in `en`, `fr`, `de`, `es`, `it`, `nl`, `pt`, `tr`, `id`, `ms`, `sv` or `mul`. Names are labels, aliases, the monolingual statements P1559, P1705, P1448 and P1813, and P2440 transliterations with their P459 method. Wikimedia categories, templates, disambiguation and list pages and scholarly articles are left out. People whose latest possible birth year (from P569 at any rank, or a birth year or childhood named in a description) is 2008 or later are left out; birth dates and descriptions are read only for this rule.
- **Strings.** Each string is classified by script (Unicode `Script=Arabic` and `Script=Latin`); mixed-script strings, other scripts, strings with digits, web addresses or handles and encoding damage ($ or the square root sign) are dropped; parenthesised disambiguators are removed, and leading honorifics are removed from people's names.
- **People's strings.** Wikidata labels and aliases are free text, and on a person some hold a description or a statement instead of a name. Three checks run on people's rows. (1) `text.human_string_reason`: a lexicon of words and phrases in every language present, sentence punctuation, ten or more words with no name particle, and a limit of 90 characters. (2) The same-name rule, for people who may be alive (latest possible birth year 1900 or later, or none known): a pair is kept only when the two strings read as the same name, so a `translation_exonym` pair goes, and so does a pair with a word of three or more letters on either side that is neither a listed particle or title nor matched in a word pair of any row of the entity. This also removes many real names, such as nicknames and full names with a middle name. (3) Strings flagged in a separate check by Bayanat Labs that the first two checks miss. `build --drop-list` removes such strings, given as pairs of a QID and the SHA-256 of the normalised string. A rebuild from this repository does not repeat that check, so it keeps those strings.
- **Normalisation.** Arabic `text_norm` uses arabic-normalize, policy `bayanat-norm-v1`. `latin_fold` is casefolded Latin text without diacritics or apostrophe-like marks.
- **Romanisation fields.** `buckwalter` is a fixed one-character-per-letter table (Tim Buckwalter's, with پ P, چ J, ڤ V, گ G), applied to the raw Arabic; it is null when a character is outside the table. ALA-LC is only filled when attested (a P2440 transliteration with P459 ALA-LC romanization, or a Latin name of the entity in macron and underdot letters that matches; such a name may follow IJMES, BGN/PCGN or another system and is not verified as ALA-LC) or derived by rule from an Arabic form that carries its short vowels; `ala_lc_origin` says which, and it is never derived from text without vowel marks. DIN 31635 is filled only when attested by P2440 with P459 DIN 31635 as a qualifier on an Arabic name; a free-standing value shows in `method_qids`.
- **Relation filter.** Both strings are reduced to consonant skeletons (Arabic letters to phonetic classes; Latin digraphs such as kh, gh, sh, ch, th, dh, dj and dsch read as one sound) and compared with a weighted edit distance in which usual renderings cost nothing (ج as j, g, dj or Turkish c; ق as q, k or g). Similarity of at least 0.7 is `transliteration`, below 0.4 `translation_exonym`, otherwise `mixed`. The word alignment then refines the label: a transliteration that leaves a word unmatched on either side (an extra given name, a title) becomes `mixed`, and a translation in which one Arabic word of two or more consonants has a Latin word at similarity 0.8 or more ("Kingdom of Bahrain" for البحرين) becomes `mixed`.
- **Alignment.** Arabic compounds are merged first (عبد + next word, أبو/أبي/أبا + next word, a word + الدين or الله, آل + next word); the article ال stays on its word, and Latin article forms (al-, el-, ad-, ash-, ul- ...) stay on the following word. Equal unit counts are aligned by position; otherwise a dynamic programme allows one-to-two and two-to-one merges and skipped units. Alignments scoring at least 0.65 feed the token tables, and inside them only word pairs with a similarity of at least 0.7.
- **Variant tables.** `token_variants` is computed from the word pairs in `entity_variants`. It groups, for each Arabic word (`text_norm`), its aligned Latin words by `latin_fold`, with entity and language counts, a rank and a heuristic style tag. Format characters, edge punctuation and a Persian ezafe in front or behind are removed first, a final dot does not separate spellings, and a Latin unit with a lone initial or a generational suffix beside another word ("Charles A.") gives no spelling. Arabic units lose edge punctuation too, and a lone Arabic initial gives no spelling. A spelling is listed when at least two entities give it. `clusters` groups full names on identical Arabic `text_norm` across entities, from `transliteration` pairs; `canonical_latin` is the spelling with the most entities, then the most languages, then one attested in English.
- **Benchmark.** 10% of entities are held out by a fixed hash of the QID. 5,000 positive pairs (an Arabic name and a transliteration of the same held-out entity) and 5,000 hard negatives (names of different held-out entities with skeleton similarity of at least 0.5, different Arabic names, and a Latin name not attested for that Arabic name) are scored by four baselines: Buckwalter-fold exact match, Jaro-Winkler on a naive romanisation, the skeleton similarity, and a lookup in a `token_variants` table rebuilt without the held-out entities. Recall is reported at false positive rates of 1% and 0.1%, and for each scope at the 1% threshold.

## Tests

```bash
pip install ".[test]"
python -m pytest -q
```

The tests run offline on a small synthetic dump in the Wikidata format (made-up QIDs).

## Licence

Code: Apache License 2.0 (`LICENSE`). Data derived with it from Wikidata is CC0 1.0 (Wikidata's structured data is CC0). See `NOTICE`.

## Cite

See `CITATION.cff`.

## Contact

Questions, corrections and removal requests: hello@bayanatlabs.com
