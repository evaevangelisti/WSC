# Annotation guidelines

We label word offsets to find out how often the collector marks the right words, and what it misses.

## What you see

A lemma with its part of speech, and a sentence in which every occurrence the collector found is set in bold.

The same sentence comes round more than once, marked as a different run left it. Judge each on its own and do not try to remember the last one.

## Is every marking an occurrence of the lemma?

**All correct.** Every bold span is the lemma, in that part of speech, and runs from its first character to its last.

**Some wrong.** At least one bold span is not.

Violation: `bank` as a noun, and `She **banked** the plane.`
Violation: `give up` as a verb, and `She **gave** the money up.`

**Nothing marked.** No span is bold.

## Is any occurrence left unmarked?

**Nothing missing.** Every occurrence of the lemma in that part of speech is bold. A sentence that never uses the lemma has nothing to miss.

**Something missing.** At least one occurrence is not bold.

Violation: `bank` as a noun, and `The **bank** raised its rates, so I left the bank.`

## When unsure

- Judge the sentence as written, not as intended.
- Judge the lemma in the part of speech named, and leave the rest of the sentence alone.
- A word talked about rather than used is still an occurrence.
- A lemma of several words runs from its first word to its last, whatever sits between them.
- Write anything the labels miss in the notes.
