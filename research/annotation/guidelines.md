# Annotation guidelines

We label sentences to find out whether the pipeline marks the right occurrences of a headword, and how often it does not.

## What you see

The headword with its part of speech, the sense the sentence illustrates, and the sentence with every occurrence the pipeline found in bold.

The pipeline looks for the headword and its inflections, whatever the case, wherever a whole word stands. It knows nothing of part of speech.

## Occurrences

Say what is wrong with the bold, if anything.

**All right.** Every occurrence of the headword is in bold, and nothing else is.

**One was missed.** The headword is in the sentence, outside the bold.
Violation: the headword is `don't`, the sentence writes `don’t`.

**One is not the headword.** Something in bold is another word spelled the same.
Violation: the headword is `may`, the sentence names the month.

**One stops in the wrong place.** The bold covers too much of a word, or too little.
Violation: `banking` marked for the headword `bank`.

**Not in the sentence.** Nothing is in bold, and nothing should be.
Violation: the headword is `give up`, the sentence writes `gave it up`.

Mark the first fault that fits, reading down the list.

## Part of speech

Say whether the marked word carries the part of speech of the sense.

Judge the word as it is used, not as it could be used. Where nothing is marked, say so.

## When unsure

- Judge what is shown, not what was meant.
- Open the entry when the sense alone does not settle it.
- Mark a fault only when you can name it.
- When something is wrong and no label fits, write it in the notes.
