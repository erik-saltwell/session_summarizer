# File formats
========
## RTTM
-------
Rich Transcription Time Marked (RTTM) files are space-delimited text files
containing one turn per line, each line containing ten fields:

- ``Type``  --  segment type; should always by ``SPEAKER``
- ``File ID``  --  file name; basename of the recording minus extension (e.g.,
  ``rec1_a``)
- ``Channel ID``  --  channel (1-indexed) that turn is on; should always be
  ``1``
- ``Turn Onset``  --  onset of turn in seconds from beginning of recording
- ``Turn Duration``  -- duration of turn in seconds
- ``Orthography Field`` --  should always by ``<NA>``
- ``Speaker Type``  --  should always be ``<NA>``
- ``Speaker Name``  --  name of speaker of turn; should be unique within scope
  of each file
- ``Confidence Score``  --  system confidence (probability) that information
  is correct; should always be ``<NA>``
- ``Signal Lookahead Time``  --  should always be ``<NA>``

For instance:

    SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>
    SPEAKER CMU_20020319-1400_d01_NONE 1 157.610000 3.060 <NA> <NA> tbc <NA> <NA>
    SPEAKER CMU_20020319-1400_d01_NONE 1 130.490000 0.450 <NA> <NA> chek <NA> <NA>

If you would like to confirm that a set of RTTM files are valid, use the
included ``validate_rttm.py`` script. For instance, if you have RTTMs
``fn1.rttm``, ``fn2.rttm``, ..., then

     python validate_rttm.py fn1.rttm fn2.rttm ...

will iterate over each line of each file and warn on any that do not match the
spec.

## UEM
------
Un-partitioned evaluation map (UEM) files are used to specify the scoring
regions within each recording. For each scoring region, the UEM file contains
a line with the following four space-delimited fields

- ``File ID``  --  file name; basename of the recording minus extension (e.g.,
  ``rec1_a``)
- ``Channel ID``  --  channel (1-indexed) that scoring region is on; ignored by
  ``score.py``
- ``Onset``  --  onset of scoring region in seconds from beginning of recording
- ``Offset``  --  offset of scoring region in seconds from beginning of
  recording

For instance:

    CMU_20020319-1400_d01_NONE 1 125.000000 727.090000
    CMU_20020320-1500_d01_NONE 1 111.700000 615.330000
    ICSI_20010208-1430_d05_NONE 1 97.440000 697.290000


## SEGment-wise Long-form Speech Transcription annotation (SegLST)

The SegLST format was used in the [CHiME-7 challenge](https://www.chimechallenge.org/challenges/chime7/task1/index) and is the default format for `MeetEval`.

The SegLST format is stored in JSON format and contains a list of segments.
Each segment should have a minimum set of keys `"session_id"` and `"words"`.
Depending on the metric, additional keys may be required (`"speaker"`, `"start_time"`, `"end_time"`).

An example is shown below:
```python
[
    {
        "session_id": "recordingA", # Required
        "words": "The quick brown fox jumps over the lazy dog", # Required for WER metrics
        "speaker": "Alice", # Required for metrics that use speaker information (cpWER, ORC WER, MIMO WER)
        "start_time": 0,    # Required for time-constrained metrics (tcpWER, tcORC-WER, DER, ...)
        "end_time": 1,      # Required for time-constrained metrics (tcpWER, tcORC-WER, DER, ...)
        "audio_path": "path/to/recordingA.wav"  # Any additional keys can be included
    },
    ...
]
```
Another example can be found [here](example_files/hyp.seglst.json).

## [Segmental Time Mark (STM)](https://github.com/usnistgov/SCTK/blob/master/doc/infmts.htm#L75)
Each line in an `STM` file represents one "utterance" and is defined as

```
STM :== <filename> <channel> <speaker_id> <begin_time> <end_time> <transcript>
```
where
- `filename`: name of the recording
- `channel`: ignored by MeetEval
- `speaker_id`: ID of the speaker or system output stream/channel (not microphone channel)
- `begin_time`: in seconds, used to find the order of the utterances
- `end_time`: in seconds
- `transcript`: space-separated list of words

for example:
```STM
recording1 1 Alice 0 0 Hello Bob.
recording1 1 Bob 1 0 Hello Alice.
recording1 1 Alice 2 0 How are you?
recording2 1 Alice 0 0 Hello Carol.
;; ...
```

An example `STM` file can be found in [here](example_files/ref.stm).

## [Time Marked Conversation (CTM)](https://github.com/usnistgov/SCTK/blob/master/doc/infmts.htm#L286)
The CTM format is defined as

```
CTM :== <filename> <channel> <begin_time> <duration> <word> [<confidence>]
```

for the hypothesis (one file per speaker).
You have to supply one `CTM` file for each system output channel using multiple `-h` arguments since `CTM` files don't encode speaker or system output channel information (the `channel` field has a different meaning: left or right microphone).
For example:

```shell
meeteval-wer orcwer -h hyp1.ctm -h hyp2.ctm -r reference.stm
```

> [!NOTE]
> Note that the `LibriCSS` baseline recipe produces one `CTM` file which merges the speakers, so that it cannot be applied straight away. We recommend to use `STM` or `SegLST` files.
