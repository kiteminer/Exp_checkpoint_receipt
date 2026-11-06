# Batch Footprint

Did every training sample actually reach the optimizer before and after a restart? Compare a restarted run against an uninterrupted one, sample by sample.

Run in the same folder with Python 3.8+ (on Windows use `py` in place of `python3`):

```sh
python3 demo.py start
python3 demo.py resume
```

The first process trains a one-weight regression model on samples 1–8 and saves a JSON checkpoint. It has already fetched samples 9–12 into memory. The second process independently restores the same checkpoint twice. Resuming from the fetched cursor skips 9–12. Resuming from the consumed cursor reproduces the uninterrupted receipt and final weight exactly.

The checkpoint is preserved for replay. For a fresh start, pass the same new `--file run2.json` to both commands. This intentionally small experiment models prefetch at a clean checkpoint boundary on one CPU. It does not simulate a crash during a write or a distributed loader.
