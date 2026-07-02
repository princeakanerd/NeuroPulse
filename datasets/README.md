# Local dataset layout

- `WESAD/S*.pkl`: original WESAD subject recordings, used by the corrected scripts. Preserved locally and excluded from ordinary Git tracking because individual files exceed 100 MiB.
- `utd/`: local UTD-MHAD recordings. Configure the notebook's dataset path for your environment.
- `muse_v3.csv`: MuSe song descriptors used by the corrected notebook.
- `Behavioral-features - per minute.xlsx`: supplied SWELL-related behavioral features. This is not a substitute for the HRV train.csv expected by the original SWELL experiment notebook.

No raw datasets were altered during repository cleanup. Review the providers' redistribution terms before publishing third-party data. If you intend to put WESAD on GitHub, configure Git LFS explicitly; do not remove its ignore rule and try an ordinary Git push.
