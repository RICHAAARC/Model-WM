# Source and interface

No new model is selected. Reuse the completed SD1.4 LaWa call chain:
- LaWa author source commit `251ac11aa223e6ff12553364a65a75add5b1aecf`; author paired checkpoint URL recorded in `lawa_ab/assets.py`.
- CompVis SD1.4 component revision `133a221b8aa7292a167afc5127cb63fb5005638b`.
- Original marked forward, clean VAE, post_quant_conv once, terminal latent division by .18215 once, signed 48-bit payload, six-branch coarse rule, continuous reader unchanged.
- Only the globally uniform gain grid is extended as preregistered; hooks restore after every call/failure.
- Locked Python and package setup, current-runtime environment probe, local asset receipts and actual component/tokenizer parsers precede model loading. CPU synthetic tests do not validate real-weight GPU output.
- No real weights are loaded or downloaded in local engineering checks. Real weights are fetched/reused only when the user runs Colab and stay under /content.

Canonical provenance is the prior completed matched-comparison source audit and pinned dependency files bundled with this entry. No reinterpretation of prior scientific gates is made.
