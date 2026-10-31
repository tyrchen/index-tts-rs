# Upstream model assets

`emo_matrix.safetensors` contains `spk_matrix [73, 192]` and `emo_matrix [73, 1280]`,
converted from `feat1.pt` and `feat2.pt` in
[IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5).
The original model rights belong to bilibili / the IndexTTS authors.

These tensors are embedded in the core library and resulting binaries. They are
covered by the accompanying [bilibili Model Use License Agreement](LICENSE-INDEXTTS),
not the MIT license for this project's Rust code. The same distinction applies to
model-derived golden fixtures in `../fixtures/`. Full inference weights are downloaded
separately from `yunfengwang/IndexTTS-2.5-mlx`.

Any modifications made to the original model in this Derivative Work are not endorsed,
warranted, or guaranteed by the original right-holder of the original model, and the
original right-holder disclaims all liability related to this Derivative Work.

Retain this notice and LICENSE-INDEXTTS when redistributing these assets or binaries
that contain them. See the agreement for the full usage and redistribution terms.
