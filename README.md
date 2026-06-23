FSDFuse
Official repository for FSDFuse: Physics-Informed Frequency Decoupling and Data-Adaptive Spatial Decoupling for SAR-Visible Image Fusion.
FSDFuse is a SAR-visible image fusion framework built around a physical-data joint-driven decoupling idea. Instead of learning all decomposition boundaries only from data, FSDFuse assigns different roles to different domains: frequency-domain decoupling is guided by SAR imaging physics, while spatial-domain decoupling remains data-adaptive to capture scene-specific structures.
News
This repository currently releases the core design code of:
FrequencyDecouple
Spatial_decouple
The complete training and testing code, configuration files, pretrained models, and reproduction scripts will be released after the paper is accepted.
Method Overview
The central motivation of FSDFuse is that SAR and visible images differ not only in appearance, but also in their physical imaging mechanisms. SAR images contain coherent speckle and backscatter responses, while visible images provide rich reflectance and texture information. A purely data-driven decomposition may mistakenly treat SAR speckle as transferable detail, whereas a fixed handcrafted frequency split cannot adapt to different scenes.
FSDFuse addresses this problem with a dual-domain decoupling strategy.
FrequencyDecouple
FrequencyDecouple corresponds to the physics-informed frequency decoupling design. It is motivated by the spectral behavior of coherent SAR imaging, where speckle and structural backscatter can overlap in the frequency domain.
Rather than using a fixed low/high-frequency split, the module learns input-adaptive Fourier masks. The low-frequency and high-frequency masks are non-complementary, leaving a learnable uncertainty region between confident spectral assignments. This design allows the network to process more reliable sub-bands separately and reduces the risk of forcing ambiguous SAR spectral content into an incorrect branch.
In the full FSDFuse framework, this frequency pathway is used to separate global energy distributions and provide physically supported representations for fusion.
Spatial_decouple
Spatial_decouple corresponds to the data-adaptive spatial decoupling design. Since local texture, object layout, directional context, and geometric distortions vary strongly across scenes, FSDFuse does not impose a fixed spatial prior.
The spatial branch combines local convolutional modeling with long-range state-space modeling. A CNN pathway captures fine textures and local structures, while a Mamba-style pathway models broader spatial topology. Directional gating is used to suppress unreliable scan directions, so the model can retain useful contextual information while reducing the propagation of noisy or artifact-corrupted responses.
In the full FSDFuse framework, this spatial pathway complements the frequency pathway by learning scene-dependent structural information that cannot be specified by SAR imaging physics alone.
Design Principle
FSDFuse follows a simple principle:
Use physical priors where they are reliable, and use data-adaptive learning where the scene structure is too variable to be predefined.

Under this principle, frequency-domain decoupling provides interpretability and physical support, while spatial-domain decoupling provides flexibility and adaptability. The two branches are designed to be complementary rather than redundant.
Code Availability
At this stage, the repository is intended to share the core architectural ideas behind the method. It is not yet a complete reproduction package.
Currently available:
Core frequency decoupling code
Core spatial decoupling code
Key module design for understanding FSDFuse
Coming after paper acceptance:
Full model implementation
Training and inference scripts
Configuration files
Evaluation protocol
Pretrained weights, if permitted
Additional documentation for reproducing the reported experiments
Citation
If you find this project helpful, please consider citing our paper after it is accepted. The BibTeX entry will be updated here once the final publication information is available.
Contact
For questions about the method or code release, please open an issue in this repository.
