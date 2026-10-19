# Public release decisions

This file records decisions that must be completed by the project owner before a public
candidate is prepared. It is intentionally not a license substitute and does not authorize any
publication operation.

| Decision           | Current implementation                                                                                             | Owner action                                                                                         |
| ------------------ | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Support            | Windows 11 x64; Python >=3.11,<3.14                                                                                | Confirmed on 2026-08-09 by the 3.11-3.13 matrix                                                      |
| Distribution       | Public source repository / source ZIP; no external binaries                                                        | Confirmed on 2026-08-10                                                                              |
| Names              | repository/distribution/CLI `video-enhance-lite`; package `video_enhance_lite`                                     | Confirmed on 2026-08-09                                                                              |
| venv               | user-local venv bound to an install receipt                                                                        | Confirm the user-local policy                                                                        |
| Stage count        | 11 built-in + optional CodeFormer face restore                                                                     | Confirm the public count wording                                                                     |
| Project license    | MIT; `Copyright (c) 2026 Ryoichi-R and contributors`                                                               | Confirmed on 2026-08-10; holder name updated to Ryoichi-R on 2026-10-18; third-party terms separate  |
| Security contact   | GitHub Private Vulnerability Reporting only                                                                        | Confirmed on 2026-08-10; no public issue or email reporting channel                                  |
| Public destination | `Ryoichi-R/video-enhance-lite`, GitHub repository, public visibility                                               | Confirmed on 2026-08-10                                                                              |
| Dependency lock    | Windows x64 wheel-only exact pins with SHA-256 hashes; Python 3.11 uses NumPy 2.4.6 and Python 3.12/3.13 use 2.5.1 | Confirmed on 2026-08-09; update all hashes together and rerun the three-version clean-install matrix |

Any remaining owner action must be resolved before the corresponding public release gate passes.
