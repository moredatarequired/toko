# Security Policy

Report a suspected vulnerability privately, by email, to hugh.wimberly@gmail.com. Do
not open a public issue for it; an issue is world-readable the moment it is filed.

GitHub's private advisory reporting is the preferred channel once it is enabled on
this repository. It is not enabled today, so email is the way in.

Fixes ship in the next release cut from `main`. Toko is pre-1.0 and has no
maintenance branches, so there is no older line to backport to: the current release
on PyPI is the one that gets patched.

A useful report says what the attacker controls, what they end up with, and the
shortest command or input that demonstrates it.

For scope, toko reads local files and directories, fetches URLs you hand it, and
sends what it reads to provider APIs using a key from the environment or
`~/.config/toko/config.toml`. The interesting classes are therefore credential
handling — a key reaching stdout, an error message, a cache file, or a host it was
not meant for — and path and URL handling, where toko reads or fetches something the
invoking command did not intend.
