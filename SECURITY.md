# Security policy

## Supported versions

The current development line is `0.2.x`. Support status and the first public release version
are documented in the repository release notes.

## Reporting

Report vulnerabilities only through GitHub Private Vulnerability Reporting:

https://github.com/Ryoichi-R/video-enhance-lite/security/advisories/new

Do not publish sensitive details in an issue, discussion, chat, pull request, or normal support
channel. Maintainers will acknowledge a valid private report as soon as practical, investigate
its impact, and coordinate disclosure through the private advisory. No fixed response-time SLA
is offered.

## Safe reports

Include the smallest reproducible description, affected version, operating system, and whether
an external binary or model was involved. Remove usernames, input video paths, tokens, API keys,
private URLs, and full environment dumps from logs before sharing them.

The application does not intentionally download external engines or send input media to a
remote service. If a suspected incident involves a credential, revoke it from a clean device and
report the scope separately.
