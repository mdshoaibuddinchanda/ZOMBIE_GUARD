# Security policy

ZombieGuard processes attacker-controlled archive bytes. Parser bugs, denial-of-service
conditions, unsafe defaults, and false-clean outcomes are security-sensitive.

## Supported versions

| Version | Security fixes |
|---|---|
| Latest `main` branch | Yes |
| Older commits and unpublished builds | No |

Until the project publishes a stable release, security fixes are made on `main` only.

## Reporting a vulnerability

Please use GitHub's **Security → Report a vulnerability** flow for this repository. If
private vulnerability reporting is unavailable, open a public issue containing only a
request for a private contact channel.

Include, when safe:

- the affected commit and Python version;
- the smallest non-sensitive reproducer or a generator for it;
- observed and expected behavior;
- security impact and the configured input-size limit;
- a SHA-256 digest instead of a live malware attachment.

Do not publish working bypass details, credentials, personal data, or malware samples in a
public issue. The maintainer will acknowledge a complete report, reproduce it, and
coordinate disclosure where practical. No bounty is currently offered.

## Safe operation

- Scan untrusted archives in a sandbox or restricted service account.
- Keep the scanner's input-size limit enabled. Add OS/container memory and time limits when
  operating it as a service.
- Do not extract or execute payloads merely because ZombieGuard reports a clean result.
- Treat malformed, unsupported, encrypted, and size-limited inputs as unresolved—not clean.
- Keep quarantined samples outside the repository and record only hashes and non-sensitive
  metadata.
- Supply service credentials through environment variables or a secret manager. Never put
  them in source, fixtures, logs, or generated reports.

ZombieGuard is a defense-in-depth signal, not a general malware detector or a security
guarantee.
