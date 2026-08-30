# Security Policy

## Supported versions

Until the first stable release, only the latest commit on the default branch is supported with security fixes.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability.

Use GitHub's **Security → Advisories → Report a vulnerability** flow for this repository. Include the affected version or commit, reproduction steps, impact, and any suggested mitigation. If private vulnerability reporting is unavailable, contact the repository owner privately through their GitHub profile and ask for a secure reporting channel without including exploit details in the first message.

Maintainers should acknowledge a complete report within 7 days and provide a status update within 14 days. Timelines may change with severity and maintainer availability. Please allow a reasonable remediation window before public disclosure.

Model behavior, detection quality, and adversarial examples are usually not software vulnerabilities by themselves. Reports involving arbitrary code execution, unsafe deserialization, path traversal, dependency compromise, private data exposure, or untrusted model files are in scope.

Only load model weights from sources you trust. PyTorch `.pt` files may use Python serialization and can be unsafe when obtained from an untrusted party.
