FinGuard documentation has moved to [README.md](README.md), which contains the current setup, implementation overview, limitations, and security guidance.

**Security notice:** the application source currently contains hard-coded SMTP credentials. Treat them as exposed: revoke/rotate them immediately, move replacement credentials into environment variables or a secret manager, and never commit secrets again.

Do not rely on the performance claims in this legacy file until they are reproducible from documented training and evaluation scripts.