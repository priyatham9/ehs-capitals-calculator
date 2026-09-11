# Security Policy

## Scope

The calculator performs arithmetic operations on user-supplied inputs and returns cost-benefit analysis results. It stores no data beyond a single request.

The security surface consists of:

1. **Input validation** - numeric ranges and type checks
2. **Numerical computation** - arithmetic correctness and overflow handling
3. **Output formatting** - no injection vectors in JSON or text output

The calculator does not:
- Accept external data sources or network input
- Persist state or user data
- Execute user-supplied code
- Authenticate users or manage secrets
- Interact with external services

## Reporting a Vulnerability

If you discover a security issue, please report it privately to the repository owner. Do not open a public issue.

Steps:

1. Check if the issue is already known by searching closed issues
2. Contact the maintainer via GitHub's security advisory feature
3. Provide a clear description of the vulnerability and reproduction steps
4. Allow reasonable time for a response and fix

We will aim to acknowledge reports within 14 days and publish fixes within reasonable timeframes based on severity.

## Assumptions

- Python's standard library numeric operations are secure
- NumPy array operations are used as documented
- Users provide well-formed input and understand model limitations

The model's arithmetic is intentionally simple and fully transparent to support verification and auditability.
