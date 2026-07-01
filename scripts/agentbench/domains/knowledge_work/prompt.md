You are given an occupational task to complete. Read the task description carefully and produce the required deliverable(s).

## Task

{prompt}

## Reference Files

{reference_section}

## Workspace

Save all output files to: `{workspace_dir}`

Make sure to create the exact deliverable files requested in the task. Use appropriate tools (e.g. Python scripts) to generate files in the required format (Excel, PDF, Word, etc.).

## PDF Reference Files

When reading `.pdf` files listed above, **do not use the `read` tool** — it returns raw PDF binary and wastes context.

Extract text with `exec` instead:

- **Preferred:** `pdftotext "<path>" -` (`/usr/bin/pdftotext` is available)
- **Alternative:** `python3` with `pypdf` (installed on system Python)

Example:

```bash
pdftotext "/path/to/file.pdf" -
```
