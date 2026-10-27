#!/usr/bin/env python3
"""Check citation and archive metadata (stdlib only).

Validates CITATION.cff against the required Citation File Format 1.2.0
fields, validates .zenodo.json and codemeta.json, and checks that
CITATION.cff, .zenodo.json, codemeta.json and LICENSE agree on title,
authors, keywords and license.

This is a structural check, not the full CFF JSON schema (cffconvert is not
needed). It reads the small YAML subset that CITATION.cff uses here: top-level
scalars, lists of scalars, and lists of flat mappings.

Usage: python tools/check_citation.py [repo_root]   (exit 1 on any problem)
"""

import json
import re
import sys
from pathlib import Path

CFF_REQUIRED = ("cff-version", "message", "title", "authors")
CFF_TYPES = {"software", "dataset"}
RELATIONS = {
    "isCitedBy", "cites", "isSupplementTo", "isSupplementedBy", "isContinuedBy",
    "continues", "isDescribedBy", "describes", "hasMetadata", "isMetadataFor",
    "isNewVersionOf", "isPreviousVersionOf", "isPartOf", "hasPart",
    "isReferencedBy", "references", "isDocumentedBy", "documents",
    "isCompiledBy", "compiles", "isVariantFormOf", "isOriginalFormOf",
    "isIdenticalTo", "isAlternateIdentifier", "isReviewedBy", "reviews",
    "isDerivedFrom", "isSourceOf", "requires", "isRequiredBy",
    "isObsoletedBy", "obsoletes", "isPublishedIn",
}
# LICENSE text marker -> (CFF SPDX id, Zenodo id, codemeta URL)
LICENSES = {
    "MIT License": ("MIT", "mit", "https://spdx.org/licenses/MIT"),
}
PROGRAMME = "https://priyatham9.github.io/grounded"


def _scalar(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def parse_cff(text):
    """Parse the CITATION.cff subset into {key: scalar | [scalar] | [dict]}."""
    data, key, item = {}, None, None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if indent == 0:
            k, _, v = line.partition(":")
            key, item = k.strip(), None
            data[key] = _scalar(v) if v.strip() else []
        elif isinstance(data.get(key), list):
            if line.startswith("- "):
                body = line[2:].strip()
                if ":" in body and not body.startswith(("\"", "'")):
                    k, _, v = body.partition(":")
                    item = {k.strip(): _scalar(v)}
                    data[key].append(item)
                else:
                    item = None
                    data[key].append(_scalar(body))
            elif item is not None and ":" in line and indent > 2:
                k, _, v = line.partition(":")
                item[k.strip()] = _scalar(v)
    return data


def cff_authors(cff):
    out = []
    for a in cff.get("authors", []):
        if isinstance(a, dict) and "family-names" in a:
            out.append(f"{a['family-names']}, {a.get('given-names', '')}".strip(", "))
        elif isinstance(a, dict) and "name" in a:
            out.append(a["name"])
    return out


def check_cff(cff, problems):
    for k in CFF_REQUIRED:
        if not cff.get(k):
            problems.append(f"CITATION.cff: missing required field '{k}'")
    if cff.get("cff-version") != "1.2.0":
        problems.append("CITATION.cff: cff-version must be 1.2.0")
    if cff.get("type", "software") not in CFF_TYPES:
        problems.append("CITATION.cff: type must be software or dataset")
    authors = cff.get("authors", [])
    if not isinstance(authors, list):
        authors = []
    for i, a in enumerate(authors):
        if not isinstance(a, dict) or not (
            "name" in a or ("family-names" in a and "given-names" in a)
        ):
            problems.append(f"CITATION.cff: authors[{i}] needs family-names and given-names, or name")
        elif "orcid" in a and not re.fullmatch(
            r"https://orcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX]", a["orcid"]
        ):
            problems.append(f"CITATION.cff: authors[{i}] orcid is not a valid ORCID URL")
    if "date-released" in cff and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cff["date-released"]):
        problems.append("CITATION.cff: date-released must be YYYY-MM-DD")
    for k in ("repository-code", "url"):
        if k in cff and not cff[k].startswith("https://"):
            problems.append(f"CITATION.cff: {k} must be an https URL")
    if "doi" in cff and not re.fullmatch(r"10\.\d{4,9}/\S+", cff["doi"]):
        problems.append("CITATION.cff: doi is not a valid DOI")


def check_zenodo(z, cff, lic, problems):
    if z.get("upload_type") != "software":
        problems.append(".zenodo.json: upload_type must be 'software'")
    for k in ("title", "description"):
        if not isinstance(z.get(k), str) or not z[k].strip():
            problems.append(f".zenodo.json: missing '{k}'")
    if z.get("title") != cff.get("title"):
        problems.append(".zenodo.json: title differs from CITATION.cff")
    creators = z.get("creators")
    if not isinstance(creators, list) or not creators:
        problems.append(".zenodo.json: creators must be a non-empty list")
        creators = []
    names = [c.get("name") for c in creators if isinstance(c, dict)]
    if names != cff_authors(cff):
        problems.append(f".zenodo.json: creators {names} differ from CITATION.cff authors {cff_authors(cff)}")
    for c in creators:
        if "orcid" in c:
            problems.append(".zenodo.json: orcid must not be set (none supplied by the owner)")
    for banned in ("doi", "prereserve_doi", "conceptdoi"):
        if banned in z:
            problems.append(f".zenodo.json: must not carry '{banned}' (Zenodo assigns it)")
    if lic and z.get("license") != lic[1]:
        problems.append(f".zenodo.json: license {z.get('license')!r} differs from LICENSE ({lic[1]!r})")
    kw = z.get("keywords")
    if not isinstance(kw, list) or not kw or not all(isinstance(x, str) and x for x in kw):
        problems.append(".zenodo.json: keywords must be a non-empty list of strings")
    elif "keywords" in cff and sorted(kw) != sorted(cff["keywords"]):
        problems.append(".zenodo.json: keywords differ from CITATION.cff")
    rel = z.get("related_identifiers")
    if not isinstance(rel, list) or not rel:
        problems.append(".zenodo.json: related_identifiers must be a non-empty list")
        rel = []
    for r in rel:
        if r.get("relation") not in RELATIONS:
            problems.append(f".zenodo.json: unknown relation {r.get('relation')!r}")
        if r.get("scheme") != "url" or not str(r.get("identifier", "")).startswith("https://"):
            problems.append(f".zenodo.json: related identifier must be an https url: {r}")
    ids = [r.get("identifier") for r in rel]
    if len(ids) != len(set(ids)):
        problems.append(".zenodo.json: duplicate related_identifiers")
    if cff.get("repository-code") in ids:
        problems.append(".zenodo.json: related_identifiers must not point at the repository itself")
    if not any(str(i).startswith(PROGRAMME) for i in ids):
        problems.append(".zenodo.json: programme site not in related_identifiers")


def check_codemeta(c, cff, lic, problems):
    if "codemeta" not in str(c.get("@context", "")):
        problems.append("codemeta.json: @context must reference CodeMeta")
    if c.get("@type") != "SoftwareSourceCode":
        problems.append("codemeta.json: @type must be SoftwareSourceCode")
    if c.get("name") != cff.get("title"):
        problems.append("codemeta.json: name differs from CITATION.cff title")
    for k in ("description", "codeRepository"):
        if not c.get(k):
            problems.append(f"codemeta.json: missing '{k}'")
    if c.get("codeRepository") != cff.get("repository-code"):
        problems.append("codemeta.json: codeRepository differs from CITATION.cff repository-code")
    if c.get("version") != cff.get("version"):
        problems.append("codemeta.json: version differs from CITATION.cff")
    names = []
    for a in c.get("author", []):
        names.append(f"{a.get('familyName')}, {a.get('givenName')}")
        if "orcid.org" in str(a.get("@id", "")):
            problems.append("codemeta.json: ORCID must not be set (none supplied by the owner)")
    if names != cff_authors(cff):
        problems.append(f"codemeta.json: author {names} differs from CITATION.cff authors")
    if lic and c.get("license") != lic[2]:
        problems.append(f"codemeta.json: license {c.get('license')!r} differs from LICENSE ({lic[2]!r})")
    if "keywords" in cff and sorted(c.get("keywords", [])) != sorted(cff["keywords"]):
        problems.append("codemeta.json: keywords differ from CITATION.cff")


def check(root):
    root, problems = Path(root), []
    cff_path = root / "CITATION.cff"
    if not cff_path.exists():
        return ["CITATION.cff: missing"]
    cff = parse_cff(cff_path.read_text(encoding="utf-8"))
    check_cff(cff, problems)
    lic = None
    lic_path = root / "LICENSE"
    if lic_path.exists():
        head = lic_path.read_text(encoding="utf-8").lstrip()[:200]
        lic = next((v for k, v in LICENSES.items() if head.startswith(k)), None)
        if lic is None:
            problems.append("LICENSE: unrecognised licence text (extend LICENSES in this tool)")
        elif cff.get("license") != lic[0]:
            problems.append(f"CITATION.cff: license {cff.get('license')!r} differs from LICENSE ({lic[0]!r})")
    else:
        problems.append("LICENSE: missing")
    for name, fn in ((".zenodo.json", check_zenodo), ("codemeta.json", check_codemeta)):
        p = root / name
        if not p.exists():
            problems.append(f"{name}: missing")
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            problems.append(f"{name}: invalid JSON ({e})")
            continue
        fn(doc, cff, lic, problems)
    return problems


def main(argv):
    root = argv[1] if len(argv) > 1 else Path(__file__).resolve().parent.parent
    problems = check(root)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("citation metadata OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
