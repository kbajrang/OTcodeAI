from __future__ import annotations

import ast
import difflib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import requests

from src.config.settings import settings
from src.embeddings.bge import BGEEmbedder
from src.graph.store import load_graph
from src.utils.logging import get_logger
from src.utils.paths import PROJECT_ROOT
from src.vector_store.faiss_store import FaissStore
from src.vector_store.metadata import load_metadata

logger = get_logger(__name__)


class GraphRAGPipeline:
    def __init__(self) -> None:
        self._embedder: BGEEmbedder | None = None
        self.graph = None
        self.vector_store = None
        self.metadata: list[dict[str, Any]] = []
        self._graph_mtime: float | None = None
        self._vector_mtime: float | None = None
        self._metadata_mtime: float | None = None
        self._stem_to_indices: dict[str, list[int]] = {}
        self._known_stems: list[str] = []
        self._path_lowers: list[str] = []
        self._load_indexes()

    def _get_embedder(self) -> BGEEmbedder:
        if self._embedder is None:
            logger.info("Loading embeddings model: %s", settings.embedding_model)
            self._embedder = BGEEmbedder()
        return self._embedder

    def _load_indexes(self) -> None:
        try:
            self.graph = load_graph()
            self._graph_mtime = os.path.getmtime(settings.graph_path)
        except FileNotFoundError:
            logger.warning("Graph file not found. Run the indexer first.")
            self.graph = None
            self._graph_mtime = None
        try:
            self.metadata = load_metadata()
            self.vector_store = FaissStore()
            self.vector_store.load()
            self._metadata_mtime = os.path.getmtime(settings.vector_metadata_path)
            self._vector_mtime = os.path.getmtime(settings.vector_db_path)
            self._build_retrieval_indexes()
        except FileNotFoundError:
            logger.warning("Vector index not found. Run the indexer first.")
            self.vector_store = None
            self._metadata_mtime = None
            self._vector_mtime = None
            self._stem_to_indices = {}
            self._known_stems = []
            self._path_lowers = []

    def _build_retrieval_indexes(self) -> None:
        self._stem_to_indices = {}
        self._path_lowers = [""] * len(self.metadata)

        for idx, item in enumerate(self.metadata):
            file_path = (item.get("file_path") or "").replace("\\", "/")
            file_path_lower = file_path.lower()
            self._path_lowers[idx] = file_path_lower

            if not file_path:
                continue

            file_name = PurePosixPath(file_path).name
            stem = file_name.rsplit(".", 1)[0].strip()
            if not stem:
                continue
            stem_lower = stem.lower()
            self._stem_to_indices.setdefault(stem_lower, []).append(idx)

        for stem_lower, indices in self._stem_to_indices.items():
            indices.sort(key=lambda i: (self.metadata[i].get("start_line") or 0))

        self._known_stems = sorted(self._stem_to_indices.keys())

    def _reload_if_stale(self) -> None:
        try:
            graph_mtime = os.path.getmtime(settings.graph_path)
            metadata_mtime = os.path.getmtime(settings.vector_metadata_path)
            vector_mtime = os.path.getmtime(settings.vector_db_path)
        except OSError:
            return
        if (
            self._graph_mtime != graph_mtime
            or self._metadata_mtime != metadata_mtime
            or self._vector_mtime != vector_mtime
        ):
            logger.info("Detected updated index files. Reloading indexes.")
            self._load_indexes()

    def _max_prompt_chars(self) -> int:
        available_tokens = max(256, settings.llm_num_ctx - settings.prompt_reserved_tokens)
        return int(available_tokens * settings.approx_chars_per_token)

    def _llm_timeout(self) -> float | None:
        if self._is_local_llm_base():
            return None
        timeout_s = float(settings.llama_timeout or 0)
        return None if timeout_s <= 0 else timeout_s

    def _is_local_llm_base(self) -> bool:
        base = (settings.llama_api_base or "").strip()
        if not base:
            return False
        candidate = base
        if "://" not in candidate:
            candidate = f"http://{candidate}"
        try:
            parsed = urlparse(candidate)
        except Exception:
            return False
        host = (parsed.hostname or "").strip().lower()
        return host in {"localhost", "127.0.0.1", "::1"}

    def _llm_health_check(self) -> None:
        """Fast preflight to avoid long hangs when the LLM endpoint is down."""
        base = (settings.llama_api_base or "").strip().rstrip("/")
        if not base:
            raise ValueError("LLAMA_API_BASE is not set (or empty).")

        provider = (settings.llm_provider or "").strip().lower()
        timeout_setting = float(settings.llama_timeout or 0)
        timeout = 5.0 if timeout_setting <= 0 else min(5.0, timeout_setting)

        # Choose a lightweight endpoint per provider.
        if provider == "ollama":
            url = f"{base}/api/tags"
            headers = {"Accept": "application/json"}
        else:
            # OpenAI-compatible: models listing.
            url = f"{base}/models" if not base.endswith("/models") else base
            headers = {"Accept": "application/json"}

        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code >= 400:
                raise ValueError(f"LLM preflight failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as exc:
            raise ValueError(f"LLM endpoint not reachable at {url}: {exc}") from exc

    def _extract_query_terms(self, question: str) -> list[str]:
        raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question)
        stop = {
            "what",
            "does",
            "do",
            "is",
            "are",
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "with",
            "for",
            "from",
            "about",
            "explain",
            "describe",
            "how",
            "why",
            "where",
            "when",
            "file",
            "module",
            "class",
            "function",
            "method",
            "code",
        }
        terms: list[str] = []
        seen: set[str] = set()
        for tok in raw_tokens:
            low = tok.lower()
            if low in stop:
                continue
            if len(tok) <= 3 and low.isalpha():
                continue
            if low in seen:
                continue
            seen.add(low)
            terms.append(tok)

        terms.sort(
            key=lambda t: (
                any(c.isupper() for c in t),
                len(t),
            ),
            reverse=True,
        )
        return terms[:12]

    def _approximate_stem_matches(self, term: str) -> list[str]:
        if not term or not self._known_stems:
            return []
        term_lower = term.lower()
        if term_lower in self._stem_to_indices:
            return [term_lower]
        if term_lower.endswith("contoller"):
            fixed = term_lower.replace("contoller", "controller")
            if fixed in self._stem_to_indices:
                return [fixed]
        return difflib.get_close_matches(term_lower, self._known_stems, n=3, cutoff=0.82)

    def _path_bias(self, file_path_lower: str, *, question: str) -> float:
        bias = 0.0
        if "/src/main/" in file_path_lower or "\\src\\main\\" in file_path_lower:
            bias += 0.05
        if "/src/test/" in file_path_lower or "\\src\\test\\" in file_path_lower:
            if not re.search(r"\b(test|tests|ut|unit|integration|ct)\b", question.lower()):
                bias -= 0.15
        return bias

    def _lexical_bonus(self, item: dict[str, Any], terms: list[str], *, question: str) -> float:
        if not terms:
            return 0.0
        file_path = (item.get("file_path") or "").replace("\\", "/")
        file_path_lower = file_path.lower()
        file_name_lower = PurePosixPath(file_path).name.lower()
        code = item.get("code") or ""
        question_lower = question.lower()

        bonus = 0.0
        for term in terms:
            term_lower = term.lower()
            if not term_lower:
                continue

            if term_lower in file_path_lower:
                bonus += 0.35
            if term_lower in file_name_lower:
                bonus += 0.45

            if term_lower.endswith("controller") and (
                "/controller/" in file_path_lower
                or "/controllers/" in file_path_lower
                or "\\controller\\" in file_path_lower
                or "\\controllers\\" in file_path_lower
            ):
                bonus += 0.25

            # Prefer non-test definitions when asking about a symbol.
            if "test" in file_name_lower and "test" not in term_lower and "test" not in question_lower:
                bonus -= 0.25

            # Big boost if this chunk contains a definition for the symbol.
            if re.search(
                rf"\b(class|interface|enum)\s+{re.escape(term)}\b",
                code,
                flags=re.IGNORECASE,
            ):
                bonus += 1.0
            elif re.search(rf"\b{re.escape(term)}\b", code, flags=re.IGNORECASE):
                bonus += 0.12

        bonus += self._path_bias(file_path_lower, question=question)
        return bonus

    def _hybrid_search(self, question: str, *, k: int) -> tuple[list[float], list[int], dict[str, Any]]:
        if not self._path_lowers or len(self._path_lowers) != len(self.metadata):
            self._build_retrieval_indexes()

        query_vector = self._get_embedder().encode([question])[0]
        candidate_k = min(max(k * 10, 60), max(1, len(self.metadata)))
        vector_scores, vector_indices = self.vector_store.search(query_vector, k=candidate_k)
        vec_by_idx = {
            idx: float(score)
            for score, idx in zip(vector_scores, vector_indices, strict=False)
            if idx >= 0
        }

        terms = self._extract_query_terms(question)
        matched_stems: list[str] = []
        for term in terms:
            matched_stems.extend(self._approximate_stem_matches(term))
        matched_stems = list(dict.fromkeys(matched_stems))

        score_terms_lower = list(dict.fromkeys([t.lower() for t in terms] + matched_stems))

        lexical_candidates: set[int] = set()
        for stem_lower in matched_stems:
            for idx in self._stem_to_indices.get(stem_lower, [])[:6]:
                lexical_candidates.add(idx)

        for term_lower in score_terms_lower[:6]:
            if len(term_lower) < 4:
                continue
            for idx, path_lower in enumerate(self._path_lowers):
                if term_lower in path_lower:
                    lexical_candidates.add(idx)

        candidate_indices = {idx for idx in vector_indices if idx >= 0} | lexical_candidates

        scored: list[tuple[float, int]] = []
        for idx in candidate_indices:
            item = self.metadata[idx]
            vec = vec_by_idx.get(idx, 0.0)
            lex = self._lexical_bonus(item, score_terms_lower, question=question)
            scored.append((vec + lex, idx))

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:k]

        meta: dict[str, Any] = {
            "retrieval": "hybrid(vector+lexical)",
            "query_terms": terms,
            "matched_file_stems": matched_stems[:10],
            "vector_candidates": len([i for i in vector_indices if i >= 0]),
            "lexical_candidates": len(lexical_candidates),
            "combined_candidates": len(candidate_indices),
            "candidate_k": candidate_k,
        }
        return [s for s, _ in top], [i for _, i in top], meta

    def _try_read_file_text(self, *, module: str | None, file_path: str | None) -> str | None:
        if not file_path:
            return None

        normalized = file_path.replace("\\", "/")
        rel = normalized
        if module and normalized.startswith(f"{module}/"):
            rel = normalized[len(module) + 1 :]

        # Prefer reading from the imported workspace modules (when available).
        modules_root = PROJECT_ROOT / settings.modules_dir
        if module:
            base = modules_root / module
        else:
            base = PROJECT_ROOT

        candidate = (base / Path(rel)).resolve()
        try:
            # Basic safety: avoid path traversal out of the workspace root.
            if module and not candidate.is_relative_to(base.resolve()):
                return None
        except OSError:
            return None

        if not candidate.exists() or not candidate.is_file():
            return None
        try:
            return candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return candidate.read_text(encoding="latin-1")

    def _extract_java_interface_routes(
        self, *, interface_name: str, module_hint: str | None
    ) -> tuple[str | None, list[dict[str, str | None]]]:
        idxs = self._stem_to_indices.get(interface_name.lower()) or []
        if not idxs:
            return None, []

        chosen_item: dict[str, Any] | None = None
        for idx in idxs:
            item = self.metadata[idx]
            file_path = (item.get("file_path") or "").replace("\\", "/").lower()
            if "/src/test/" in file_path:
                continue
            chosen_item = item
            break
        if chosen_item is None:
            chosen_item = self.metadata[idxs[0]]

        interface_file_path = chosen_item.get("file_path")
        interface_module = chosen_item.get("module") or module_hint
        code = self._try_read_file_text(module=interface_module, file_path=interface_file_path) or (
            chosen_item.get("code") or ""
        )
        if not code:
            return interface_file_path, []

        routes: list[dict[str, str | None]] = []
        lines = code.splitlines()
        i = 0
        while i < len(lines):
            if "@RequestMapping" not in lines[i]:
                i += 1
                continue
            block_lines = [lines[i]]
            j = i + 1
            while j < len(lines) and ")" not in lines[j]:
                block_lines.append(lines[j])
                j += 1
            if j < len(lines):
                block_lines.append(lines[j])
            block = "\n".join(block_lines)

            path_match = re.search(r'value\s*=\s*"(?P<path>[^"]+)"', block)
            method_match = re.search(r"method\s*=\s*RequestMethod\.(?P<method>[A-Z]+)", block)
            path = path_match.group("path") if path_match else None
            http_method = method_match.group("method") if method_match else None

            handler: str | None = None
            k = j + 1
            while k < len(lines):
                if "ResponseEntity" in lines[k] and "(" in lines[k]:
                    name_match = re.search(r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(", lines[k])
                    if name_match:
                        handler = name_match.group("name")
                    break
                if lines[k].strip().startswith("@"):
                    k += 1
                    continue
                if lines[k].strip():
                    break
                k += 1

            if path and http_method:
                routes.append({"method": http_method, "path": path, "handler": handler})

            i = j + 1

        return interface_file_path, routes

    def _extract_java_file_facts(
        self, *, code: str, file_path: str, module: str | None, question: str
    ) -> dict[str, Any]:
        facts: dict[str, Any] = {"file_path": file_path, "module": module}

        class_match = re.search(
            r"(?m)^\s*public\s+class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:implements\s+(?P<impl>[^{]+))?",
            code,
        )
        class_name = class_match.group("name") if class_match else None
        implements_raw = (class_match.group("impl") if class_match else None) or ""

        implements: list[str] = []
        for part in implements_raw.split(","):
            tokens = part.strip().split()
            if not tokens:
                continue
            name = tokens[0].strip()
            if not name:
                continue
            name = re.sub(r"<.*?>", "", name).strip()
            if name:
                implements.append(name)

        if class_name:
            facts["class"] = class_name
        if implements:
            facts["implements"] = implements

        annotations: list[str] = []
        for ann in (
            "RestController",
            "Controller",
            "RequestMapping",
            "GetMapping",
            "PostMapping",
            "PutMapping",
            "DeleteMapping",
            "PatchMapping",
            "Authorized",
        ):
            if f"@{ann}" in code:
                annotations.append(f"@{ann}")
        if annotations:
            facts["annotations"] = annotations

        allowed_params: dict[str, list[str]] = {}
        for match in re.finditer(
            r"ALLOWED_PARAMS_(?P<name>[A-Za-z0-9_]+)\s*=\s*Set\.of\((?P<items>[^;]+)\);",
            code,
        ):
            name = match.group("name")
            items = re.findall(r"\"([^\"]+)\"", match.group("items"))
            if items:
                allowed_params[name.lower()] = items
        if allowed_params:
            facts["allowed_params"] = allowed_params

        # Autowired fields.
        fields: list[dict[str, str]] = []
        autowired_next = False
        for line in code.splitlines():
            if "@Autowired" in line:
                autowired_next = True
                continue
            if not autowired_next:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("@"):
                continue
            field_match = re.match(
                r"(?:public|private|protected)?\s*(?P<type>[A-Za-z_][A-Za-z0-9_<>\[\]]*)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;",
                stripped,
            )
            if field_match:
                fields.append({"name": field_match.group("name"), "type": field_match.group("type")})
            autowired_next = False
        if fields:
            facts["autowired"] = fields

        # Methods + call highlights (lightweight, best-effort).
        method_sig_re = re.compile(
            r"(?m)^\s*(?:@\w+(?:\([^)]*\))?\s*)*(?:public|protected|private)\s+[A-Za-z0-9_<>\[\],\s\?]+\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
        )
        call_re = re.compile(r"\b(?P<obj>[A-Za-z_][A-Za-z0-9_]*)\.(?P<method>[A-Za-z_][A-Za-z0-9_]*)\s*\(")

        method_facts: list[dict[str, Any]] = []
        for match in method_sig_re.finditer(code):
            name = match.group("name")
            # Skip constructors.
            if class_name and name == class_name:
                continue
            brace_start = code.find("{", match.end())
            if brace_start == -1:
                continue
            depth = 0
            end = None
            for pos in range(brace_start, len(code)):
                ch = code[pos]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = pos + 1
                        break
            if end is None:
                continue
            body = code[brace_start:end]

            calls: dict[str, set[str]] = {}
            for call in call_re.finditer(body):
                obj = call.group("obj")
                method_name = call.group("method")
                if obj == "logger":
                    continue
                calls.setdefault(obj, set()).add(method_name)

            pre = code[max(0, match.start() - 300) : match.start()]
            method_facts.append(
                {
                    "name": name,
                    "authorized": "@Authorized" in pre,
                    "calls": {k: sorted(v) for k, v in sorted(calls.items(), key=lambda t: t[0].lower())},
                    "hints": {
                        "offset_based": "isOffsetBasedRequest" in body,
                        "expanded_view": "isExpanded" in body,
                        "triggers_rights": "triggerRightsComputation" in body,
                        "validates_query": "validateQueryParams" in body,
                        "validates_filter": "validateDeviceFilter" in body,
                        "validates_response": "validateDeviceResponse" in body,
                        "builds_spec": "DeviceSpecification" in body or "Specification.where" in body,
                    },
                }
            )

        if method_facts:
            facts["methods"] = method_facts

        # If the class implements an API interface, extract route mappings from it.
        if implements:
            api_routes: list[dict[str, Any]] = []
            related_files: list[str] = []
            for iface in implements[:3]:
                iface_path, routes = self._extract_java_interface_routes(
                    interface_name=iface, module_hint=module
                )
                if iface_path:
                    related_files.append(iface_path)
                for route in routes:
                    api_routes.append({"interface": iface, **route})
            if api_routes:
                facts["api_routes"] = api_routes
            if related_files:
                facts["related_files"] = related_files

        return facts

    def _format_extracted_facts(self, facts: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.append("EXTRACTED SUMMARY (auto)")
        file_path = facts.get("file_path")
        if file_path:
            lines.append(f"Focus file: {file_path}")

        class_name = facts.get("class")
        implements = facts.get("implements") or []
        if class_name:
            impl = f" implements {', '.join(implements)}" if implements else ""
            lines.append(f"Class: {class_name}{impl}")

        routes = facts.get("api_routes") or []
        if routes:
            lines.append("Routes (from interface annotations):")
            for route in routes[:12]:
                handler = route.get("handler")
                handler_suffix = f" -> {handler}()" if handler else ""
                lines.append(
                    f"- {route.get('method')} {route.get('path')}{handler_suffix} (via {route.get('interface')})"
                )

        autowired = facts.get("autowired") or []
        if autowired:
            deps = ", ".join([f"{f['name']}:{f['type']}" for f in autowired[:12]])
            lines.append(f"Injected deps: {deps}")

        allowed = facts.get("allowed_params") or {}
        if allowed:
            for key, items in allowed.items():
                lines.append(f"Allowed query params ({key}): {', '.join(items)}")

        methods = facts.get("methods") or []
        if methods:
            lines.append("Method call highlights (best-effort):")
            for m in methods[:10]:
                name = m.get("name")
                flags = []
                if m.get("authorized"):
                    flags.append("@Authorized")
                hints = m.get("hints") or {}
                if hints.get("triggers_rights"):
                    flags.append("triggers_rights")
                if hints.get("builds_spec"):
                    flags.append("builds_spec")
                if hints.get("expanded_view"):
                    flags.append("expanded_view")
                if hints.get("offset_based"):
                    flags.append("offset_based")
                flag_text = f" [{' '.join(flags)}]" if flags else ""

                # Show key dependency calls first.
                calls = m.get("calls") or {}
                call_parts: list[str] = []
                for obj, methods_called in calls.items():
                    if obj.lower() in {"logger"}:
                        continue
                    if len(call_parts) >= 6:
                        break
                    call_parts.append(f"{obj}.{','.join(methods_called[:5])}")
                call_text = "; ".join(call_parts)
                lines.append(f"- {name}(){flag_text}: {call_text}")

        related_files = facts.get("related_files") or []
        if related_files:
            lines.append("Related files:")
            for fp in related_files[:8]:
                lines.append(f"- {fp}")

        return "\n".join(lines).strip() + "\n"

    def _build_context_preface(
        self, *, question: str, indices: list[int], retrieval_meta: dict[str, Any]
    ) -> str:
        if not indices:
            return ""
        if not self.metadata:
            return ""
        idx0 = indices[0]
        if idx0 < 0 or idx0 >= len(self.metadata):
            return ""
        focus = self.metadata[idx0]
        module = focus.get("module")
        file_path = focus.get("file_path")
        if not file_path or not isinstance(file_path, str):
            return ""
        if not file_path.lower().endswith(".java"):
            return ""

        code = self._try_read_file_text(module=module, file_path=file_path)
        if not code:
            # Fall back to the retrieved chunk itself.
            code = focus.get("code") or ""
        if not code:
            return ""

        facts = self._extract_java_file_facts(code=code, file_path=file_path, module=module, question=question)
        if not facts:
            return ""

        preface = self._format_extracted_facts(facts)
        return preface

    def _prompt_without_context(self, question: str) -> str:
        return (
            "You are a code intelligence assistant for a large codebase.\n"
            "Use ONLY the provided context.\n\n"
            "Requirements:\n"
            "- Be concrete and specific (name the exact classes/methods you see).\n"
            "- Prefer production code (src/main) over tests (src/test), unless the user asks about tests.\n"
            "- Do NOT invent routes/behaviors not present in the context.\n"
            "- When stating behavior, cite the snippet using its 'File:' and 'Lines:' from the context headers.\n\n"
            "Answer format:\n"
            "1) One-sentence summary\n"
            "2) Key responsibilities (bullets)\n"
            "3) Key methods + call flow (bullets)\n"
            "4) Related files to open next (bullets)\n\n"
            "Context:\n\n"
            f"Question: {question}\nAnswer:"
        )

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "You are a code intelligence assistant for a large codebase.\n"
            "Use ONLY the provided context.\n\n"
            "Requirements:\n"
            "- Be concrete and specific (name the exact classes/methods you see).\n"
            "- Prefer production code (src/main) over tests (src/test), unless the user asks about tests.\n"
            "- Do NOT invent routes/behaviors not present in the context.\n"
            "- When stating behavior, cite the snippet using its 'File:' and 'Lines:' from the context headers.\n\n"
            "Answer format:\n"
            "1) One-sentence summary\n"
            "2) Key responsibilities (bullets)\n"
            "3) Key methods + call flow (bullets)\n"
            "4) Related files to open next (bullets)\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

    def _format_context_item(
        self, *, item: dict[str, Any], score: float | None = None, include_code: bool = True
    ) -> str:
        file_path = item.get("file_path")
        module = item.get("module")
        start_line = item.get("start_line")
        end_line = item.get("end_line")
        name = item.get("name")
        typ = item.get("type")
        item_id = item.get("id")

        header_parts = [
            f"ID: {item_id}",
            f"Type: {typ}",
            f"Name: {name}" if name else None,
            f"Module: {module}" if module else None,
            f"File: {file_path}" if file_path else None,
            f"Lines: {start_line}-{end_line}" if start_line and end_line else None,
            f"Score: {score:.4f}" if score is not None else None,
        ]
        header = "\n".join([p for p in header_parts if p])
        code = item.get("code") or ""
        if not include_code:
            return header
        return f"{header}\n```text\n{code}\n```"

    def _truncate_snippet(self, snippet: str, *, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(snippet) <= max_chars:
            return snippet
        # Keep the header + beginning of the code; the model tends to benefit most
        # from imports/signatures at the top of a chunk.
        return snippet[: max_chars - 20] + "\n...<truncated>...\n"

    def _collect_graph_expansions(self, seed_id: str) -> list[dict[str, Any]]:
        if not self.graph or seed_id not in self.graph:
            return []
        expansions: list[dict[str, Any]] = []
        seen: set[str] = set()

        def maybe_add(node_id: str) -> None:
            if node_id in seen:
                return
            if node_id not in self.graph:
                return
            node_data = dict(self.graph.nodes[node_id])
            node_data.setdefault("id", node_id)
            code = node_data.get("code")
            if not code:
                return
            # Never include full files by default; chunks already cover the content.
            if node_data.get("type") == "file":
                return
            expansions.append(node_data)
            seen.add(node_id)

        # Prefer structural neighbors first (contains edges).
        for predecessor in self.graph.predecessors(seed_id):
            if predecessor not in self.graph:
                continue
            if self.graph.nodes[predecessor].get("type") == "file":
                # For chunk seeds, a file predecessor is expected. Include lightweight
                # structure from that file (function/class signatures) and the first
                # chunk of a few imported files (helps follow interface-based routes).
                added = 0
                imported = 0
                for successor in self.graph.successors(predecessor):
                    if successor not in self.graph:
                        continue
                    successor_type = self.graph.nodes[successor].get("type")
                    if successor_type == "file":
                        successor_path = (self.graph.nodes[successor].get("file_path") or "").replace(
                            "\\", "/"
                        )
                        successor_path_lower = successor_path.lower()
                        if "/src/test/" in successor_path_lower:
                            continue
                        maybe_add(f"{successor}::chunk::0")
                        imported += 1
                        if imported >= 6:
                            continue
                    if successor_type in {"function", "class"}:
                        maybe_add(successor)
                        added += 1
                        if added >= 12:
                            break
                continue
            maybe_add(predecessor)
        for successor in self.graph.successors(seed_id):
            maybe_add(successor)

        # Add sibling chunks (previous/next) if present.
        if "::chunk::" in seed_id:
            prefix, _, idx_str = seed_id.rpartition("::chunk::")
            try:
                idx = int(idx_str)
            except ValueError:
                idx = None
            if idx is not None:
                maybe_add(f"{prefix}::chunk::{idx - 1}")
                maybe_add(f"{prefix}::chunk::{idx + 1}")

        return expansions

    def _build_context(
        self,
        scores: list[float],
        top_indices: list[int],
        *,
        question: str,
        max_chars: int,
        preface: str = "",
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        base_prompt_len = len(self._prompt_without_context(question))
        context_budget = max(0, max_chars - base_prompt_len)

        included_ids: set[str] = set()
        retrieved_items: list[dict[str, Any]] = []
        parts: list[str] = []
        used_chars = 0
        preface_chars = 0

        if preface:
            preface_text = preface.strip()
            if preface_text:
                if len(preface_text) > context_budget:
                    preface_text = self._truncate_snippet(preface_text, max_chars=context_budget)
                if preface_text:
                    parts.append(preface_text)
                    used_chars += len(preface_text) + 5
                    preface_chars = len(preface_text)

        # Primary: vector hits in score order.
        for score, idx in zip(scores, top_indices, strict=False):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            item_id = item.get("id")
            if not item_id or item_id in included_ids:
                continue
            snippet = self._format_context_item(item=item, score=score)
            if used_chars + len(snippet) > context_budget:
                remaining = context_budget - used_chars
                if remaining <= 0:
                    break
                snippet = self._truncate_snippet(snippet, max_chars=remaining)
                if not snippet:
                    break
            parts.append(snippet)
            used_chars += len(snippet) + 5
            included_ids.add(item_id)
            retrieved_items.append(
                {
                    "id": item_id,
                    "score": float(score),
                    "type": item.get("type"),
                    "name": item.get("name"),
                    "module": item.get("module"),
                    "file_path": item.get("file_path"),
                    "start_line": item.get("start_line"),
                    "end_line": item.get("end_line"),
                    "chars": len(item.get("code") or ""),
                }
            )

        # Secondary: graph expansions from each primary hit (within remaining budget).
        for item in list(retrieved_items):
            seed_id = item.get("id")
            if not seed_id:
                continue
            for expanded in self._collect_graph_expansions(seed_id):
                expanded_id = expanded.get("id")
                if not expanded_id or expanded_id in included_ids:
                    continue
                snippet = self._format_context_item(item=expanded, include_code=True)
                if used_chars + len(snippet) > context_budget:
                    remaining = context_budget - used_chars
                    if remaining <= 0:
                        break
                    snippet = self._truncate_snippet(snippet, max_chars=remaining)
                    if not snippet:
                        break
                parts.append(snippet)
                used_chars += len(snippet) + 5
                included_ids.add(expanded_id)
                retrieved_items.append(
                    {
                        "id": expanded_id,
                        "score": None,
                        "type": expanded.get("type"),
                        "name": expanded.get("name"),
                        "module": expanded.get("module"),
                        "file_path": expanded.get("file_path"),
                        "start_line": expanded.get("start_line"),
                        "end_line": expanded.get("end_line"),
                        "chars": len(expanded.get("code") or ""),
                    }
                )

        context = "\n\n---\n\n".join(parts).strip()
        meta = {
            "max_prompt_chars": max_chars,
            "base_prompt_chars": base_prompt_len,
            "context_budget_chars": context_budget,
            "context_chars": len(context),
            "included_items": len(retrieved_items),
            "preface_chars": preface_chars,
        }
        return context, retrieved_items, meta

    def _try_answer_simple_math(self, question: str) -> str | None:
        raw = (question or "").strip()
        if not raw:
            return None

        normalized = re.sub(r"[\s\?=]+$", "", raw).strip()
        if not normalized:
            return None

        match = re.fullmatch(
            r"(?is)\s*(?:what\s+is|calculate|compute|eval(?:uate)?)\s+(?P<expr>.+?)\s*",
            normalized,
        )
        expr = (match.group("expr") if match else normalized).strip()
        if not expr or len(expr) > 200:
            return None

        if re.search(r"[A-Za-z_]", expr):
            return None
        if not re.fullmatch(r"[0-9\.\s\+\-\*\/%\(\)]+", expr):
            return None

        try:
            value = self._safe_eval_arithmetic(expr)
        except Exception:
            return None

        if isinstance(value, float):
            if value == 0.0:
                return "0"
            if value.is_integer():
                return str(int(value))
        return str(value)

    def _safe_eval_arithmetic(self, expr: str) -> int | float:
        tree = ast.parse(expr, mode="eval")

        def eval_node(node: ast.AST) -> int | float:
            if isinstance(node, ast.Constant):
                value = node.value
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("unsupported literal")
                return value
            if isinstance(node, ast.Num):  # pragma: no cover (py<3.8)
                value = node.n
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("unsupported literal")
                return value
            if isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +operand
                if isinstance(node.op, ast.USub):
                    return -operand
                raise ValueError("unsupported unary op")
            if isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                op = node.op
                if isinstance(op, ast.Add):
                    return left + right
                if isinstance(op, ast.Sub):
                    return left - right
                if isinstance(op, ast.Mult):
                    return left * right
                if isinstance(op, ast.Div):
                    return left / right
                if isinstance(op, ast.FloorDiv):
                    return left // right
                if isinstance(op, ast.Mod):
                    return left % right
                if isinstance(op, ast.Pow):
                    if isinstance(right, (int, float)) and abs(right) > 10_000:
                        raise ValueError("power too large")
                    return left**right
                raise ValueError("unsupported binary op")
            raise ValueError("unsupported expression")

        return eval_node(tree.body)

    def _llm_endpoint_candidates(self) -> list[str]:
        base = (settings.llama_api_base or "").strip()
        if not base:
            return []
        base = base.rstrip("/")

        provider = (settings.llm_provider or "").strip().lower()
        if provider == "ollama":
            # Prefer native Ollama endpoints; fall back to OpenAI-compatible if user points to /v1.
            if base.lower().endswith("/v1"):
                return [f"{base}/chat/completions"]
            if base.lower().endswith("/api"):
                return [f"{base}/chat", f"{base}/generate"]
            return [f"{base}/api/chat", f"{base}/api/generate", f"{base}/v1/chat/completions"]

        if re.search(r"(?i)(/chat/completions|/completions|:generatecontent)$", base):
            return [base]

        candidates = [
            f"{base}/v1/chat/completions",
            f"{base}/openai/v1/chat/completions",
            f"{base}/chat/completions",
            f"{base}/v1/completions",
            f"{base}/openai/v1/completions",
            base,
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for url in candidates:
            if url in seen:
                continue
            seen.add(url)
            ordered.append(url)
        return ordered

    def _llm_auth_header_sets(self) -> list[dict[str, str]]:
        base_headers = {"Content-Type": "application/json"}
        api_key = (settings.llama_api_key or "").strip()
        if not api_key:
            return [base_headers]

        provider = (settings.llm_provider or "").strip().lower()
        if provider == "gemini":
            return [
                {**base_headers, "x-goog-api-key": api_key},
                {**base_headers, "Authorization": f"Bearer {api_key}"},
                {**base_headers, "x-api-key": api_key},
            ]

        return [
            {**base_headers, "Authorization": f"Bearer {api_key}"},
            {**base_headers, "x-api-key": api_key},
            {**base_headers, "x-goog-api-key": api_key},
        ]

    def _extract_text_from_llm_response(self, data: Any) -> str:
        if isinstance(data, str):
            return data.strip()

        if not isinstance(data, dict):
            return ""

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            if isinstance(first, dict):
                message = first.get("message") or {}
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                text = first.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
                delta = first.get("delta") or {}
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

        response_text = data.get("response")
        if isinstance(response_text, str) and response_text.strip():
            return response_text.strip()

        output_text = data.get("output")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        # Gemini-style response: candidates[].content.parts[].text
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates:
            cand0 = candidates[0] or {}
            if isinstance(cand0, dict):
                content = cand0.get("content") or {}
                if isinstance(content, dict):
                    parts = content.get("parts")
                    if isinstance(parts, list):
                        texts: list[str] = []
                        for part in parts:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                text = part.get("text") or ""
                                if text.strip():
                                    texts.append(text.strip())
                        if texts:
                            return "\n".join(texts).strip()

        return ""

    def _query_llm(self, question: str, context: str) -> str:
        endpoints = self._llm_endpoint_candidates()
        if not endpoints:
            raise ValueError("LLAMA_API_BASE is not set (or empty).")

        # Quick preflight to fail fast if the endpoint is down.
        self._llm_health_check()

        prompt = self._build_prompt(question, context)
        model_name = (settings.llama_model_name or "").strip()
        if not model_name:
            raise ValueError("LLAMA_MODEL_NAME is not set (or empty).")

        provider = (settings.llm_provider or "").strip().lower()
        api_key_present = bool((settings.llama_api_key or "").strip())
        last_error: Exception | None = None

        for url in endpoints:
            is_chat = url.rstrip("/").lower().endswith("/chat/completions")
            is_completion = (
                url.rstrip("/").lower().endswith("/completions") and not is_chat
            )
            is_gemini_generate = "generatecontent" in url.lower()
            is_ollama_chat = url.rstrip("/").lower().endswith("/api/chat")
            is_ollama_generate = url.rstrip("/").lower().endswith("/api/generate")

            if is_ollama_chat or (provider == "ollama" and not (is_chat or is_completion or is_gemini_generate)):
                payload: dict[str, Any] = {
                    "model": model_name,
                    "stream": False,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"num_ctx": int(settings.llm_num_ctx)},
                }
                header_sets = [{"Content-Type": "application/json"}]
            elif is_ollama_generate:
                payload = {
                    "model": model_name,
                    "stream": False,
                    "prompt": prompt,
                    "options": {"num_ctx": int(settings.llm_num_ctx)},
                }
                header_sets = [{"Content-Type": "application/json"}]
            elif is_gemini_generate:
                payload: dict[str, Any] = {
                    "model": model_name,
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                }
                header_sets = self._llm_auth_header_sets()
            elif is_completion:
                payload = {"model": model_name, "prompt": prompt}
                header_sets = self._llm_auth_header_sets()
            else:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                header_sets = self._llm_auth_header_sets()

            for headers in header_sets:
                response: requests.Response | None = None
                try:
                    response = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self._llm_timeout(),
                    )

                    if response.status_code == 404:
                        break

                    if response.status_code in (401, 403) and api_key_present:
                        last_error = requests.HTTPError(
                            f"HTTP {response.status_code} (auth failed)", response=response
                        )
                        continue

                    try:
                        response.raise_for_status()
                    except requests.HTTPError as exc:
                        body = (response.text or "").strip()
                        detail = body[:800] + ("..." if len(body) > 800 else "")
                        raise requests.HTTPError(
                            f"{exc} - {detail}" if detail else str(exc),
                            response=response,
                        ) from None

                    try:
                        data = response.json()
                    except ValueError:
                        data = {"raw": (response.text or "").strip()}

                    text = self._extract_text_from_llm_response(data)
                    if text:
                        return text

                    # Ollama native format: {"message": {"content": "..."}}
                    if isinstance(data, dict):
                        message = data.get("message")
                        if isinstance(message, dict):
                            content = message.get("content")
                            if isinstance(content, str) and content.strip():
                                return content.strip()
                        resp_text = data.get("response")
                        if isinstance(resp_text, str) and resp_text.strip():
                            return resp_text.strip()

                    last_error = ValueError("LLM response did not contain any text.")
                except Exception as exc:
                    last_error = exc
                finally:
                    if response is not None:
                        response.close()

        if last_error is not None:
            raise last_error
        raise ValueError("LLM request failed: no working endpoint found.")

    def answer_with_debug(self, question: str, *, k: int = 5, debug: bool = False) -> str | dict[str, Any]:
        logger.info("Received question: %s", question)
        self._reload_if_stale()

        math_answer = self._try_answer_simple_math(question)
        if math_answer is not None:
            payload = {"answer": math_answer, "retrieved": [], "meta": {"fast_path": "math"}}
            return payload if debug else math_answer

        if not (settings.llama_api_base or "").strip():
            message = "LLAMA_API_BASE is not set (or empty). Set it in `.env` (or as an environment variable) and restart the API."
            payload = {
                "answer": message,
                "retrieved": [],
                "meta": {
                    "llm_provider": settings.llm_provider,
                    "llama_api_base": settings.llama_api_base,
                    "llama_model_name": settings.llama_model_name,
                    "llama_timeout_s": settings.llama_timeout,
                },
            }
            return payload if debug else message

        if not self.vector_store or not self.metadata:
            # Try to reload in case indexing completed after app startup.
            self._load_indexes()
        if not self.vector_store or not self.metadata:
            message = (
                "Vector index is missing. Run the indexer to build embeddings and graph data."
            )
            return {"answer": message} if debug else message

        # Detect embedding dimension mismatches (e.g., switching embedding backends).
        try:
            probe_vec = self._get_embedder().encode(["dimension_probe"])[0]
            expected_dim = len(probe_vec)
        except Exception as exc:
            message = f"Embedding backend failed to initialize: {exc}"
            return {"answer": message} if debug else message

        actual_dim = int(getattr(self.vector_store, "dim", 0) or 0)
        if actual_dim and expected_dim and actual_dim != expected_dim:
            message = (
                "Embedding dimension mismatch between the saved FAISS index and the current embedding backend. "
                f"Index dim={actual_dim}, embedder dim={expected_dim}. "
                "Re-run indexing to rebuild the vector store with the current settings."
            )
            payload = {"answer": message, "retrieved": [], "meta": {"index_dim": actual_dim, "embedder_dim": expected_dim}}
            return payload if debug else message

        scores, indices, retrieval_meta = self._hybrid_search(question, k=k)

        max_chars = self._max_prompt_chars()
        preface = self._build_context_preface(
            question=question, indices=indices, retrieval_meta=retrieval_meta
        )
        context, retrieved, meta = self._build_context(
            scores, indices, question=question, max_chars=max_chars, preface=preface
        )
        meta.update(
            {
                "k": k,
                "llm_provider": settings.llm_provider,
                "llama_api_base": settings.llama_api_base,
                "llama_model_name": settings.llama_model_name,
                "llama_timeout_s": settings.llama_timeout,
                "llm_num_ctx": settings.llm_num_ctx,
            }
        )
        meta.update(retrieval_meta)
        if not context:
            message = "No relevant context found for the question."
            return {"answer": message, "retrieved": retrieved, "meta": meta} if debug else message

        try:
            answer = self._query_llm(question, context)
        except Exception as exc:
            logger.error("LLM request failed: %s", exc)
            error_message = f"LLM request failed: {exc}"
            return (
                {
                    "answer": error_message,
                    "retrieved": retrieved,
                    "context": context,
                    "prompt": self._build_prompt(question, context),
                    "meta": meta,
                }
                if debug
                else error_message
            )

        if not debug:
            return answer

        return {
            "answer": answer,
            "retrieved": retrieved,
            "context": context,
            "prompt": self._build_prompt(question, context),
            "meta": meta,
        }

    def answer(self, question: str, *, k: int = 5, debug: bool = False) -> str | dict[str, Any]:
        return self.answer_with_debug(question, k=k, debug=debug)
