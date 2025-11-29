"""Structural scraping powered by Docling."""
from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from docling.datamodel.document import DoclingDocument, DocumentStream




SCRAPE_DIR = Path("./SAVED")
SAVE_SCRAPE = True
from .loader import iter_document_payloads
from .types import (
    DocumentPayload,
    Paragraph,
    ScrapeArtifact,
    Section,
    Subsection,
    TableRowPayload,
)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

try:  # Docling >=1.5.0 exposes RapidOcrOptions for higher-accuracy OCR.
    from docling.datamodel.pipeline_options import RapidOcrOptions
except ImportError:  # pragma: no cover - fallback for older Docling versions.
    RapidOcrOptions = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
import os

os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/5/"


class StructuralScraper:
    """Extract a section/subsection/paragraph tree using Docling's document model."""

    def __init__(
        self,
        *,
        converter: Optional[DocumentConverter] = None,
        min_paragraph_chars: int = 20,
        subheading_ratio_threshold: float = 0.5,
    ) -> None:
        """Initialize the Docling-backed structural scraper.

        Args:
            converter: Optional pre-built Docling converter for tests.
            min_paragraph_chars: Minimum characters before keeping a paragraph.
            subheading_ratio_threshold: Heading height ratio that triggers
                demotion to a subsection to reduce false section breaks.
        """

        self.min_paragraph_chars = min_paragraph_chars
        self.converter = converter or self._build_docling_converter()
        self.heading_labels = {"title", "section_header"}
        self.paragraph_labels = {"paragraph", "text", "list_item"}
        self.table_labels = {"table", "document_index"}
        self.subheading_ratio_threshold = subheading_ratio_threshold

    def _build_docling_converter(self) -> DocumentConverter:
        """Construct a Docling converter with OCR + table structure enabled."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options = RapidOcrOptions(force_full_page_ocr=True)

        pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)
        return DocumentConverter(
            format_options={
                InputFormat.PDF: pdf_format_option,
            }
        )

    def scrape(self, payload: DocumentPayload) -> ScrapeArtifact:
        """Convert a PDF payload into hierarchical text plus breadcrumbs."""

        logger.debug("Scraping document %s via Docling", payload.doc_id)
        dl_document = self._convert_payload(payload)
        doc_dict = dl_document.model_dump()

        sections, breadcrumbs_index, table_rows = self._build_sections(
            payload.doc_id, doc_dict
        )
        if not sections:
            placeholder = Paragraph(
                paragraph_id=f"{payload.doc_id}_p0",
                text="No textual content extracted.",
                page=None,
            )
            sections = [
                Section(
                    heading="Document",
                    order=1,
                    subsections=[Subsection(subheading="Body", paragraphs=[placeholder])],
                )
            ]
            breadcrumbs_index = {}
            table_rows = []

        tables = self._summarize_structures(doc_dict.get("tables", []))
        figures = self._summarize_structures(doc_dict.get("pictures", []))

        artifact = ScrapeArtifact(
            doc_id=payload.doc_id,
            sections=sections,
            tables=tables,
            figures=figures,
            source_path=payload.path,
            docling_json=doc_dict,
            doc_item_breadcrumbs=breadcrumbs_index,
            table_rows=table_rows,
        )
        logger.info(
            "Docling scraped %s → %d sections, %d subsections, %d paragraphs",
            payload.doc_id,
            len(artifact.sections),
            sum(len(sec.subsections) for sec in artifact.sections),
            self._count_paragraphs(artifact.sections),
        )
        return artifact

    def save_artifact(self, artifact: ScrapeArtifact) -> Path:
        """Persist the scraped artifact to disk for reproducibility."""

        SCRAPE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = SCRAPE_DIR / f"{artifact.doc_id}.json"
        output_path.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2))
        logger.info("Wrote scrape artifact to %s", output_path)
        return output_path

    def _convert_payload(self, payload: DocumentPayload) -> DoclingDocument:
        """Run the Docling converter against bytes or a file path."""

        source: str | DocumentStream
        if payload.bytes:
            name = payload.path.name if payload.path else f"{payload.doc_id}.pdf"
            source = DocumentStream(name=name, stream=BytesIO(payload.bytes))
        elif payload.path and payload.path.exists():
            source = str(payload.path)
        else:
            raise FileNotFoundError(f"Unable to open PDF for payload {payload.doc_id}")

        result = self.converter.convert(source)
        if result.document is None:
            raise RuntimeError(f"Docling returned no document for payload {payload.doc_id}")
        return result.document

    def _build_sections(
        self, doc_id: str, doc_dict: Dict[str, object]
    ) -> tuple[List[Section], Dict[str, List[str]], List[TableRowPayload]]:
        """Traverse Docling's body tree and emit Section/Subsection objects."""

        body = doc_dict.get("body")
        if not isinstance(body, dict):
            return [], {}, []
        ref_index = self._build_ref_index(doc_dict)

        sections: List[Section] = []
        current_section: Optional[Section] = None
        current_subsection: Optional[Subsection] = None
        current_section_size: Optional[float] = None
        breadcrumbs_index: Dict[str, List[str]] = {}
        paragraph_counter = 0
        table_rows_cache: Dict[str, List[TableRowPayload]] = {}
        table_rows_accumulator: List[TableRowPayload] = []
        table_order: Dict[str, int] = {}
        for node in self._walk_body(body, ref_index):
            label = (node.get("label") or "").lower()
            text = self._normalize_text(node.get("text") or "")

            if self._is_table_node(label, node.get("self_ref")):
                table_ref = node.get("self_ref")
                if not table_ref:
                    continue
                rows = table_rows_cache.get(table_ref)
                if rows is None:
                    rows = self._extract_table_rows(node)
                    table_rows_cache[table_ref] = rows
                    table_rows_accumulator.extend(rows)
                if not rows:
                    continue
                current_section, current_subsection = self._ensure_targets(
                    sections, current_section, current_subsection
                )
                table_position = table_order.setdefault(table_ref, len(table_order) + 1)
                self._record_breadcrumb(
                    breadcrumbs_index,
                    table_ref,
                    current_section,
                    current_subsection,
                )
                for row in rows:
                    paragraph_counter += 1
                    row_text = json.dumps(
                        {
                            "headers": row.headers,
                            "row": row.values,
                            "table_ref": row.table_ref,
                        },
                        separators=(",", ":"),
                    )
                    paragraph = Paragraph(
                        paragraph_id=f"{doc_id}_table_{table_position}_row_{row.row_index}",
                        text=row_text,
                        page=row.page,
                        doc_item_ref=f"{row.table_ref}::row::{row.row_index}",
                    )
                    current_subsection.paragraphs.append(paragraph)
                    self._record_breadcrumb(
                        breadcrumbs_index,
                        paragraph.doc_item_ref,
                        current_section,
                        current_subsection,
                    )
                continue

            if not text:
                continue

            if label in self.heading_labels:
                level = node.get("level") or 1
                heading_size = self._estimate_font_height(node)
                if self._should_promote_to_subheading(
                    level, heading_size, current_section_size, current_section
                ):
                    level = max(level, 2)
                current_section, current_subsection = self._update_headings(
                    sections, text, level
                )
                if level <= 1:
                    current_section_size = heading_size
                self._record_breadcrumb(
                    breadcrumbs_index,
                    node.get("self_ref"),
                    current_section,
                    current_subsection,
                    fallback=text,
                )
                continue

            if label in self.paragraph_labels:
                if len(text) < self.min_paragraph_chars:
                    continue
                paragraph_counter += 1
                paragraph = Paragraph(
                    paragraph_id=f"{doc_id}_p{paragraph_counter}",
                    text=text,
                    page=self._extract_page(node),
                    doc_item_ref=node.get("self_ref"),
                )
                current_section, current_subsection = self._ensure_targets(
                    sections, current_section, current_subsection
                )
                current_subsection.paragraphs.append(paragraph)
                self._record_breadcrumb(
                    breadcrumbs_index,
                    node.get("self_ref"),
                    current_section,
                    current_subsection,
                )

        return sections, breadcrumbs_index, table_rows_accumulator

    def _build_ref_index(self, doc_dict: Dict[str, object]) -> Dict[str, Dict[str, object]]:
        """Map Docling self_ref identifiers to their concrete nodes."""

        ref_index: Dict[str, Dict[str, object]] = {}
        for value in doc_dict.values():
            if isinstance(value, dict):
                ref = value.get("self_ref")
                if ref:
                    ref_index[ref] = value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        ref = item.get("self_ref")
                        if ref:
                            ref_index[ref] = item
        return ref_index

    def _walk_body(
        self, node: Dict[str, object], ref_index: Dict[str, Dict[str, object]]
    ) -> Iterator[Dict[str, object]]:
        """Yield each body node referenced by the tree, avoiding cycles."""

        visited: set[str] = set()

        def _walk(current: Dict[str, object]) -> Iterator[Dict[str, object]]:
            """Depth-first traversal over Docling body references without repeats."""

            for child in current.get("children", []) or []:
                ref = child.get("$ref") or child.get("cref")
                if not ref or ref in visited:
                    continue
                target = ref_index.get(ref)
                if not target:
                    continue
                visited.add(ref)
                yield target
                if target.get("children"):
                    yield from _walk(target)

        yield from _walk(node)

    def _update_headings(
        self, sections: List[Section], text: str, level: int
    ) -> tuple[Section, Optional[Subsection]]:
        """Create or update the active Section/Subsection from a heading node."""

        if level <= 1 or not sections:
            section = Section(heading=text, order=len(sections) + 1, subsections=[])
            sections.append(section)
            return section, None
        section = sections[-1]
        subsection = Subsection(subheading=text, paragraphs=[])
        section.subsections.append(subsection)
        return section, subsection

    def _ensure_targets(
        self,
        sections: List[Section],
        current_section: Optional[Section],
        current_subsection: Optional[Subsection],
    ) -> tuple[Section, Subsection]:
        """Guarantee that we have section/subsection containers for paragraphs."""

        if current_section is None:
            current_section = Section(heading="Document", order=len(sections) + 1, subsections=[])
            sections.append(current_section)
        if current_subsection is None:
            current_subsection = Subsection(subheading="Body", paragraphs=[])
            current_section.subsections.append(current_subsection)
        return current_section, current_subsection

    def _record_breadcrumb(
        self,
        index: Dict[str, List[str]],
        ref: Optional[str],
        section: Optional[Section],
        subsection: Optional[Subsection],
        *,
        fallback: Optional[str] = None,
    ) -> None:
        """Store breadcrumb trails for Docling refs so chunking can reuse them."""

        if not ref:
            return
        breadcrumbs: List[str] = []
        if section:
            breadcrumbs.append(section.heading)
        if subsection:
            breadcrumbs.append(subsection.subheading)
        if not breadcrumbs and fallback:
            breadcrumbs.append(fallback)
        if breadcrumbs:
            index.setdefault(ref, breadcrumbs)

    def _estimate_font_height(self, node: Dict[str, object]) -> Optional[float]:
        """Approximate visual prominence using Docling provenance bounding boxes."""

        provenance = node.get("prov") or []
        if not provenance or not isinstance(provenance, list):
            return None
        bbox = provenance[0].get("bbox") or {}
        top = bbox.get("t")
        bottom = bbox.get("b")
        if isinstance(top, (int, float)) and isinstance(bottom, (int, float)):
            height = abs(float(top) - float(bottom))
            return height if height > 0 else None
        return None

    def _should_promote_to_subheading(
        self,
        level: int,
        heading_size: Optional[float],
        current_section_size: Optional[float],
        current_section: Optional[Section],
    ) -> bool:
        """Decide whether a level-1 heading is really a subsection title."""

        if level > 1:
            return False
        if (
            current_section is None
            or heading_size is None
            or current_section_size is None
            or current_section_size == 0
        ):
            return False
        ratio = heading_size / current_section_size
        return ratio <= self.subheading_ratio_threshold

    def _summarize_structures(self, items: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        """Extract lightweight table/figure metadata for downstream audits."""

        summaries: List[Dict[str, object]] = []
        for item in items or []:
            label = item.get("label")
            ref = item.get("self_ref")
            page = self._extract_page(item)
            summaries.append({"label": label, "self_ref": ref, "page": page})
        return summaries

    def _is_table_node(self, label: str, ref: Optional[str]) -> bool:
        """Return True when a Docling node represents a table structure."""

        if ref and str(ref).startswith("#/tables/"):
            return True
        return label in self.table_labels

    def _extract_table_rows(self, table_node: Dict[str, object]) -> List[TableRowPayload]:
        """Reconstruct ordered rows from Docling table cells."""

        data = table_node.get("data") or {}
        if not isinstance(data, dict):
            return []
        table_cells = data.get("table_cells") or []
        if not table_cells:
            return []

        num_cols = (
            data.get("num_cols")
            or data.get("n_cols")
            or self._infer_dimension(table_cells, axis="col")
        )
        headers = self._build_headers(table_cells, num_cols)
        header_rows = {
            int(cell.get("start_row_offset_idx") or 0)
            for cell in table_cells
            if cell.get("column_header")
        }

        rows_by_idx: Dict[int, Dict[int, str]] = {}
        for cell in table_cells:
            if cell.get("column_header"):
                continue
            text = self._normalize_text(cell.get("text") or "")
            if not text:
                continue
            start_row = int(cell.get("start_row_offset_idx") or 0)
            end_row = int(
                cell.get("end_row_offset_idx")
                or (start_row + max(int(cell.get("row_span") or 1), 1))
            )
            start_col = int(cell.get("start_col_offset_idx") or 0)
            end_col = int(
                cell.get("end_col_offset_idx")
                or (start_col + max(int(cell.get("col_span") or 1), 1))
            )
            for row_idx in range(start_row, end_row):
                row = rows_by_idx.setdefault(row_idx, {})
                for col_idx in range(start_col, end_col):
                    row[col_idx] = text

        page = self._extract_page(table_node)
        label_text = self._extract_table_label(table_node)
        sorted_row_indices = sorted(idx for idx in rows_by_idx if idx not in header_rows)
        payloads: List[TableRowPayload] = []
        for logical_index, row_idx in enumerate(sorted_row_indices):
            row_cells = rows_by_idx[row_idx]
            values: Dict[str, str] = {}
            for col_idx, header in enumerate(headers):
                value = row_cells.get(col_idx)
                if value:
                    values[header or f"column_{col_idx + 1}"] = value
            if not values:
                continue
            payloads.append(
                TableRowPayload(
                    table_ref=table_node.get("self_ref") or "",
                    row_index=logical_index,
                    headers=list(headers),
                    values=values,
                    page=page,
                    label=label_text,
                )
            )
        return payloads

    def _build_headers(
        self, cells: Sequence[Dict[str, object]], num_cols: Optional[int]
    ) -> List[str]:
        """Derive ordered header names using Docling cell metadata."""

        if num_cols is None:
            inferred = self._infer_dimension(cells, axis="col")
            num_cols = inferred or 0
        headers: List[str] = [""] * max(num_cols or 0, 0)
        for cell in cells:
            if not cell.get("column_header"):
                continue
            text = self._normalize_text(cell.get("text") or "")
            start_col = int(cell.get("start_col_offset_idx") or 0)
            end_col = int(
                cell.get("end_col_offset_idx")
                or (start_col + max(int(cell.get("col_span") or 1), 1))
            )
            if end_col > len(headers):
                headers.extend([""] * (end_col - len(headers)))
            for col_idx in range(start_col, end_col):
                existing = headers[col_idx]
                headers[col_idx] = text if not existing else f"{existing} | {text}"
        for idx, name in enumerate(headers):
            if not name:
                headers[idx] = f"column_{idx + 1}"
        return headers

    @staticmethod
    def _infer_dimension(
        cells: Sequence[Dict[str, object]], *, axis: str
    ) -> Optional[int]:
        """Infer docling row/column counts when missing."""

        key = "end_col_offset_idx" if axis == "col" else "end_row_offset_idx"
        values = [int(cell.get(key) or 0) for cell in cells]
        if not values:
            return None
        return max(values)

    @staticmethod
    def _extract_table_label(node: Dict[str, object]) -> Optional[str]:
        """Prefer caption text, falling back to Docling labels."""

        captions = node.get("captions") or []
        if captions and isinstance(captions, list):
            first = captions[0]
            if isinstance(first, dict):
                text = first.get("text")
                if text:
                    return " ".join(str(text).split())
        meta = node.get("meta") or {}
        if isinstance(meta, dict):
            title = meta.get("title") or meta.get("name")
            if title:
                return " ".join(str(title).split())
        label = node.get("label")
        if isinstance(label, str) and label:
            return label
        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse whitespace so heading comparisons remain stable."""

        return " ".join(text.split())

    @staticmethod
    def _extract_page(node: Dict[str, object]) -> Optional[int]:
        """Pull the first page number from Docling provenance if available."""

        provenance = node.get("prov") or []
        if provenance and isinstance(provenance, list):
            page = provenance[0].get("page_no")
            if isinstance(page, int):
                return page
        return None

    @staticmethod
    def _count_paragraphs(sections: Iterable[Section]) -> int:
        """Return the total number of paragraphs for logging/QA."""

        return sum(len(sub.paragraphs) for sec in sections for sub in sec.subsections)


__all__ = [
    "StructuralScraper",
    "ScrapeArtifact",
    "Section",
    "Subsection",
    "Paragraph",
]


if __name__ == "__main__":
    scraper = StructuralScraper()
    for payload in iter_document_payloads(load_bytes=True):
        artifact = scraper.scrape(payload)
        if SAVE_SCRAPE:
            scraper.save_artifact(artifact)
        logger.info(
            "Saved scrape for %s: sections=%d",
            payload.doc_id,
            len(artifact.sections),
        )
