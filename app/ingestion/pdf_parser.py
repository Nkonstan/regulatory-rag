import fitz  # PyMuPDF
import pdfplumber
import re
from typing import List, Dict, Tuple
from pathlib import Path


class PDFParser:
    """Extract text and sections from regulatory PDFs."""
    
    # Patterns for detecting section headings
    SECTION_PATTERNS = [

        # NEW — Q&A boundaries (colon or period delimiter only, not space)
        r'^(Q\d+)[\.:]\s*(.*)$',
        r'^(A\d+)[\.:]\s*(.*)$',

        # Roman numerals: I. INTRODUCTION (1.0)
        r'^([IVX]+)\.\s+([A-Z][A-Z\s\-:]+?)(?:\s*\([0-9.]+\))?\s*$',
        
        # Letters (uppercase): A. GENERAL SCHEME (1.1) or B. PURPOSE OF CONTROL GROUP
        r'^([A-Z])\.\s+([A-Za-z][A-Za-z\s\-:]+?)(?:\s*\([0-9.]+\))?\s*$',
        
        # Numbered sections with title in caps: 1. DESCRIPTION (2.3.1)
        r'^(\d+)\.\s+([A-Z][A-Za-z\s\-:]+?)(?:\s*\([0-9.]+\))?\s*$',
        
        # Decimal sections: 2.1 Placebo Control or 2.1. Placebo Control
        r'^(\d+\.\d+)\.?\s+(.+?)(?:\s*\([0-9.]+\))?\s*$',
        
        # Deep nesting: 2.1.3 Ethical Issues
        r'^(\d+\.\d+\.\d+)\.?\s+(.+?)(?:\s*\([0-9.]+\))?\s*$',
        
        # Even deeper: 2.1.3.1 Historical Evidence
        r'^(\d+\.\d+\.\d+\.\d+)\.?\s+(.+?)(?:\s*\([0-9.]+\))?\s*$',
        
        # Lowercase letters for sub-sub-sections: a. Efficiency (2.3.6.1)
        r'^([a-z])\.\s+(.+?)(?:\s*\([0-9.]+\))?\s*$',
    ]
    
    def __init__(self):
        self.current_section = "0"
        self.section_hierarchy = []

    def _overlaps_any(self, bx0: float, by0: float, bx1: float, by1: float,
                    bboxes: list) -> bool:
        """
        Return True if the block (bx0,by0,bx1,by1) overlaps with any
        pdfplumber table bbox. pdfplumber uses top-of-page origin, same
        as fitz, so coordinates are directly comparable.
        """
        for (tx0, ty0, tx1, ty1) in bboxes:
            # Standard rectangle overlap check
            if bx0 < tx1 and bx1 > tx0 and by0 < ty1 and by1 > ty0:
                return True
        return False

    
    def extract_text_with_sections(self, pdf_path: Path) -> List[Dict[str, str]]:
        """
        Extract text from PDF and identify sections with hierarchical tracking.
        """
        doc = fitz.open(pdf_path)
        sections = []
        current_section = {
            'section': '0',
            'section_title': 'Document Start',
            'text': '',
            'page_start': 1,
            'chunk_type': 'text'
        }

        section_hierarchy = []

        # Pre-extract all tables using pdfplumber before the main fitz loop.
        # page_tables maps page_num (1-indexed) -> list of Markdown table strings.
        page_tables = {}      # pg_num -> list of markdown strings
        page_table_bboxes = {}  # pg_num -> list of (x0, top, x1, bottom)

        with pdfplumber.open(pdf_path) as pdf_pl:
            for pg_num, pg in enumerate(pdf_pl.pages, start=1):
                # Try bordered tables first, fall back to text-based detection
                found = pg.find_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                })

                if found:
                    page_table_bboxes[pg_num] = [t.bbox for t in found]
                    page_tables[pg_num] = []
                    for t in found:
                        if not t.extract():
                            continue
                        table_md = self._table_to_markdown(t.extract())
                        x0, top, x1, bottom = t.bbox
                        page_height = pg.height

                        # Read 40pt above the table border
                        above_text = ''
                        if top > 0:
                            above_region = pg.crop((x0, max(0, top - 40), x1, top))
                            above_text = (above_region.extract_text() or '').strip()

                        # Read 40pt below the table border
                        below_text = ''
                        if bottom < page_height:
                            below_region = pg.crop((x0, bottom, x1, min(page_height, bottom + 40)))
                            below_text = (below_region.extract_text() or '').strip()

                        # Pick whichever contains a table/figure title keyword
                        title = ''
                        for candidate in [above_text, below_text]:
                            if any(kw in candidate for kw in ('Table', 'Figure', 'table', 'figure')):
                                title = candidate
                                break

                        if title:
                            table_md = f"{title}\n{table_md}"

                        page_tables[pg_num].append(table_md)



        in_toc = False
        toc_end_page = 0
        toc_found = False
        for page_num, page in enumerate(doc, start=1):

            # Get text as blocks with coordinates so we can filter table regions
            bboxes_to_exclude = page_table_bboxes.get(page_num, [])

            if not bboxes_to_exclude:
                text = page.get_text()
            else:
                # Collect only text blocks that don't overlap with any table bbox
                prose_parts = []
                for block in page.get_text("blocks"):
                    # block = (x0, y0, x1, y1, text, block_no, block_type)
                    bx0, by0, bx1, by1, block_text, _, block_type = block
                    if block_type != 0:   # 0 = text block, 1 = image — skip images
                        continue
                    if not self._overlaps_any(bx0, by0, bx1, by1, bboxes_to_exclude):
                        prose_parts.append(block_text)
                text = "\n".join(prose_parts)

            lines = text.split('\n')

            
            # Detect TOC
            page_text_upper = text.upper()
            if 'TABLE OF CONTENTS' in page_text_upper and not toc_found:
                in_toc = True
                toc_found = True  # prevents any re-triggering on later pages
                toc_end_page = page_num + 1
                print(f"📋 Detected Table of Contents on page {page_num}")
            
            # Skip TOC pages
            if in_toc and page_num <= toc_end_page:
                current_section['text'] += text + ' '
                continue
            else:
                in_toc = False
            
            # Inject structured table chunks before processing prose on this page.
            if page_num in page_tables:
                for table_md in page_tables[page_num]:

                    if current_section['text'].strip():
                        current_section['page_end'] = page_num
                        sections.append(current_section.copy())
                        current_section = {
                            **current_section,
                            'text': '',
                            'page_start': page_num,
                            'chunk_type': 'text'
                        }

                    sections.append({
                        'section': current_section['section'],
                        'section_title': current_section['section_title'],
                        'text': table_md,  # merged
                        'page_start': page_num,
                        'page_end': page_num,
                        'chunk_type': 'table'
                    })
                    print(f"📊 Injected table chunk in section "
                        f"{current_section['section']} (page {page_num})")


            # Process lines
            i = 0
            while i < len(lines):

                line = lines[i].strip()
                
                if not line:
                    i += 1
                    continue
                
                # Try merging with next line for section detection
                merged_line = line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if len(line) < 10 and next_line:
                        merged_line = line + ' ' + next_line
                
                # Try to detect section
                section_match = self._detect_section(merged_line)
                
                if section_match:
                    if current_section['section'] != '0' or current_section['text'].strip():
                        current_section['page_end'] = page_num - 1
                        sections.append(current_section.copy())
                    
                    section_num, section_title = section_match
                    
                    # NEW: Update hierarchy based on section level
                    section_hierarchy = self._update_hierarchy(
                        section_hierarchy, 
                        section_num, 
                        section_title
                    )
                    
                    # NEW: Build full hierarchical section ID
                    full_section_id = self._build_section_id(
                        section_hierarchy, 
                        section_num
                    )
                    
                    print(f"✓ Found section: {full_section_id} - {section_title} (page {page_num})")
                    
                    current_section = {
                        'section': full_section_id,
                        'section_title': section_title,
                        'text': '',
                        'page_start': page_num,
                        'chunk_type': 'text'
                    }

                    
                    # Skip merged lines
                    if len(line) < 10 and i + 1 < len(lines):
                        i += 2
                    else:
                        i += 1
                else:
                    current_section['text'] += line + ' '
                    i += 1
        
        # Add final section
        if current_section['text'].strip():
            current_section['page_end'] = len(doc)
            sections.append(current_section)
        
        doc.close()
        
        print(f"\n📊 Extraction Summary:")
        print(f"   Total sections: {len(sections)}")
        print(f"   Sections with content: {sum(1 for s in sections if s['section'] != '0')}")
        
        return sections
        

    def _detect_section(self, line: str) -> Tuple[str, str] | None:
        """
        Detect if a line is a section heading.
        
        Returns:
            (section_number, section_title) or None
        """
        # Skip very long lines
        if len(line) > 200:
            return None
        
        # Clean the line first
        line = ' '.join(line.split())
        
        # Skip lines that look like Table of Contents (have dots leading to page numbers)
        if re.search(r'\.{3,}\s*\d+$', line):
            return None  # This is a TOC line, not a real section
        
        # PRIORITY 1: Look for parenthetical section numbers like (1.0), (2.1.6)
        # These are the REAL section numbers in ICH documents
        paren_match = re.search(r'\((\d+(?:\.\d+)*)\)', line)
        
        # Try each pattern
        for pattern in self.SECTION_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                # Prefer parenthetical number if found
                if paren_match:
                    section_num = paren_match.group(1)  # e.g., "2.1.6"
                else:
                    section_num = match.group(1)  # Use matched number
                
                section_title = match.group(2).strip()
                
                # Clean the section title
                # Remove parenthetical numbers
                section_title = re.sub(r'\s*\(\d+(?:\.\d+)*\)', '', section_title)
                # Remove trailing dots and page numbers
                section_title = re.sub(r'\s*\.+\s*\d*$', '', section_title)
                # Remove (See Section X.X) references
                section_title = re.sub(r'\s*\(SEE SECTION[^)]+\)', '', section_title, flags=re.IGNORECASE)
                section_title = section_title.strip()
                
                
                # Q/A patterns skip heading validation — they are always sections
                is_qa = re.match(r'^[QA]\d+$', section_num)
                # Validate it looks like a heading
                if is_qa or self._looks_like_heading(line, section_title):
                    return (section_num, section_title)
        
        return None

    def _get_section_level(self, section_num: str) -> int:
        """
        Determine the hierarchical level of a section.
        
        Returns:
            0 = Roman numerals (I, II, III, IV, V)
            1 = Capital letters (A, B, C, D)
            2 = Numbers (1, 2, 3, 4)
            3 = Lowercase letters (a, b, c)
            4 = Deep nesting (1.1, 2.3.1, etc.)
        """
        # Roman numerals
        if re.match(r'^[IVX]+$', section_num):
            return 0
        
        # Single capital letter
        if re.match(r'^[A-Z]$', section_num):
            return 1
        
        # Lowercase letter
        if re.match(r'^[a-z]$', section_num):
            return 3
        
        # Decimal numbering (1.2, 2.3.1, etc.)
        if '.' in section_num:
            return 4
        
        # Single digit or number
        if section_num.isdigit():
            return 2
        
        return 2  # Default to level 2

    def _update_hierarchy(self, current_hierarchy: List[str], 
                        new_section_num: str, 
                        section_title: str) -> List[str]:
        """
        Update the section hierarchy stack based on new section.
        
        Args:
            current_hierarchy: Current hierarchy stack, e.g., ["III", "A"]
            new_section_num: New section number, e.g., "2"
            section_title: Section title for context
            
        Returns:
            Updated hierarchy stack
        """
        new_level = self._get_section_level(new_section_num)
        
        # If this is a top-level section (Roman numeral), reset hierarchy
        if new_level == 0:
            return [new_section_num]
        
        # If hierarchy is empty, start it
        if not current_hierarchy:
            return [new_section_num]
        
        # Determine how many levels to keep based on new section level
        # Level 0 (Roman) → keep none
        # Level 1 (Letter) → keep level 0 (Roman)
        # Level 2 (Number) → keep levels 0-1 (Roman, Letter)
        # Level 3 (lowercase) → keep levels 0-2
        # Level 4 (decimal) → keep all
        
        if new_level == 1:
            # Capital letter - keep only Roman numerals
            hierarchy = [h for h in current_hierarchy if self._get_section_level(h) == 0]
            hierarchy.append(new_section_num)
            return hierarchy
        
        elif new_level == 2:
            # Number - keep Roman and Letters
            hierarchy = [h for h in current_hierarchy if self._get_section_level(h) <= 1]
            hierarchy.append(new_section_num)
            return hierarchy
        
        elif new_level == 3:
            # Lowercase letter - keep Roman, Letters, Numbers
            hierarchy = [h for h in current_hierarchy if self._get_section_level(h) <= 2]
            hierarchy.append(new_section_num)
            return hierarchy
        
        elif new_level == 4:
            # Decimal numbering - this is self-contained, use as-is
            return [new_section_num]
        
        # Default: append to current hierarchy
        return current_hierarchy + [new_section_num]

    def _build_section_id(self, hierarchy: List[str], current_section: str) -> str:
        """
        Build a full hierarchical section ID.
        
        Args:
            hierarchy: Current hierarchy stack
            current_section: Current section number
            
        Returns:
            Full section ID, e.g., "III.A.2" or just "2.3.1"
        """
        # If current section is decimal notation (1.2.3), it's already complete
        if '.' in current_section and current_section[0].isdigit():
            return current_section
        
        # If hierarchy is just the current section, return it
        if len(hierarchy) == 1 and hierarchy[0] == current_section:
            return current_section
        
        # Build hierarchical ID
        if len(hierarchy) > 1:
            return '.'.join(hierarchy)
        elif len(hierarchy) == 1:
            return hierarchy[0]
        else:
            return current_section
    
    def _looks_like_heading(self, line: str, title: str) -> bool:
        """Heuristics to validate if something is really a heading."""
        
        # Remove dots and page numbers for length check
        clean_line = re.sub(r'\s*\.+\s*\d*$', '', line)
        
        # Headings are usually short (after cleaning)
        if len(clean_line) > 150:
            return False
        
        # Title shouldn't be empty
        if not title or len(title) < 2:
            return False
        
        # Headings are short — long sentences are body text, not headings
        words = title.split()
        if len(words) > 12:
            return False

        # Check capitalization (at least 40% of words should start with capital)
        if len(words) > 2:
            capitalized = sum(1 for w in words if w and w[0].isupper())
            if capitalized / len(words) < 0.4:
                return False
        
        return True


    def _table_to_markdown(self, table: list) -> str:
        if not table or not table[0]:
            return ""

        def clean(cell):
            if cell is None:
                return ""
            return str(cell).replace("\n", " ").strip()

        # Find the first row that has at least 50% non-empty cells — use as header
        header_idx = 0
        for i, row in enumerate(table):
            filled = sum(1 for c in row if clean(c))
            if filled >= len(row) * 0.5:
                header_idx = i
                break

        header = [clean(c) for c in table[header_idx]]
        data_rows = table[header_idx + 1:]

        rows = []
        rows.append('| ' + ' | '.join(header) + ' |')
        rows.append('|' + '|'.join(['---'] * len(header)) + '|')

        for row in data_rows:
            cleaned = [clean(c) for c in row]
            while len(cleaned) < len(header):
                cleaned.append('')
            cleaned = cleaned[:len(header)]
            if any(cleaned):  # skip fully empty rows
                rows.append('| ' + ' | '.join(cleaned) + ' |')

        return '\n'.join(rows)



    
    def extract_metadata(self, pdf_path: Path) -> Dict[str, str]:
        """Extract PDF metadata."""
        doc = fitz.open(pdf_path)
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'pages': doc.page_count
        }
        doc.close()
        return metadata
