# region BuildScript
import sys
from pathlib import Path

try:
    import typst
except ImportError:
    print("[PaperBuilder-build_pdf] Error: 'typst' package not installed. Run with: uv run --with typst python paper/build_pdf.py")
    sys.exit(1)


def build_paper_pdf() -> Path:
    """Compiles paper.typ into a publication-ready PDF using the Typst compiler."""
    script_dir = Path(__file__).resolve().parent
    source_file = script_dir / "paper.typ"
    output_file = script_dir / "paper.pdf"

    if not source_file.exists():
        raise FileNotFoundError(f"[PaperBuilder-build_pdf] Source file not found: {source_file}")

    print(f"[PaperBuilder-build_pdf] Compiling {source_file.name} to {output_file.name}...")
    typst.compile(str(source_file), output=str(output_file))

    if not output_file.exists():
        raise RuntimeError("[PaperBuilder-build_pdf] PDF compilation failed: output file was not generated.")

    size_kb = output_file.stat().st_size / 1024
    print(f"[PaperBuilder-build_pdf] Success! Generated {output_file.name} ({size_kb:.1f} KB)")
    return output_file


if __name__ == "__main__":
    build_paper_pdf()
# endregion
