"""Command-line entry point:  python -m geneweaver scan genome.fa --guide ... """
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple


def read_guides(args) -> Tuple[List[str], List[str]]:
    names, seqs = [], []
    for i, g in enumerate(args.guide or [], 1):
        if ":" in g:
            n, s = g.split(":", 1)
        else:
            n, s = f"guide{i}", g
        names.append(n)
        seqs.append(s.strip().upper())
    if args.guides_file:
        with open(args.guides_file) as fh:
            for j, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 1:
                    names.append(f"guide_f{j}")
                    seqs.append(parts[0].upper())
                else:
                    names.append(parts[0])
                    seqs.append(parts[1].upper())
    return names, seqs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="geneweaver", description="GPU-accelerated CRISPR off-target search")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan a FASTA for off-target sites")
    s.add_argument("fasta", help="genome FASTA (.fa/.fasta, optionally .gz)")
    s.add_argument("-g", "--guide", action="append", metavar="[NAME:]SEQ",
                   help="guide RNA/DNA sequence (20 nt, no PAM); repeatable")
    s.add_argument("--guides-file", help="text file: 'SEQ' or 'NAME SEQ' per line")
    s.add_argument("--pam", default="NGG", help="PAM motif, IUPAC (default NGG = SpCas9)")
    s.add_argument("-m", "--max-mismatch", type=int, default=4)
    s.add_argument("-o", "--out", default="geneweaver_hits.tsv", help="TSV output path")
    s.add_argument("--chunk-size", type=int, default=32_000_000, help="windows per GPU chunk")
    s.add_argument("--backend", choices=["auto", "cuda", "cpu"], default="auto")
    s.add_argument("--cudasim", action="store_true",
                   help="run CUDA kernels in Numba's CPU simulator (no GPU needed; very slow, for testing)")
    s.add_argument("--scheduler", choices=["dask", "sync"], default="dask")
    s.add_argument("--workers", type=int, help="Dask worker processes (default: 1 per GPU / CPU core)")
    s.add_argument("--dask-address", help="connect to an existing Dask scheduler, e.g. tcp://host:8786")
    s.add_argument("--no-tui", action="store_true", help="plain text progress instead of the dashboard")
    s.add_argument("--exit-when-done", action="store_true", help="close the dashboard automatically when finished")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Must happen before numba is imported anywhere (also inherited by Dask workers).
    if args.cudasim:
        os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

    from .engine import QuerySpec
    from .scheduler import ScanConfig, run_scan

    names, seqs = read_guides(args)
    if not seqs:
        print("error: provide at least one --guide or --guides-file", file=sys.stderr)
        return 2
    try:
        spec = QuerySpec(names=names, guides=seqs, max_mismatch=args.max_mismatch, pam=args.pam)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not os.path.exists(args.fasta):
        print(f"error: FASTA not found: {args.fasta}", file=sys.stderr)
        return 2

    cfg = ScanConfig(fasta=args.fasta, spec=spec, chunk_size=args.chunk_size, backend=args.backend,
                     mode=args.scheduler, n_workers=args.workers, dask_address=args.dask_address)

    if args.no_tui:
        return _run_plain(cfg, args.out)

    from .tui import GeneWeaverApp
    cluster = None
    if cfg.mode == "dask" and not cfg.dask_address:
        # Start workers before Textual grabs the terminal (see start_local_cluster).
        from .engine import detect_backend
        from .scheduler import start_local_cluster
        print("Starting Dask workers ...", file=sys.stderr)
        cluster = start_local_cluster(detect_backend(cfg.backend), cfg.n_workers)
        cfg.dask_address = cluster.scheduler_address
    try:
        app = GeneWeaverApp(cfg, out_path=args.out, auto_exit=args.exit_when_done)
        app.run()
    finally:
        if cluster is not None:
            cluster.close()
    return 1 if app.last_error else 0


def _run_plain(cfg, out_path: str) -> int:
    from .results import write_tsv
    from .scheduler import run_scan

    state = {"done": 0, "total": None}

    def on_event(kind, payload):
        if kind == "status":
            print(payload, file=sys.stderr)
        elif kind == "start":
            state["total"] = payload.get("total_windows")
            print(f"backend={payload['backend']} windows={state['total']}", file=sys.stderr)
        elif kind == "chunk":
            state["done"] += payload.n_windows
            pct = f"{100 * state['done'] / state['total']:.1f}%" if state["total"] else "?"
            print(f"  chunk {payload.chunk_index} {payload.contig}:{payload.offset} "
                  f"hits={payload.total_hits} [{pct}]", file=sys.stderr)

    summary = run_scan(cfg, on_event=on_event)
    n = write_tsv(out_path, summary.hits)
    print(f"{summary.windows_scanned:,} bp in {summary.seconds:.2f}s "
          f"({summary.throughput / 1e6:.2f} Mbp/s, backend={summary.backend}); "
          f"{summary.total_hits:,} hits -> {out_path} ({n} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
