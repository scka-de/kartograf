from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

from .models import (
    CorpusItem,
    CoverageReport,
    Decision,
    EvalCase,
    EvalRunResult,
    Gap,
    GeneratedCase,
    RedundantEval,
    Region,
)

DATA_DIR = Path("data")
AUDITS_DIR = DATA_DIR / "audits"
DB_PATH = AUDITS_DIR / "cartograph.db"


def audit_dir(audit_id: str) -> Path:
    path = AUDITS_DIR / audit_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_db() -> None:
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS audits (
                id TEXT PRIMARY KEY,
                target_agent TEXT NOT NULL,
                corpus_source TEXT NOT NULL,
                corpus_size INTEGER,
                noise_fraction REAL,
                coverage_score REAL,
                eval_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                warnings_json TEXT
            );
            CREATE TABLE IF NOT EXISTS regions (
                audit_id TEXT REFERENCES audits(id),
                region_id TEXT,
                label TEXT,
                density REAL,
                heterogeneity REAL,
                eval_density REAL,
                member_count INTEGER,
                member_corpus_ids_json TEXT,
                exemplar_corpus_ids_json TEXT,
                eval_case_ids_json TEXT,
                centroid_json TEXT,
                centroid_path TEXT,
                PRIMARY KEY (audit_id, region_id)
            );
            CREATE TABLE IF NOT EXISTS corpus_items (
                audit_id TEXT REFERENCES audits(id),
                item_id TEXT,
                text TEXT,
                source TEXT,
                metadata_json TEXT,
                region_id TEXT,
                PRIMARY KEY (audit_id, item_id)
            );
            CREATE TABLE IF NOT EXISTS eval_cases (
                audit_id TEXT REFERENCES audits(id),
                case_id TEXT,
                source_path TEXT,
                content TEXT,
                raw_json TEXT,
                region_id TEXT,
                assignment_distance REAL,
                PRIMARY KEY (audit_id, case_id)
            );
            CREATE TABLE IF NOT EXISTS generated_cases (
                audit_id TEXT REFERENCES audits(id),
                case_id TEXT,
                region_id TEXT,
                content TEXT,
                raw_json TEXT,
                grounded_in_json TEXT,
                validation_status TEXT,
                validation_details_json TEXT,
                accepted BOOLEAN,
                PRIMARY KEY (audit_id, case_id)
            );
            CREATE TABLE IF NOT EXISTS gaps (
                audit_id TEXT REFERENCES audits(id),
                region_id TEXT,
                risk_score REAL,
                components_json TEXT,
                rank INTEGER,
                PRIMARY KEY (audit_id, region_id)
            );
            CREATE TABLE IF NOT EXISTS redundant_evals (
                audit_id TEXT REFERENCES audits(id),
                case_id TEXT,
                duplicate_of_case_id TEXT,
                similarity REAL,
                PRIMARY KEY (audit_id, case_id, duplicate_of_case_id)
            );
            CREATE TABLE IF NOT EXISTS eval_runs (
                audit_id TEXT REFERENCES audits(id),
                label TEXT,
                command_json TEXT,
                exit_code INTEGER,
                pass_count INTEGER,
                fail_count INTEGER,
                pass_rate REAL,
                stdout TEXT,
                stderr TEXT,
                duration_seconds REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (audit_id, label)
            );
            CREATE TABLE IF NOT EXISTS decisions (
                audit_id TEXT REFERENCES audits(id),
                step INTEGER,
                timestamp TIMESTAMP,
                action TEXT,
                inputs_json TEXT,
                outputs_json TEXT,
                reason TEXT,
                PRIMARY KEY (audit_id, step)
            );
            """
        )


def create_audit(audit_id: str, target_agent: str, corpus_source: str) -> None:
    init_db()
    audit_dir(audit_id)
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO audits
            (id, target_agent, corpus_source, corpus_size, noise_fraction, coverage_score,
             eval_count, warnings_json)
            VALUES (?, ?, ?, 0, 0, 0, 0, ?)
            """,
            (audit_id, target_agent, corpus_source, "[]"),
        )


def save_corpus_items(audit_id: str, items: list[CorpusItem]) -> None:
    init_db()
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO corpus_items
            (audit_id, item_id, text, source, metadata_json, region_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (audit_id, item.id, item.text, item.source, _json(item.metadata), item.region_id)
                for item in items
            ],
        )
        conn.execute("UPDATE audits SET corpus_size=? WHERE id=?", (len(items), audit_id))


def load_corpus_items(audit_id: str) -> list[CorpusItem]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT item_id, text, source, metadata_json, region_id
            FROM corpus_items WHERE audit_id=? ORDER BY item_id
            """,
            (audit_id,),
        ).fetchall()
    return [
        CorpusItem(
            id=row["item_id"],
            text=row["text"],
            source=row["source"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            region_id=row["region_id"],
        )
        for row in rows
    ]


def save_corpus_embeddings(
    audit_id: str,
    full: np.ndarray,
    reduced_30d: np.ndarray,
    reduced_2d: np.ndarray,
) -> None:
    path = audit_dir(audit_id)
    np.save(path / "embeddings_corpus_full.npy", full)
    np.save(path / "embeddings_corpus_30d.npy", reduced_30d)
    np.save(path / "embeddings_corpus_2d.npy", reduced_2d)


def save_eval_embeddings(audit_id: str, full: np.ndarray, reduced_30d: np.ndarray) -> None:
    path = audit_dir(audit_id)
    np.save(path / "embeddings_evals_full.npy", full)
    np.save(path / "embeddings_evals_30d.npy", reduced_30d)


def save_generated_embeddings(audit_id: str, full: np.ndarray, reduced_30d: np.ndarray) -> None:
    path = audit_dir(audit_id)
    np.save(path / "embeddings_generated_full.npy", full)
    np.save(path / "embeddings_generated_30d.npy", reduced_30d)


def save_cluster_labels(audit_id: str, labels: np.ndarray) -> None:
    np.save(audit_dir(audit_id) / "cluster_labels.npy", labels)


def save_reducer(audit_id: str, reducer: Any, kind: Literal["30d", "2d"]) -> None:
    path = audit_dir(audit_id) / f"reducer_{kind}.joblib"
    if joblib is not None:
        joblib.dump(reducer, path)
    else:
        import pickle

        with path.open("wb") as handle:
            pickle.dump(reducer, handle)


def load_reducer(audit_id: str, kind: Literal["30d", "2d"]) -> Any:
    path = audit_dir(audit_id) / f"reducer_{kind}.joblib"
    if joblib is not None:
        return joblib.load(path)
    import pickle

    with path.open("rb") as handle:
        return pickle.load(handle)


def save_regions(audit_id: str, regions: list[Region]) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM regions WHERE audit_id=?", (audit_id,))
        for region in regions:
            centroid_path = audit_dir(audit_id) / f"{region.id}_centroid.npy"
            np.save(centroid_path, np.asarray(region.centroid, dtype=float))
            conn.execute(
                """
                INSERT INTO regions
                (audit_id, region_id, label, density, heterogeneity, eval_density,
                 member_count, member_corpus_ids_json, exemplar_corpus_ids_json,
                 eval_case_ids_json, centroid_json, centroid_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    region.id,
                    region.label,
                    region.density,
                    region.heterogeneity,
                    region.eval_density,
                    region.member_count,
                    _json(region.member_corpus_ids),
                    _json(region.exemplar_corpus_ids),
                    _json(region.eval_case_ids),
                    _json(region.centroid),
                    str(centroid_path),
                ),
            )


def load_regions(audit_id: str) -> list[Region]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM regions WHERE audit_id=? ORDER BY region_id",
            (audit_id,),
        ).fetchall()
    return [_region_from_row(row) for row in rows]


def save_eval_cases(audit_id: str, cases: list[EvalCase]) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM eval_cases WHERE audit_id=?", (audit_id,))
        conn.executemany(
            """
            INSERT INTO eval_cases
            (audit_id, case_id, source_path, content, raw_json, region_id, assignment_distance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    audit_id,
                    case.id,
                    case.source_path,
                    case.content,
                    _json(case.raw),
                    case.region_id,
                    case.assignment_distance,
                )
                for case in cases
            ],
        )
        conn.execute("UPDATE audits SET eval_count=? WHERE id=?", (len(cases), audit_id))


def load_eval_cases(audit_id: str) -> list[EvalCase]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_cases WHERE audit_id=? ORDER BY case_id",
            (audit_id,),
        ).fetchall()
    return [
        EvalCase(
            id=row["case_id"],
            source_path=row["source_path"],
            content=row["content"],
            raw=json.loads(row["raw_json"] or "{}"),
            region_id=row["region_id"],
            assignment_distance=row["assignment_distance"],
        )
        for row in rows
    ]


def save_gaps(audit_id: str, gaps: list[Gap]) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM gaps WHERE audit_id=?", (audit_id,))
        conn.executemany(
            """
            INSERT INTO gaps
            (audit_id, region_id, risk_score, components_json, rank)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (audit_id, gap.region_id, gap.risk_score, _json(gap.components), gap.rank)
                for gap in gaps
            ],
        )


def save_redundant_evals(audit_id: str, redundant: list[RedundantEval]) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM redundant_evals WHERE audit_id=?", (audit_id,))
        conn.executemany(
            """
            INSERT INTO redundant_evals
            (audit_id, case_id, duplicate_of_case_id, similarity) VALUES (?, ?, ?, ?)
            """,
            [
                (audit_id, item.case_id, item.duplicate_of_case_id, item.similarity)
                for item in redundant
            ],
        )


def save_generated_cases(audit_id: str, cases: list[GeneratedCase]) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM generated_cases WHERE audit_id=?", (audit_id,))
        conn.executemany(
            """
            INSERT INTO generated_cases
            (audit_id, case_id, region_id, content, raw_json, grounded_in_json,
             validation_status, validation_details_json, accepted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    audit_id,
                    case.id,
                    case.region_id,
                    case.content,
                    _json(case.raw),
                    _json(case.grounded_in_corpus_ids),
                    case.validation_status,
                    _json(case.validation_details),
                    int(case.accepted),
                )
                for case in cases
            ],
        )


def save_eval_run(audit_id: str, result: EvalRunResult) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_runs
            (audit_id, label, command_json, exit_code, pass_count, fail_count, pass_rate,
             stdout, stderr, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                result.label,
                _json(result.command),
                result.exit_code,
                result.pass_count,
                result.fail_count,
                result.pass_rate,
                result.stdout,
                result.stderr,
                result.duration_seconds,
            ),
        )


def save_decision(audit_id: str, decision: Decision) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO decisions
            (audit_id, step, timestamp, action, inputs_json, outputs_json, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                decision.step,
                decision.timestamp.isoformat(),
                decision.action,
                _json(decision.inputs),
                _json(decision.outputs),
                decision.reason,
            ),
        )


def update_audit_summary(
    audit_id: str,
    coverage_score: float,
    noise_fraction: float,
    warnings: list[str],
    completed: bool = False,
) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE audits SET coverage_score=?, noise_fraction=?, warnings_json=?,
            completed_at=CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE completed_at END
            WHERE id=?
            """,
            (coverage_score, noise_fraction, _json(warnings), int(completed), audit_id),
        )


def latest_audit_id() -> str | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT id FROM audits ORDER BY created_at DESC LIMIT 1").fetchone()
    return None if row is None else str(row["id"])


def list_audits(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audits ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) | {"warnings": json.loads(row["warnings_json"] or "[]")} for row in rows]


def load_coverage_report(audit_id: str) -> CoverageReport:
    init_db()
    with _connect() as conn:
        audit = conn.execute("SELECT * FROM audits WHERE id=?", (audit_id,)).fetchone()
        if audit is None:
            raise KeyError(f"unknown audit_id: {audit_id}")
        gaps = [
            Gap(
                region_id=row["region_id"],
                risk_score=row["risk_score"],
                components=json.loads(row["components_json"] or "{}"),
                rank=row["rank"],
            )
            for row in conn.execute(
                "SELECT * FROM gaps WHERE audit_id=? ORDER BY rank", (audit_id,)
            )
        ]
        redundant = [
            RedundantEval(
                case_id=row["case_id"],
                duplicate_of_case_id=row["duplicate_of_case_id"],
                similarity=row["similarity"],
            )
            for row in conn.execute("SELECT * FROM redundant_evals WHERE audit_id=?", (audit_id,))
        ]
        generated = [
            GeneratedCase(
                id=row["case_id"],
                region_id=row["region_id"],
                content=row["content"],
                raw=json.loads(row["raw_json"] or "{}"),
                grounded_in_corpus_ids=json.loads(row["grounded_in_json"] or "[]"),
                validation_status=row["validation_status"],
                validation_details=json.loads(row["validation_details_json"] or "{}"),
                accepted=bool(row["accepted"]),
            )
            for row in conn.execute("SELECT * FROM generated_cases WHERE audit_id=?", (audit_id,))
        ]
        eval_runs = {
            row["label"]: EvalRunResult(
                label=row["label"],
                command=json.loads(row["command_json"] or "[]"),
                exit_code=row["exit_code"],
                pass_count=row["pass_count"],
                fail_count=row["fail_count"],
                pass_rate=row["pass_rate"],
                stdout=row["stdout"] or "",
                stderr=row["stderr"] or "",
                duration_seconds=row["duration_seconds"] or 0.0,
            )
            for row in conn.execute("SELECT * FROM eval_runs WHERE audit_id=?", (audit_id,))
        }
        decisions = [
            Decision(
                step=row["step"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                action=row["action"],
                inputs=json.loads(row["inputs_json"] or "{}"),
                outputs=json.loads(row["outputs_json"] or "{}"),
                reason=row["reason"],
            )
            for row in conn.execute(
                "SELECT * FROM decisions WHERE audit_id=? ORDER BY step", (audit_id,)
            )
        ]

    return CoverageReport(
        audit_id=audit_id,
        target_agent=audit["target_agent"],
        corpus_source=audit["corpus_source"],
        corpus_size=audit["corpus_size"] or 0,
        noise_fraction=audit["noise_fraction"] or 0.0,
        eval_count=audit["eval_count"] or 0,
        coverage_score=audit["coverage_score"] or 0.0,
        regions=load_regions(audit_id),
        gaps=gaps,
        redundant_evals=redundant,
        generated_cases=generated,
        eval_run_before=eval_runs.get("before"),
        eval_run_after=eval_runs.get("after"),
        decisions=decisions,
        warnings=json.loads(audit["warnings_json"] or "[]"),
        created_at=datetime.fromisoformat(str(audit["created_at"])),
    )


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _json(value: object) -> str:
    return json.dumps(value, default=str)


def _region_from_row(row: sqlite3.Row) -> Region:
    return Region(
        id=row["region_id"],
        label=row["label"],
        centroid=json.loads(row["centroid_json"]),
        density=row["density"],
        heterogeneity=row["heterogeneity"],
        member_count=row["member_count"],
        member_corpus_ids=json.loads(row["member_corpus_ids_json"] or "[]"),
        exemplar_corpus_ids=json.loads(row["exemplar_corpus_ids_json"] or "[]"),
        eval_case_ids=json.loads(row["eval_case_ids_json"] or "[]"),
        eval_density=row["eval_density"] or 0.0,
    )
