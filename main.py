import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.logging import RichHandler

from Core.configs.dataset_config import DatasetConfig, load_dataset_config
from Core.configs.system_config import SystemConfig, load_system_config
from Core.construct_index import (
    compute_mm_reranker,
    construct_GBC_index,
    construct_vdb,
    rebuild_graph_vdb,
)
from Core.inference import inference
from Core.provider.TokenTracker import TokenTracker


log = logging.getLogger(__name__)


# ============================================================================
# CONFIG / SECRET SAFETY
# ============================================================================

SENSITIVE_CONFIG_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "access_token",
    "auth_token",
    "password",
    "secret",
    "client_secret",
}


def redact_sensitive_data(value: Any) -> Any:
    """
    Recursively redact secrets before logging or saving config snapshots.

    This keeps runtime configuration intact while preventing API keys and
    similar credentials from being written to logs / YAML snapshots.
    """
    if isinstance(value, dict):
        redacted = {}

        for key, item in value.items():
            key_lower = str(key).lower()

            if key_lower in SENSITIVE_CONFIG_KEYS:
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = redact_sensitive_data(item)

        return redacted

    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]

    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)

    return value


def get_redacted_config_dict(config: SystemConfig) -> dict:
    """
    Convert a SystemConfig to a dictionary and redact secrets.
    """
    return redact_sensitive_data(config.model_dump())


def save_redacted_config_snapshot(
    config: SystemConfig,
    output_path: Path,
) -> None:
    """
    Save a reproducibility snapshot without exposing credentials.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.dump(
            get_redacted_config_dict(config),
            file,
            allow_unicode=True,
            default_flow_style=False,
        )


# ============================================================================
# CLI
# ============================================================================


def create_args():
    """
    Configure command-line arguments.

    Commands:
        index  -> build one or more BookRAG index stages
        rag    -> run RAG inference
    """

    parser = argparse.ArgumentParser(
        description=(
            "A command-line interface for building a BookRAG index "
            "or running RAG inference."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Global arguments.
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help=(
            "Path to the main system configuration file "
            "(for example config/self_correcting.yaml)."
        ),
    )

    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        required=False,
        help=(
            "(Optional) Path to the dataset configuration file "
            "for batch processing."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )

    parser.add_argument(
        "--nsplit",
        type=int,
        default=2,
        help="Total number of splits for dataset batch processing.",
    )

    # Default changed to 1 for safe local/single-machine use.
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Current split number (1-indexed) for dataset batch processing.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands",
    )

    # ------------------------------------------------------------------
    # index
    # ------------------------------------------------------------------

    parser_index = subparsers.add_parser(
        "index",
        help="Build BookRAG index components.",
    )

    parser_index.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "tree",
            "graph",
            "vdb",
            "all",
            "mm_reranker",
            "rebuild_graph_vdb",
        ],
        help=(
            "Index stage to execute:\n"
            "  tree              Build/save the document tree only.\n"
            "  graph             Build/save the knowledge graph.\n"
            "  vdb               Build/save the vector database.\n"
            "  all               Run tree, graph, and vdb stages.\n"
            "  mm_reranker       Build multimodal reranker embeddings.\n"
            "  rebuild_graph_vdb Rebuild graph vector database."
        ),
    )

    # ------------------------------------------------------------------
    # rag
    # ------------------------------------------------------------------

    parser_rag = subparsers.add_parser(
        "rag",
        help="Run RAG inference using a pre-built index.",
    )

    parser_rag.add_argument(
        "-q",
        "--query",
        type=str,
        help=(
            "(Optional) A single query to run. "
            "If using a dataset config, dataset questions are used."
        ),
    )

    return parser.parse_args()


# ============================================================================
# INDEXING
# ============================================================================


def build_index(
    config: SystemConfig,
    stage: str = "all",
    data_df: pd.DataFrame = None,
):
    """
    Build one selected part of the BookRAG index, or all stages.
    """

    log.info(
        " - build_index called. Indexing '%s' into '%s'",
        config.pdf_path,
        config.save_path,
    )

    # Stage 1: document tree.
    if stage in ["tree", "all"]:
        log.info(" - STAGE: Building Document Tree...")
        construct_GBC_index(
            config,
            tree_only=True,
        )

    # Stage 2: knowledge graph.
    if stage in ["graph", "all"]:
        log.info(" - STAGE: Building Knowledge Graph...")

        # graph_only=True means load the existing tree and build only graph.
        graph_only = stage == "graph"

        construct_GBC_index(
            config,
            graph_only=graph_only,
        )

    # Stage 3: vector database.
    if stage in ["vdb", "all"]:
        log.info(" - STAGE: Building Vector Database...")
        construct_vdb(config)

    if stage == "mm_reranker":
        log.info(" - STAGE: Building MM Reranker Embedding...")
        compute_mm_reranker(
            config,
            data_df,
        )

    if stage == "rebuild_graph_vdb":
        log.info(" - STAGE: Rebuilding Graph VDB...")
        rebuild_graph_vdb(config)


# ============================================================================
# INFERENCE
# ============================================================================


def run_inference(
    config: SystemConfig,
    data_df: pd.DataFrame,
    dataset_name: str,
):
    """
    Run BookRAG inference.

    Core.inference.inference() expects a DataFrame with a `question` column.
    """
    log.info(
        " - run_inference called. Using index from '%s'",
        config.save_path,
    )

    inference(
        cfg=config,
        data_df=data_df,
        dataset_name=dataset_name,
    )


# ============================================================================
# LOGGING
# ============================================================================


def setup_logging(
    save_path: str,
    config_to_log: SystemConfig,
):
    """
    Configure console + file logging.

    Secrets are redacted before configuration is written to the log.
    """

    log_dir = Path(save_path) / "logs"
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
        log_dir
        / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )

    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.INFO)

    console_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
    )

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="[%X]",
    )

    file_handler.setFormatter(
        file_formatter
    )

    root_logger.addHandler(
        console_handler
    )

    root_logger.addHandler(
        file_handler
    )

    setup_log = logging.getLogger(
        "LoggerSetup"
    )

    setup_log.info(
        "Logging initialized. Log file will be saved to: %s",
        log_file,
    )

    config_yaml_string = yaml.dump(
        get_redacted_config_dict(
            config_to_log
        ),
        allow_unicode=True,
        default_flow_style=False,
    )

    setup_log.info(
        "--- Starting Run with Configuration ---\n%s",
        config_yaml_string,
    )


# ============================================================================
# LOCAL RESOURCE POLICY
# ============================================================================


def process_resource(
    base_system_cfg: SystemConfig,
    args,
):
    """
    Preserve configuration values for this local CPU-oriented project.

    Upstream BookRAG contains server-specific CUDA overrides (for example
    cuda:3 / cuda:4). Those overrides are intentionally disabled here because
    this project is running on a laptop without a discrete NVIDIA GPU.

    Device/backend choices therefore come directly from the YAML config.
    """
    return base_system_cfg


# ============================================================================
# MAIN
# ============================================================================


def main():
    args = create_args()

    # This early log may occur before setup_logging; that is fine.
    log.info("--- Arguments Loaded ---")
    log.info("Main Config Path: %s", args.config)
    log.info("Selected Command: %s", args.command)
    log.info("------------------------\n")

    base_system_cfg: SystemConfig = load_system_config(
        args.config
    )

    token_tracker = TokenTracker.get_instance()
    token_tracker.reset()

    # Preserve local YAML resource/device choices.
    base_system_cfg = process_resource(
        base_system_cfg,
        args,
    )

    # =====================================================================
    # DATASET / BATCH MODE
    # =====================================================================

    if args.dataset_config:
        dataset_cfg: DatasetConfig = load_dataset_config(
            args.dataset_config
        )

        log.info(
            " - Loading dataset from: %s",
            dataset_cfg.dataset_path,
        )

        try:
            df = pd.read_json(
                dataset_cfg.dataset_path
            )

            print(
                f"Dataset shape: {df.shape}"
            )

        except FileNotFoundError:
            log.error(
                "ERROR: Dataset file not found at '%s'",
                dataset_cfg.dataset_path,
            )
            return

        except Exception as error:
            log.error(
                "ERROR: Failed to parse JSON file. Reason: %s",
                error,
            )
            return

        document_groups = df.groupby(
            [
                "doc_uuid",
                "doc_path",
            ]
        )

        print(
            f" - Found {len(document_groups)} "
            "unique documents in the dataset."
        )

        all_groups = list(
            document_groups
        )

        total_docs = len(
            all_groups
        )

        if (
            args.num > args.nsplit
            or args.num <= 0
        ):
            print(
                f" - ERROR: --num ({args.num}) "
                f"must be between 1 and "
                f"--nsplit ({args.nsplit})."
            )
            return

        items_per_split = math.ceil(
            total_docs / args.nsplit
        )

        start_index = (
            (args.num - 1)
            * items_per_split
        )

        end_index = min(
            start_index
            + items_per_split,
            total_docs,
        )

        docs_to_process = all_groups[
            start_index:end_index
        ]

        print(
            f" This worker "
            f"(split {args.num}/{args.nsplit}) "
            f"will process {len(docs_to_process)} "
            f"documents "
            f"(from index {start_index} "
            f"to {end_index - 1})."
        )

        index_error_list = []
        rag_error_list = []

        for (
            doc_uuid,
            doc_path,
        ), group in docs_to_process:

            current_config = (
                base_system_cfg.model_copy(
                    deep=True
                )
            )

            pdf_full_path = Path(
                doc_path
            )

            output_full_path = (
                Path(
                    dataset_cfg.working_dir
                )
                / str(doc_uuid)
            )

            current_config.pdf_path = str(
                pdf_full_path
            )

            current_config.save_path = str(
                output_full_path
            )

            setup_logging(
                save_path=(
                    current_config.save_path
                ),
                config_to_log=(
                    current_config
                ),
            )

            log.info(
                "--- Processing Document UUID: %s ---",
                doc_uuid,
            )

            log.info(
                " - PDF Path: %s",
                pdf_full_path,
            )

            log.info(
                " - Save Path: %s",
                output_full_path,
            )

            output_full_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            # -------------------------------------------------------------
            # INDEX
            # -------------------------------------------------------------

            if args.command == "index":
                config_snapshot_path = (
                    output_full_path
                    / "run_config.yaml"
                )

                save_redacted_config_snapshot(
                    current_config,
                    config_snapshot_path,
                )

                log.info(
                    " - Saved redacted index config snapshot to: %s",
                    config_snapshot_path,
                )

                try:
                    data_df = (
                        group.reset_index(
                            drop=True
                        )
                    )

                    build_index(
                        config=current_config,
                        stage=args.stage,
                        data_df=data_df,
                    )

                except Exception as error:
                    log.error(
                        " - ERROR: Failed to build index. Reason: %s",
                        error,
                    )

                    index_error_list.append(
                        (
                            doc_uuid,
                            str(error),
                        )
                    )

            # -------------------------------------------------------------
            # RAG
            # -------------------------------------------------------------

            elif args.command == "rag":
                rag_strategy = (
                    current_config
                    .rag
                    .strategy_config
                    .strategy
                )

                config_snapshot_filename = (
                    f"rag_config_{rag_strategy}.yaml"
                )

                config_snapshot_path = (
                    output_full_path
                    / config_snapshot_filename
                )

                save_redacted_config_snapshot(
                    current_config,
                    config_snapshot_path,
                )

                log.info(
                    " - Saved redacted RAG config snapshot to: %s",
                    config_snapshot_path,
                )

                dataset_name = (
                    dataset_cfg.dataset_name
                )

                data_df = (
                    group.reset_index(
                        drop=True
                    )
                )

                try:
                    run_inference(
                        config=current_config,
                        data_df=data_df,
                        dataset_name=(
                            dataset_name
                        ),
                    )

                except Exception as error:
                    log.error(
                        " - ERROR: Failed to run inference. Reason: %s",
                        error,
                    )

                    rag_error_list.append(
                        (
                            doc_uuid,
                            str(error),
                        )
                    )

        # -----------------------------------------------------------------
        # Batch error reports
        # -----------------------------------------------------------------

        base_dir = Path(
            __file__
        ).resolve().parent

        dataset_name = (
            dataset_cfg.dataset_name
        )

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        if index_error_list:
            error_log_path = (
                base_dir
                / (
                    f"{dataset_name}"
                    f"-index_error_split_"
                    f"{args.num}_{timestamp}.txt"
                )
            )

            with open(
                error_log_path,
                "w",
                encoding="utf-8",
            ) as file:
                for item in index_error_list:
                    file.write(
                        f"{item}\n"
                    )

            log.info(
                " - Indexing completed with errors. See %s for details.",
                error_log_path,
            )

        if rag_error_list:
            error_log_path = (
                base_dir
                / (
                    f"{dataset_name}"
                    f"-rag_error_split_"
                    f"{args.num}_{timestamp}.txt"
                )
            )

            with open(
                error_log_path,
                "w",
                encoding="utf-8",
            ) as file:
                for item in rag_error_list:
                    file.write(
                        f"{item}\n"
                    )

            log.info(
                " - RAG completed with errors. See %s for details.",
                error_log_path,
            )

        log.info(
            "--- All documents in this split processed. ---\n"
        )

        return

    # =====================================================================
    # SINGLE FILE MODE
    # =====================================================================

    setup_logging(
        save_path=base_system_cfg.save_path,
        config_to_log=base_system_cfg,
    )

    log.info(
        "🚀 No dataset config provided. Starting SINGLE mode..."
    )

    log.info(
        "  - Using paths and settings from '%s'",
        args.config,
    )

    # ---------------------------------------------------------------------
    # SINGLE FILE INDEXING
    # ---------------------------------------------------------------------

    if args.command == "index":
        build_index(
            config=base_system_cfg,
            stage=args.stage,
        )

        return

    # ---------------------------------------------------------------------
    # SINGLE QUERY RAG
    # ---------------------------------------------------------------------

    if args.command == "rag":
        if not args.query:
            log.error(
                " - ERROR: RAG command in single mode requires a query."
            )

            log.error(
                " - Please provide one with the -q/--query argument."
            )

            return

        log.info(
            "  - Running inference on single query: '%s'",
            args.query,
        )

        # IMPORTANT:
        # Core.inference.inference() / run_rag() expects a DataFrame whose
        # records contain a "question" field.
        single_query_df = pd.DataFrame(
            [
                {
                    "question": (
                        args.query
                    )
                }
            ]
        )

        strategy_cfg = (
            base_system_cfg
            .rag
            .strategy_config
        )

        self_correcting_enabled = bool(
            getattr(
                strategy_cfg,
                "self_correcting_enabled",
                False,
            )
        )

        # Keep baseline and self-correcting result folders separate.
        dataset_name = (
            "single_query_sc_on"
            if self_correcting_enabled
            else "single_query_sc_off"
        )

        run_inference(
            config=base_system_cfg,
            data_df=single_query_df,
            dataset_name=dataset_name,
        )

        return


if __name__ == "__main__":
    main()
