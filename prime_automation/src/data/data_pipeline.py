from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.feather
from omegaconf import OmegaConf

from src.data.datatypes import Data
from src.settings import ID_COLUMN, TARGET_COLUMN, TEXT_COLUMN

COLUMNS = [ID_COLUMN, TEXT_COLUMN, TARGET_COLUMN, "split", "num_words", "num_targets"]
SCHEMA = pa.schema(
    [
        pa.field(ID_COLUMN, pa.int64()),
        pa.field(TEXT_COLUMN, pa.large_utf8()),
        pa.field(TARGET_COLUMN, pa.list_(pa.large_string())),
        pa.field("split", pa.large_string()),
        pa.field("num_words", pa.int64()),
        pa.field("num_targets", pa.int64()),
    ]
)


def get_code_system2code_counts(
    table: pa.Table, code_systems: list[str]
) -> dict[str, dict[str, int]]:
    """Count the occurrences of each code in each code system.

    Args:
        table (pa.Table): The dataset
        code_systems (list[str]): list of code systems to get counts for
    Returns:
        dict[str, dict[str, int]]: A dictionary with code systems as keys and a dictionary of code counts as values
    """
    code_system2code_counts = {}
    for col in code_systems:
        codes = table.column(col).combine_chunks().flatten().to_pylist()
        code_system2code_counts[col] = dict(Counter(codes))
    return code_system2code_counts


def data_pipeline(config: OmegaConf) -> Data:
    """The data pipeline.

    Args:
        config (OmegaConf): The config.

    Returns:
        Data: The data.
    """
    table = pyarrow.feather.read_table(
        Path(config.dir) / config.data_filename,
        columns=COLUMNS + list(config.code_column_names),
    )
    code_system2code_counts = get_code_system2code_counts(
        table, config.code_column_names
    )
    return Data(table.select(COLUMNS).cast(SCHEMA), code_system2code_counts)
