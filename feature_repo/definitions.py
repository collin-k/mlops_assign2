"""Feast feature definitions."""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"

TTL = timedelta(days=3650)

athlete = Entity(
    name="athlete",
    join_keys=["athlete_id"],
    description="A unique CrossFit Open competitor.",
)

# V1
v1_source = FileSource(
    name="athlete_features_v1_source",
    path=str(DATA_DIR / "athlete_features_v1.parquet"),
    timestamp_field="event_timestamp",
)

athlete_features_v1 = FeatureView(
    name="athlete_features_v1",
    entities=[athlete],
    ttl=TTL,
    schema=[
        Field(name="age", dtype=Float64),
        Field(name="height", dtype=Float64),
        Field(name="weight", dtype=Float64),
        Field(name="gender_male", dtype=Int64),
    ],
    source=v1_source,
    online=True,
    tags={"version": "v1", "description": "baseline"},
)

# V2
v2_source = FileSource(
    name="athlete_features_v2_source",
    path=str(DATA_DIR / "athlete_features_v2.parquet"),
    timestamp_field="event_timestamp",
)

athlete_features_v2 = FeatureView(
    name="athlete_features_v2",
    entities=[athlete],
    ttl=TTL,
    schema=[
        Field(name="age", dtype=Float64),
        Field(name="height", dtype=Float64),
        Field(name="weight", dtype=Float64),
        Field(name="gender_male", dtype=Int64),
        Field(name="bmi", dtype=Float64),
        Field(name="weight_to_height", dtype=Float64),
        Field(name="age_bucket", dtype=Int64),
        Field(name="is_experienced", dtype=Int64),
    ],
    source=v2_source,
    online=True,
    tags={"version": "v2", "description": "base + engineered features"},
)
