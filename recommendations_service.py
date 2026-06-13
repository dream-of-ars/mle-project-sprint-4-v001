from __future__ import annotations

import os
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Deque

import boto3
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

REC_PREFIX = "recsys/recommendations"
SPLIT_DATE = pd.Timestamp("2022-12-16")
MAX_ONLINE_HISTORY = 50
ONLINE_NEIGHBORS_PER_SEED = 3


class SourcesPayload(BaseModel):
    offline_count: int
    online_count: int
    fallback: str | None


class RecommendationPayload(BaseModel):
    user_id: int
    track_ids: list[int]
    limit: int
    strategy: str
    sources: SourcesPayload


class HealthPayload(BaseModel):
    status: str
    details: dict[str, int]
    error: str | None


class DebugUsersPayload(BaseModel):
    with_personal_no_online: int | None
    with_personal_and_online: int | None


@dataclass
class DataState:
    ready: bool = False
    error: str | None = None
    offline_recommendations: dict[int, list[int]] = field(default_factory=dict)
    personal_recommendations: dict[int, list[int]] = field(default_factory=dict)
    top_popular: list[int] = field(default_factory=list)
    similar_tracks: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    online_history: dict[int, list[int]] = field(default_factory=dict)
    with_personal_no_online: int | None = None
    with_personal_and_online: int | None = None


state = DataState()


def _build_s3_client():
    bucket = os.getenv("S3_BUCKET_NAME")
    key = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not bucket or not key or not secret:
        raise RuntimeError(
            "Не заданы переменные S3_BUCKET_NAME, AWS_ACCESS_KEY_ID или AWS_SECRET_ACCESS_KEY."
        )
    endpoint = os.getenv("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
    return bucket, boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def _try_get_local_file(filename: str) -> str | None:
    local_path = os.path.abspath(filename)
    if os.path.exists(local_path):
        return local_path

    try:
        bucket, s3 = _build_s3_client()
        key = f"{REC_PREFIX}/{filename}"
        s3.download_file(bucket, key, local_path)
        return local_path
    except Exception as exc:
        print(f"[service] {filename} не найден локально/S3, будет fallback: {exc}")
        return None


def _group_user_tracks(df: pd.DataFrame) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    sorted_df = df.sort_values(["user_id", "rank"])
    for user_id, group in sorted_df.groupby("user_id", sort=False):
        grouped[int(user_id)] = [int(track_id) for track_id in group["track_id"].tolist()]
    return grouped


def _load_online_history(events_path: str) -> dict[int, list[int]]:
    events_file = pq.ParquetFile(events_path)
    history: dict[int, Deque[int]] = defaultdict(lambda: deque(maxlen=MAX_ONLINE_HISTORY))

    for batch in events_file.iter_batches(
        batch_size=2_000_000, columns=["user_id", "track_id", "started_at"]
    ):
        chunk = batch.to_pandas()
        chunk = chunk[chunk["started_at"] >= SPLIT_DATE]
        if chunk.empty:
            continue
        users = chunk["user_id"].to_numpy()
        tracks = chunk["track_id"].to_numpy()
        for user_id, track_id in zip(users, tracks):
            history[int(user_id)].append(int(track_id))

    deduplicated: dict[int, list[int]] = {}
    for user_id, track_ids in history.items():
        seen: set[int] = set()
        unique_reversed: list[int] = []
        for track_id in reversed(track_ids):
            if track_id in seen:
                continue
            seen.add(track_id)
            unique_reversed.append(track_id)
        deduplicated[user_id] = list(reversed(unique_reversed))
    return deduplicated


def _dedupe_recent(track_ids: list[int], limit: int) -> list[int]:
    seen: set[int] = set()
    unique_reversed: list[int] = []
    for track_id in reversed(track_ids):
        if track_id in seen:
            continue
        seen.add(track_id)
        unique_reversed.append(track_id)
        if len(unique_reversed) >= limit:
            break
    return list(reversed(unique_reversed))


def _build_fallback_from_events(
    events_path: str,
) -> tuple[dict[int, list[int]], dict[int, list[int]], list[int], dict[int, list[int]]]:
    events_file = pq.ParquetFile(events_path)
    pre_history: dict[int, Deque[int]] = defaultdict(lambda: deque(maxlen=80))
    online_history: dict[int, Deque[int]] = defaultdict(lambda: deque(maxlen=MAX_ONLINE_HISTORY))
    popularity: defaultdict[int, int] = defaultdict(int)

    for batch in events_file.iter_batches(
        batch_size=2_000_000, columns=["user_id", "track_id", "started_at"]
    ):
        chunk = batch.to_pandas()
        pre_chunk = chunk[chunk["started_at"] < SPLIT_DATE]
        online_chunk = chunk[chunk["started_at"] >= SPLIT_DATE]

        if not pre_chunk.empty:
            for user_id, track_id in zip(pre_chunk["user_id"], pre_chunk["track_id"]):
                user = int(user_id)
                track = int(track_id)
                pre_history[user].append(track)
                popularity[track] += 1

        if not online_chunk.empty:
            for user_id, track_id in zip(online_chunk["user_id"], online_chunk["track_id"]):
                online_history[int(user_id)].append(int(track_id))

    personal: dict[int, list[int]] = {}
    offline: dict[int, list[int]] = {}
    for user_id, tracks in pre_history.items():
        deduped = _dedupe_recent(list(tracks), limit=30)
        if not deduped:
            continue
        personal[user_id] = deduped
        offline[user_id] = deduped[:10]

    popular_sorted = sorted(popularity.items(), key=lambda item: item[1], reverse=True)
    top_popular = [track_id for track_id, _ in popular_sorted[:100]]

    online_deduped = {
        user_id: _dedupe_recent(list(tracks), limit=MAX_ONLINE_HISTORY)
        for user_id, tracks in online_history.items()
    }

    return offline, personal, top_popular, online_deduped


def _fallback_similar_from_top_popular(top_popular: list[int]) -> dict[int, list[tuple[int, float]]]:
    similar_map: dict[int, list[tuple[int, float]]] = {}
    if not top_popular:
        return similar_map

    max_score = float(len(top_popular))
    for track_id in top_popular:
        candidates: list[tuple[int, float]] = []
        score = max_score
        for candidate in top_popular:
            if candidate == track_id:
                continue
            candidates.append((candidate, score))
            score -= 1.0
            if len(candidates) >= 10:
                break
        similar_map[track_id] = candidates
    return similar_map


def _load_data() -> None:
    if not os.path.exists("events.parquet"):
        raise RuntimeError("Не найден файл events.parquet в корне проекта.")

    recommendations_path = _try_get_local_file("recommendations.parquet")
    personal_path = _try_get_local_file("personal_als.parquet")
    similar_path = _try_get_local_file("similar.parquet")
    top_popular_path = _try_get_local_file("top_popular.parquet")

    need_fallback = (
        recommendations_path is None
        or personal_path is None
        or top_popular_path is None
    )

    fallback_offline: dict[int, list[int]] = {}
    fallback_personal: dict[int, list[int]] = {}
    fallback_top_popular: list[int] = []
    fallback_online_history: dict[int, list[int]] = {}
    if need_fallback:
        (
            fallback_offline,
            fallback_personal,
            fallback_top_popular,
            fallback_online_history,
        ) = _build_fallback_from_events("events.parquet")

    if recommendations_path is not None:
        recommendations_df = pd.read_parquet(
            recommendations_path, columns=["user_id", "track_id", "rank"]
        )
        state.offline_recommendations = _group_user_tracks(recommendations_df)
    else:
        state.offline_recommendations = fallback_offline

    if personal_path is not None:
        personal_df = pd.read_parquet(personal_path, columns=["user_id", "track_id", "rank"])
        state.personal_recommendations = _group_user_tracks(personal_df)
    else:
        state.personal_recommendations = fallback_personal

    if top_popular_path is not None:
        top_popular_df = pd.read_parquet(top_popular_path, columns=["track_id", "rank"])
        state.top_popular = [
            int(track_id)
            for track_id in top_popular_df.sort_values("rank")["track_id"].tolist()
        ]
    else:
        state.top_popular = fallback_top_popular

    if similar_path is not None:
        similar_df = pd.read_parquet(
            similar_path, columns=["track_id", "similar_track_id", "score"]
        ).sort_values(["track_id", "score"], ascending=[True, False])
        similar_map: dict[int, list[tuple[int, float]]] = {}
        for track_id, group in similar_df.groupby("track_id", sort=False):
            candidates = [
                (int(similar_id), float(score))
                for similar_id, score in zip(group["similar_track_id"], group["score"])
            ]
            similar_map[int(track_id)] = candidates
        state.similar_tracks = similar_map
    else:
        state.similar_tracks = _fallback_similar_from_top_popular(state.top_popular)

    if need_fallback:
        state.online_history = fallback_online_history
    else:
        state.online_history = _load_online_history("events.parquet")

    rec_users = set(state.offline_recommendations.keys())
    online_users = set(state.online_history.keys())
    no_online_users = rec_users - online_users
    mixed_users = rec_users & online_users
    state.with_personal_no_online = min(no_online_users) if no_online_users else None
    state.with_personal_and_online = min(mixed_users) if mixed_users else None


def _collect_online_candidates(user_id: int, online_seeds: int) -> list[int]:
    history = state.online_history.get(user_id, [])
    if not history or online_seeds == 0:
        return []

    seed_tracks = history[-online_seeds:]
    scores: defaultdict[int, float] = defaultdict(float)
    for seed_track_id in seed_tracks:
        neighbors = state.similar_tracks.get(seed_track_id, [])
        if not neighbors:
            neighbors = [(track_id, 1.0) for track_id in state.top_popular[:10]]
        for similar_id, score in neighbors[:ONLINE_NEIGHBORS_PER_SEED]:
            scores[similar_id] += score
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [track_id for track_id, _ in ordered]


def _offline_for_user(user_id: int) -> tuple[list[int], str | None]:
    if user_id in state.offline_recommendations:
        return state.offline_recommendations[user_id], None
    if user_id in state.personal_recommendations:
        return state.personal_recommendations[user_id], "personal_als"
    return state.top_popular, "top_popular"


def _merge_tracks(
    offline_candidates: list[int],
    online_candidates: list[int],
    history: list[int],
    limit: int,
) -> tuple[list[int], int, int]:
    result: list[int] = []
    used: set[int] = set()
    seen_history = set(history)
    offline_count = 0
    online_count = 0
    online_quota = min(len(online_candidates), int(round(limit * 0.3))) if limit > 0 else 0
    offline_quota = max(limit - online_quota, 0)

    def add_tracks(candidates: list[int], quota: int | None, source: str) -> None:
        nonlocal offline_count, online_count
        added = 0
        for track_id in candidates:
            if track_id in used or track_id in seen_history:
                continue
            used.add(track_id)
            result.append(track_id)
            if source == "offline":
                offline_count += 1
            else:
                online_count += 1
            added += 1
            if quota is not None and added >= quota:
                break
            if len(result) >= limit:
                break

    add_tracks(offline_candidates, offline_quota, "offline")
    if len(result) < limit:
        add_tracks(online_candidates, online_quota, "online")
    if len(result) < limit:
        add_tracks(offline_candidates, None, "offline")
    if len(result) < limit:
        add_tracks(state.top_popular, None, "offline")
    if len(result) < limit:
        add_tracks(online_candidates, None, "online")

    return result[:limit], offline_count, online_count


def _strategy_name(online_count: int, fallback: str | None) -> str:
    if online_count > 0:
        return "mixed"
    if fallback == "top_popular":
        return "popular_fallback"
    return "offline_only"


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv()
    try:
        _load_data()
        state.ready = True
        state.error = None
    except Exception as exc:
        state.ready = False
        state.error = str(exc)
    yield


app = FastAPI(title="Recommendations Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthPayload)
def health() -> HealthPayload:
    status = "ok" if state.ready else "error"
    return HealthPayload(
        status=status,
        details={
            "offline_users": len(state.offline_recommendations),
            "online_users": len(state.online_history),
            "top_popular_tracks": len(state.top_popular),
        },
        error=state.error,
    )


@app.get("/debug/users", response_model=DebugUsersPayload)
def debug_users() -> DebugUsersPayload:
    if not state.ready:
        raise HTTPException(status_code=503, detail=state.error or "Сервис не готов.")
    return DebugUsersPayload(
        with_personal_no_online=state.with_personal_no_online,
        with_personal_and_online=state.with_personal_and_online,
    )


@app.get("/recommendations/{user_id}", response_model=RecommendationPayload)
def recommendations(
    user_id: int,
    limit: int = Query(default=10, ge=1, le=100),
    online_seeds: int = Query(default=5, ge=0, le=20),
) -> RecommendationPayload:
    if not state.ready:
        raise HTTPException(status_code=503, detail=state.error or "Сервис не готов.")

    offline_candidates, fallback = _offline_for_user(user_id)
    history = state.online_history.get(user_id, [])
    online_candidates = _collect_online_candidates(user_id, online_seeds)

    track_ids, offline_count, online_count = _merge_tracks(
        offline_candidates=offline_candidates,
        online_candidates=online_candidates,
        history=history,
        limit=limit,
    )

    return RecommendationPayload(
        user_id=user_id,
        track_ids=track_ids,
        limit=limit,
        strategy=_strategy_name(online_count=online_count, fallback=fallback),
        sources=SourcesPayload(
            offline_count=offline_count,
            online_count=online_count,
            fallback=fallback,
        ),
    )
