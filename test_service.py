from __future__ import annotations

import os
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv

BASE_URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{BASE_URL}/health"
DEBUG_USERS_URL = f"{BASE_URL}/debug/users"
RECS_URL = f"{BASE_URL}/recommendations"
START_TIMEOUT_SEC = 300


def _wait_for_service() -> None:
    started_at = time.time()
    last_error = "нет ответа"
    while time.time() - started_at < START_TIMEOUT_SEC:
        try:
            response = requests.get(HEALTH_URL, timeout=10)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
            last_error = response.text
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Сервис не запустился за {START_TIMEOUT_SEC} секунд: {last_error}")


def _start_service() -> subprocess.Popen:
    env = os.environ.copy()
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "recommendations_service:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def _stop_service(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def _request_recommendations(user_id: int, **params) -> dict:
    response = requests.get(f"{RECS_URL}/{user_id}", params=params, timeout=30)
    if response.status_code != 200:
        raise AssertionError(
            f"Ожидали 200 для user_id={user_id}, получили {response.status_code}: {response.text}"
        )
    payload = response.json()
    track_ids = payload.get("track_ids", [])
    if len(track_ids) != len(set(track_ids)):
        raise AssertionError(f"В ответе есть дубликаты track_id для user_id={user_id}")
    if len(track_ids) > params.get("limit", 10):
        raise AssertionError(f"Ответ длиннее limit для user_id={user_id}")
    return payload


def _pick_users() -> tuple[int, int, int]:
    debug_response = requests.get(DEBUG_USERS_URL, timeout=15)
    if debug_response.status_code != 200:
        raise RuntimeError(f"Не удалось получить тестовых пользователей: {debug_response.text}")
    debug_payload = debug_response.json()
    no_online_user = debug_payload.get("with_personal_no_online")
    mixed_user = debug_payload.get("with_personal_and_online")
    if no_online_user is None:
        raise RuntimeError("Не найден пользователь с персональными рекомендациями без онлайн-истории")
    if mixed_user is None:
        raise RuntimeError("Не найден пользователь с персональными рекомендациями и онлайн-историей")
    return -1, int(no_online_user), int(mixed_user)


def run_tests() -> None:
    process = _start_service()
    try:
        _wait_for_service()
        without_personal, no_online_user, mixed_user = _pick_users()

        print(f"[CASE 1] user_id={without_personal}: без персональных рекомендаций")
        case_1 = _request_recommendations(without_personal, limit=10)
        assert case_1["sources"]["fallback"] == "top_popular", (
            "Для пользователя без персональных рекомендаций ожидаем fallback top_popular"
        )
        print(
            f"status=ok strategy={case_1['strategy']} online_count={case_1['sources']['online_count']} "
            f"tracks={case_1['track_ids'][:5]}"
        )

        print(f"[CASE 2] user_id={no_online_user}: персональные есть, онлайн-истории нет")
        case_2 = _request_recommendations(no_online_user, limit=10)
        assert case_2["strategy"] == "offline_only", "Ожидали offline_only для пользователя без онлайн-истории"
        assert case_2["sources"]["online_count"] == 0, "Для offline_only ожидаем online_count=0"
        print(
            f"status=ok strategy={case_2['strategy']} online_count={case_2['sources']['online_count']} "
            f"tracks={case_2['track_ids'][:5]}"
        )

        print(f"[CASE 3] user_id={mixed_user}: персональные есть и онлайн-история есть")
        case_3 = _request_recommendations(mixed_user, limit=10)
        case_3_offline_only = _request_recommendations(mixed_user, limit=10, online_seeds=0)
        assert case_3["strategy"] == "mixed", "Ожидали mixed для пользователя с онлайн-историей"
        assert case_3["sources"]["online_count"] > 0, "Для mixed ожидаем online_count>0"
        assert case_3["track_ids"] != case_3_offline_only["track_ids"], (
            "Смешанный ответ должен отличаться от чисто офлайн-выдачи"
        )
        print(
            f"status=ok strategy={case_3['strategy']} online_count={case_3['sources']['online_count']} "
            f"tracks={case_3['track_ids'][:5]}"
        )

        print("TEST_STATUS: PASS")
    finally:
        _stop_service(process)


if __name__ == "__main__":
    load_dotenv()
    run_tests()
